"""
Training loop for Late Disambiguation Lag experiments.

Handles:
- Training with AdamW / SGD and warmup
- Periodic logging with gradient norms and optimizer-aware dissipation metrics
- Callback hooks for experiment-specific instrumentation
- Checkpointing
"""

import math
import os
from pathlib import Path
from typing import Callable, Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from tqdm import tqdm

from .checkpoint import save_checkpoint, get_checkpoint_dir
from ..data import CharTokenizer, MappingData


@dataclass
class TrainingMetrics:
    """Container for training metrics at a single step."""
    step: int
    train_loss: float
    train_accuracy: Optional[float] = None
    learning_rate: float = 0.0


@dataclass
class TrainingCallbacks:
    """Optional hooks invoked at specific points in the training loop.

    Each callback receives the current training state and can record
    arbitrary metrics.  Return values are ignored.
    """
    on_after_backward: Optional[Callable] = None
    on_after_step: Optional[Callable] = None
    on_checkpoint: Optional[Callable] = None


def shuffle_z_in_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Create a corrupted batch where z tokens are shuffled across the batch dimension.
    
    This preserves B and A but randomizes which z selector each example sees.
    Used to test whether the model is actually using z information.
    """
    input_ids = batch["input_ids"].clone()
    z_positions = batch["z_positions"]
    z_end_positions = batch["z_end_positions"]
    
    batch_size = input_ids.shape[0]
    device = input_ids.device
    
    # Generate a random permutation that guarantees no element stays in place.
    # This ensures every example's z is mismatched (true corruption signal).
    perm = torch.randperm(batch_size, device=device)
    # Fix any fixed points (where perm[i] == i)
    for i in range(batch_size):
        if perm[i] == i:
            # Swap with next element (wrapping around)
            j = (i + 1) % batch_size
            perm[i], perm[j] = perm[j].clone(), perm[i].clone()
    
    # Gather z tokens from permuted indices so we can swap them cleanly.
    z_tokens_original = []
    for i in range(batch_size):
        z_start = z_positions[i].item()
        z_end = z_end_positions[i].item()
        z_tokens_original.append(input_ids[i, z_start:z_end].clone())
    
    # Apply permutation: example i gets z from example perm[i].
    # This keeps B/A fixed but breaks the correct z selector.
    for i in range(batch_size):
        z_start = z_positions[i].item()
        z_end = z_end_positions[i].item()
        src_idx = perm[i].item()
        input_ids[i, z_start:z_end] = z_tokens_original[src_idx]
    
    # Return new batch with shuffled z (other fields unchanged).
    # This is used as a read-only diagnostic (no gradients).
    shuffled_batch = batch.copy()
    shuffled_batch["input_ids"] = input_ids
    return shuffled_batch


def compute_loss(
    model: HookedTransformer,
    batch: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, float, float]:
    """
    Compute cross-entropy loss for next-token prediction.
    
    Only computes loss on target tokens (where labels != -100).
    
    Returns:
        (loss tensor for backprop, accuracy float, first_target_loss float)
    """
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    
    # Forward pass
    logits = model(input_ids)
    
    # Shift for next-token prediction
    # logits: (batch, seq, vocab) -> predict position i+1 from position i
    # labels: (batch, seq) -> target at each position
    
    # We want: logits at position i predicts token at position i+1
    # So shift logits left by 1, labels right by 1... actually,
    # TransformerLens returns logits where logits[i] predicts token[i+1]
    # So we need: loss between logits[:, :-1] and labels[:, 1:]
    
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    
    # Flatten
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)
    
    # Compute loss (ignoring -100 positions)
    loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
    
    # Compute accuracy on non-ignored positions, excluding EOS
    eos_id = 2  # tokenizer.eos_token_id (pad=0, bos=1, eos=2, sep=3)
    mask = (shift_labels != -100) & (shift_labels != eos_id)
    if mask.sum() > 0:
        predictions = shift_logits.argmax(dim=-1)
        correct = (predictions == shift_labels) & mask
        accuracy = correct.sum().float() / mask.sum().float()
    else:
        accuracy = torch.tensor(0.0)
    
    # Compute loss on the first target token only (sanity check metric)
    if "target_start_positions" in batch:
        target_start = batch["target_start_positions"].to(logits.device)
        batch_idx = torch.arange(logits.size(0), device=logits.device)
        pred_pos = target_start - 1  # predicts token at target_start
        valid = pred_pos >= 0
        if valid.any():
            logits_first = logits[batch_idx[valid], pred_pos[valid]]
            labels_first = labels[batch_idx[valid], target_start[valid]]
            first_target_loss = F.cross_entropy(logits_first, labels_first)
        else:
            first_target_loss = torch.tensor(0.0, device=logits.device)
    else:
        first_target_loss = torch.tensor(0.0, device=logits.device)
    
    return loss, accuracy.item(), first_target_loss.item()


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    scheduler_type: str = "cosine",
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Create learning rate scheduler.
    
    Args:
        scheduler_type: "cosine" for linear warmup + cosine decay,
                        "constant" for flat LR (no warmup, no decay).
    """
    if scheduler_type == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    
    # Default: linear warmup + cosine decay
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        else:
            # Cosine decay
            progress = (step - warmup_steps) / (max_steps - warmup_steps)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(
    model: HookedTransformer,
    train_loader: DataLoader,
    probe_loader: DataLoader,
    cfg,
    output_dir: Path,
    optimizer_type: str = "adamw",
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
    grad_clip: Optional[float] = 1.0,
    callbacks: Optional[TrainingCallbacks] = None,
    record_dissipation: bool = False,
    mapping_data: Optional[MappingData] = None,
    tokenizer: Optional[CharTokenizer] = None,
    candidate_eval_n: int = 32,
) -> Dict[str, Any]:
    """
    Main training loop.

    Args:
        model: HookedTransformer (or compatible) to train
        train_loader: Training data loader
        probe_loader: Probe data loader (held-out subset for analysis)
        cfg: Hydra config
        output_dir: Directory for outputs
        optimizer_type: "adamw" or "sgd"
        optimizer_kwargs: Extra kwargs forwarded to the optimizer constructor
            (e.g. ``{"momentum": 0.9}`` for SGD).
        grad_clip: Max gradient norm for clipping.  ``None`` disables clipping.
        callbacks: Optional :class:`TrainingCallbacks` with hooks.
        record_dissipation: If ``True``, record per-step Q_work, Q_update,
            and eta_eff for optimizer-aware dissipation analysis (Exp 4).
        mapping_data: If provided (along with ``tokenizer``), candidate-
            normalized loss and accuracy are evaluated every ``eval_every``
            steps instead of first-target loss.
        tokenizer: Required when ``mapping_data`` is provided.
        candidate_eval_n: Number of B groups to sample for candidate eval.

    Returns:
        Training history dict
    """
    device = model.cfg.device
    if callbacks is None:
        callbacks = TrainingCallbacks()
    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    _do_candidate_eval = mapping_data is not None and tokenizer is not None
    if _do_candidate_eval:
        from ..analysis.candidate_eval import run_candidate_eval

    # ---- Setup optimizer ----
    base_optim_kwargs = dict(
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    if optimizer_type == "sgd":
        base_optim_kwargs.pop("weight_decay", None)
        base_optim_kwargs.update(optimizer_kwargs)
        optimizer = torch.optim.SGD(model.parameters(), **base_optim_kwargs)
    else:
        base_optim_kwargs.update(optimizer_kwargs)
        optimizer = torch.optim.AdamW(model.parameters(), **base_optim_kwargs)

    # ---- Setup scheduler ----
    scheduler_type = getattr(cfg.training, "scheduler", "cosine")
    scheduler = get_lr_scheduler(
        optimizer,
        cfg.training.warmup_steps,
        cfg.training.max_steps,
        scheduler_type=scheduler_type,
    )

    # ---- Checkpointing ----
    checkpoint_dir = get_checkpoint_dir(str(output_dir), cfg.experiment.name)

    # ---- Training history ----
    history: Dict[str, List] = {
        "train_loss": [],
        "train_accuracy": [],
        "first_target_loss": [],
        "loss_z_shuffled": [],
        "grad_norm_sq": [],
        "grad_norm_sq_clipped": [],
        "clipping_active": [],
        "steps": [],
    }
    if _do_candidate_eval:
        history["candidate_loss"] = []
        history["candidate_accuracy"] = []
    if record_dissipation:
        history["q_work"] = []
        history["q_update"] = []
        history["eta_eff"] = []

    _first_z_shuffle_logged = False

    # ---- Training loop ----
    model.train()
    step = 0
    epoch = 0
    running_loss = 0.0
    running_acc = 0.0
    running_first_loss = 0.0
    running_grad_norm_sq = 0.0
    running_grad_norm_sq_clipped = 0.0
    running_clip_count = 0
    n_batches = 0

    # Dissipation accumulators (running sums for the current eval window)
    running_q_work = 0.0
    running_q_update = 0.0
    running_eta_eff_sum = 0.0

    # Early stopping: stop when candidate_loss < 1% of log(K) for 20
    # consecutive evals (1000 steps).  This is strictly more conservative
    # than the 5% analysis threshold — all measurements (τ, success/fail)
    # are locked in before this fires.  Only active when candidate eval is
    # enabled and early_stop_threshold is set in the config.
    _early_stop_threshold = None
    _early_stop_patience = 20  # consecutive evals below threshold
    _early_stop_counter = 0
    _early_stopped = False
    if _do_candidate_eval and hasattr(cfg.data, "k"):
        _es_frac = getattr(cfg.training, "early_stop_convergence_frac", None)
        if _es_frac is not None:
            _early_stop_threshold = _es_frac * math.log(cfg.data.k)

    pbar = tqdm(total=cfg.training.max_steps, desc="Training")

    while step < cfg.training.max_steps and not _early_stopped:
        epoch += 1

        for batch in train_loader:
            if step >= cfg.training.max_steps or _early_stopped:
                break

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Forward + backward
            optimizer.zero_grad()
            loss, train_acc, first_target_loss = compute_loss(model, batch)
            current_train_acc = train_acc
            loss.backward()

            # ---- Gradient norm (pre-clip) ----
            _grad_norm_sq = sum(
                p.grad.data.norm() ** 2 for p in model.parameters() if p.grad is not None
            ).item()

            # ---- Callback: after backward (before clip & step) ----
            if callbacks.on_after_backward is not None:
                callbacks.on_after_backward(model=model, batch=batch,
                                            optimizer=optimizer, step=step,
                                            grad_norm_sq=_grad_norm_sq)

            # ---- Dissipation: snapshot grads & params before step ----
            if record_dissipation:
                _grads = [p.grad.data.clone() for p in model.parameters() if p.grad is not None]
                _params_before = [p.data.clone() for p in model.parameters()]

            # ---- Gradient clipping ----
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                _grad_norm_sq_clipped = sum(
                    p.grad.data.norm() ** 2 for p in model.parameters() if p.grad is not None
                ).item()
                _clip_active = _grad_norm_sq_clipped < _grad_norm_sq * 0.999
            else:
                _grad_norm_sq_clipped = _grad_norm_sq
                _clip_active = False

            optimizer.step()
            scheduler.step()

            # ---- Dissipation: compute per-step quantities ----
            if record_dissipation:
                _params_after = [p.data.clone() for p in model.parameters()]
                _q_work_t = sum(
                    (g * (pa - pb)).sum()
                    for g, pa, pb in zip(_grads, _params_after, _params_before)
                ).item()
                _q_update_t = sum(
                    (pa - pb).norm() ** 2
                    for pa, pb in zip(_params_after, _params_before)
                ).item()
                _gnorm = math.sqrt(sum(g.norm() ** 2 for g in _grads).item() + 1e-30)
                _eta_eff_t = math.sqrt(_q_update_t) / _gnorm
                running_q_work += _q_work_t
                running_q_update += _q_update_t
                running_eta_eff_sum += _eta_eff_t
                del _grads, _params_before, _params_after

            # ---- Callback: after optimizer step ----
            if callbacks.on_after_step is not None:
                callbacks.on_after_step(model=model, batch=batch,
                                        optimizer=optimizer, step=step + 1)

            # ---- Accumulate running stats ----
            running_loss += loss.item()
            running_acc += train_acc
            running_first_loss += first_target_loss
            running_grad_norm_sq += _grad_norm_sq
            running_grad_norm_sq_clipped += _grad_norm_sq_clipped
            running_clip_count += int(_clip_active)
            n_batches += 1
            step += 1

            # ---- Periodic logging ----
            if step % cfg.training.eval_every == 0:
                avg_train_loss = running_loss / n_batches
                avg_train_acc = running_acc / n_batches
                avg_first_loss = running_first_loss / n_batches
                avg_grad_norm_sq = running_grad_norm_sq / n_batches
                avg_grad_norm_sq_clipped = running_grad_norm_sq_clipped / n_batches
                frac_clipped = running_clip_count / n_batches

                # Z-shuffle diagnostic (read-only, no gradients)
                with torch.no_grad():
                    model.eval()
                    shuffled_batch = shuffle_z_in_batch(batch)
                    _, _, shuffled_first_loss = compute_loss(model, shuffled_batch)
                    loss_z_shuffled_val = shuffled_first_loss
                    model.train()

                # Candidate-normalized eval (K-way disambiguation)
                _cand_loss_val = None
                _cand_acc_val = None
                if _do_candidate_eval:
                    model.eval()
                    cand_result = run_candidate_eval(
                        model=model,
                        tokenizer=tokenizer,
                        mapping_data=mapping_data,
                        n_examples=candidate_eval_n,
                        task=cfg.data.task,
                        device=str(device),
                        seed=step,
                    )
                    _cand_loss_val = cand_result["candidate_loss"]
                    _cand_acc_val = cand_result["candidate_accuracy"]
                    model.train()

                if not _first_z_shuffle_logged:
                    msg = (f"\n[Z-Shuffle Probe] Step {step} | "
                           f"Clean Loss: {first_target_loss:.4f} | "
                           f"Shuffled Loss: {loss_z_shuffled_val:.4f}")
                    if _cand_loss_val is not None:
                        msg += f" | Cand Loss: {_cand_loss_val:.4f}"
                    print(msg, flush=True)
                    _first_z_shuffle_logged = True

                history["train_loss"].append(avg_train_loss)
                history["train_accuracy"].append(avg_train_acc)
                history["first_target_loss"].append(avg_first_loss)
                history["loss_z_shuffled"].append(loss_z_shuffled_val)
                if _do_candidate_eval:
                    history["candidate_loss"].append(_cand_loss_val)
                    history["candidate_accuracy"].append(_cand_acc_val)

                    if _early_stop_threshold is not None and _cand_loss_val is not None:
                        if _cand_loss_val < _early_stop_threshold:
                            _early_stop_counter += 1
                            if _early_stop_counter >= _early_stop_patience:
                                print(f"\n[Early stop] Converged at step {step} "
                                      f"(cand_loss={_cand_loss_val:.6f} < "
                                      f"{_early_stop_threshold:.6f} for "
                                      f"{_early_stop_patience} evals)")
                                _early_stopped = True
                        else:
                            _early_stop_counter = 0

                history["grad_norm_sq"].append(avg_grad_norm_sq)
                history["grad_norm_sq_clipped"].append(avg_grad_norm_sq_clipped)
                history["clipping_active"].append(frac_clipped)
                history["steps"].append(step)

                if record_dissipation:
                    history["q_work"].append(running_q_work)
                    history["q_update"].append(running_q_update)
                    history["eta_eff"].append(running_eta_eff_sum / n_batches)

                _postfix = {
                    "loss": f"{avg_train_loss:.4f}",
                    "1st": f"{avg_first_loss:.4f}",
                    "z_shuf": f"{loss_z_shuffled_val:.4f}",
                    "acc": f"{avg_train_acc:.2%}",
                    "gnorm": f"{math.sqrt(avg_grad_norm_sq):.2f}",
                }
                if _cand_loss_val is not None:
                    _postfix["cand"] = f"{_cand_loss_val:.4f}"
                    _postfix["cacc"] = f"{_cand_acc_val:.2%}"
                pbar.set_postfix(_postfix)

                running_loss = 0.0
                running_acc = 0.0
                running_first_loss = 0.0
                running_grad_norm_sq = 0.0
                running_grad_norm_sq_clipped = 0.0
                running_clip_count = 0
                n_batches = 0
                if record_dissipation:
                    running_q_work = 0.0
                    running_q_update = 0.0
                    running_eta_eff_sum = 0.0

            # ---- Checkpointing ----
            if step % cfg.training.checkpoint_every == 0:
                if history["train_loss"]:
                    _ckpt_loss = history["train_loss"][-1]
                    _ckpt_acc = history["train_accuracy"][-1]
                else:
                    _ckpt_loss = loss.item()
                    _ckpt_acc = current_train_acc

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    train_loss=_ckpt_loss,
                    train_accuracy=_ckpt_acc,
                    checkpoint_dir=checkpoint_dir,
                )

                if callbacks.on_checkpoint is not None:
                    callbacks.on_checkpoint(model=model, optimizer=optimizer,
                                            step=step, history=history)

                history_path = output_dir / cfg.experiment.name / "training_history.json"
                history_path.parent.mkdir(parents=True, exist_ok=True)
                with open(history_path, "w") as f:
                    json.dump(history, f, indent=2)

            pbar.update(1)

    pbar.close()

    if _early_stopped:
        history["early_stopped"] = True
        history["early_stopped_step"] = step

    # ---- Final checkpoint ----
    final_train_loss = history["train_loss"][-1] if history["train_loss"] else 0.0
    final_train_acc = history["train_accuracy"][-1] if history["train_accuracy"] else 0.0
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        step=step,
        train_loss=final_train_loss,
        train_accuracy=final_train_acc,
        checkpoint_dir=checkpoint_dir,
    )

    history_path = output_dir / cfg.experiment.name / "training_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    return history
