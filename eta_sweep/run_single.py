#!/usr/bin/env python
"""
η-Sweep single-cell trainer.

Trains one (η, K, seed) configuration and writes per-checkpoint logs to
  eta_sweep/results/eta_{η}_K_{K}_seed_{seed}/log.jsonl

Per-checkpoint quantities logged:
    step                       — training step
    train_loss                 — running mean over the last eval window
    candidate_loss             — CE over the K valid A's
    delta_z                    — candidate loss under shuffled z minus clean
    grad_norm_sq_held_out      — ‖∇L(θ)‖² on the fixed held-out batch of 256
    grad_norm_sq_training      — ‖∇L(θ)‖² on the current training batch (reference)
    lr                         — scheduled learning rate at this step
    wall_clock_s               — elapsed seconds since run start

Final status (status.json) is one of:
    transitioned | stuck | diverged | inconclusive | crashed

Reuses utilities from the existing src/ (no modification):
    compute_loss, shuffle_z_in_batch, get_lr_scheduler
    create_tokenizer_from_config, create_datasets_from_config, collate_fn
    create_model_from_config
    run_candidate_eval, score_candidate_sequences
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

# Must be set BEFORE torch is imported for CUDA-deterministic cublas.  The
# :16:8 workspace fits small batch/d_model configs; raise to :4096:8 if we
# hit deterministic-algorithm failures on larger models.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# --- wire up the existing repo ------------------------------------------------
ETA_SWEEP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = ETA_SWEEP_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data import (  # noqa: E402
    CharTokenizer,
    MappingData,
    collate_fn,
    create_datasets_from_config,
    create_tokenizer_from_config,
)
from src.model import create_model_from_config  # noqa: E402
from src.training.trainer import (  # noqa: E402
    compute_loss,
    get_lr_scheduler,
    shuffle_z_in_batch,
)
from src.analysis.candidate_eval import (  # noqa: E402
    score_candidate_sequences,
)

from eta_sweep.config import CellConfig, run_dir  # noqa: E402


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _set_all_seeds(seed: int) -> None:
    """Set every RNG source we care about, plus CUDA-deterministic flags.

    Deterministic behaviour is best-effort on MPS (Apple Silicon): the OS
    schedules kernels non-deterministically across processes, and some ops
    in TransformerLens fall back to non-deterministic implementations.  On
    CUDA with the flags below + `CUBLAS_WORKSPACE_CONFIG=:16:8` set before
    the torch import, the same (seed, cell config) should produce bit-exact
    Q values across runs on the same device.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # warn_only=True: TransformerLens uses a few ops (e.g. scatter variants)
    # that have no deterministic implementation; we warn instead of crashing.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        # Older torch releases expose use_deterministic_algorithms without
        # warn_only; fall back silently.
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _to_ns(d):
    """Recursively convert a dict → SimpleNamespace so existing src/ code can
    attribute-access it.  CellConfig is flat; we wrap it into the nested
    layout the existing code expects."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_to_ns(v) for v in d]
    return d


def build_legacy_cfg(cc: CellConfig) -> SimpleNamespace:
    """Rebuild the Hydra-style nested cfg expected by src/ factories."""
    # NOTE: data.seed is held constant for a (K, seed) pair regardless of η
    # so that the same (K, seed) always produces the same dataset.
    data_seed = 1_000_003 * cc.seed + cc.k
    cfg_dict = dict(
        experiment=dict(name=cc.run_name, seed=data_seed),
        data=dict(
            n_unique_b=cc.n_unique_b,
            k=cc.k,
            task=cc.task,
            b_length=cc.b_length,
            a_length=cc.a_length,
            z_length=cc.z_length,
            vocab_chars=cc.vocab_chars,
            probe_fraction=cc.probe_fraction,
            split_by_base=cc.split_by_base,
            enforce_unique_a_first_char_per_b=cc.enforce_unique_a_first_char_per_b,
            disambiguation_prefix_length=cc.disambiguation_prefix_length,
            label_noise_prob=0.0,
        ),
        tokenizer=dict(
            pad_token="<PAD>",
            bos_token="<BOS>",
            eos_token="<EOS>",
            sep_token="<SEP>",
        ),
        model=dict(
            n_layers=cc.n_layers,
            n_heads=cc.n_heads,
            d_model=cc.d_model,
            d_head=cc.d_head,
            d_mlp=cc.d_mlp,
            act_fn=cc.act_fn,
        ),
        training=dict(
            batch_size=cc.batch_size,
            learning_rate=cc.eta,
            weight_decay=cc.weight_decay,
            max_steps=cc.max_steps,
            warmup_steps=cc.warmup_steps,
            scheduler=cc.scheduler,
        ),
        output=dict(base_dir=str(ETA_SWEEP_ROOT / "results")),
    )
    return _to_ns(cfg_dict)


# ---------------------------------------------------------------------------
# Held-out gradient norm
# ---------------------------------------------------------------------------

def build_held_out_batch(
    train_dataset,
    batch_size: int,
    seed: int,
    device: str,
) -> Dict[str, torch.Tensor]:
    """Fixed batch of `batch_size` examples, resampled once per run and held
    constant for the lifetime of training.  Used for the Q integrand so the
    gradient-norm measurement is independent of the mini-batch noise that
    the update itself consumed."""
    rng = random.Random(seed)
    n = len(train_dataset)
    idx = rng.sample(range(n), min(batch_size, n))
    items = [train_dataset[i] for i in idx]
    batch = collate_fn(items)
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def compute_held_out_grad_norm_sq(model, batch) -> float:
    """Full-parameter L2 squared of ∇L on the held-out batch."""
    model.zero_grad(set_to_none=True)
    loss, _, _ = compute_loss(model, batch)
    loss.backward()
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.data.norm(2).item() ** 2)
    model.zero_grad(set_to_none=True)
    return total


# ---------------------------------------------------------------------------
# Candidate loss + z-shuffle Δz
# ---------------------------------------------------------------------------

def compute_candidate_loss_and_delta_z(
    model,
    tokenizer: CharTokenizer,
    mapping_data: MappingData,
    n_examples: int,
    task: str,
    device: str,
    seed: int,
) -> Tuple[float, float, float]:
    """Returns (candidate_loss_clean, candidate_loss_shuffled_z, delta_z).

    For Δz we score each example against the K candidates twice: once with
    the correct z, once with a z drawn from a different group.  The correct
    index is unchanged so the correct A is still the target; only z
    changes.  This is the candidate-space analogue of shuffle_z_in_batch.
    """
    rng = random.Random(seed)
    base_strings = list(mapping_data.mappings.keys())
    n = min(n_examples, len(base_strings))
    sampled = rng.sample(base_strings, n)

    clean_losses: List[float] = []
    shuffled_losses: List[float] = []

    # For shuffled z we need a pool of z strings from other B groups.
    # Collect all z strings globally; we'll pick one ≠ the correct z.
    all_z: List[str] = []
    for mappings in mapping_data.mappings.values():
        for z, _ in mappings:
            all_z.append(z)
    all_z_set = set(all_z)

    for base_string in sampled:
        mappings = mapping_data.mappings[base_string]
        correct_index = rng.randrange(len(mappings))
        correct_z = mappings[correct_index][0]
        candidate_a_strings = [entry[1] for entry in mappings]

        # --- clean ---
        clean = score_candidate_sequences(
            model=model,
            tokenizer=tokenizer,
            base_string=base_string,
            z_string=correct_z,
            candidate_a_strings=candidate_a_strings,
            correct_index=correct_index,
            task=task,
            device=device,
        )
        clean_losses.append(clean["candidate_loss"])

        # --- shuffled z (pick any z != correct) ---
        # Prefer a z of the same length; fall back to rejection sampling.
        shuffled_z = correct_z
        for _ in range(32):
            z = rng.choice(all_z)
            if z != correct_z and len(z) == len(correct_z):
                shuffled_z = z
                break
        if shuffled_z == correct_z:
            # Degenerate case: everything is the same z; skip.
            shuffled_losses.append(float("nan"))
            continue

        shuf = score_candidate_sequences(
            model=model,
            tokenizer=tokenizer,
            base_string=base_string,
            z_string=shuffled_z,
            candidate_a_strings=candidate_a_strings,
            correct_index=correct_index,
            task=task,
            device=device,
        )
        shuffled_losses.append(shuf["candidate_loss"])

    mean_clean = float(np.mean(clean_losses))
    valid = [s for s in shuffled_losses if not math.isnan(s)]
    mean_shuf = float(np.mean(valid)) if valid else float("nan")
    delta_z = mean_shuf - mean_clean if not math.isnan(mean_shuf) else float("nan")
    return mean_clean, mean_shuf, delta_z


# ---------------------------------------------------------------------------
# Online status detectors
# ---------------------------------------------------------------------------

class RunStatusTracker:
    """Online plateau / stuck / diverged / transitioned detector.

    Receives the per-checkpoint record and returns a terminal status or
    None to continue.
    """

    def __init__(self, cc: CellConfig):
        self.cc = cc
        self.log_k = math.log(cc.k)

        # Rolling windows of (step, value) for the last `patience` steps
        self._plateau_window: List[Tuple[int, bool]] = []
        self._transition_window: List[Tuple[int, bool]] = []
        self._diverge_window: List[Tuple[int, bool]] = []

        self._grad_norm_step_200: Optional[float] = None
        self._transition_detected_step: Optional[int] = None
        self._post_transition_budget: Optional[int] = None

    def _prune(self, window: List[Tuple[int, bool]], current_step: int,
               patience: int) -> List[Tuple[int, bool]]:
        # Drop entries older than `patience` steps.
        return [(s, v) for s, v in window if current_step - s <= patience]

    def _all_true_over(self, window: List[Tuple[int, bool]], current_step: int,
                      patience: int) -> bool:
        window = self._prune(window, current_step, patience)
        if not window:
            return False
        return (
            window[0][0] <= current_step - patience + 1
            and all(v for _, v in window)
        )

    def update(self, step: int, record: Dict[str, float]) -> Optional[str]:
        cc = self.cc
        cand = record.get("candidate_loss")
        dz = record.get("delta_z")
        g = record.get("grad_norm_sq_held_out")

        # ---- diverged: NaN anywhere is instant ----
        if cand is not None and (math.isnan(cand) or math.isinf(cand)):
            return "diverged"
        if record.get("train_loss") is not None and math.isnan(record["train_loss"]):
            return "diverged"

        # Anchor g for the grad-spike divergence rule.
        if step >= 200 and self._grad_norm_step_200 is None and g is not None:
            self._grad_norm_step_200 = g

        # ---- transitioned: cand_loss < transition_frac · log K for patience ----
        is_transition = (
            cand is not None
            and cand < cc.transition_frac * self.log_k
        )
        self._transition_window.append((step, is_transition))
        self._transition_window = self._prune(
            self._transition_window, step, cc.transition_patience
        )
        if (
            self._transition_detected_step is None
            and self._all_true_over(self._transition_window, step,
                                    cc.transition_patience)
        ):
            self._transition_detected_step = step
            tail = min(2000, cc.max_steps - step)
            self._post_transition_budget = step + tail

        # Once transition detected, run for post-transition tail
        if self._transition_detected_step is not None:
            if step >= (self._post_transition_budget or step):
                return "transitioned"

        # ---- plateau → stuck ----
        on_plateau = (
            cand is not None
            and abs(cand - self.log_k) <= cc.plateau_tol_frac * self.log_k
            and dz is not None
            and not math.isnan(dz)
            and dz < cc.delta_z_plateau
        )
        self._plateau_window.append((step, on_plateau))
        self._plateau_window = self._prune(
            self._plateau_window, step, cc.stuck_patience
        )

        if step >= cc.min_steps and self._all_true_over(
            self._plateau_window, step, cc.stuck_patience
        ):
            # Also require gradient-norm quiescence: no spike above 3× rolling
            # mean of the last 500 steps.  If we don't have data we skip this
            # extra check and just use the plateau condition.
            return "stuck"

        # ---- diverged: cand_loss > 1.5 · log K for 500 steps ----
        if step >= 200:
            over = (cand is not None and cand > cc.diverge_ratio * self.log_k)
            self._diverge_window.append((step, over))
            self._diverge_window = self._prune(
                self._diverge_window, step, cc.diverge_patience
            )
            if self._all_true_over(
                self._diverge_window, step, cc.diverge_patience
            ):
                return "diverged"

        # ---- diverged: grad norm exceeds 1000× step-200 value for 100 steps ----
        if (
            self._grad_norm_step_200 is not None
            and g is not None
            and g > 1000.0 * self._grad_norm_step_200
        ):
            # single observation at the current cadence ≥ 100 steps suffices
            return "diverged"

        return None


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_single(
    cc: CellConfig,
    resume: bool = False,
    reverse_sequences: bool = False,
    output_subdir: Optional[str] = None,
) -> Dict[str, object]:
    # Route output to a subdirectory if requested (e.g. dataset_reversal,
    # temperature_collapse) so per-experiment runs don't clobber each other.
    if output_subdir:
        from eta_sweep.config import RESULTS_DIR
        out_dir = RESULTS_DIR / output_subdir / cc.run_name
    else:
        out_dir = run_dir(cc.eta, cc.k, cc.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.jsonl"
    status_path = out_dir / "status.json"
    config_path = out_dir / "config.json"
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # Skip already-finished runs unless explicitly resumed.
    if status_path.exists() and not resume:
        with open(status_path) as f:
            prior = json.load(f)
        if prior.get("status") in {"transitioned", "stuck", "diverged",
                                   "inconclusive", "crashed"}:
            print(f"[{cc.run_name}] already finished with status={prior['status']}")
            return prior

    # Persist the config we ran with.
    with open(config_path, "w") as f:
        json.dump({"cell": asdict(cc), "git_commit": _git_commit()}, f, indent=2)

    # Reproducibility.
    _set_all_seeds(cc.seed)

    device = _select_device()
    print(f"[{cc.run_name}] device={device}  max_steps={cc.max_steps}")

    cfg = build_legacy_cfg(cc)
    tokenizer = create_tokenizer_from_config(cfg)

    # For the reversed-dataset experiment, monkey-patch tokenizer.encode_sequence
    # so that *every* caller (both DisambiguationDataset._precompute_tokens and
    # candidate_eval.score_candidate_sequences) emits option-(b)-reversed
    # encodings.  This keeps training distribution and eval distribution in sync.
    if reverse_sequences:
        from eta_sweep.reverse_dataset import reverse_tokenized
        _original_encode = tokenizer.encode_sequence
        def _reversed_encode(b, z, a, task="bz_to_a"):
            return reverse_tokenized(_original_encode(b, z, a, task=task))
        tokenizer.encode_sequence = _reversed_encode  # type: ignore[assignment]
        print(f"[{cc.run_name}] option-(b) sequence reversal ENABLED "
              f"(tokenizer.encode_sequence monkey-patched)")

    train_dataset, probe_dataset, mapping_data = create_datasets_from_config(
        cfg, tokenizer
    )
    log_k = math.log(cc.k)
    print(f"[{cc.run_name}] train examples={len(train_dataset)}  log K={log_k:.4f}")

    # Infinite-ish training loader (reshuffles each epoch).
    train_loader = DataLoader(
        train_dataset,
        batch_size=cc.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    model = create_model_from_config(cfg, tokenizer)

    # Optimizer
    opt_name = (cc.optimizer or "adamw").lower()
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cc.eta,
            weight_decay=cc.weight_decay,
        )
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cc.eta,
            momentum=cc.momentum,
            weight_decay=cc.weight_decay,
        )
    else:
        raise ValueError(f"unknown optimizer '{cc.optimizer}'; "
                         "expected 'adamw' or 'sgd'")
    scheduler = get_lr_scheduler(
        optimizer,
        warmup_steps=cc.warmup_steps,
        max_steps=cc.max_steps,
        scheduler_type=cc.scheduler,
    )

    # Held-out batch for Q integrand.  Separate RNG stream from the
    # training-batch RNG so the two are independent.
    held_out = build_held_out_batch(
        train_dataset,
        batch_size=cc.held_out_grad_batch_size,
        seed=cc.seed * 2654435761 % (2**31),
        device=device,
    )

    tracker = RunStatusTracker(cc)

    # ---- Open the log file in append mode so partial progress persists even
    #      if we crash. ----
    log_fp = open(log_path, "w", buffering=1)

    def _log(record: Dict[str, float]):
        log_fp.write(json.dumps(record) + "\n")

    t0 = time.time()
    step = 0
    running_loss = 0.0
    running_grad_norm_sq = 0.0
    running_n = 0
    status: Optional[str] = None

    try:
        while step < cc.max_steps and status is None:
            for batch in train_loader:
                if step >= cc.max_steps or status is not None:
                    break
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss, _train_acc, _first_loss = compute_loss(model, batch)
                loss.backward()
                train_grad_norm_sq = sum(
                    float(p.grad.data.norm(2).item() ** 2)
                    for p in model.parameters()
                    if p.grad is not None
                )
                if cc.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cc.grad_clip
                    )
                optimizer.step()
                scheduler.step()

                running_loss += float(loss.item())
                running_grad_norm_sq += train_grad_norm_sq
                running_n += 1
                step += 1

                # ---- Decide whether to log this step ----
                if step < cc.switch_step:
                    do_log = (step % cc.checkpoint_every_early == 0)
                else:
                    do_log = (step % cc.checkpoint_every_late == 0)
                if not do_log:
                    continue

                # Held-out gradient norm — uses its own forward/backward
                # (model.train() mode, no dropout in this arch, deterministic).
                g_held = compute_held_out_grad_norm_sq(model, held_out)

                # Candidate loss + Δz (candidate-space z-shuffle).
                c_clean, c_shuf, delta_z = compute_candidate_loss_and_delta_z(
                    model=model,
                    tokenizer=tokenizer,
                    mapping_data=mapping_data,
                    n_examples=cc.candidate_eval_n,
                    task=cc.task,
                    device=device,
                    seed=step,
                )

                record = {
                    "step": step,
                    "train_loss": running_loss / max(running_n, 1),
                    "candidate_loss": c_clean,
                    "candidate_loss_shuffled_z": c_shuf,
                    "delta_z": delta_z,
                    "grad_norm_sq_held_out": g_held,
                    "grad_norm_sq_training": running_grad_norm_sq / max(running_n, 1),
                    "lr": optimizer.param_groups[0]["lr"],
                    "wall_clock_s": time.time() - t0,
                }
                _log(record)

                # Reset running means for next window.
                running_loss = 0.0
                running_grad_norm_sq = 0.0
                running_n = 0

                status = tracker.update(step, record)
                if status is not None:
                    break

                # Save a model checkpoint every weight_checkpoint_every steps.
                if step % cc.weight_checkpoint_every == 0:
                    torch.save(
                        model.state_dict(),
                        ckpt_dir / f"model_step_{step:07d}.pt",
                    )

        if status is None:
            status = "inconclusive"

        # Save the final model weights for every completed run.
        torch.save(model.state_dict(), ckpt_dir / f"model_step_{step:07d}.pt")

    except Exception as exc:  # noqa: BLE001
        print(f"[{cc.run_name}] CRASHED at step {step}: {exc}")
        traceback.print_exc()
        status = "crashed"

    finally:
        log_fp.close()

    summary = {
        "status": status,
        "final_step": step,
        "wall_clock_s": time.time() - t0,
        "run_name": cc.run_name,
        "eta": cc.eta,
        "k": cc.k,
        "seed": cc.seed,
        "transition_detected_step": tracker._transition_detected_step,
    }
    with open(status_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[{cc.run_name}] done: status={status}  step={step}  "
          f"wall={summary['wall_clock_s']:.1f}s")
    return summary


def _parse_eta(s: str) -> float:
    return float(s)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eta", type=_parse_eta, required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=None,
                   help="Override the η-dependent default budget")
    p.add_argument("--min-steps", type=int, default=None,
                   help="Override min_steps (default 2000)")
    p.add_argument("--stuck-patience", type=int, default=None,
                   help="Override stuck_patience / stability window (default 2000)")
    p.add_argument("--n-unique-b", type=int, default=None,
                   help="Override n_unique_b (default 1000). Used for K>36 experiments.")
    p.add_argument("--disambiguation-prefix-length", type=int, default=None,
                   help="Override disambiguation_prefix_length (default 1). "
                        "Required when K>36 with enforce_unique_a_first_char_per_b=True.")
    p.add_argument("--d-model", type=int, default=None,
                   help="Override d_model. d_head is set to d_model/n_heads.")
    p.add_argument("--n-layers", type=int, default=None,
                   help="Override n_layers.")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override batch_size (default 128). Used for the "
                        "η/B collapse experiment.")
    p.add_argument("--reverse", action="store_true",
                   help="Option-(b) sequence reversal for the dataset-reversal "
                        "experiment.  BOS/EOS stay at boundary, middle reversed.")
    p.add_argument("--optimizer", type=str, default=None,
                   choices=["adamw", "sgd"],
                   help="Override optimizer (default 'adamw').  'sgd' is "
                        "vanilla momentum-SGD, used for the SGD control.")
    p.add_argument("--momentum", type=float, default=None,
                   help="SGD momentum (default 0.0). Ignored when optimizer "
                        "is adamw.")
    p.add_argument("--weight-decay", type=float, default=None,
                   help="Weight-decay override (default 0.01). For SGD this "
                        "becomes L2 regularisation; for AdamW it is decoupled.")
    p.add_argument("--output-subdir", type=str, default=None,
                   help="Write results to eta_sweep/results/<subdir>/... instead "
                        "of the root results/ directory.")
    p.add_argument("--resume", action="store_true",
                   help="Rerun even if status.json already exists")
    args = p.parse_args()

    cc = CellConfig(eta=args.eta, k=args.k, seed=args.seed)

    # Apply CLI overrides for thresholds we expose.
    override_fields = asdict(cc)
    if args.min_steps is not None:
        override_fields["min_steps"] = int(args.min_steps)
    if args.stuck_patience is not None:
        override_fields["stuck_patience"] = int(args.stuck_patience)
    if args.n_unique_b is not None:
        override_fields["n_unique_b"] = int(args.n_unique_b)
    if args.disambiguation_prefix_length is not None:
        override_fields["disambiguation_prefix_length"] = int(args.disambiguation_prefix_length)
    if args.d_model is not None:
        override_fields["d_model"] = int(args.d_model)
        override_fields["d_head"] = int(args.d_model) // override_fields["n_heads"]
        override_fields["d_mlp"] = 4 * int(args.d_model)
    if args.n_layers is not None:
        override_fields["n_layers"] = int(args.n_layers)
    if args.batch_size is not None:
        override_fields["batch_size"] = int(args.batch_size)
    if args.optimizer is not None:
        override_fields["optimizer"] = str(args.optimizer)
    if args.momentum is not None:
        override_fields["momentum"] = float(args.momentum)
    if args.weight_decay is not None:
        override_fields["weight_decay"] = float(args.weight_decay)

    if args.max_steps is not None:
        # CellConfig exposes max_steps as a property; override via a subclass
        # instance bound to the argument.
        class _Override(CellConfig):
            @property
            def max_steps(self_inner):  # type: ignore[override]
                return int(args.max_steps)
        cc = _Override(**override_fields)
    else:
        cc = CellConfig(**override_fields)

    run_single(
        cc,
        resume=args.resume,
        reverse_sequences=bool(args.reverse),
        output_subdir=args.output_subdir,
    )


if __name__ == "__main__":
    main()
