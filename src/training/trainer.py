"""
Training loop for Late Disambiguation Lag experiments.

Handles:
- Training with AdamW and warmup
- Periodic logging
- Checkpointing
- Logging
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from tqdm import tqdm

from .checkpoint import save_checkpoint, get_checkpoint_dir


@dataclass
class TrainingMetrics:
    """Container for training metrics at a single step."""
    step: int
    train_loss: float
    train_accuracy: Optional[float] = None
    learning_rate: float = 0.0


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
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Create learning rate scheduler with linear warmup and cosine decay.
    """
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
) -> Dict[str, Any]:
    """
    Main training loop.
    
    Args:
        model: HookedTransformer to train
        train_loader: Training data loader
        probe_loader: Probe data loader (held-out subset for analysis)
        cfg: Hydra config
        output_dir: Directory for outputs
        
    Returns:
        Training history dict
    """
    device = model.cfg.device
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    
    # Setup scheduler
    scheduler = get_lr_scheduler(
        optimizer,
        cfg.training.warmup_steps,
        cfg.training.max_steps,
    )
    
    # Setup checkpointing
    checkpoint_dir = get_checkpoint_dir(str(output_dir), cfg.experiment.name)
    
    # Training history
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "first_target_loss": [],
        "steps": [],
    }
    
    # Training loop
    model.train()
    step = 0
    epoch = 0
    running_loss = 0.0
    running_acc = 0.0
    running_first_loss = 0.0
    n_batches = 0
    
    pbar = tqdm(total=cfg.training.max_steps, desc="Training")
    
    while step < cfg.training.max_steps:
        epoch += 1
        
        for batch in train_loader:
            if step >= cfg.training.max_steps:
                break
                
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            # Forward + backward
            optimizer.zero_grad()
            loss, train_acc, first_target_loss = compute_loss(model, batch)
            current_train_acc = train_acc
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Track loss
            running_loss += loss.item()
            running_acc += train_acc
            running_first_loss += first_target_loss
            n_batches += 1
            step += 1
            
            # Log periodically (training-only)
            if step % cfg.training.eval_every == 0:
                avg_train_loss = running_loss / n_batches
                avg_train_acc = running_acc / n_batches
                avg_first_loss = running_first_loss / n_batches
                
                history["train_loss"].append(avg_train_loss)
                history["train_accuracy"].append(avg_train_acc)
                history["first_target_loss"].append(avg_first_loss)
                history["steps"].append(step)
                
                pbar.set_postfix({
                    "train_loss": f"{avg_train_loss:.4f}",
                    "first_loss": f"{avg_first_loss:.4f}",
                    "train_acc": f"{avg_train_acc:.2%}",
                })
                
                running_loss = 0.0
                running_acc = 0.0
                running_first_loss = 0.0
                n_batches = 0
            
            # Checkpoint periodically
            if step % cfg.training.checkpoint_every == 0:
                # Get current metrics
                if history["train_loss"]:
                    train_loss = history["train_loss"][-1]
                    train_acc = history["train_accuracy"][-1]
                else:
                    train_loss = loss.item()
                    train_acc = current_train_acc
                
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    train_loss=train_loss,
                    train_accuracy=train_acc,
                    checkpoint_dir=checkpoint_dir,
                )
            
            pbar.update(1)
    
    pbar.close()
    
    # Final checkpoint
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
    
    # Save training history
    history_path = output_dir / cfg.experiment.name / "training_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    
    return history
