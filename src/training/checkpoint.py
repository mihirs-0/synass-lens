"""
Checkpoint management for experiment reproducibility.

Saves model weights, optimizer state, and training metadata.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import torch
from transformer_lens import HookedTransformer


def get_checkpoint_dir(base_dir: str, experiment_name: str) -> Path:
    """Get the checkpoint directory for an experiment."""
    path = Path(base_dir) / experiment_name / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_checkpoint(
    model: HookedTransformer,
    optimizer: torch.optim.Optimizer,
    step: int,
    train_loss: float,
    train_accuracy: float,
    checkpoint_dir: Path,
    config: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save a training checkpoint.
    
    Saves:
    - model_step_{step}.pt: Model weights
    - optimizer_step_{step}.pt: Optimizer state
    - metadata_step_{step}.json: Training metrics
    
    Returns:
        Path to the saved checkpoint directory
    """
    step_dir = checkpoint_dir / f"step_{step:06d}"
    step_dir.mkdir(exist_ok=True)
    
    # Save model weights
    model_path = step_dir / "model.pt"
    torch.save(model.state_dict(), model_path)
    
    # Save optimizer state
    optimizer_path = step_dir / "optimizer.pt"
    torch.save(optimizer.state_dict(), optimizer_path)
    
    # Save metadata
    metadata = {
        "step": step,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
    }
    if config is not None:
        metadata["config"] = config
        
    metadata_path = step_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    return step_dir


def load_checkpoint(
    model: HookedTransformer,
    optimizer: Optional[torch.optim.Optimizer],
    checkpoint_dir: Path,
    step: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Load a checkpoint.
    
    If step is None, loads the latest checkpoint.
    
    Returns:
        Metadata dict with step, losses, etc.
    """
    if step is None:
        # Find latest checkpoint
        step_dirs = sorted(checkpoint_dir.glob("step_*"))
        if not step_dirs:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        step_dir = step_dirs[-1]
    else:
        step_dir = checkpoint_dir / f"step_{step:06d}"
        
    if not step_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found: {step_dir}")
    
    # Load model weights
    model_path = step_dir / "model.pt"
    model.load_state_dict(torch.load(model_path, map_location=model.cfg.device))
    
    # Load optimizer state if provided
    if optimizer is not None:
        optimizer_path = step_dir / "optimizer.pt"
        if optimizer_path.exists():
            optimizer.load_state_dict(torch.load(optimizer_path))
    
    # Load metadata
    metadata_path = step_dir / "metadata.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    return metadata


def list_checkpoints(checkpoint_dir: Path) -> list:
    """List all checkpoint steps in a directory."""
    step_dirs = sorted(checkpoint_dir.glob("step_*"))
    steps = []
    for d in step_dirs:
        try:
            step = int(d.name.split("_")[1])
            steps.append(step)
        except:
            continue
    return steps
