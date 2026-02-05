#!/usr/bin/env python
"""
Main training script for Late Disambiguation Lag experiments.

Usage:
    python scripts/train.py --config-name=k10_n1000
    python scripts/train.py experiment.name=my_experiment data.k=50
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader

from src.data import create_tokenizer_from_config, create_datasets_from_config, collate_fn
from src.model import create_model_from_config
from src.training import train


@hydra.main(version_base=None, config_path="../configs/experiments", config_name="k10_n1000")
def main(cfg: DictConfig):
    """Main training entry point."""
    
    print("=" * 60)
    print(f"Experiment: {cfg.experiment.name}")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)
    
    # Set seed
    torch.manual_seed(cfg.experiment.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.experiment.seed)
    
    # Device (prefer cuda, then mps, else cpu)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    # Create tokenizer
    tokenizer = create_tokenizer_from_config(cfg)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    
    # Create datasets
    train_dataset, probe_dataset, mapping_data = create_datasets_from_config(cfg, tokenizer)
    print(f"Train examples: {len(train_dataset)}")
    print(f"Probe examples: {len(probe_dataset)}")
    print(f"Task: {cfg.data.task}")
    if cfg.data.task in {"bz_to_a", "b_to_a"}:
        print(f"Unique B strings: {mapping_data.n_unique_b}")
        print(f"Unique A strings: {mapping_data.n_unique_a}")
        print(f"K (A per B): {mapping_data.k}")
    else:
        print(f"Unique B strings: {mapping_data.n_unique_b}")
        print(f"Unique A strings: {mapping_data.n_unique_a}")
        if cfg.data.task == "az_to_b":
            print(f"K (A per B, z redundant): {mapping_data.k}")
        else:
            print(f"K (A per B, no z): {mapping_data.k}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Keep simple for now
    )
    
    probe_loader = DataLoader(
        probe_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    # Create model
    model = create_model_from_config(cfg, tokenizer)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup output directory
    output_dir = Path(cfg.output.base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_path = output_dir / cfg.experiment.name / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    
    # Train!
    history = train(
        model=model,
        train_loader=train_loader,
        probe_loader=probe_loader,
        cfg=cfg,
        output_dir=output_dir,
    )
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Final train accuracy: {history['train_accuracy'][-1]:.2%}")
    print(f"Outputs saved to: {output_dir / cfg.experiment.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
