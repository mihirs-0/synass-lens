#!/usr/bin/env python
"""
Experiment 1 — Group-Specific Selector Control.

Runs:
  Condition A (shared z)     × 5 seeds
  Condition B (private z)    × 5 seeds
  Condition C (super-group z, G ∈ {10, 50, 100, 500, 1000}) × 5 seeds

For every run the enhanced trainer records loss, candidate loss (first-target
z-shuffle gap), gradient norm, and checkpoints every 50 steps.

Results are saved under  outputs/exp1_<condition>_seed<S>/
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import torch
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader

from src.data import create_tokenizer_from_config, create_datasets_from_config, collate_fn
from src.model import create_model_from_config
from src.training import train


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_cfg() -> DictConfig:
    """Load base config and set Exp-1 defaults."""
    base = OmegaConf.load(Path(__file__).parent.parent / "configs" / "base.yaml")
    base.data.k = 20
    base.data.n_unique_b = 1000
    base.data.task = "bz_to_a"
    base.training.max_steps = 30000
    base.training.warmup_steps = 500
    base.training.checkpoint_every = 50
    base.training.eval_every = 50
    base.training.batch_size = 128
    base.training.learning_rate = 1e-3
    base.training.weight_decay = 0.01
    base.training.scheduler = "cosine"
    return base


def run_single(
    name: str,
    seed: int,
    z_sharing: str = "shared",
    n_supergroups: int = 1,
):
    cfg = _base_cfg()
    cfg.experiment.name = name
    cfg.experiment.seed = seed
    cfg.data.z_sharing = z_sharing
    cfg.data.n_supergroups = n_supergroups

    output_dir = Path("outputs")
    history_path = output_dir / name / "training_history.json"
    if history_path.exists():
        print(f"[SKIP] {name} already exists", flush=True)
        return

    print(f"\n{'='*60}\n  {name}  (seed={seed}, z_sharing={z_sharing}, G={n_supergroups})\n{'='*60}", flush=True)
    torch.manual_seed(seed)

    tokenizer = create_tokenizer_from_config(cfg)
    train_ds, probe_ds, mapping_data = create_datasets_from_config(cfg, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size,
                              shuffle=True, collate_fn=collate_fn, num_workers=0)
    probe_loader = DataLoader(probe_ds, batch_size=cfg.training.batch_size,
                              shuffle=False, collate_fn=collate_fn, num_workers=0)

    model = create_model_from_config(cfg, tokenizer)
    train(model, train_loader, probe_loader, cfg, output_dir,
          mapping_data=mapping_data, tokenizer=tokenizer)

    # Save mapping metadata for analysis
    meta = {"z_sharing": z_sharing, "n_supergroups": n_supergroups,
            "seed": seed, "k": cfg.data.k, "n_unique_b": cfg.data.n_unique_b}
    meta_path = output_dir / name / "exp1_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

SEEDS = [42, 43, 44, 45, 46]
SUPERGROUP_G_VALUES = [10, 50, 100, 500, 1000]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Single seed (42), skip multi-seed sweep")
    args = parser.parse_args()

    seeds = [42] if args.quick else SEEDS

    # Condition A: shared z
    for seed in seeds:
        run_single(f"exp1_shared_z_seed{seed}", seed, z_sharing="shared")

    # Condition B: private z
    for seed in seeds:
        run_single(f"exp1_private_z_seed{seed}", seed, z_sharing="private")

    # Condition C: super-group z
    for G in SUPERGROUP_G_VALUES:
        for seed in seeds:
            run_single(f"exp1_supergroup_G{G}_seed{seed}", seed,
                       z_sharing="supergroup", n_supergroups=G)

    print("\n[Exp 1] All runs complete.", flush=True)


if __name__ == "__main__":
    main()
