#!/usr/bin/env python
"""
Experiment 4 — Optimizer-Aware Dissipation Measure.

Three optimizer configurations, each swept over K ∈ {10, 15, 20, 25, 36}
with 5 seeds:

  Config 1: AdamW (baseline), grad_clip=1.0
  Config 2: SGD + momentum 0.9, grad_clip=1.0
  Config 3: AdamW, NO gradient clipping

All runs record per-step dissipation quantities (Q_work, Q_update, eta_eff)
via the ``record_dissipation=True`` flag in the enhanced trainer.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import torch
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader

from src.data import create_tokenizer_from_config, create_datasets_from_config, collate_fn
from src.model import create_model_from_config
from src.training import train

# ---------------------------------------------------------------------------
K_VALUES = [10, 15, 20, 25, 36]
SEEDS = [42, 43, 44, 45, 46]
MAX_STEPS = 60_000

CONFIGS = {
    "adamw": {"optimizer_type": "adamw", "grad_clip": 1.0, "optimizer_kwargs": {},
              "lr": 1e-3, "scheduler": "cosine"},
    "sgd": {"optimizer_type": "sgd", "grad_clip": 1.0,
            "optimizer_kwargs": {"momentum": 0.9},
            "lr": 0.1, "scheduler": "constant"},
    "adamw_noclip": {"optimizer_type": "adamw", "grad_clip": None,
                     "optimizer_kwargs": {}, "lr": 1e-3, "scheduler": "cosine"},
}

# ---------------------------------------------------------------------------


def _base_cfg(k: int, seed: int, lr: float, scheduler: str) -> DictConfig:
    base = OmegaConf.load(Path(__file__).parent.parent / "configs" / "base.yaml")
    base.data.k = k
    base.data.n_unique_b = 1000
    base.data.task = "bz_to_a"
    base.training.max_steps = MAX_STEPS
    base.training.warmup_steps = 500 if scheduler == "cosine" else 0
    base.training.checkpoint_every = 200
    base.training.eval_every = 50
    base.training.batch_size = 128
    base.training.learning_rate = lr
    base.training.weight_decay = 0.01
    base.training.scheduler = scheduler
    base.experiment.seed = seed
    return base


def run_single(config_name: str, k: int, seed: int):
    conf = CONFIGS[config_name]
    name = f"exp4_{config_name}_k{k}_seed{seed}"
    cfg = _base_cfg(k, seed, conf["lr"], conf["scheduler"])
    cfg.experiment.name = name

    output_dir = Path("outputs")
    history_path = output_dir / name / "training_history.json"
    if history_path.exists():
        print(f"[SKIP] {name}")
        return

    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    torch.manual_seed(seed)

    tokenizer = create_tokenizer_from_config(cfg)
    train_ds, probe_ds, mapping_data = create_datasets_from_config(cfg, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size,
                              shuffle=True, collate_fn=collate_fn, num_workers=0)
    probe_loader = DataLoader(probe_ds, batch_size=cfg.training.batch_size,
                              shuffle=False, collate_fn=collate_fn, num_workers=0)
    model = create_model_from_config(cfg, tokenizer)

    train(
        model, train_loader, probe_loader, cfg, output_dir,
        optimizer_type=conf["optimizer_type"],
        optimizer_kwargs=conf["optimizer_kwargs"],
        grad_clip=conf["grad_clip"],
        record_dissipation=True,
    )

    meta = {"config": config_name, "k": k, "seed": seed,
            "optimizer_type": conf["optimizer_type"],
            "grad_clip": conf["grad_clip"]}
    with open(output_dir / name / "exp4_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def main():
    for config_name in CONFIGS:
        for k in K_VALUES:
            for seed in SEEDS:
                run_single(config_name, k, seed)
    print("\n[Exp 4] All runs complete.")


if __name__ == "__main__":
    main()
