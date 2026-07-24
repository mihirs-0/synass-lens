#!/usr/bin/env python
"""
Experiment 5 — Multihead Specialization Analysis.

Single heavily-instrumented run (K=20, |B|=1000, seed 42).  Runs the
MultiheadDecompositionProbe at every checkpoint to capture per-head
attention, value contribution, and MLP logit decomposition across training.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.data import create_tokenizer_from_config, create_datasets_from_config, collate_fn
from src.model import create_model_from_config
from src.training import train, TrainingCallbacks
from src.probes.multihead_decomposition import MultiheadDecompositionProbe

OUTPUT_DIR = Path("outputs")


def main():
    cfg = OmegaConf.load(Path(__file__).parent.parent / "configs" / "base.yaml")
    cfg.experiment.name = "exp5_multihead"
    cfg.experiment.seed = 42
    cfg.data.k = 20
    cfg.data.n_unique_b = 1000
    cfg.data.task = "bz_to_a"
    cfg.training.max_steps = 30_000
    cfg.training.warmup_steps = 500
    cfg.training.checkpoint_every = 100
    cfg.training.eval_every = 25
    cfg.training.batch_size = 128
    cfg.training.learning_rate = 1e-3
    cfg.training.weight_decay = 0.01
    cfg.training.scheduler = "cosine"

    exp_dir = OUTPUT_DIR / cfg.experiment.name
    history_path = exp_dir / "training_history.json"
    if history_path.exists():
        print(f"[SKIP] {cfg.experiment.name} already exists")
        return

    torch.manual_seed(cfg.experiment.seed)

    tokenizer = create_tokenizer_from_config(cfg)
    train_ds, probe_ds, mapping_data = create_datasets_from_config(cfg, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size,
                              shuffle=True, collate_fn=collate_fn, num_workers=0)
    probe_loader = DataLoader(probe_ds, batch_size=cfg.training.batch_size,
                              shuffle=False, collate_fn=collate_fn, num_workers=0)

    # Build a small probe dataloader (256 examples)
    from torch.utils.data import Subset
    probe_subset = Subset(train_ds, list(range(min(256, len(train_ds)))))
    probe_dl = DataLoader(probe_subset, batch_size=64, shuffle=False,
                          collate_fn=collate_fn, num_workers=0)

    model = create_model_from_config(cfg, tokenizer)
    device = model.cfg.device
    decomp_probe = MultiheadDecompositionProbe()

    probe_results = []

    def on_checkpoint(model, optimizer, step, history, **kw):
        model.eval()
        res = decomp_probe.run(model, probe_dl, device=device)
        res["step"] = step
        probe_results.append(res)
        model.train()

    callbacks = TrainingCallbacks(on_checkpoint=on_checkpoint)

    train(model, train_loader, probe_loader, cfg, OUTPUT_DIR, callbacks=callbacks)

    # Save probe trajectory
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "multihead_probe_results.json", "w") as f:
        json.dump(probe_results, f, indent=2)
    print(f"[Exp 5] Saved {len(probe_results)} probe snapshots.")


if __name__ == "__main__":
    main()
