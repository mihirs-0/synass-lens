#!/usr/bin/env python
"""
Experiment 7 — Isotropy Verification (standalone).

Single run (K=20, |B|=1000, seed 42) with dense checkpoints.  The isotropy
probe runs at every checkpoint, recording:

  7A: Across-group cosine similarity of group centroids.
  7B: Within-group spread.
  7C: Effective rank of centroid matrix.
  7D: Selector-index alignment.
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
from src.probes.isotropy import IsotropyProbe

OUTPUT_DIR = Path("outputs")


def main():
    cfg = OmegaConf.load(Path(__file__).parent.parent / "configs" / "base.yaml")
    cfg.experiment.name = "exp7_isotropy"
    cfg.experiment.seed = 42
    cfg.data.k = 20
    cfg.data.n_unique_b = 1000
    cfg.data.task = "bz_to_a"
    cfg.training.max_steps = 30_000
    cfg.training.warmup_steps = 500
    cfg.training.checkpoint_every = 100
    cfg.training.eval_every = 50
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

    model = create_model_from_config(cfg, tokenizer)
    device = model.cfg.device

    iso_probe = IsotropyProbe()
    probe_trajectory: list = []

    # Record at init
    init_res = iso_probe.run_with_mappings(model, tokenizer, mapping_data, device)
    init_res["step"] = 0
    probe_trajectory.append(init_res)

    def on_checkpoint(model, optimizer, step, history, **kw):
        # Dense (every 100) for first 5000, sparse (every 500) after
        if step <= 5000 or step % 500 == 0:
            model.eval()
            res = iso_probe.run_with_mappings(model, tokenizer, mapping_data, device)
            res["step"] = step
            # Trim the cosine sample to save space at frequent checkpoints
            if "across_group_cosine" in res and step <= 5000:
                res["across_group_cosine"]["sample"] = res["across_group_cosine"]["sample"][:200]
            probe_trajectory.append(res)
            model.train()

    callbacks = TrainingCallbacks(on_checkpoint=on_checkpoint)

    train(model, train_loader, probe_loader, cfg, OUTPUT_DIR, callbacks=callbacks)

    # Save probe trajectory
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "isotropy_trajectory.json", "w") as f:
        json.dump(probe_trajectory, f, indent=2,
                  default=lambda o: round(o, 8) if isinstance(o, float) else o)
    print(f"[Exp 7] Saved {len(probe_trajectory)} isotropy snapshots.")


if __name__ == "__main__":
    main()
