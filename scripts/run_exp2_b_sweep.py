#!/usr/bin/env python
"""
Experiment 2 — |B| Sweep with Direct Gradient SNR Measurement.

Sweeps |B| ∈ {50, 100, 250, 500, 1000, 2000} with K=20 fixed.
For each value, runs 5 seeds × 60 000 steps.

At every 200 steps during the plateau (candidate loss ≈ log K, z-gap < 0.5)
the gradient SNR probe is run.  Isotropy probe runs at init, mid-plateau,
and post-convergence.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import math
import torch
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader

from src.data import (
    create_tokenizer_from_config, create_datasets_from_config,
    collate_fn, MappingData, CharTokenizer,
)
from src.model import create_model_from_config
from src.training import train, TrainingCallbacks
from src.probes.gradient_snr import GradientSNRProbe
from src.probes.isotropy import IsotropyProbe


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
B_VALUES = [50, 100, 250, 500, 1000, 2000]
SEEDS = [42, 43, 44, 45, 46]
K = 20
LOG_K = math.log(K)
MAX_STEPS = 60_000
SNR_EVERY = 200
ISOTROPY_CHECKPOINTS = {"init", "mid_plateau", "post_convergence"}


# ---------------------------------------------------------------------------

def _base_cfg(n_b: int, seed: int) -> DictConfig:
    base = OmegaConf.load(Path(__file__).parent.parent / "configs" / "base.yaml")
    base.data.k = K
    base.data.n_unique_b = n_b
    base.data.task = "bz_to_a"
    base.training.max_steps = MAX_STEPS
    base.training.warmup_steps = 500
    base.training.checkpoint_every = 200
    base.training.eval_every = 50
    base.training.batch_size = 128
    base.training.learning_rate = 1e-3
    base.training.weight_decay = 0.01
    base.training.scheduler = "cosine"
    base.experiment.seed = seed
    return base


def run_single(n_b: int, seed: int):
    name = f"exp2_B{n_b}_seed{seed}"
    cfg = _base_cfg(n_b, seed)
    cfg.experiment.name = name

    output_dir = Path("outputs")
    exp_dir = output_dir / name
    history_path = exp_dir / "training_history.json"
    if history_path.exists():
        print(f"[SKIP] {name} already exists")
        return

    print(f"\n{'='*60}\n  {name}  (|B|={n_b}, seed={seed})\n{'='*60}")
    torch.manual_seed(seed)

    tokenizer = create_tokenizer_from_config(cfg)
    train_ds, probe_ds, mapping_data = create_datasets_from_config(cfg, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size,
                              shuffle=True, collate_fn=collate_fn, num_workers=0)
    probe_loader = DataLoader(probe_ds, batch_size=cfg.training.batch_size,
                              shuffle=False, collate_fn=collate_fn, num_workers=0)

    model = create_model_from_config(cfg, tokenizer)
    device = model.cfg.device

    # ---- Probes ----
    snr_probe = GradientSNRProbe({"n_sample_groups": min(200, n_b)})
    iso_probe = IsotropyProbe()

    snr_results = []
    iso_results = {}

    # Isotropy at init
    iso_results["init"] = iso_probe.run_with_mappings(model, tokenizer, mapping_data, device)

    # State tracking for plateau detection (online)
    _in_plateau = [False]
    _plateau_entered = [False]
    _converged = [False]

    def _on_checkpoint(model, optimizer, step, history, **kw):
        # Determine plateau status from recent history
        if not history["first_target_loss"]:
            return
        recent_loss = history["first_target_loss"][-1]
        recent_z_shuf = history["loss_z_shuffled"][-1]
        recent_first = history["first_target_loss"][-1]
        z_gap = recent_z_shuf - recent_first

        in_plateau = abs(recent_loss - LOG_K) < 0.3 * LOG_K and z_gap < 0.5
        if in_plateau:
            _in_plateau[0] = True
            _plateau_entered[0] = True
        else:
            _in_plateau[0] = False
            if _plateau_entered[0] and z_gap > 0.5:
                _converged[0] = True

        # SNR during plateau
        if _in_plateau[0] and step % SNR_EVERY == 0:
            snr = snr_probe.run_with_mappings(model, tokenizer, mapping_data, device, seed=step)
            snr_results.append({"step": step, **snr})

        # Isotropy at mid-plateau (once)
        if _in_plateau[0] and "mid_plateau" not in iso_results:
            iso_results["mid_plateau"] = iso_probe.run_with_mappings(model, tokenizer, mapping_data, device)
            iso_results["mid_plateau"]["step"] = step

    callbacks = TrainingCallbacks(on_checkpoint=_on_checkpoint)

    train(model, train_loader, probe_loader, cfg, output_dir, callbacks=callbacks)

    # Isotropy post-convergence
    iso_results["post_convergence"] = iso_probe.run_with_mappings(model, tokenizer, mapping_data, device)

    # Save probe results
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "snr_results.json", "w") as f:
        json.dump(snr_results, f, indent=2)
    with open(exp_dir / "isotropy_results.json", "w") as f:
        json.dump(iso_results, f, indent=2, default=lambda o: o if not isinstance(o, float) else round(o, 8))

    meta = {"n_b": n_b, "seed": seed, "k": K}
    with open(exp_dir / "exp2_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def main():
    for n_b in B_VALUES:
        for seed in SEEDS:
            run_single(n_b, seed)
    print("\n[Exp 2] All runs complete.")


if __name__ == "__main__":
    main()
