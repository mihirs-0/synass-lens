#!/usr/bin/env python
"""
Suite 2A: Label Noise Sweep.

For each (K, noise_level), train bz_to_a with within-group label noise.
Reuses existing K=20 noise runs where available.
Single seed. Parallel GPU execution with early stopping.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.experiment_helpers import (
    make_config, run_exists, run_parallel,
)
from omegaconf import OmegaConf

K_VALUES = [10, 20, 36]
NOISE_LEVELS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]
SEED = 42
LR = 1e-3
MAX_STEPS = 200000
OUTPUT_DIR = "outputs"
MAX_WORKERS = 6

EXISTING_ALIASES = {
    (20, 0.0): "landauer_dense_k20",
    (20, 0.05): "landauer_k20_noise0.05",
    (20, 0.10): "landauer_k20_noise0.10",
    (20, 0.20): "landauer_k20_noise0.20",
    (10, 0.0): "landauer_dense_k10",
    (36, 0.0): "landauer_dense_k36",
}


def canonical_name(k, noise):
    if noise == 0.0:
        return f"ziyin_noise_K{k}_p0.00"
    return f"ziyin_noise_K{k}_p{noise:.2f}"


def find_existing(k, noise):
    name = canonical_name(k, noise)
    if run_exists(name, min_steps=1, output_dir=OUTPUT_DIR):
        return name
    alias = EXISTING_ALIASES.get((k, noise))
    if alias and run_exists(alias, min_steps=1, output_dir=OUTPUT_DIR):
        return alias
    return None


def run_suite2a():
    """Run label noise sweep with parallel execution."""
    print("=" * 60)
    print("SUITE 2A: LABEL NOISE SWEEP")
    print("=" * 60)

    jobs = []
    for K in K_VALUES:
        for noise in NOISE_LEVELS:
            existing = find_existing(K, noise)
            if existing:
                print(f"  [SKIP] K={K}, noise={noise:.2f} ({existing})")
                continue

            name = canonical_name(K, noise)
            es_frac = 0.05 if noise < 0.10 else None

            cfg = make_config(
                experiment_name=name, task="bz_to_a", k=K, seed=SEED,
                lr=LR, max_steps=MAX_STEPS, eval_every=200,
                checkpoint_every=max(MAX_STEPS // 10, 5000),
                label_noise_prob=noise, early_stop_frac=es_frac,
            )
            jobs.append({"cfg_dict": OmegaConf.to_container(cfg),
                         "mapping_path": None, "output_dir": OUTPUT_DIR,
                         "name": name})

    print(f"\n  New runs needed: {len(jobs)}")
    run_parallel(jobs, max_workers=MAX_WORKERS, label="Suite2A")

    print(f"\n{'='*60}")
    print("SUITE 2A COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_suite2a()
