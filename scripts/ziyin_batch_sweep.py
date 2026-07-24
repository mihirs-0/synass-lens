#!/usr/bin/env python
"""
Suite 2C: Batch Size Sweep at K=20.

Varies batch size while keeping eta=1e-3 fixed.
Tests whether gradient noise stabilizes or destabilizes the plateau.
Parallel GPU execution with early stopping.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.experiment_helpers import (
    make_config, run_exists, run_parallel,
)
from omegaconf import OmegaConf

K = 20
BATCH_SIZES = [32, 64, 128, 256, 512]
SEED = 42
LR = 1e-3
MAX_STEPS = 200000
OUTPUT_DIR = "outputs"
MAX_WORKERS = 5  # one per batch size

EXISTING_MAP = {
    64: "temp_lr1e3_bs64",
    128: "temp_lr1e3_bs128",
    256: "temp_lr1e3_bs256",
    512: "temp_lr1e3_bs512",
}


def canonical_name(bs):
    return f"ziyin_batch_K{K}_bs{bs}"


def find_existing(bs):
    name = canonical_name(bs)
    if run_exists(name, min_steps=1, output_dir=OUTPUT_DIR):
        return name
    alias = EXISTING_MAP.get(bs)
    if alias and run_exists(alias, min_steps=1, output_dir=OUTPUT_DIR):
        return alias
    return None


def run_suite2c():
    """Run batch size sweep with parallel execution."""
    print("=" * 60)
    print(f"SUITE 2C: BATCH SIZE SWEEP (K={K}, eta={LR})")
    print("=" * 60)

    jobs = []
    for bs in BATCH_SIZES:
        existing = find_existing(bs)
        if existing:
            print(f"  [SKIP] BS={bs} ({existing})")
            continue

        name = canonical_name(bs)
        cfg = make_config(
            experiment_name=name, task="bz_to_a", k=K, seed=SEED,
            lr=LR, bs=bs, max_steps=MAX_STEPS, eval_every=200,
            checkpoint_every=max(MAX_STEPS // 10, 5000),
            early_stop_frac=0.05,
        )
        jobs.append({"cfg_dict": OmegaConf.to_container(cfg),
                     "mapping_path": None, "output_dir": OUTPUT_DIR,
                     "name": name})

    print(f"\n  New runs needed: {len(jobs)}")
    run_parallel(jobs, max_workers=MAX_WORKERS, label="Suite2C")

    print(f"\n{'='*60}")
    print("SUITE 2C COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_suite2c()
