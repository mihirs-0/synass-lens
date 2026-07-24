#!/usr/bin/env python
"""
Suite 2B: Q vs log(K) Slope Across eta.

Measures Q_transition = sum of eta * ||grad L||^2 over the transition window
for multiple (K, eta) pairs. Parallel training, then post-hoc analysis.
"""

import sys
import math
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.experiment_helpers import (
    make_config, run_exists, run_parallel, load_history, detect_tau,
)
from omegaconf import OmegaConf

K_VALUES = [5, 10, 20]
ETA_VALUES = [3e-4, 1e-3, 3e-3]
SEED = 42
MAX_STEPS = 200000
OUTPUT_DIR = "outputs"
MAX_WORKERS = 6

EXISTING_MAP = {
    (5, 1e-3): "landauer_dense_k5",
    (10, 1e-3): "landauer_dense_k10",
    (20, 1e-3): "landauer_dense_k20",
    (20, 3e-4): "lr_sweep_eta3e-4",
    (20, 3e-3): "lr_sweep_eta3e-3",
    (20, 5e-4): "lr_sweep_eta5e-4",
}


def canonical_name(k, eta):
    return f"ziyin_qslope_K{k}_eta{eta}"


def find_existing(k, eta):
    name = canonical_name(k, eta)
    if run_exists(name, min_steps=1, output_dir=OUTPUT_DIR):
        return name
    alias = EXISTING_MAP.get((k, eta))
    if alias and run_exists(alias, min_steps=1, output_dir=OUTPUT_DIR):
        return alias
    return None


def compute_q_transition(history, eta, log_k):
    """Q_transition = sum of eta * ||grad L||^2 over transition window."""
    steps = history.get("steps", [])
    gnorm_sq = history.get("grad_norm_sq", [])
    first_loss = history.get("first_target_loss", [])
    z_shuffled = history.get("loss_z_shuffled", [])

    if not steps or not gnorm_sq or not z_shuffled:
        return None

    z_gaps = [zs - ft for zs, ft in zip(z_shuffled, first_loss)]

    start_idx = end_idx = None
    for i, gap in enumerate(z_gaps):
        if gap > 0.1 and start_idx is None:
            start_idx = i
        if gap > 0.9 * log_k and end_idx is None:
            end_idx = i
            break

    if start_idx is None or end_idx is None:
        return None

    eval_every = steps[1] - steps[0] if len(steps) > 1 else 200
    return sum(eta * gnorm_sq[i] * eval_every for i in range(start_idx, end_idx + 1))


def run_suite2b():
    """Run Q vs logK slope experiment with parallel training."""
    print("=" * 60)
    print("SUITE 2B: Q vs log(K) SLOPE ACROSS ETA")
    print("=" * 60)

    # Step 1: Collect jobs and track existing runs
    run_names = {}
    jobs = []
    for eta in ETA_VALUES:
        for K in K_VALUES:
            existing = find_existing(K, eta)
            if existing:
                run_names[(K, eta)] = existing
                print(f"  [SKIP] K={K}, eta={eta:.0e} ({existing})")
                continue

            name = canonical_name(K, eta)
            run_names[(K, eta)] = name

            cfg = make_config(
                experiment_name=name, task="bz_to_a", k=K, seed=SEED,
                lr=eta, max_steps=MAX_STEPS, eval_every=200,
                checkpoint_every=max(MAX_STEPS // 10, 5000),
                early_stop_frac=0.05,
            )
            jobs.append({"cfg_dict": OmegaConf.to_container(cfg),
                         "mapping_path": None, "output_dir": OUTPUT_DIR,
                         "name": name})

    # Step 2: Run all training in parallel
    print(f"\n  New runs needed: {len(jobs)}")
    run_parallel(jobs, max_workers=MAX_WORKERS, label="Suite2B")

    # Step 3: Compute Q_transition
    print(f"\n--- Computing Q_transition ---")
    results = {}
    for eta in ETA_VALUES:
        q_vals = {}
        for K in K_VALUES:
            name = run_names.get((K, eta))
            if not name:
                continue
            h = load_history(name, output_dir=OUTPUT_DIR)
            if h is None:
                continue
            q = compute_q_transition(h, eta, math.log(K))
            if q is not None:
                q_vals[K] = q
                print(f"  K={K}, eta={eta:.0e}: Q_trans={q:.4f}")
            else:
                print(f"  K={K}, eta={eta:.0e}: transition not detected")
        results[f"{eta}"] = q_vals

    # Step 4: Fit slopes
    print(f"\n--- Q vs log(K) slopes ---")
    summary = []
    for eta in ETA_VALUES:
        q_vals = results.get(f"{eta}", {})
        if len(q_vals) < 2:
            print(f"  eta={eta:.0e}: insufficient data ({len(q_vals)} points)")
            continue
        ks = sorted(q_vals.keys())
        log_ks = np.array([math.log(k) for k in ks])
        qs = np.array([q_vals[k] for k in ks])
        slope, intercept = np.polyfit(log_ks, qs, 1)
        ss_res = np.sum((qs - (slope * log_ks + intercept)) ** 2)
        ss_tot = np.sum((qs - np.mean(qs)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        summary.append({"eta": eta, "slope": slope, "intercept": intercept,
                         "r2": r2, "n": len(ks)})
        print(f"  eta={eta:.0e}: slope={slope:.4f}, R^2={r2:.3f}")

    out_path = Path(OUTPUT_DIR) / "ziyin_q_slope_results.json"
    with open(out_path, "w") as f:
        json.dump({"results": results, "slopes": summary}, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print(f"\n{'='*60}")
    print("SUITE 2B COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_suite2b()
