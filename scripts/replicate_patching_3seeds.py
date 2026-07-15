"""Re-run analyze_cell for 3 seeds at canonical K=20, η=1e-3 (Cell A config).

Existing Cell A is seed 0; this adds seeds 1 and 2 (also re-runs seed 0 to
guarantee consistent eval-batch construction across seeds).

Outputs JSON to eta_sweep/results/multi_seed_patching_K20.json.

Usage:
    python scripts/replicate_patching_3seeds.py
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eta_sweep.analysis.mech_interp_mbc import (  # noqa: E402
    CellSpec, analyze_cell, _select_device,
)


def main():
    device = _select_device()
    print(f"device: {device}")

    cells = [
        CellSpec("CellA_seed0", 0.001, 20, 0,
                 "Canonical AdamW (β1=0.9), K=20, η=1e-3, seed=0 (= Cell A)"),
        CellSpec("CellA_seed1", 0.001, 20, 1,
                 "Canonical AdamW, K=20, η=1e-3, seed=1"),
        CellSpec("CellA_seed2", 0.001, 20, 2,
                 "Canonical AdamW, K=20, η=1e-3, seed=2"),
    ]

    all_out = {"device": device, "cells": {}}
    t_start = time.time()

    for cs in cells:
        t0 = time.time()
        try:
            cell_out = analyze_cell(cs, device,
                                    n_eval=128,
                                    n_probe_train=192,
                                    n_probe_test=64,
                                    n_ablation_batch=24,
                                    n_patch_pairs=32)
        except Exception as e:
            print(f"FAILED on {cs.label}: {e}")
            cell_out = {"error": str(e), "label": cs.label}
        dt = time.time() - t0
        cell_out["wall_clock_s"] = dt
        all_out["cells"][cs.label] = cell_out
        print(f"=== {cs.label} done in {dt:.1f}s ===")

    all_out["total_wall_clock_s"] = time.time() - t_start

    out_path = REPO_ROOT / "eta_sweep" / "results" / "multi_seed_patching_K20.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_out, f, indent=2, default=lambda o: o.tolist()
                  if hasattr(o, "tolist") else str(o))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
