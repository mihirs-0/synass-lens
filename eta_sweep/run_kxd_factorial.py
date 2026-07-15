#!/usr/bin/env python
"""
K × D factorial sweep at the 30% threshold.

Runs 78 cells in 6-way parallel through a process pool.
Output: eta_sweep/results/kxd_factorial/D{D}/eta_0.001_K_{K}_seed_{seed}/

Skips cells we already have multi-seed coverage for at this threshold:
  - K=5, D=5000  (n_b=1000)  → existing in multiseed_d_replication
  - K=10, D=10000 (n_b=1000) → existing in v_tracking
"""

from __future__ import annotations

import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "eta_sweep" / "results" / "kxd_factorial"
LOG_DIR = RESULTS_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


# Cell list: (K, D, max_steps).  n_b = D / K computed per cell.
# max_steps chosen per D for transition + 2000-step post-tail budget.
CELLS = [
    # D=5000 column (skip K=5, already done at multi-seed)
    (10, 5000,  6000),
    (15, 5000,  6000),
    (20, 5000,  6000),
    (25, 5000,  6000),
    (30, 5000,  6000),
    (36, 5000,  6000),
    # D=10000 column (skip K=10, already done at multi-seed via v_tracking)
    (5,  10000, 8000),
    (15, 10000, 8000),
    (20, 10000, 8000),
    (25, 10000, 8000),
    (30, 10000, 8000),
    (36, 10000, 8000),
    # D=20000 column
    (5,  20000, 12000),
    (10, 20000, 12000),
    (15, 20000, 12000),
    (20, 20000, 12000),
    (25, 20000, 12000),
    (30, 20000, 12000),
    (36, 20000, 12000),
    # D=40000 column (longest cells; pushed last)
    (5,  40000, 20000),
    (10, 40000, 20000),
    (15, 40000, 20000),
    (20, 40000, 20000),
    (25, 40000, 20000),
    (30, 40000, 20000),
    (36, 40000, 20000),
]
SEEDS = [0, 1, 2]
CONCURRENCY = 6


def run_one(K: int, D: int, max_steps: int, seed: int) -> dict:
    """Launch one run.  Returns status dict."""
    n_b = D // K
    sub = f"kxd_factorial/D{D}"
    log_path = LOG_DIR / f"K{K}_D{D}_seed{seed}.log"
    cmd = [
        sys.executable, "-m", "eta_sweep.run_single",
        "--eta", "0.001",
        "--k", str(K),
        "--seed", str(seed),
        "--max-steps", str(max_steps),
        "--n-unique-b", str(n_b),
        "--output-subdir", sub,
    ]
    t0 = time.time()
    with open(log_path, "w") as f:
        rc = subprocess.run(
            cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(REPO_ROOT),
        ).returncode
    return {
        "K": K, "D": D, "seed": seed, "n_b": n_b,
        "max_steps": max_steps,
        "returncode": rc,
        "wall_s": time.time() - t0,
        "log": str(log_path),
    }


def main():
    # Build full task list
    tasks = []
    for K, D, max_steps in CELLS:
        for seed in SEEDS:
            tasks.append((K, D, max_steps, seed))

    print(f"Launching {len(tasks)} runs with {CONCURRENCY}-way concurrency.")
    print(f"Output: {RESULTS_DIR}")
    sys.stdout.flush()

    t_start = time.time()
    completed = 0
    with ProcessPoolExecutor(max_workers=CONCURRENCY) as ex:
        futures = {ex.submit(run_one, *t): t for t in tasks}
        for fut in as_completed(futures):
            result = fut.result()
            completed += 1
            elapsed = time.time() - t_start
            rate = completed / max(elapsed, 1)
            eta_s = (len(tasks) - completed) / max(rate, 1e-6)
            print(
                f"[{completed}/{len(tasks)}] "
                f"K={result['K']:2d} D={result['D']:5d} seed={result['seed']} "
                f"rc={result['returncode']} wall={result['wall_s']:.0f}s "
                f"| elapsed={elapsed/60:.1f}min ETA={eta_s/60:.1f}min"
            )
            sys.stdout.flush()

    print(f"\nALL DONE: {len(tasks)} runs in {(time.time()-t_start)/60:.1f} minutes")


if __name__ == "__main__":
    main()
