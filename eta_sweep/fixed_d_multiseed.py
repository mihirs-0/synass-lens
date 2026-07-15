#!/usr/bin/env python
"""
Fixed-D multi-seed sweep — paper-payoff replication of confound v2.

Goal: replace 'fixed-D controls are single-seed' with 'fixed-D controls
replicated over 3 seeds', so the joint n_b · K^α fit on confound data
gets a real seed-replication SE on α.

Path 1 (chosen): we use eta_sweep's `run_single.py` with the standard
held-out unclipped gradient, NOT the parent-repo training-batch clipped
gradient.  This makes the new fixed-D grid directly comparable to the
rest of the eta_sweep (same Q pipeline) at the cost of breaking direct
comparability with the existing confound v2 absolute Q values.  The old
v2 cells become supporting evidence with a 'different gradient
definition' caveat.

Grid:
  D ∈ {10 000, 20 000}    (matched to v2)
  K ∈ {5, 10, 20, 36}     (matched to v2)
  n_b = D / K rounded:    {2000, 1000,  500,  278}  at D=10k
                          {4000, 2000, 1000,  556}  at D=20k
  seeds ∈ {0, 1, 2}        (eta_sweep convention)
  Total cells: 2 × 4 × 3 = 24

Per-cell config:
  η = 1e-3
  batch_size = 128
  scheduler = constant, warmup = 0
  min_steps = 20 000, stuck_patience = 5 000, max_steps = 100 000

Output layout (avoids collisions across D values for same K):
  eta_sweep/results/fixed_d_multiseed/D_<D>/eta_<η>_K_<K>_seed_<s>/

After-run analysis (next step, not in this script):
  - Refit M7 (Q ~ a · n_b · K^α + b) with seed-bootstrap CI on α
  - Test sign reversal of Q_transition vs log K at fixed D, seed-by-seed
  - Mean ± SE for τ ≈ t₁ at each (D, K)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

ETA_SWEEP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = ETA_SWEEP_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from eta_sweep.config import RESULTS_DIR  # noqa: E402


# --- Grid (matched exactly to confound v2 K values and n_b values) ---------

ETA = 1e-3
BATCH_SIZE = 128
SEEDS = [0, 1, 2]
MIN_STEPS = 20_000
STUCK_PATIENCE = 5_000
MAX_STEPS = 100_000

# (D, K, n_b) tuples — n_b values literally match confound v2.
GRID: List[Tuple[int, int, int]] = [
    (10_000,  5, 2000),
    (10_000, 10, 1000),
    (10_000, 20,  500),
    (10_000, 36,  278),
    (20_000,  5, 4000),
    (20_000, 10, 2000),
    (20_000, 20, 1000),
    (20_000, 36,  556),
]

SUBDIR_BASE = "fixed_d_multiseed"


@dataclass(frozen=True)
class Cell:
    D: int
    K: int
    n_b: int
    seed: int

    @property
    def subdir(self) -> str:
        return f"{SUBDIR_BASE}/D_{self.D}"

    @property
    def out_dir(self) -> Path:
        return (RESULTS_DIR / SUBDIR_BASE / f"D_{self.D}"
                / f"eta_{ETA:g}_K_{self.K}_seed_{self.seed}")


CELLS: List[Cell] = [
    Cell(D=D, K=K, n_b=n_b, seed=seed)
    for (D, K, n_b) in GRID
    for seed in SEEDS
]


def _has_status(cell: Cell) -> bool:
    return (cell.out_dir / "status.json").exists()


def _read_status(cell: Cell) -> dict:
    p = cell.out_dir / "status.json"
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return {}


def _launch_cell(cell: Cell, log_path: Path) -> subprocess.Popen:
    cmd = [
        sys.executable, "-u",
        str(ETA_SWEEP_ROOT / "run_single.py"),
        "--eta", str(ETA),
        "--k", str(cell.K),
        "--seed", str(cell.seed),
        "--batch-size", str(BATCH_SIZE),
        "--n-unique-b", str(cell.n_b),
        "--max-steps", str(MAX_STEPS),
        "--min-steps", str(MIN_STEPS),
        "--stuck-patience", str(STUCK_PATIENCE),
        "--output-subdir", cell.subdir,
        "--resume",
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    log_fp = open(log_path, "w", buffering=1)
    proc = subprocess.Popen(
        cmd, stdout=log_fp, stderr=subprocess.STDOUT,
        env=env, cwd=str(REPO_ROOT),
    )
    proc._log_fp = log_fp  # type: ignore[attr-defined]
    return proc


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    pending = [c for c in CELLS if not _has_status(c)]
    print(f"Fixed-D multiseed sweep: {len(pending)} / {len(CELLS)} cells "
          f"pending, {args.workers} workers")
    print(f"  η = {ETA:g}, batch_size = {BATCH_SIZE}, seeds = {SEEDS}")
    print(f"  thresholds: min_steps={MIN_STEPS}, stuck_patience={STUCK_PATIENCE},"
          f" max_steps={MAX_STEPS}")
    print()
    for c in pending:
        print(f"  D={c.D:>5}  K={c.K:>2}  n_b={c.n_b:>4}  seed={c.seed}")

    if args.dry_run:
        return
    if not pending:
        print("nothing to do")
        return

    base = RESULTS_DIR / SUBDIR_BASE
    base.mkdir(parents=True, exist_ok=True)
    log_dir = base / "logs"
    log_dir.mkdir(exist_ok=True)
    summary_fp = open(base / "summary.jsonl", "a", buffering=1)

    readme = base / "README.md"
    if not readme.exists():
        readme.write_text(
            "# Fixed-D multi-seed control sweep (3 seeds × 8 (D, K) pairs = 24 cells)\n\n"
            f"η = {ETA:g}, batch_size = {BATCH_SIZE}, seeds = {SEEDS}.  "
            "Replicates the confound v2 grid using the eta_sweep `run_single.py` "
            "trainer (held-out batch gradient, no clipping) so that the new "
            "fixed-D cells are directly comparable to the rest of the eta_sweep.\n\n"
            "After completion: refit Q ~ a · n_b · K^α + b with seed-bootstrap "
            "CIs on α; test sign reversal of Q_transition vs log K at fixed D "
            "across seeds; report τ ≈ t₁ mean ± SE per (D, K).\n"
        )

    queue = list(pending)
    running: List[Tuple[subprocess.Popen, Cell]] = []
    done: List[dict] = []
    t0 = time.time()

    def _reap() -> None:
        nonlocal running
        still = []
        for proc, cell in running:
            rc = proc.poll()
            if rc is None:
                still.append((proc, cell))
                continue
            try: proc._log_fp.close()  # type: ignore[attr-defined]
            except Exception: pass
            s = _read_status(cell)
            if not s:
                s = {"status": "crashed" if rc != 0 else "unknown",
                     "final_step": 0, "wall_clock_s": 0.0,
                     "run_name": cell.out_dir.name,
                     "eta": ETA, "k": cell.K, "seed": cell.seed,
                     "returncode": rc}
            s.update({"D": cell.D, "n_b": cell.n_b,
                      "experiment": "fixed_d_multiseed"})
            done.append(s)
            summary_fp.write(json.dumps(s) + "\n")
            elapsed = time.time() - t0
            frac = (len(done) + 0.0) / max(len(pending), 1)
            eta_s = (elapsed / max(frac, 1e-9)) - elapsed
            print(f"  [{len(done):>2}/{len(pending)}] DONE  "
                  f"D={cell.D:>5}  K={cell.K:>2}  n_b={cell.n_b:>4}  "
                  f"s={cell.seed}  status={s.get('status','?'):<14}  "
                  f"step={s.get('final_step',0):>6}  "
                  f"wall={s.get('wall_clock_s',0):>6.1f}s  "
                  f"ETA={eta_s/60:.1f}min")
        running = still

    try:
        while queue or running:
            while queue and len(running) < args.workers:
                cell = queue.pop(0)
                log_path = log_dir / (
                    f"fixedD_D_{cell.D}_K_{cell.K}_n_b_{cell.n_b}"
                    f"_seed_{cell.seed}.log"
                )
                proc = _launch_cell(cell, log_path)
                running.append((proc, cell))
                print(f"  START D={cell.D:>5}  K={cell.K:>2}  "
                      f"n_b={cell.n_b:>4}  seed={cell.seed}  pid={proc.pid}")
            time.sleep(2.0)
            _reap()
    except KeyboardInterrupt:
        print("\n[interrupted] terminating workers…")
        for proc, _ in running:
            try: proc.terminate()
            except Exception: pass
        raise

    summary_fp.close()
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"Fixed-D multiseed complete: {len(done)} cells in {elapsed/60:.1f} min")
    print("=" * 70)
    from collections import Counter
    print("  by status:", dict(Counter(d["status"] for d in done)))


if __name__ == "__main__":
    main()
