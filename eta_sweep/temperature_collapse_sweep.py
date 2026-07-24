#!/usr/bin/env python
"""
Experiment 2 — η/B collapse test.

Tests whether escape time depends on η and B only through η/B (the SGD-
as-Langevin "effective temperature" prediction that the thermodynamic
framework requires).  A bare metastable-bifurcation account doesn't
require this collapse.

Scope:
  * K=20 fixed (well-characterised critical region, moderate compute)
  * Batch-size sweep:   B ∈ {32, 64, 128, 256, 512} at fixed η = 1e-3 (5)
  * Learning-rate sweep: η ∈ {5e-4, 1e-3, 2e-3, 3e-3} at fixed B = 128 (4)
  * shared cell: (η=1e-3, B=128) — run once, counted in both sweeps
  * seeds ∈ {1, 2, 3}
  * Unique cells: (5 + 4 - 1) × 3 = 24

Crucial design point already verified by pre-flight audit:
  `trainer.py:240` and `run_single.py:494` pass `lr=cc.eta` directly
  to AdamW with NO batch-size rescaling.  So η is the literal per-step
  learning rate across batch sizes — the collapse test is well-founded.

Output layout — nested per batch size to avoid directory collisions:
  eta_sweep/results/temperature_collapse/B_32/eta_0.001_K_20_seed_1/
  eta_sweep/results/temperature_collapse/B_64/eta_0.001_K_20_seed_1/
  ...

Thresholds: main-sweep extended budget (min_steps=20k, stuck_patience=5k,
max_steps=50k).  Early transitions wrap cleanly without hitting the
full budget.
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


# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------

K_FIXED = 20
SEEDS = [1, 2, 3]

B_SWEEP_ETA = 1e-3
B_SWEEP_VALUES = [32, 64, 128, 256, 512]

ETA_SWEEP_B = 128
ETA_SWEEP_VALUES = [5e-4, 1e-3, 2e-3, 3e-3]

MIN_STEPS = 20_000
STUCK_PATIENCE = 5_000
MAX_STEPS = 50_000

SUBDIR_BASE = "temperature_collapse"


@dataclass(frozen=True)
class Cell:
    eta: float
    batch_size: int
    seed: int
    tag: str  # "B_sweep" or "eta_sweep" or "shared"


# Build the 24-cell grid (15 + 12 − 3 shared at η=1e-3 B=128 × 3 seeds).
_seen: set = set()
CELLS: List[Cell] = []
# B-sweep at η=1e-3
for B in B_SWEEP_VALUES:
    for seed in SEEDS:
        key = (B_SWEEP_ETA, B, seed)
        if key in _seen:
            continue
        _seen.add(key)
        tag = "shared" if (B == ETA_SWEEP_B and abs(B_SWEEP_ETA - ETA_SWEEP_B * 0) > 0 and abs(B_SWEEP_ETA - 1e-3) < 1e-12) else "B_sweep"
        CELLS.append(Cell(eta=B_SWEEP_ETA, batch_size=B, seed=seed, tag=tag))
# η-sweep at B=128
for eta in ETA_SWEEP_VALUES:
    for seed in SEEDS:
        key = (eta, ETA_SWEEP_B, seed)
        if key in _seen:
            continue
        _seen.add(key)
        CELLS.append(Cell(eta=eta, batch_size=ETA_SWEEP_B, seed=seed, tag="eta_sweep"))


def cell_subdir(cell: Cell) -> str:
    return f"{SUBDIR_BASE}/B_{cell.batch_size}"


def cell_out_dir(cell: Cell) -> Path:
    return (
        RESULTS_DIR
        / SUBDIR_BASE
        / f"B_{cell.batch_size}"
        / f"eta_{cell.eta:g}_K_{K_FIXED}_seed_{cell.seed}"
    )


def _has_status(cell: Cell) -> bool:
    return (cell_out_dir(cell) / "status.json").exists()


def _read_status(cell: Cell) -> dict:
    p = cell_out_dir(cell) / "status.json"
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
        "--eta", str(cell.eta),
        "--k", str(K_FIXED),
        "--seed", str(cell.seed),
        "--batch-size", str(cell.batch_size),
        "--max-steps", str(MAX_STEPS),
        "--min-steps", str(MIN_STEPS),
        "--stuck-patience", str(STUCK_PATIENCE),
        "--output-subdir", cell_subdir(cell),
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
    print(f"Temperature-collapse sweep: {len(pending)} / {len(CELLS)} cells "
          f"pending, {args.workers} workers")
    print(f"  grid fixed K={K_FIXED}, seeds={SEEDS}")
    print(f"  B-sweep: η={B_SWEEP_ETA:g}, B ∈ {B_SWEEP_VALUES}")
    print(f"  η-sweep: B={ETA_SWEEP_B}, η ∈ {ETA_SWEEP_VALUES}")
    print()
    for c in pending:
        print(f"  [{c.tag:<9}] η={c.eta:g}  B={c.batch_size:>3}  seed={c.seed}")

    if args.dry_run:
        return
    if not pending:
        print("nothing to do")
        return

    log_dir = RESULTS_DIR / SUBDIR_BASE / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_fp = open(
        RESULTS_DIR / SUBDIR_BASE / "summary.jsonl", "a", buffering=1
    )

    readme = RESULTS_DIR / SUBDIR_BASE / "README.md"
    if not readme.exists():
        readme.write_text(
            "# Experiment 2 — η/B collapse test\n\n"
            f"Fixed K={K_FIXED}.  Tests the SGD-as-Langevin prediction "
            "that escape time depends on (η, B) only through η/B.\n\n"
            f"B-sweep at η={B_SWEEP_ETA:g}: B ∈ {B_SWEEP_VALUES}\n"
            f"η-sweep at B={ETA_SWEEP_B}:   η ∈ {ETA_SWEEP_VALUES}\n"
            f"Seeds: {SEEDS}.  Shared cell: (η=1e-3, B=128).  Unique cells: "
            f"{len(CELLS)}.\n\n"
            f"Thresholds: min_steps={MIN_STEPS}, stuck_patience={STUCK_PATIENCE}, "
            f"max_steps={MAX_STEPS}.\n\n"
            "η is literal per-step LR (not batch-size-rescaled) per trainer.py:240.\n\n"
            "Analysis: escape time (`t1` or `final_step` when transitioned) vs "
            "η (at fixed B) and vs η/B (combining sweeps).  Collapse on η/B ⇒ "
            "thermodynamic framework supported.\n"
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
                     "run_name": cell_out_dir(cell).name,
                     "eta": cell.eta, "k": K_FIXED, "seed": cell.seed,
                     "returncode": rc}
            s["batch_size"] = cell.batch_size
            s["tag"] = cell.tag
            done.append(s)
            summary_fp.write(json.dumps(s) + "\n")
            elapsed = time.time() - t0
            frac = (len(done) + 0.0) / max(len(pending), 1)
            eta_s = (elapsed / max(frac, 1e-9)) - elapsed
            print(f"  [{len(done):>3}/{len(pending)}] DONE "
                  f"η={cell.eta:g} B={cell.batch_size:>3} s={cell.seed}  "
                  f"status={s.get('status','?'):<14}  "
                  f"step={s.get('final_step',0):>6}  "
                  f"wall={s.get('wall_clock_s',0):>6.1f}s  "
                  f"ETA={eta_s/60:.1f}min")
        running = still

    try:
        while queue or running:
            while queue and len(running) < args.workers:
                cell = queue.pop(0)
                log_path = log_dir / (
                    f"tempcol_eta_{cell.eta:g}_B_{cell.batch_size}"
                    f"_K_{K_FIXED}_seed_{cell.seed}.log"
                )
                proc = _launch_cell(cell, log_path)
                running.append((proc, cell))
                print(f"  START η={cell.eta:g} B={cell.batch_size:>3} "
                      f"s={cell.seed}  pid={proc.pid}")
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
    print(f"Temperature-collapse sweep complete: {len(done)} cells "
          f"in {elapsed/60:.1f} min")
    print("=" * 70)
    from collections import Counter
    print("  by status:", dict(Counter(d["status"] for d in done)))
    print("  by B:     ", dict(Counter(d["batch_size"] for d in done)))


if __name__ == "__main__":
    main()
