#!/usr/bin/env python
"""
Experiment 4 — Dataset reversal sweep.

Tests whether phase 1 (marginal acquisition) exhibits direction
dependence when the training sequence is token-reversed, while phase 2
(conditional acquisition) is expected to be physically blocked because
z is causally downstream of the A targets in the reversed sequence.

Scope:
  * K ∈ {10, 36}
  * η ∈ {1e-4, 3e-4, 7e-4, 1e-3, 2.4e-3, 2.6e-3, 2.8e-3, 3e-3}
  * seeds ∈ {1, 2, 3}
  * 48 cells total
  * option-(b) reversal: BOS/EOS stay, middle flipped

Shortened stuck thresholds vs the main sweep: since phase 2 is
physically inaccessible in the reversed direction (z is past A in the
sequence), we expect every cell to plateau on phase 1 forever.  We
classify "stuck" early to save compute:

    min_steps      = 10 000
    stuck_patience = 3 000
    max_steps      = 30 000

Output: eta_sweep/results/dataset_reversal/eta_X_K_Y_seed_Z/
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


ETAS = [1e-4, 3e-4, 7e-4, 1e-3, 2.4e-3, 2.6e-3, 2.8e-3, 3e-3]
KS = [10, 36]
SEEDS = [1, 2, 3]

MIN_STEPS = 10_000
STUCK_PATIENCE = 3_000
MAX_STEPS = 30_000

SUBDIR = "dataset_reversal"


@dataclass(frozen=True)
class Cell:
    eta: float
    k: int
    seed: int


CELLS: List[Cell] = [
    Cell(eta, k, seed)
    for eta in ETAS for k in KS for seed in SEEDS
]


def cell_out_dir(cell: Cell) -> Path:
    return RESULTS_DIR / SUBDIR / f"eta_{cell.eta:g}_K_{cell.k}_seed_{cell.seed}"


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
        "--k", str(cell.k),
        "--seed", str(cell.seed),
        "--max-steps", str(MAX_STEPS),
        "--min-steps", str(MIN_STEPS),
        "--stuck-patience", str(STUCK_PATIENCE),
        "--reverse",
        "--output-subdir", SUBDIR,
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
    print(f"Reversal sweep: {len(pending)} / {len(CELLS)} cells pending, "
          f"{args.workers} workers")
    for c in pending:
        print(f"  η={c.eta:g}  K={c.k:>2}  seed={c.seed}")

    if args.dry_run:
        return
    if not pending:
        print("nothing to do")
        return

    log_dir = RESULTS_DIR / SUBDIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_fp = open(RESULTS_DIR / SUBDIR / "summary.jsonl", "a", buffering=1)

    # Write a small README describing the run.
    readme = RESULTS_DIR / SUBDIR / "README.md"
    if not readme.exists():
        readme.write_text(
            "# Dataset reversal experiment (Experiment 4)\n\n"
            "Option-(b) reversal: BOS and EOS kept at boundary; middle "
            "(B, z, A segments) flipped.  In the reversed sequence, A tokens "
            "live at positions 1-4 and z at positions 6-7, so z is causally "
            "downstream of A — the model cannot use z to predict A.\n\n"
            f"Grid: K ∈ {KS}, η ∈ {ETAS}, seeds ∈ {SEEDS} "
            f"({len(CELLS)} cells).\n\n"
            f"Thresholds: min_steps={MIN_STEPS}, "
            f"stuck_patience={STUCK_PATIENCE}, max_steps={MAX_STEPS}.\n\n"
            "Every cell is expected to classify 'stuck' at step ~13k if the "
            "reversed task lacks phase 2.  Any cell that transitions is a "
            "surprise and indicates the reversed task has some learnable "
            "z→A path we missed.\n"
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
                     "eta": cell.eta, "k": cell.k, "seed": cell.seed,
                     "returncode": rc}
            s["experiment"] = "reversal"
            done.append(s)
            summary_fp.write(json.dumps(s) + "\n")
            elapsed = time.time() - t0
            frac = (len(done) + 0.0) / max(len(pending), 1)
            eta_s = (elapsed / max(frac, 1e-9)) - elapsed
            print(f"  [{len(done):>3}/{len(pending)}] DONE "
                  f"η={cell.eta:g} K={cell.k:>2} s={cell.seed}  "
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
                    f"reversal_eta_{cell.eta:g}_K_{cell.k}_seed_{cell.seed}.log"
                )
                proc = _launch_cell(cell, log_path)
                running.append((proc, cell))
                print(f"  START η={cell.eta:g} K={cell.k:>2} s={cell.seed}  "
                      f"pid={proc.pid}")
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
    print(f"Reversal sweep complete: {len(done)} cells in {elapsed/60:.1f} min")
    print("=" * 70)
    from collections import Counter
    print("  by status:", dict(Counter(d["status"] for d in done)))


if __name__ == "__main__":
    main()
