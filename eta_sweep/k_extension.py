#!/usr/bin/env python
"""
K-extension sweep — clean D-scaling version.

Runs 5 cells at K > 36 to test whether Q ∝ log K extends beyond K=36
without a dataset-size confound.

All cells use n_b = 1000 (matching every cell at K ≤ 36 in the sweep),
so D = n_b · K scales proportionally to K — exactly the same D(K) scaling
as the baseline data.

Cells:
    η = 3e-4 × K ∈ {50, 75, 100}   (Experiment B: extend η=3e-4 cleanly)
    η = 7e-4 × K ∈ {50, 75}        (Experiment C: test saturation at a
                                     second η value)

K > 36 requires disambiguation_prefix_length = 2 (vocab of 36 chars only
supports K ≤ 36 at prefix_length = 1).  n_b is kept at 1000.

Extended-budget config (same as rerun_stuck + extend_sweep):
    min_steps      = 20 000
    stuck_patience = 5 000
    max_steps      = 100 000
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

from eta_sweep.config import RESULTS_DIR, run_dir  # noqa: E402


@dataclass(frozen=True)
class Cell:
    eta: float
    k: int
    seed: int = 0
    n_unique_b: int = 1000                  # ← fixed, same as baseline
    disambiguation_prefix_length: int = 2   # ← required for K > 36
    max_steps: int = 100_000
    experiment: str = "K-ext"


CELLS: List[Cell] = [
    Cell(eta=3e-4, k=50),
    Cell(eta=3e-4, k=75),
    Cell(eta=3e-4, k=100),
    Cell(eta=7e-4, k=50),
    Cell(eta=7e-4, k=75),
]

MIN_STEPS_NEW = 20_000
STUCK_PATIENCE_NEW = 5_000


def _launch_cell(cell: Cell, log_path: Path) -> subprocess.Popen:
    cmd = [
        sys.executable, "-u",
        str(ETA_SWEEP_ROOT / "run_single.py"),
        "--eta", str(cell.eta),
        "--k", str(cell.k),
        "--seed", str(cell.seed),
        "--max-steps", str(cell.max_steps),
        "--min-steps", str(MIN_STEPS_NEW),
        "--stuck-patience", str(STUCK_PATIENCE_NEW),
        "--n-unique-b", str(cell.n_unique_b),
        "--disambiguation-prefix-length", str(cell.disambiguation_prefix_length),
        "--resume",
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    log_fp = open(log_path, "w", buffering=1)
    proc = subprocess.Popen(
        cmd, stdout=log_fp, stderr=subprocess.STDOUT,
        env=env, cwd=str(REPO_ROOT),
    )
    proc._log_fp = log_fp  # type: ignore[attr-defined]
    return proc


def _read_status(cell: Cell) -> dict:
    p = run_dir(cell.eta, cell.k, cell.seed) / "status.json"
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return {}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    print(f"K-extension queue: {len(CELLS)} cells, {args.workers} workers")
    for c in CELLS:
        D = c.n_unique_b * c.k
        print(f"  η={c.eta:g}  K={c.k:>3}  n_b={c.n_unique_b}  D={D:,}")

    log_dir = RESULTS_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    summary_fp = open(RESULTS_DIR / "k_extension_summary.jsonl", "a",
                      buffering=1)

    queue = list(CELLS)
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
                     "run_name": f"eta_{cell.eta:g}_K_{cell.k}_seed_{cell.seed}",
                     "eta": cell.eta, "k": cell.k, "seed": cell.seed,
                     "returncode": rc}
            done.append(s)
            summary_fp.write(json.dumps(s) + "\n")
            print(f"  DONE  η={cell.eta:g}  K={cell.k:>3}  "
                  f"status={s.get('status','?'):<14}  "
                  f"step={s.get('final_step',0):>7}  "
                  f"wall={s.get('wall_clock_s',0):>7.1f}s")
        running = still

    try:
        while queue or running:
            while queue and len(running) < args.workers:
                cell = queue.pop(0)
                log_path = log_dir / (
                    f"kext_eta_{cell.eta:g}_K_{cell.k}_seed_{cell.seed}.log"
                )
                proc = _launch_cell(cell, log_path)
                running.append((proc, cell))
                print(f"  START η={cell.eta:g}  K={cell.k:>3}  "
                      f"pid={proc.pid}  log={log_path}")
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
    print(f"K-extension complete: {len(done)} cells in {elapsed/60:.1f} min")
    print("=" * 70)
    from collections import Counter
    ct = Counter(d["status"] for d in done)
    print(dict(ct))


if __name__ == "__main__":
    main()
