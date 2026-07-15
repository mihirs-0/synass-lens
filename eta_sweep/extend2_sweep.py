#!/usr/bin/env python
"""
η-Sweep second extension: pin down η_c and test K-extension.

THREE EXPERIMENTS:

A — Pin down η_c between 2e-3 and 3e-3.
    20 cells: η ∈ {2.2e-3, 2.4e-3, 2.6e-3, 2.8e-3} × K ∈ {10, 15, 20, 25, 36}.
    Launch order: middle values first (2.4, 2.6), then outer (2.2, 2.8).

B — Extend K range at η=3e-4 to test power-law external validity.
    3 cells: η=3e-4 × K ∈ {50, 75, 100}.

C — Test whether η=1e-3 Q-saturation is η-specific.
    4 cells: η ∈ {3e-4, 7e-4} × K ∈ {50, 75}.
    Note: two of these cells overlap with Experiment B
    ((3e-4, 50), (3e-4, 75)).  We run each physical cell once.

TOTAL UNIQUE NEW CELLS: 25 (20 Experiment A + 3 Experiment B + 4
Experiment C − 2 shared between B and C).

For K > 36 we override two dataset knobs:
  * n_unique_b = 50 · K   (scales dataset to keep D = n_b · K large)
  * disambiguation_prefix_length = 2  (enforce_unique_a_first_char_per_b
    with prefix_length=1 caps K at len(vocab)=36; prefix_length=2
    caps at 36² = 1296.  We preserve the "unique prefix" property.)

All cells use the extended-budget config:
    min_steps        = 20 000
    stuck_patience   = 5 000
    max_steps        = 100 000
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


# ---------------------------------------------------------------------------
# Cell definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Cell:
    eta: float
    k: int
    seed: int = 0
    n_unique_b: int = 1000
    disambiguation_prefix_length: int = 1
    max_steps: int = 100_000
    experiment: str = "A"  # "A", "B", or "C" (or "BC" for the shared cells)


KS_CORE = [10, 15, 20, 25, 36]

# --- Recovery cells ---------------------------------------------------------
# Four cells at η=2e-3 (K=10,15,20,25) were corrupted by an accidental
# directory-name collision with the 2.4e-3 cells (old f"{eta:.0e}" formatting
# rounded both to "2e-03").  The directory-name format is now f"{eta:g}" which
# disambiguates; we retrain these four cells from scratch using the same
# extended-budget config as the surrounding extend_sweep runs.
EXP_RECOVERY_CELLS = [
    Cell(eta=2e-3, k=k, experiment="RECOVER") for k in [10, 15, 20, 25]
]

# --- Experiment A: critical-region fill-ins ---------------------------------
EXP_A_ETAS_MIDDLE = [2.4e-3, 2.6e-3]
EXP_A_ETAS_OUTER = [2.2e-3, 2.8e-3]

EXP_A_CELLS_MIDDLE = [Cell(eta=e, k=k, experiment="A") for e in EXP_A_ETAS_MIDDLE for k in KS_CORE]
EXP_A_CELLS_OUTER  = [Cell(eta=e, k=k, experiment="A") for e in EXP_A_ETAS_OUTER  for k in KS_CORE]

# --- Experiments B & C: extended-K cells ------------------------------------
# Shared between B and C: (3e-4, 50), (3e-4, 75)

def _extra_K_cell(eta: float, k: int, experiment: str) -> Cell:
    return Cell(
        eta=eta,
        k=k,
        n_unique_b=50 * k,                 # per the user's scaling rule
        disambiguation_prefix_length=2,    # required for K > 36
        experiment=experiment,
    )

EXP_B_CELLS = [
    _extra_K_cell(3e-4,  50, "BC"),
    _extra_K_cell(3e-4,  75, "BC"),
    _extra_K_cell(3e-4, 100, "B"),
]

EXP_C_CELLS = [
    _extra_K_cell(7e-4, 50, "C"),
    _extra_K_cell(7e-4, 75, "C"),
]

# Total order: recovery first, then A middle, A outer, B, C
CELLS: List[Cell] = (
    EXP_RECOVERY_CELLS
    + EXP_A_CELLS_MIDDLE
    + EXP_A_CELLS_OUTER
    + EXP_B_CELLS
    + EXP_C_CELLS
)

MIN_STEPS_NEW = 20_000
STUCK_PATIENCE_NEW = 5_000


def _fmt_eta(eta: float) -> str:
    return f"{eta:g}"


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
        "--resume",
    ]
    # Only pass K>36 overrides for extra-K cells.
    if cell.n_unique_b != 1000:
        cmd += ["--n-unique-b", str(cell.n_unique_b)]
    if cell.disambiguation_prefix_length != 1:
        cmd += ["--disambiguation-prefix-length",
                str(cell.disambiguation_prefix_length)]

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    log_fp = open(log_path, "w", buffering=1)
    proc = subprocess.Popen(
        cmd, stdout=log_fp, stderr=subprocess.STDOUT,
        env=env, cwd=str(REPO_ROOT),
    )
    proc._log_fp = log_fp  # type: ignore[attr-defined]
    return proc


def _has_status(cell: Cell) -> bool:
    return (run_dir(cell.eta, cell.k, cell.seed) / "status.json").exists()


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
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip cells that already have status.json")
    args = p.parse_args()

    if args.skip_existing:
        pending = [c for c in CELLS if not _has_status(c)]
    else:
        pending = list(CELLS)

    print(f"Extension-2 queue: {len(pending)} / {len(CELLS)} cells, "
          f"{args.workers} workers")
    for c in pending:
        extra = ""
        if c.n_unique_b != 1000:
            extra = f"  n_b={c.n_unique_b}  prefix_len={c.disambiguation_prefix_length}"
        print(f"  [{c.experiment}] η={_fmt_eta(c.eta)}  K={c.k:>3}  "
              f"max_steps={c.max_steps:,}{extra}")

    log_dir = RESULTS_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    summary_fp = open(RESULTS_DIR / "extend2_summary.jsonl", "a", buffering=1)

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
                     "run_name": f"eta_{cell.eta:g}_K_{cell.k}_seed_{cell.seed}",
                     "eta": cell.eta, "k": cell.k, "seed": cell.seed,
                     "returncode": rc}
            s["experiment"] = cell.experiment
            done.append(s)
            summary_fp.write(json.dumps(s) + "\n")
            print(f"  DONE  [{cell.experiment}]  η={_fmt_eta(cell.eta)}  "
                  f"K={cell.k:>3}  status={s.get('status','?'):<14}  "
                  f"step={s.get('final_step',0):>7}  "
                  f"wall={s.get('wall_clock_s',0):>7.1f}s")
        running = still

    try:
        while queue or running:
            while queue and len(running) < args.workers:
                cell = queue.pop(0)
                log_path = log_dir / (
                    f"extend2_{cell.experiment}_eta_{_fmt_eta(cell.eta)}"
                    f"_K_{cell.k}_seed_{cell.seed}.log"
                )
                proc = _launch_cell(cell, log_path)
                running.append((proc, cell))
                print(f"  START [{cell.experiment}]  η={_fmt_eta(cell.eta)}  "
                      f"K={cell.k:>3}  pid={proc.pid}  log={log_path}")
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
    print(f"Extension-2 complete: {len(done)} cells in {elapsed/60:.1f} min")
    print("=" * 70)
    from collections import Counter
    ct = Counter(d["status"] for d in done)
    print(dict(ct))


if __name__ == "__main__":
    main()
