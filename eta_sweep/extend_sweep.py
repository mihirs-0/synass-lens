#!/usr/bin/env python
"""
η-Sweep extension.

Adds 25 new cells to the sweep:
  • 15 stable-regime fill-ins: η ∈ {5e-4, 7e-4, 2e-3} × K ∈ {10, 15, 20, 25, 36}
  • 10 phase-boundary cells:   η ∈ {5e-3, 7e-3}     × K ∈ {10, 15, 20, 25, 36}

All new cells run with the extended-budget config (same as rerun_stuck.py):
    min_steps        = 20 000
    stuck_patience   = 5 000
    max_steps (per-η, consistent with rerun pattern):
        5e-4, 7e-4 → 100 000
        2e-3       → 100 000
        5e-3, 7e-3 → 100 000   (enough room if they transition slowly)

Launch order: stable-regime first (high value for the c(η) fit), then
phase-boundary cells.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

ETA_SWEEP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = ETA_SWEEP_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from eta_sweep.config import RESULTS_DIR, run_dir  # noqa: E402

# --- Extension grid ----------------------------------------------------------

STABLE_ETAS = [5e-4, 7e-4, 2e-3]
PHASE_ETAS = [5e-3, 7e-3]
KS = [10, 15, 20, 25, 36]

CELLS: List[Tuple[float, int]] = (
    [(eta, k) for eta in STABLE_ETAS for k in KS]
    + [(eta, k) for eta in PHASE_ETAS for k in KS]
)

MIN_STEPS_NEW = 20_000
STUCK_PATIENCE_NEW = 5_000

MAX_STEPS_BY_ETA = {
    5e-4: 100_000,
    7e-4: 100_000,
    2e-3: 100_000,
    5e-3: 100_000,
    7e-3: 100_000,
}


def _max_steps_for(eta: float) -> int:
    for k, v in MAX_STEPS_BY_ETA.items():
        if abs(eta - k) / k < 1e-6:
            return v
    raise ValueError(f"unexpected η: {eta}")


def _fmt_eta(eta: float) -> str:
    return f"{eta:g}"


def _launch_cell(
    eta: float, k: int, seed: int, log_path: Path,
) -> subprocess.Popen:
    cmd = [
        sys.executable, "-u",
        str(ETA_SWEEP_ROOT / "run_single.py"),
        "--eta", str(eta),
        "--k", str(k),
        "--seed", str(seed),
        "--max-steps", str(_max_steps_for(eta)),
        "--min-steps", str(MIN_STEPS_NEW),
        "--stuck-patience", str(STUCK_PATIENCE_NEW),
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


def _has_status(eta: float, k: int, seed: int) -> bool:
    return (run_dir(eta, k, seed) / "status.json").exists()


def _read_status(eta: float, k: int, seed: int) -> dict:
    p = run_dir(eta, k, seed) / "status.json"
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
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip cells that already have status.json")
    args = p.parse_args()

    all_cells = [(eta, k, args.seed) for (eta, k) in CELLS]
    if args.skip_existing:
        pending = [c for c in all_cells if not _has_status(*c)]
    else:
        pending = all_cells

    print(f"Extension queue: {len(pending)} / {len(all_cells)} cells, "
          f"{args.workers} workers")
    for eta, k, seed in pending:
        print(f"  η={_fmt_eta(eta)}  K={k:>2}  seed={seed}  "
              f"max_steps={_max_steps_for(eta):,}")

    log_dir = RESULTS_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    summary_fp = open(RESULTS_DIR / "extend_summary.jsonl", "a", buffering=1)

    queue = list(pending)
    running: List[Tuple[subprocess.Popen, Tuple[float, int, int]]] = []
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
            eta, k, seed = cell
            s = _read_status(eta, k, seed)
            if not s:
                s = {"status": "crashed" if rc != 0 else "unknown",
                     "final_step": 0, "wall_clock_s": 0.0,
                     "run_name": f"eta_{eta:g}_K_{k}_seed_{seed}",
                     "eta": eta, "k": k, "seed": seed, "returncode": rc}
            done.append(s)
            summary_fp.write(json.dumps(s) + "\n")
            print(f"  DONE  η={_fmt_eta(eta)}  K={k:>2}  "
                  f"status={s.get('status','?'):<14}  "
                  f"step={s.get('final_step',0):>7}  "
                  f"wall={s.get('wall_clock_s',0):>7.1f}s")
        running = still

    try:
        while queue or running:
            while queue and len(running) < args.workers:
                eta, k, seed = queue.pop(0)
                log_path = log_dir / f"extend_eta_{_fmt_eta(eta)}_K_{k}_seed_{seed}.log"
                proc = _launch_cell(eta, k, seed, log_path)
                running.append((proc, (eta, k, seed)))
                print(f"  START η={_fmt_eta(eta)} K={k:>2} seed={seed}  "
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
    print(f"Extension complete: {len(done)} cells in {elapsed/60:.1f} min")
    print("=" * 70)
    from collections import Counter
    ct = Counter(d["status"] for d in done)
    print(dict(ct))


if __name__ == "__main__":
    main()
