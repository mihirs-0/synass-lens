#!/usr/bin/env python
"""
η-Sweep orchestrator.

Runs every (η, K, seed) cell in the sweep grid.  Cells are independent so
can be parallelised with ``--workers N``; each worker is a separate
Python subprocess (avoids torch/MPS multi-threading weirdness).  Runs are
resumable — a cell with an existing status.json is skipped unless
``--resume-all`` is passed.

Usage:
    python eta_sweep/launch_sweep.py                   # full 30-cell sweep, 1 worker
    python eta_sweep/launch_sweep.py --workers 3       # 3 parallel workers
    python eta_sweep/launch_sweep.py --seeds 0 1 2     # 90-cell triple-seed
    python eta_sweep/launch_sweep.py --etas 1e-3 3e-3  # subset
    python eta_sweep/launch_sweep.py --ks 10 20        # subset
    python eta_sweep/launch_sweep.py --dry-run         # list cells only
    python eta_sweep/launch_sweep.py --workers 2 \\
        --max-steps-override 300                       # benchmark / smoke
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

from eta_sweep.config import RESULTS_DIR, run_dir, sweep_cells  # noqa: E402


def _fmt_eta(eta: float) -> str:
    return f"{eta:g}"


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


def _launch_cell(
    eta: float,
    k: int,
    seed: int,
    max_steps_override: Optional[int],
    log_path: Path,
) -> subprocess.Popen:
    """Start a subprocess that runs run_single for this cell."""
    cmd = [
        sys.executable,
        "-u",                        # unbuffered stdout
        str(ETA_SWEEP_ROOT / "run_single.py"),
        "--eta", str(eta),
        "--k", str(k),
        "--seed", str(seed),
        "--resume",                  # allow overwrite if resuming
    ]
    if max_steps_override is not None:
        cmd += ["--max-steps", str(max_steps_override)]

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    log_fp = open(log_path, "w", buffering=1)
    proc = subprocess.Popen(
        cmd,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=str(REPO_ROOT),
    )
    proc._log_fp = log_fp  # type: ignore[attr-defined]
    return proc


def _status_line(eta: float, k: int, seed: int, status: str,
                 final_step: int, wall: float) -> str:
    return (
        f"η={_fmt_eta(eta):>6}  K={k:>2}  seed={seed:>2}  "
        f"status={status:<14}  step={final_step:>7}  wall={wall:>7.1f}s"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--etas", type=float, nargs="*", default=None)
    p.add_argument("--ks", type=int, nargs="*", default=None)
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--workers", type=int, default=1,
                   help="Number of parallel worker subprocesses")
    p.add_argument("--resume-all", action="store_true",
                   help="Rerun cells that already have a status.json")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-steps-override", type=int, default=None,
                   help="Override max_steps per cell (for benchmarking)")
    args = p.parse_args()

    cells: List[Tuple[float, int, int]] = sweep_cells(
        etas=args.etas, ks=args.ks, seeds=args.seeds
    )

    # Filter out already-finished cells unless --resume-all.
    if not args.resume_all:
        pending = [c for c in cells if not _has_status(*c)]
    else:
        pending = list(cells)

    print(f"Sweep grid: {len(cells)} cells total, {len(pending)} pending, "
          f"{args.workers} worker(s)")
    for eta, k, seed in pending:
        print(f"  η={_fmt_eta(eta)}  K={k:>2}  seed={seed}")

    if args.dry_run:
        return
    if not pending:
        print("nothing to do")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    per_cell_log_dir = RESULTS_DIR / "logs"
    per_cell_log_dir.mkdir(exist_ok=True)
    summary_path = RESULTS_DIR / "sweep_summary.jsonl"
    summary_fp = open(summary_path, "a", buffering=1)

    t_sweep = time.time()

    running: List[Tuple[subprocess.Popen, Tuple[float, int, int]]] = []
    queue = list(pending)
    done: List[dict] = []

    def _reap_finished() -> None:
        nonlocal running
        still: List[Tuple[subprocess.Popen, Tuple[float, int, int]]] = []
        for proc, cell in running:
            rc = proc.poll()
            if rc is None:
                still.append((proc, cell))
                continue
            # closed — collect status
            try:
                proc._log_fp.close()  # type: ignore[attr-defined]
            except Exception:
                pass
            eta, k, seed = cell
            summary = _read_status(eta, k, seed)
            if not summary:
                summary = {
                    "status": "crashed" if rc != 0 else "unknown",
                    "final_step": 0, "wall_clock_s": 0.0,
                    "run_name": f"eta_{eta:g}_K_{k}_seed_{seed}",
                    "eta": eta, "k": k, "seed": seed,
                    "returncode": rc,
                }
            done.append(summary)
            summary_fp.write(json.dumps(summary) + "\n")
            print("  DONE  " + _status_line(
                eta, k, seed,
                summary.get("status", "?"),
                summary.get("final_step", 0),
                summary.get("wall_clock_s", 0.0),
            ))
        running = still

    # Scheduler loop: keep up to args.workers subprocesses alive.
    try:
        while queue or running:
            while queue and len(running) < args.workers:
                eta, k, seed = queue.pop(0)
                log_name = f"eta_{eta:g}_K_{k}_seed_{seed}.log"
                log_path = per_cell_log_dir / log_name
                proc = _launch_cell(
                    eta, k, seed, args.max_steps_override, log_path
                )
                running.append((proc, (eta, k, seed)))
                print(f"  START η={_fmt_eta(eta)} K={k} seed={seed}  "
                      f"pid={proc.pid}  log={log_path}")
            time.sleep(2.0)
            _reap_finished()
    except KeyboardInterrupt:
        print("\n[interrupted] terminating workers…")
        for proc, _ in running:
            try:
                proc.terminate()
            except Exception:
                pass
        raise

    summary_fp.close()

    # ---- Print final table ----
    elapsed = time.time() - t_sweep
    print("\n" + "=" * 70)
    print(f"Sweep complete: {len(done)} cells in {elapsed/60:.1f} min "
          f"({args.workers} workers)")
    print("=" * 70)
    by_status: dict = {}
    for s in done:
        by_status.setdefault(s.get("status", "?"), []).append(s)
    for status, entries in sorted(by_status.items()):
        print(f"  {status:<14}  {len(entries)}")
    print("=" * 70)
    for s in done:
        print(_status_line(
            s["eta"], s["k"], s["seed"], s.get("status", "?"),
            s.get("final_step", 0), s.get("wall_clock_s", 0.0),
        ))


if __name__ == "__main__":
    main()
