#!/usr/bin/env python
"""
Rerun the 14 cells that were prematurely classified 'stuck' with the
original sweep's min_steps=2000 / stuck_patience=2000 detector.

New thresholds for these reruns:
  min_steps        = 20 000
  stuck_patience   = 5 000
  max_steps        = 200 000  (η = 1e-4)
                     100 000  (η ∈ {3e-4, 1e-3, 3e-3})
                      50 000  (η ∈ {1e-2, 3e-2})

Cells:
  K=36 at η ∈ {1e-4, 3e-4, 1e-3, 3e-3}
  K ∈ {10, 15, 20, 25, 36} at η ∈ {1e-2, 3e-2}
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

# --- Rerun grid --------------------------------------------------------------

CELLS: List[Tuple[float, int]] = [
    # K=36 at four η values
    (1e-4, 36), (3e-4, 36), (1e-3, 36), (3e-3, 36),
    # η=1e-3 K=10: corrupted by an accidental smoke-test overwrite.  Retrain
    # from scratch; the transition at step ~4500 is well under min_steps=20k
    # so the new stuck thresholds won't change its classification.
    (1e-3, 10),
    # All K values at two high η
    (1e-2, 10), (1e-2, 15), (1e-2, 20), (1e-2, 25), (1e-2, 36),
    (3e-2, 10), (3e-2, 15), (3e-2, 20), (3e-2, 25), (3e-2, 36),
]

MIN_STEPS_NEW = 20_000
STUCK_PATIENCE_NEW = 5_000


def _max_steps_for(eta: float) -> int:
    if abs(eta - 1e-4) < 1e-10:
        return 200_000
    if eta in (3e-4, 1e-3, 3e-3):
        return 100_000
    if eta in (1e-2, 3e-2):
        return 50_000
    raise ValueError(f"unexpected η: {eta}")


def _fmt_eta(eta: float) -> str:
    return f"{eta:g}"


def _launch_cell(
    eta: float, k: int, seed: int, log_path: Path,
) -> subprocess.Popen:
    max_steps = _max_steps_for(eta)
    cmd = [
        sys.executable, "-u",
        str(ETA_SWEEP_ROOT / "run_single.py"),
        "--eta", str(eta),
        "--k", str(k),
        "--seed", str(seed),
        "--max-steps", str(max_steps),
        "--min-steps", str(MIN_STEPS_NEW),
        "--stuck-patience", str(STUCK_PATIENCE_NEW),
        "--resume",
    ]
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
    args = p.parse_args()

    pending = [(eta, k, args.seed) for (eta, k) in CELLS]
    print(f"Rerun queue: {len(pending)} cells, {args.workers} workers")
    for eta, k, seed in pending:
        print(f"  η={_fmt_eta(eta)}  K={k:>2}  seed={seed}  "
              f"max_steps={_max_steps_for(eta):,}  "
              f"min_steps={MIN_STEPS_NEW}  "
              f"stuck_patience={STUCK_PATIENCE_NEW}")

    log_dir = RESULTS_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    summary_fp = open(RESULTS_DIR / "rerun_summary.jsonl", "a", buffering=1)

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
                log_path = log_dir / f"rerun_eta_{_fmt_eta(eta)}_K_{k}_seed_{seed}.log"
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
    print(f"Rerun complete: {len(done)} cells in {elapsed/60:.1f} min")
    print("=" * 70)
    from collections import Counter
    ct = Counter(d["status"] for d in done)
    print(dict(ct))


if __name__ == "__main__":
    main()
