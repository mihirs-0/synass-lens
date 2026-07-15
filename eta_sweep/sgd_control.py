#!/usr/bin/env python
"""
SGD control — minimum viable scout.

Question: does the K-dependence of η_c (and the plateau→escape phenomenology
itself) survive when AdamW is replaced by vanilla SGD?

  AdamW result, multi-seed: η_c(K=20) ≈ 3.0e-3.  η_c(K=36) < η_c(K=20).
  (Single-seed earlier fits suggested η_c ∝ K^{-0.83}; that figure was
  retracted in paper_claim_matrix_revised.md but the *sign* of the
  K-dependence is supported.)

Hypothesis A: K-dependence is a property of the loss landscape.
  -> SGD should also exhibit the plateau→escape, with η_c(K) decreasing in K.
  -> Absolute η_c is allowed to differ (SGD typically needs ~10–100× the LR
     of Adam to match step-size in well-conditioned directions).

Hypothesis B: K-dependence is an Adam-preconditioner artefact.
  -> SGD η_c may show a different K-dependence (no decrease, or increase).
  -> Or SGD may not produce a clean plateau→escape at all.

This scout is sized to discriminate at the lowest-cost level: 8 cells.

Stage 1 grid (this script):
  K     ∈ {5, 36}                       # widest K bracket we have
  η     ∈ {0.01, 0.03, 0.1, 0.3}        # 1.5 decades log-spaced; SGD scale
  seed  ∈ {0}
  Total: 2 × 4 × 1 = 8 cells.

Per-cell config:
  optimizer       = "sgd"
  momentum        = 0.0       # vanilla SGD; clean preconditioner test
  weight_decay    = 0.01      # match AdamW arm; applied as L2 for SGD
  batch_size      = 128
  scheduler       = constant, warmup = 0
  min_steps       = 2000
  stuck_patience  = 5000      # be generous; SGD is slower
  max_steps       = 20000     # scout budget; not a final-figure budget

If Stage 1 shows the transition exists at some η for at least K=5 (denser
gradient signal) but breaks the K-decreasing pattern from AdamW, that is a
clear signal that K^(-0.83) is preconditioner-mediated.  If the transition
also shows up at K=36 with lower η, the AdamW pattern is corroborated.

After-run analysis (separate script): for each cell, extract t1, t2,
candidate-loss-floor, and grad-norm trajectory.  Compare to the AdamW
fixed-D multiseed cells at matched (K, n_b).

Output layout:
  eta_sweep/results/sgd_control/eta_<η>_K_<K>_seed_<s>/
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
from typing import List, Tuple

ETA_SWEEP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = ETA_SWEEP_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from eta_sweep.config import RESULTS_DIR  # noqa: E402

# --- Stage 1 grid ----------------------------------------------------------

K_VALUES = [5, 36]
ETA_VALUES = [1e-2, 3e-2, 1e-1, 3e-1]
SEEDS = [0]

OPTIMIZER = "sgd"
MOMENTUM = 0.0
WEIGHT_DECAY = 0.01
BATCH_SIZE = 128

MIN_STEPS = 2_000
STUCK_PATIENCE = 5_000
MAX_STEPS = 20_000

SUBDIR = "sgd_control"


@dataclass(frozen=True)
class Cell:
    eta: float
    K: int
    seed: int

    @property
    def out_dir(self) -> Path:
        return (RESULTS_DIR / SUBDIR
                / f"eta_{self.eta:g}_K_{self.K}_seed_{self.seed}")


CELLS: List[Cell] = [
    Cell(eta=eta, K=K, seed=seed)
    for eta in ETA_VALUES
    for K in K_VALUES
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
        "--eta", str(cell.eta),
        "--k", str(cell.K),
        "--seed", str(cell.seed),
        "--batch-size", str(BATCH_SIZE),
        "--max-steps", str(MAX_STEPS),
        "--min-steps", str(MIN_STEPS),
        "--stuck-patience", str(STUCK_PATIENCE),
        "--optimizer", OPTIMIZER,
        "--momentum", str(MOMENTUM),
        "--weight-decay", str(WEIGHT_DECAY),
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
    p.add_argument("--workers", type=int, default=2,
                   help="parallel run_single.py workers (default 2)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    pending = [c for c in CELLS if not _has_status(c)]
    print(f"SGD control (Stage 1): {len(pending)} / {len(CELLS)} cells "
          f"pending, {args.workers} workers")
    print(f"  optimizer = {OPTIMIZER}, momentum = {MOMENTUM}, "
          f"weight_decay = {WEIGHT_DECAY}")
    print(f"  batch_size = {BATCH_SIZE}")
    print(f"  thresholds: min_steps={MIN_STEPS}, "
          f"stuck_patience={STUCK_PATIENCE}, max_steps={MAX_STEPS}")
    print(f"  K values: {K_VALUES}")
    print(f"  η values: {ETA_VALUES}")
    print(f"  seeds: {SEEDS}")
    print()
    for c in pending:
        print(f"  η={c.eta:>5.2g}  K={c.K:>2}  seed={c.seed}")

    if args.dry_run:
        return
    if not pending:
        print("nothing to do")
        return

    base = RESULTS_DIR / SUBDIR
    base.mkdir(parents=True, exist_ok=True)
    log_dir = base / "logs"
    log_dir.mkdir(exist_ok=True)
    summary_fp = open(base / "summary.jsonl", "a", buffering=1)

    readme = base / "README.md"
    if not readme.exists():
        readme.write_text(
            "# SGD control — minimum-viable scout (Stage 1)\n\n"
            "8 cells (K ∈ {5, 36} × η ∈ {0.01, 0.03, 0.1, 0.3} × seed 0).\n\n"
            "Vanilla SGD (momentum=0), weight_decay=0.01, batch_size=128, "
            "constant LR, no warmup, no clipping.\n\n"
            "Purpose: discriminate whether the K-dependence of η_c "
            "(observed under AdamW) survives the optimizer change.  "
            "AdamW arm is the fixed-D multi-seed sweep at η=1e-3; the SGD "
            "η scale is unknown a priori, so this scout brackets it widely.\n"
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
                     "eta": cell.eta, "k": cell.K, "seed": cell.seed,
                     "returncode": rc}
            s.update({"experiment": "sgd_control_stage1",
                      "optimizer": OPTIMIZER,
                      "momentum": MOMENTUM,
                      "weight_decay": WEIGHT_DECAY})
            done.append(s)
            summary_fp.write(json.dumps(s) + "\n")
            elapsed = time.time() - t0
            frac = (len(done) + 0.0) / max(len(pending), 1)
            eta_s = (elapsed / max(frac, 1e-9)) - elapsed
            print(f"  [{len(done):>2}/{len(pending)}] DONE  "
                  f"η={cell.eta:>5.2g}  K={cell.K:>2}  s={cell.seed}  "
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
                    f"sgd_eta_{cell.eta:g}_K_{cell.K}"
                    f"_seed_{cell.seed}.log"
                )
                proc = _launch_cell(cell, log_path)
                running.append((proc, cell))
                print(f"  START  η={cell.eta:>5.2g}  K={cell.K:>2}  "
                      f"seed={cell.seed}  pid={proc.pid}")
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
    print(f"SGD control Stage 1 complete: {len(done)} cells in "
          f"{elapsed/60:.1f} min")
    print("=" * 70)
    from collections import Counter
    print("  by status:", dict(Counter(d["status"] for d in done)))


if __name__ == "__main__":
    main()
