#!/usr/bin/env python
"""
Multi-seed η-sweep for the NeurIPS 2026 submission.

Implements the boundary-focused asymmetric-seed matrix from the revised
plan at /Users/mihir/.claude/plans/sunny-dazzling-cray.md.

Matrix:
  Tier A — stable-regime replication
      η ∈ {3e-4, 7e-4, 1e-3} × K ∈ {10, 20, 36} × seeds {0, 1, 2}
      → 27 cells.  Purpose: confirm Q ∝ log K scaling isn't seed-noise.

  Tier B — critical region (HIGH-DENSITY seeds)
      η ∈ {2.4e-3, 2.6e-3, 2.8e-3, 3.0e-3} × K ∈ {10, 20, 36}
         × seeds {0, 1, 2, 42, 1337, 2718, 31415, 271828}
      → 96 cells.  Purpose: seed-averaged transition probability curve,
        logistic fit for η_c with proper CI.

  Tier C — failure confirmation
      η ∈ {5e-3, 1e-2} × K ∈ {10, 20, 36} × seeds {0, 1, 2}
      → 18 cells.  Purpose: confirm failure is seed-robust.

  Tier D — architecture sanity check
      η ∈ {3e-4, 2.8e-3} × K=20 × seeds {0, 1, 2} × (d=64) × 1 depth
                                                 × (n_layers=2) × 1 width
      → 12 cells.  Purpose: single cheap check.

Total: 153 cells.

All cells use the extended-budget config (min_steps=20k, stuck_patience=5k)
and per-η max_steps consistent with the prior extended runs.

Launch order: **Tier B first** (most important for the headline result),
then A, C, D.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
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
    seed: int
    tier: str                                # "A", "B", "C", or "D"
    max_steps: int = 100_000
    min_steps: int = 20_000
    stuck_patience: int = 5_000
    # K > 36 support (unused here but preserved for forward compat)
    n_unique_b: int = 1000
    disambiguation_prefix_length: int = 1
    # Tier D architecture overrides (None ⇒ default)
    d_model: Optional[int] = None
    n_layers: Optional[int] = None


# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------

KS_CORE = [10, 20, 36]
SEEDS_BASE = [0, 1, 2]
SEEDS_BOUNDARY = [0, 1, 2, 42, 1337, 2718, 31415, 271828]


def _max_steps_for(eta: float) -> int:
    # Consistent with earlier extended runs: 100k for η ≤ 5e-3, 50k above.
    return 50_000 if eta >= 5e-3 else 100_000


# --- Tier B: critical region (launched FIRST) --------------------------------
TIER_B: List[Cell] = [
    Cell(eta=eta, k=k, seed=seed, tier="B",
         max_steps=_max_steps_for(eta))
    for eta in [2.4e-3, 2.6e-3, 2.8e-3, 3.0e-3]
    for k in KS_CORE
    for seed in SEEDS_BOUNDARY
]

# --- Tier A: stable-regime replication ---------------------------------------
TIER_A: List[Cell] = [
    Cell(eta=eta, k=k, seed=seed, tier="A",
         max_steps=_max_steps_for(eta))
    for eta in [3e-4, 7e-4, 1e-3]
    for k in KS_CORE
    for seed in SEEDS_BASE
]

# --- Tier C: failure confirmation --------------------------------------------
TIER_C: List[Cell] = [
    Cell(eta=eta, k=k, seed=seed, tier="C",
         max_steps=_max_steps_for(eta))
    for eta in [5e-3, 1e-2]
    for k in KS_CORE
    for seed in SEEDS_BASE
]

# --- Tier D: architecture sanity check ---------------------------------------
# Two sub-variants: (a) half-width d_model=64, (b) half-depth n_layers=2
# Each: 2 η × K=20 × 3 seeds = 6 cells; total D = 12 cells.
TIER_D: List[Cell] = (
    [Cell(eta=eta, k=20, seed=seed, tier="D",
          max_steps=_max_steps_for(eta), d_model=64)
     for eta in [3e-4, 2.8e-3] for seed in SEEDS_BASE]
    +
    [Cell(eta=eta, k=20, seed=seed, tier="D",
          max_steps=_max_steps_for(eta), n_layers=2)
     for eta in [3e-4, 2.8e-3] for seed in SEEDS_BASE]
)

# Final order: B first (most valuable), then A, C, D.
CELLS: List[Cell] = TIER_B + TIER_A + TIER_C + TIER_D


# ---------------------------------------------------------------------------
# Per-cell directory naming — distinguish architecture variants
# ---------------------------------------------------------------------------

def _cell_suffix(cell: Cell) -> str:
    """Suffix appended when architecture is non-default, so Tier D cells
    don't collide with the baseline architecture runs at the same (η, K, seed).
    """
    bits = []
    if cell.d_model is not None and cell.d_model != 128:
        bits.append(f"d{cell.d_model}")
    if cell.n_layers is not None and cell.n_layers != 4:
        bits.append(f"L{cell.n_layers}")
    return ("_" + "_".join(bits)) if bits else ""


def cell_dir(cell: Cell) -> Path:
    # Baseline architecture → matches config.run_dir() naming.
    if cell.d_model in (None, 128) and cell.n_layers in (None, 4):
        return run_dir(cell.eta, cell.k, cell.seed)
    # Architecture variant → append suffix to avoid collision.
    return RESULTS_DIR / (
        f"eta_{cell.eta:g}_K_{cell.k}_seed_{cell.seed}{_cell_suffix(cell)}"
    )


# ---------------------------------------------------------------------------
# Launcher
# ---------------------------------------------------------------------------

def _launch_cell(cell: Cell, log_path: Path) -> subprocess.Popen:
    cmd = [
        sys.executable, "-u",
        str(ETA_SWEEP_ROOT / "run_single.py"),
        "--eta", str(cell.eta),
        "--k", str(cell.k),
        "--seed", str(cell.seed),
        "--max-steps", str(cell.max_steps),
        "--min-steps", str(cell.min_steps),
        "--stuck-patience", str(cell.stuck_patience),
        "--resume",
    ]
    if cell.n_unique_b != 1000:
        cmd += ["--n-unique-b", str(cell.n_unique_b)]
    if cell.disambiguation_prefix_length != 1:
        cmd += ["--disambiguation-prefix-length",
                str(cell.disambiguation_prefix_length)]
    if cell.d_model is not None:
        cmd += ["--d-model", str(cell.d_model)]
    if cell.n_layers is not None:
        cmd += ["--n-layers", str(cell.n_layers)]

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


def _has_status(cell: Cell) -> bool:
    # run_single.py always writes status.json at the baseline cell dir;
    # architecture variants write to the suffixed dir.
    return (cell_dir(cell) / "status.json").exists()


def _read_status(cell: Cell) -> dict:
    p = cell_dir(cell) / "status.json"
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return {}


def _compact_cell_id(cell: Cell) -> str:
    base = f"η={cell.eta:g} K={cell.k:>2} s={cell.seed:<6}"
    suf = _cell_suffix(cell)
    return f"[{cell.tier}] {base}{' ' + suf if suf else ''}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--tiers", type=str, nargs="*", default=["B", "A", "C", "D"],
                   help="Subset of tiers to run, in any order")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-existing", action="store_true", default=True,
                   help="Skip cells that already have status.json (default)")
    p.add_argument("--rerun-all", action="store_true",
                   help="Rerun cells even if status.json exists")
    p.add_argument("--smoke-max-steps", type=int, default=None,
                   help="Override max_steps for every cell (for smoke tests only)")
    p.add_argument("--limit", type=int, default=None,
                   help="Only run the first N cells (for smoke tests)")
    args = p.parse_args()

    all_cells = [c for c in CELLS if c.tier in set(args.tiers)]
    if args.rerun_all:
        pending = list(all_cells)
    else:
        pending = [c for c in all_cells if not _has_status(c)]

    # Smoke-test overrides (never used in production).
    if args.smoke_max_steps is not None:
        pending = [Cell(**{**c.__dict__, "max_steps": args.smoke_max_steps})
                   for c in pending]
    if args.limit is not None:
        pending = pending[:args.limit]

    print(f"Multiseed queue: {len(pending)} / {len(all_cells)} cells across "
          f"tiers {args.tiers}, {args.workers} workers")
    tier_counts = {t: 0 for t in "ABCD"}
    for c in pending:
        tier_counts[c.tier] += 1
    for t, n in tier_counts.items():
        print(f"  Tier {t}: {n} pending")
    print()

    if args.dry_run:
        for c in pending:
            extra = []
            if c.d_model is not None: extra.append(f"d_model={c.d_model}")
            if c.n_layers is not None: extra.append(f"n_layers={c.n_layers}")
            print(f"  {_compact_cell_id(c):<55}  max_steps={c.max_steps:>6}  "
                  + ("  ".join(extra) if extra else ""))
        return
    if not pending:
        print("nothing to do")
        return

    log_dir = RESULTS_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    summary_fp = open(RESULTS_DIR / "multiseed_summary.jsonl", "a",
                      buffering=1)

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
                     "run_name": cell_dir(cell).name,
                     "eta": cell.eta, "k": cell.k, "seed": cell.seed,
                     "returncode": rc}
            s["tier"] = cell.tier
            if cell.d_model is not None: s["d_model"] = cell.d_model
            if cell.n_layers is not None: s["n_layers"] = cell.n_layers
            done.append(s)
            summary_fp.write(json.dumps(s) + "\n")
            elapsed_total = time.time() - t0
            frac = (len(done) + 0.0) / max(len(pending), 1)
            eta_s = (elapsed_total / max(frac, 1e-9)) - elapsed_total
            print(f"  [{len(done):>3}/{len(pending)}] DONE {_compact_cell_id(cell):<55}  "
                  f"status={s.get('status','?'):<14}  "
                  f"step={s.get('final_step',0):>7}  "
                  f"wall={s.get('wall_clock_s',0):>7.1f}s  "
                  f"ETA={eta_s/60:.1f}min")
        running = still

    try:
        while queue or running:
            while queue and len(running) < args.workers:
                cell = queue.pop(0)
                log_path = log_dir / (
                    f"multiseed_{cell.tier}_"
                    f"eta_{cell.eta:g}_K_{cell.k}_seed_{cell.seed}"
                    f"{_cell_suffix(cell)}.log"
                )
                proc = _launch_cell(cell, log_path)
                running.append((proc, cell))
                print(f"  START {_compact_cell_id(cell):<55}  pid={proc.pid}")
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
    print(f"Multiseed sweep complete: {len(done)} cells in {elapsed/60:.1f} min")
    print("=" * 70)
    from collections import Counter
    ct = Counter(d["status"] for d in done)
    print("  by status:", dict(ct))
    ct_tier = Counter(d["tier"] for d in done)
    print("  by tier:  ", dict(ct_tier))


if __name__ == "__main__":
    main()
