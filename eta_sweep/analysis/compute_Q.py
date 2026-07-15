#!/usr/bin/env python
"""
Aggregate all per-run log.jsonl files into a single dataframe.

For each run:
  - Detect window [t₁, t₂] from Δz trajectory.
  - Integrate Q = ∫ η · ‖∇L‖² dt via trapezoidal over the window.
  - Compute ΔL, Q − ΔL, snap width.
  - If the run has no detected window, Q is NaN with a reason string.

Output: a single parquet at eta_sweep/results/aggregated_runs.parquet.
        One row per run.  Columns:
          run_name, eta, k, seed, status, final_step, wall_clock_s,
          t1, t2, snap_width_steps, snap_width_tokens,
          delta_z_max, cand_loss_t1, cand_loss_t2,
          Q, delta_L, Q_minus_delta_L, frac_excess, reason

Usage:
    python eta_sweep/analysis/compute_Q.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ETA_SWEEP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ETA_SWEEP_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from eta_sweep.config import (  # noqa: E402
    AGGREGATED_PARQUET,
    CellConfig,
    RESULTS_DIR,
)


# ---------------------------------------------------------------------------
# Window detection
# ---------------------------------------------------------------------------

def detect_window(
    steps: np.ndarray,
    delta_z: np.ndarray,
    threshold: float = 0.5,
    frac: float = 0.9,
) -> Tuple[Optional[int], Optional[int], str]:
    """Return (t1, t2, reason).

    t1 = first step at which Δz > 0.5 nats.
    t2 = first step at which Δz > 0.9 · max(Δz).

    If Δz never crosses 0.5 nats   → ("no transition detected").
    If Δz crosses 0.5 but max Δz < 0.5 / 0.9 so the 0.9·max check can fire
    before t1 → clamp t2 to argmax(Δz) and flag it.
    """
    if len(delta_z) == 0 or np.all(np.isnan(delta_z)):
        return None, None, "no Δz observations"

    valid = ~np.isnan(delta_z)
    if not np.any(valid):
        return None, None, "no Δz observations"

    delta_z_v = delta_z[valid]
    steps_v = steps[valid]
    max_dz = float(np.max(delta_z_v))

    above_thresh = delta_z_v > threshold
    if not np.any(above_thresh):
        return None, None, "Δz never crosses 0.5 nats"

    t1_idx = int(np.argmax(above_thresh))  # first True
    t1 = int(steps_v[t1_idx])

    above_frac = delta_z_v > frac * max_dz
    if np.any(above_frac):
        t2_idx = int(np.argmax(above_frac))
    else:
        t2_idx = int(np.argmax(delta_z_v))

    t2 = int(steps_v[t2_idx])

    reason = ""
    if t2 < t1:
        # Degenerate edge case: clamp to argmax(Δz) at/after t1.
        after = delta_z_v[t1_idx:]
        t2 = int(steps_v[t1_idx + int(np.argmax(after))])
        reason = "t2 clamped to argmax(Δz) ≥ t1"

    return t1, t2, reason


# ---------------------------------------------------------------------------
# Q integral
# ---------------------------------------------------------------------------

def integrate_Q(
    steps: np.ndarray,
    grad_norm_sq: np.ndarray,
    lr: np.ndarray,
    t1: int,
    t2: int,
) -> float:
    """Trapezoidal integration of η(t) · ‖∇L(t)‖² over [t1, t2] in step units.

    We integrate using per-checkpoint η values when the scheduler is not
    constant.  With the default eta_sweep config (constant LR, warmup=0)
    lr is a constant and this reduces to η · ∫ ‖∇L‖² dt.
    """
    mask = (steps >= t1) & (steps <= t2)
    x = steps[mask].astype(np.float64)
    y = (lr[mask] * grad_norm_sq[mask]).astype(np.float64)
    if len(x) < 2:
        return float("nan")
    return float(np.trapz(y, x))


# ---------------------------------------------------------------------------
# Per-run processing
# ---------------------------------------------------------------------------

def _read_log(path: Path) -> pd.DataFrame:
    rows = []
    bad_lines = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # A still-writing worker can occasionally flush a partial line;
                # skip it rather than crash the whole pipeline.
                bad_lines += 1
    if bad_lines > 0:
        print(f"  [warn] skipped {bad_lines} malformed line(s) in {path}")
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _process_run(run_dir: Path) -> Dict:
    cfg_path = run_dir / "config.json"
    status_path = run_dir / "status.json"
    log_path = run_dir / "log.jsonl"

    out = {
        "run_name": run_dir.name,
        "eta": float("nan"),
        "k": -1,
        "seed": -1,
        "status": "missing",
        "final_step": 0,
        "wall_clock_s": float("nan"),
        "t1": None,
        "t2": None,
        "snap_width_steps": None,
        "snap_width_tokens": None,
        "delta_z_max": float("nan"),
        "cand_loss_t1": float("nan"),
        "cand_loss_t2": float("nan"),
        "Q": float("nan"),
        "delta_L": float("nan"),
        "Q_minus_delta_L": float("nan"),
        "frac_excess": float("nan"),
        "reason": "",
    }

    if not cfg_path.exists() or not log_path.exists():
        out["reason"] = "missing config or log"
        return out

    with open(cfg_path) as f:
        cfg = json.load(f)
    cell = cfg["cell"]
    out["eta"] = float(cell["eta"])
    out["k"] = int(cell["k"])
    out["seed"] = int(cell["seed"])
    batch_size = int(cell.get("batch_size", 128))

    if status_path.exists():
        with open(status_path) as f:
            st = json.load(f)
        out["status"] = st.get("status", "unknown")
        out["final_step"] = int(st.get("final_step", 0))
        out["wall_clock_s"] = float(st.get("wall_clock_s", float("nan")))

    df = _read_log(log_path)
    if df.empty:
        out["reason"] = "empty log"
        return out

    steps = df["step"].to_numpy()
    delta_z = df["delta_z"].to_numpy(dtype=np.float64)
    cand = df["candidate_loss"].to_numpy(dtype=np.float64)
    g = df["grad_norm_sq_held_out"].to_numpy(dtype=np.float64)
    lr = df["lr"].to_numpy(dtype=np.float64)

    out["delta_z_max"] = float(np.nanmax(delta_z)) if np.any(~np.isnan(delta_z)) else float("nan")

    t1, t2, reason = detect_window(steps, delta_z)
    if t1 is None or t2 is None:
        out["reason"] = reason
        return out

    out["t1"] = t1
    out["t2"] = t2
    out["snap_width_steps"] = t2 - t1
    out["snap_width_tokens"] = (t2 - t1) * batch_size

    cand_t1 = float(np.interp(t1, steps, cand))
    cand_t2 = float(np.interp(t2, steps, cand))
    out["cand_loss_t1"] = cand_t1
    out["cand_loss_t2"] = cand_t2
    out["delta_L"] = cand_t1 - cand_t2

    Q = integrate_Q(steps, g, lr, t1, t2)
    out["Q"] = Q
    if not math.isnan(Q):
        out["Q_minus_delta_L"] = Q - out["delta_L"]
        out["frac_excess"] = (
            out["Q_minus_delta_L"] / Q if Q > 0 else float("nan")
        )
    out["reason"] = reason

    return out


def main() -> None:
    run_dirs = [p for p in RESULTS_DIR.iterdir() if p.is_dir() and p.name.startswith("eta_")]
    run_dirs.sort()
    if not run_dirs:
        print(f"No run directories found under {RESULTS_DIR}")
        return

    rows = [_process_run(d) for d in run_dirs]
    df = pd.DataFrame(rows)

    AGGREGATED_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(AGGREGATED_PARQUET, index=False)

    print(f"Wrote {len(df)} rows to {AGGREGATED_PARQUET}")
    print("\nStatus summary:")
    print(df["status"].value_counts().to_string())
    print("\nRuns with valid Q:")
    print(df.dropna(subset=["Q"])[["run_name", "eta", "k", "Q", "delta_L"]].to_string(index=False))


if __name__ == "__main__":
    main()
