#!/usr/bin/env python
"""
Feedback-loop vs saddle signature from existing logs.

Hypothesis (mechanism we want to demonstrate or fail to demonstrate):
  During the marginal plateau, gradient-norm-squared is held at a *non-zero*
  floor while the z-shuffle improvement Δz ≈ 0.  When the model breaks
  symmetry and starts using z, Δz climbs and grad-norm-sq spikes.  This is
  the signature of a feedback-loop quasi-fixed-point, not a saddle.

  Saddle prediction (null):  grad-norm-sq → 0 as the trajectory approaches
  the saddle, then a single rapid descent.  No sustained finite floor.

  Feedback-loop prediction (alternative):  grad-norm-sq pinned at a finite
  symmetry-imposed floor throughout plateau, regardless of duration; Δz
  near zero throughout plateau; both rise at snap onset.

Inputs:  log.jsonl files from the fixed-D multi-seed sweep
         (24 cells, 3 seeds × 8 (D, K) pairs).
Outputs:
  - eta_sweep/results/feedback_loop_signature.json  (per-cell + aggregates)
  - eta_sweep/results/figures/feedback_loop_signature.png (paper figure)

Definitions used here (matched to fixed_d_multiseed_analysis.py):
  prior_end :  earliest step where candidate_loss < 1.1 · log K
  t1        :  earliest step (≥ prior_end) where Δz > 0.5
  t2        :  earliest step (≥ t1)  where Δz > 0.9 · max(Δz)
  plateau   :  [prior_end, t1)
  transition:  [t1, t2]
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt

ETA_SWEEP_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ETA_SWEEP_ROOT / "results"
SWEEP_DIR = RESULTS_DIR / "fixed_d_multiseed"

DELTA_Z_T1 = 0.5
DELTA_Z_T2_FRAC = 0.9
PRIOR_TOL = 1.1


@dataclass
class CellLog:
    D: int
    K: int
    n_b: int
    seed: int
    steps: np.ndarray
    grad_norm_sq_held: np.ndarray   # held-out, no clip
    grad_norm_sq_train: np.ndarray  # training-batch, no clip in this sweep
    cand_loss: np.ndarray
    delta_z: np.ndarray
    prior_end: Optional[int]
    t1: Optional[int]
    t2: Optional[int]


def _load_cell(D_dir: Path, name: str) -> Optional[CellLog]:
    log_path = D_dir / name / "log.jsonl"
    cfg_path = D_dir / name / "config.json"
    if not log_path.exists() or not cfg_path.exists():
        return None
    cfg = json.loads(cfg_path.read_text())["cell"]
    K = cfg["k"]
    seed = cfg["seed"]
    n_b = cfg["n_unique_b"]
    D = K * n_b

    rows = []
    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if len(rows) < 5:
        return None

    steps = np.array([r["step"] for r in rows])
    g_h = np.array([r["grad_norm_sq_held_out"] for r in rows])
    g_t = np.array([r["grad_norm_sq_training"] for r in rows])
    cand = np.array([r["candidate_loss"] for r in rows])
    dz = np.array([r["delta_z"] for r in rows])

    log_K = math.log(K)
    prior_mask = cand < PRIOR_TOL * log_K
    prior_end = int(steps[prior_mask][0]) if prior_mask.any() else None

    t1 = None
    if prior_end is not None:
        post = steps >= prior_end
        cross = post & (dz > DELTA_Z_T1)
        if cross.any():
            t1 = int(steps[cross][0])

    t2 = None
    if t1 is not None:
        post = steps >= t1
        target = DELTA_Z_T2_FRAC * dz[post].max() if post.any() else None
        if target is not None:
            cross = post & (dz > target)
            if cross.any():
                t2 = int(steps[cross][0])

    return CellLog(
        D=D, K=K, n_b=n_b, seed=seed,
        steps=steps, grad_norm_sq_held=g_h, grad_norm_sq_train=g_t,
        cand_loss=cand, delta_z=dz,
        prior_end=prior_end, t1=t1, t2=t2,
    )


def _load_all() -> List[CellLog]:
    cells: List[CellLog] = []
    for D_dir in sorted(SWEEP_DIR.glob("D_*")):
        if not D_dir.is_dir():
            continue
        for name in sorted(p.name for p in D_dir.iterdir()
                           if p.is_dir() and p.name.startswith("eta_")):
            c = _load_cell(D_dir, name)
            if c is not None:
                cells.append(c)
    return cells


def _floor_stats(c: CellLog) -> Dict[str, float]:
    """Plateau-floor statistics: grad-norm-sq, Δz on [prior_end, t1)."""
    if c.prior_end is None or c.t1 is None:
        return {}
    mask = (c.steps >= c.prior_end) & (c.steps < c.t1)
    if not mask.any():
        return {}
    return {
        "plateau_steps": int(mask.sum()),
        "grad_h_floor_med": float(np.median(c.grad_norm_sq_held[mask])),
        "grad_h_floor_std": float(np.std(c.grad_norm_sq_held[mask])),
        "grad_t_floor_med": float(np.median(c.grad_norm_sq_train[mask])),
        "grad_t_floor_std": float(np.std(c.grad_norm_sq_train[mask])),
        "delta_z_plateau_med": float(np.median(c.delta_z[mask])),
        "delta_z_plateau_std": float(np.std(c.delta_z[mask])),
        "delta_z_plateau_max": float(np.max(c.delta_z[mask])),
    }


def _peak_stats(c: CellLog) -> Dict[str, float]:
    """Transition-window peak statistics: max grad-norm-sq, max Δz."""
    if c.t1 is None or c.t2 is None:
        return {}
    mask = (c.steps >= c.t1) & (c.steps <= c.t2)
    if not mask.any():
        return {}
    return {
        "transition_steps": int(mask.sum()),
        "grad_h_peak": float(np.max(c.grad_norm_sq_held[mask])),
        "grad_t_peak": float(np.max(c.grad_norm_sq_train[mask])),
        "delta_z_peak": float(np.max(c.delta_z[mask])),
    }


def _saddle_test(c: CellLog) -> Dict[str, float]:
    """Within-plateau saddle-vs-feedback discriminator.

    The cleanest signature: regress log|g|^2 on step *inside* the
    plateau window [prior_end, t1).

      Saddle approach prediction: slope < 0
        (|g|^2 decays toward the saddle; the trajectory is still
         relaxing toward a critical point throughout plateau).
      Feedback-loop floor prediction: slope ≈ 0
        (|g|^2 is held by symmetry at a non-zero floor for the
         entire plateau; no relaxation).

    We also report a flatness summary: the coefficient of variation
    (std / mean) of |g|^2 over the plateau window.  Low CV = flat.
    """
    out = {}
    if c.prior_end is None or c.t1 is None:
        return out
    mask = (c.steps >= c.prior_end) & (c.steps < c.t1)
    n = int(mask.sum())
    if n < 4:
        return out
    g = c.grad_norm_sq_held[mask]
    s = c.steps[mask].astype(float)
    log_g = np.log(np.maximum(g, 1e-30))
    # least-squares fit of log_g = slope * s + b
    # slope normalised to "per 1000 steps" so units are interpretable
    s_centered = s - s.mean()
    slope_per_step = float(np.sum(s_centered * (log_g - log_g.mean()))
                           / max(np.sum(s_centered**2), 1e-30))
    # decade per 1k steps
    slope_decade_per_1k = slope_per_step * 1000.0 / math.log(10.0)
    cv = float(g.std() / max(g.mean(), 1e-30))
    out["plateau_floor"] = float(np.median(g))
    out["plateau_floor_cv"] = cv
    out["plateau_log_slope_decade_per_1k"] = slope_decade_per_1k
    out["plateau_n_points"] = n
    return out


def main() -> None:
    cells = _load_all()
    print(f"loaded {len(cells)} cells from {SWEEP_DIR}")
    if not cells:
        raise SystemExit("no cells found")

    per_cell_records: List[dict] = []
    for c in cells:
        rec = {
            "D": c.D, "K": c.K, "n_b": c.n_b, "seed": c.seed,
            "n_log_rows": int(len(c.steps)),
            "prior_end": c.prior_end, "t1": c.t1, "t2": c.t2,
            **_floor_stats(c),
            **_peak_stats(c),
            **_saddle_test(c),
        }
        # spike ratio = peak / floor (held-out grad)
        if "grad_h_peak" in rec and "grad_h_floor_med" in rec:
            rec["grad_h_spike_ratio"] = rec["grad_h_peak"] / max(
                rec["grad_h_floor_med"], 1e-30
            )
        per_cell_records.append(rec)

    # aggregates
    floors = [r["grad_h_floor_med"] for r in per_cell_records
              if "grad_h_floor_med" in r]
    peaks = [r["grad_h_peak"] for r in per_cell_records if "grad_h_peak" in r]
    spikes = [r["grad_h_spike_ratio"] for r in per_cell_records
              if "grad_h_spike_ratio" in r]
    plateau_slopes = [r["plateau_log_slope_decade_per_1k"]
                      for r in per_cell_records
                      if "plateau_log_slope_decade_per_1k" in r]
    plateau_cvs = [r["plateau_floor_cv"] for r in per_cell_records
                   if "plateau_floor_cv" in r]
    plateau_dz = [r["delta_z_plateau_max"] for r in per_cell_records
                  if "delta_z_plateau_max" in r]
    peak_dz = [r["delta_z_peak"] for r in per_cell_records
               if "delta_z_peak" in r]

    aggregates = {
        "n_cells": len(per_cell_records),
        "grad_h_floor": {
            "median_across_cells": float(np.median(floors)),
            "iqr": [float(np.quantile(floors, 0.25)),
                    float(np.quantile(floors, 0.75))],
            "min": float(np.min(floors)),
            "max": float(np.max(floors)),
        },
        "grad_h_peak": {
            "median_across_cells": float(np.median(peaks)),
            "iqr": [float(np.quantile(peaks, 0.25)),
                    float(np.quantile(peaks, 0.75))],
        },
        "grad_h_spike_ratio_peak_over_floor": {
            "median_across_cells": float(np.median(spikes)),
            "iqr": [float(np.quantile(spikes, 0.25)),
                    float(np.quantile(spikes, 0.75))],
            "min": float(np.min(spikes)),
            "max": float(np.max(spikes)),
        },
        "saddle_test_plateau_log_slope": {
            "median_across_cells": float(np.median(plateau_slopes)),
            "iqr": [float(np.quantile(plateau_slopes, 0.25)),
                    float(np.quantile(plateau_slopes, 0.75))],
            "min": float(np.min(plateau_slopes)),
            "max": float(np.max(plateau_slopes)),
            "units": "decades of |g|^2 per 1000 steps (within plateau window)",
            "interpretation": (
                "Within-plateau log-linear slope of |g|^2 vs step.  "
                "Saddle approach prediction: significantly < 0 (decaying "
                "toward critical point).  Feedback-loop floor prediction: "
                "≈ 0 (symmetry-imposed floor, no relaxation)."
            ),
        },
        "plateau_floor_cv": {
            "median_across_cells": float(np.median(plateau_cvs)),
            "iqr": [float(np.quantile(plateau_cvs, 0.25)),
                    float(np.quantile(plateau_cvs, 0.75))],
            "interpretation": (
                "Coefficient of variation (std/mean) of |g|^2 within "
                "plateau.  Low (<~0.5) means flat floor; high means "
                "trending."
            ),
        },
        "delta_z_plateau_max": {
            "median_across_cells": float(np.median(plateau_dz)),
            "max_across_cells": float(np.max(plateau_dz)),
            "interpretation": (
                "Largest Δz observed during plateau window.  Should be "
                "near 0 throughout plateau; non-zero only at t1 onset."
            ),
        },
        "delta_z_peak_at_transition": {
            "median_across_cells": float(np.median(peak_dz)),
        },
    }

    out = {
        "method_note": (
            "Computed from log.jsonl per-step records (every 100 steps) "
            "in eta_sweep/results/fixed_d_multiseed/.  No checkpoint "
            "loading required.  Held-out gradient is over a fresh batch "
            "of 256 examples; training gradient is over the live 128 "
            "training batch."
        ),
        "definitions": {
            "prior_end": "first step where candidate_loss < 1.1 * log K",
            "t1": "first step (>= prior_end) where Δz > 0.5",
            "t2": "first step (>= t1) where Δz > 0.9 * max Δz",
            "plateau_window": "[prior_end, t1)",
            "transition_window": "[t1, t2]",
        },
        "aggregates": aggregates,
        "per_cell": per_cell_records,
    }

    out_json = RESULTS_DIR / "feedback_loop_signature.json"
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_json}")

    # ---------------- figure ----------------
    fig_dir = RESULTS_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # panel A: |g|^2 trajectory, overlaid by (D, K), 3 seeds each
    color_map = {
        (10000, 5):  "tab:blue",
        (10000, 10): "tab:cyan",
        (10000, 20): "tab:green",
        (10000, 36): "tab:olive",
        (20000, 5):  "tab:red",
        (20000, 10): "tab:orange",
        (20000, 20): "tab:purple",
        (20000, 36): "tab:brown",
    }

    def _D_bin(c: CellLog) -> int:
        return int(round(c.D / 1000) * 1000)

    ax = axes[0, 0]
    for c in cells:
        Db = _D_bin(c)
        col = color_map.get((Db, c.K), "k")
        ax.plot(c.steps, c.grad_norm_sq_held, color=col, alpha=0.45, lw=0.9)
    ax.set_xlabel("step")
    ax.set_ylabel(r"$\|g\|^2$ (held-out)")
    ax.set_yscale("log")
    ax.set_title("A. Held-out gradient norm trajectories\n(24 cells: 3 seeds × 8 (D,K))")
    # legend per (D,K)
    handles = [plt.Line2D([0], [0], color=col, lw=2,
                          label=f"D={D}, K={K}") for (D, K), col in color_map.items()]
    ax.legend(handles=handles, fontsize=7, ncol=2, loc="lower right")

    # panel B: Δz trajectory
    ax = axes[0, 1]
    for c in cells:
        Db = _D_bin(c)
        col = color_map.get((Db, c.K), "k")
        ax.plot(c.steps, c.delta_z, color=col, alpha=0.45, lw=0.9)
    ax.axhline(0.0, color="0.6", lw=0.6, ls="--")
    ax.axhline(DELTA_Z_T1, color="k", lw=0.6, ls=":", label=r"$t_1$ threshold")
    ax.set_xlabel("step")
    ax.set_ylabel(r"$\Delta_z = L(\text{shuffle-}z) - L$")
    ax.set_title(r"B. z-shuffle improvement $\Delta_z$ trajectories")
    ax.legend(fontsize=8)

    # panel C: spike ratio per cell
    ax = axes[1, 0]
    sr_by_K = {}
    for r in per_cell_records:
        if "grad_h_spike_ratio" not in r:
            continue
        Db = int(round(r["D"] / 1000) * 1000)
        key = (Db, r["K"])
        sr_by_K.setdefault(key, []).append(r["grad_h_spike_ratio"])
    keys = sorted(sr_by_K.keys())
    xs = np.arange(len(keys))
    means = [np.mean(sr_by_K[k]) for k in keys]
    sds = [np.std(sr_by_K[k]) for k in keys]
    colors = [color_map[k] for k in keys]
    ax.bar(xs, means, yerr=sds, color=colors, alpha=0.85, capsize=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"D={D}\nK={K}" for (D, K) in keys],
                       rotation=0, fontsize=7)
    ax.set_ylabel(r"peak $|g|^2$ / plateau-floor $|g|^2$")
    ax.set_title("C. Spike ratio at snap (held-out grad)\nfeedback prediction: ratio >> 1")
    ax.set_yscale("log")
    ax.axhline(1.0, color="k", lw=0.6, ls="--")

    # panel D: within-plateau log-slope (saddle vs feedback test)
    ax = axes[1, 1]
    sl_by_K = {}
    for r in per_cell_records:
        if "plateau_log_slope_decade_per_1k" not in r:
            continue
        Db = int(round(r["D"] / 1000) * 1000)
        key = (Db, r["K"])
        sl_by_K.setdefault(key, []).append(
            r["plateau_log_slope_decade_per_1k"]
        )
    keys = sorted(sl_by_K.keys())
    xs = np.arange(len(keys))
    means = [np.mean(sl_by_K[k]) for k in keys]
    sds = [np.std(sl_by_K[k]) for k in keys]
    colors = [color_map[k] for k in keys]
    ax.bar(xs, means, yerr=sds, color=colors, alpha=0.85, capsize=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"D={D}\nK={K}" for (D, K) in keys],
                       rotation=0, fontsize=7)
    ax.set_ylabel(r"d log $|g|^2$ / d step  (decades / 1k steps)")
    ax.set_title("D. Within-plateau log-slope (saddle vs feedback)\n"
                 "saddle: << 0 (decaying); feedback: ≈ 0 (flat floor)")
    ax.axhline(0.0, color="k", lw=0.8, ls="--",
               label=r"feedback prediction ($\approx 0$)")
    ax.legend(fontsize=8)

    fig.suptitle(
        "Feedback-loop vs saddle: plateau is dynamically active, not at a critical point",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_png = fig_dir / "feedback_loop_signature.png"
    fig.savefig(out_png, dpi=160)
    print(f"wrote {out_png}")

    # ---------------- console summary ----------------
    print()
    print("=" * 70)
    print("Feedback-loop signature — summary")
    print("=" * 70)
    a = aggregates
    print(f"n_cells: {a['n_cells']}")
    print(f"  plateau-floor |g|^2 (held-out, median across cells):"
          f" {a['grad_h_floor']['median_across_cells']:.4f}")
    print(f"  peak-at-snap |g|^2:"
          f" {a['grad_h_peak']['median_across_cells']:.4f}")
    sr = a["grad_h_spike_ratio_peak_over_floor"]
    print(f"  spike ratio peak/floor: median {sr['median_across_cells']:.2f}x"
          f"  IQR [{sr['iqr'][0]:.2f}, {sr['iqr'][1]:.2f}]")
    sl = a["saddle_test_plateau_log_slope"]
    print(f"  within-plateau log-slope: median"
          f" {sl['median_across_cells']:+.3f} decades/1k steps"
          f"  IQR [{sl['iqr'][0]:+.3f}, {sl['iqr'][1]:+.3f}]")
    cv = a["plateau_floor_cv"]
    print(f"  plateau-floor CV (std/mean):"
          f" median {cv['median_across_cells']:.2f}"
          f"  IQR [{cv['iqr'][0]:.2f}, {cv['iqr'][1]:.2f}]")
    print(f"  Δz max during plateau (median across cells):"
          f" {a['delta_z_plateau_max']['median_across_cells']:.4f}")
    print(f"  Δz peak at transition (median across cells):"
          f" {a['delta_z_peak_at_transition']['median_across_cells']:.4f}")
    print()
    print("Reading:")
    print(" - If plateau log-slope is significantly negative:"
          " trajectory is approaching a critical point, saddle picture.")
    print(" - If plateau log-slope is ≈ 0:"
          " floor is sustained; feedback-loop picture.")
    print(" - Spike ratio >> 1 + Δz plateau ≈ 0 + Δz peak >> 0:"
          " snap = onset of z-use, not noise.")


if __name__ == "__main__":
    main()
