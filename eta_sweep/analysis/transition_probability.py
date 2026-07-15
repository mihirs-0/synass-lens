#!/usr/bin/env python
"""
Seed-averaged transition probability + logistic fit for η_c.

For each (η, K) cell, compute success_fraction = (n_transitioned / n_seeds).
Fit a logistic of the form
    p(η; η_c, s) = σ(s · (η_c − η))          (rising-then-falling in η)
per K and on pooled data across K values.  Report η_c estimate with
standard error from nonlinear least squares.

Fallback when the logistic fit fails (too few seed-averaged points,
separable data, etc.): report an "η_c bracket" = [largest η with p ≥ 0.5,
smallest η with p < 0.5].

Output: eta_sweep/results/transition_probability.json

Usage:
    python eta_sweep/analysis/transition_probability.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

ETA_SWEEP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ETA_SWEEP_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from eta_sweep.config import AGGREGATED_PARQUET, RESULTS_DIR  # noqa: E402

OUT_PATH = RESULTS_DIR / "transition_probability.json"


# ---------------------------------------------------------------------------
# Logistic model
# ---------------------------------------------------------------------------

def _logistic(eta: np.ndarray, eta_c: float, slope: float) -> np.ndarray:
    # p = σ(slope · (eta_c − eta)).  slope > 0 ⇒ p decreases with η.
    return 1.0 / (1.0 + np.exp(-slope * (eta_c - eta)))


def _fit_logistic(etas: np.ndarray, fractions: np.ndarray,
                  weights: Optional[np.ndarray] = None
                  ) -> Dict[str, Optional[float]]:
    """Returns {eta_c, slope, eta_c_se, slope_se, R2} or None fields on fail."""
    if len(etas) < 3:
        return {"eta_c": None, "slope": None, "eta_c_se": None,
                "slope_se": None, "R2": None, "reason": "insufficient points"}
    # Warm start: midpoint between lowest η with p=1 and highest η with p=0.
    above = etas[fractions >= 0.5]
    below = etas[fractions < 0.5]
    if len(above) == 0 or len(below) == 0:
        # All-above or all-below — can't bracket η_c.
        return {"eta_c": None, "slope": None, "eta_c_se": None,
                "slope_se": None, "R2": None,
                "reason": "no crossing of 0.5"}
    eta_c0 = 0.5 * (above.max() + below.min())
    slope0 = 10_000.0  # order-of-magnitude guess for sharp transition

    sigma = None
    if weights is not None and np.all(weights > 0):
        # Turn seed counts into per-point sigma (approximate — assumes binomial).
        p = fractions
        sigma = np.sqrt(np.maximum(p * (1 - p), 1e-6) / weights)

    try:
        popt, pcov = curve_fit(
            _logistic, etas, fractions, p0=[eta_c0, slope0],
            sigma=sigma, absolute_sigma=sigma is not None,
            maxfev=5000,
        )
        eta_c, slope = popt
        perr = np.sqrt(np.diag(pcov))
        yhat = _logistic(etas, *popt)
        ss_res = float(np.sum((fractions - yhat) ** 2))
        ss_tot = float(np.sum((fractions - fractions.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
        return {"eta_c": float(eta_c), "slope": float(slope),
                "eta_c_se": float(perr[0]), "slope_se": float(perr[1]),
                "R2": r2, "reason": "ok"}
    except Exception as exc:
        return {"eta_c": None, "slope": None, "eta_c_se": None,
                "slope_se": None, "R2": None, "reason": f"fit failed: {exc}"}


def _bracket(etas: np.ndarray, fractions: np.ndarray) -> Dict[str, Optional[float]]:
    """Fallback bracket = [largest η with p ≥ 0.5, smallest η with p < 0.5]."""
    above = etas[fractions >= 0.5]
    below = etas[fractions < 0.5]
    lo = float(above.max()) if len(above) else None
    hi = float(below.min()) if len(below) else None
    midpoint = 0.5 * (lo + hi) if (lo is not None and hi is not None) else None
    return {"lo": lo, "hi": hi, "midpoint": midpoint}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    args = p.parse_args()

    if not AGGREGATED_PARQUET.exists():
        print(f"Missing {AGGREGATED_PARQUET}; run compute_Q.py first.")
        return
    df = pd.read_parquet(AGGREGATED_PARQUET)
    # K ≤ 36 only for the main η_c analysis.
    df = df[df["k"].isin({10, 15, 20, 25, 36})].copy()

    # Seed-aggregated success fraction per (η, K).
    agg = (
        df.assign(success=(df["status"] == "transitioned").astype(int))
          .groupby(["eta", "k"])
          .agg(n_seeds=("seed", "nunique"),
               n_success=("success", "sum"))
          .reset_index()
    )
    agg["fraction"] = agg["n_success"] / agg["n_seeds"]

    # Per-K logistic fit
    per_K: Dict[int, Dict] = {}
    for k in sorted(agg["k"].unique()):
        sub = agg[agg["k"] == k].sort_values("eta")
        etas = sub["eta"].to_numpy(dtype=float)
        fracs = sub["fraction"].to_numpy(dtype=float)
        weights = sub["n_seeds"].to_numpy(dtype=float)
        fit = _fit_logistic(etas, fracs, weights=weights)
        bracket = _bracket(etas, fracs)
        per_K[int(k)] = {
            "logistic_fit": fit,
            "bracket": bracket,
            "eta_values": etas.tolist(),
            "fraction": fracs.tolist(),
            "n_seeds": weights.tolist(),
        }

    # Pooled (all K combined by weighted seed count)
    pooled = (
        df.assign(success=(df["status"] == "transitioned").astype(int))
          .groupby("eta")
          .agg(n_seeds=("seed", "count"),
               n_success=("success", "sum"))
          .reset_index()
          .sort_values("eta")
    )
    pooled["fraction"] = pooled["n_success"] / pooled["n_seeds"]
    pooled_fit = _fit_logistic(
        pooled["eta"].to_numpy(dtype=float),
        pooled["fraction"].to_numpy(dtype=float),
        weights=pooled["n_seeds"].to_numpy(dtype=float),
    )
    pooled_bracket = _bracket(
        pooled["eta"].to_numpy(dtype=float),
        pooled["fraction"].to_numpy(dtype=float),
    )

    out = {
        "method": {
            "model": "p(η) = σ(slope · (η_c − η))",
            "weighting": "binomial stderr from per-cell seed count",
            "K_range": [10, 36],
        },
        "pooled": {
            "eta_values": pooled["eta"].tolist(),
            "fraction": pooled["fraction"].tolist(),
            "n_seeds": pooled["n_seeds"].tolist(),
            "logistic_fit": pooled_fit,
            "bracket": pooled_bracket,
        },
        "per_K": per_K,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2,
                  default=lambda o: None if isinstance(o, float)
                                  and (math.isnan(o) or math.isinf(o)) else o)
    print(f"Wrote {OUT_PATH}")

    # Compact stdout
    print()
    print("Pooled transition-probability fit:")
    pf = pooled_fit
    if pf["eta_c"] is not None:
        print(f"  η_c = {pf['eta_c']:.4g} ± {pf['eta_c_se']:.2g}  "
              f"slope = {pf['slope']:.2g} ± {pf['slope_se']:.2g}  "
              f"R² = {pf['R2']:.3f}")
    else:
        print(f"  logistic fit failed ({pf['reason']})")
    pb = pooled_bracket
    print(f"  bracket: [{pb['lo']}, {pb['hi']}]  midpoint = {pb['midpoint']}")

    print()
    print("Per-K:")
    for k, entry in per_K.items():
        lf = entry["logistic_fit"]
        if lf["eta_c"] is not None:
            print(f"  K={k:>2}  η_c = {lf['eta_c']:.4g} ± {lf['eta_c_se']:.2g}  "
                  f"R² = {lf['R2']:.3f}")
        else:
            b = entry["bracket"]
            print(f"  K={k:>2}  [logistic failed: {lf['reason']}]  "
                  f"bracket = [{b['lo']}, {b['hi']}]")


if __name__ == "__main__":
    main()
