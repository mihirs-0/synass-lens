#!/usr/bin/env python
"""
Per-η Q vs log K fits.

For each η, fit:
  (1) Affine:        Q = c · log K + q₀
  (2) Regime-aware:  Q = c · log(K / K*)        K* ∈ [2, 30]

Requires at least 3 valid K points per η; otherwise reports "insufficient data".

Writes eta_sweep/results/per_eta_fits.json with schema:
    {
      "3e-04": {
        "affine":  {"c": ..., "q0": ..., "R2": ..., "c_stderr": ...},
        "regime":  {"c": ..., "K_star": ..., "R2": ...},
        "n_valid_K": int,
        "K_values": [...],
        "Q_values": [...]
      },
      ...
    }

Usage:
    python eta_sweep/analysis/fit_per_eta.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

ETA_SWEEP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ETA_SWEEP_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from eta_sweep.config import (  # noqa: E402
    AGGREGATED_PARQUET,
    PER_ETA_FITS_JSON,
)


def _r2(y: np.ndarray, y_hat: np.ndarray) -> float:
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _fit_affine(logK: np.ndarray, Q: np.ndarray) -> Dict[str, float]:
    A = np.vstack([logK, np.ones_like(logK)]).T
    (c, q0), _res, _rank, _sv = np.linalg.lstsq(A, Q, rcond=None)
    y_hat = c * logK + q0
    r2 = _r2(Q, y_hat)
    # Simple stderr on c from residuals (no covariance needed for reporting).
    n = len(logK)
    if n > 2:
        resid = Q - y_hat
        sigma_sq = float(np.sum(resid ** 2)) / (n - 2)
        cov = sigma_sq * np.linalg.inv(A.T @ A)
        c_stderr = float(math.sqrt(max(cov[0, 0], 0.0)))
    else:
        c_stderr = float("nan")
    return {"c": float(c), "q0": float(q0), "R2": float(r2),
            "c_stderr": c_stderr}


def _fit_regime(K: np.ndarray, Q: np.ndarray) -> Dict[str, float]:
    """Q = c · log(K / K*).  K* ∈ [2, 30].  Uses scipy.curve_fit."""
    def model(K_, c, K_star):
        return c * np.log(K_ / K_star)

    # Initial guess: K* ≈ 2, c from a simple least-squares against log K.
    logK = np.log(K)
    c0 = float(np.polyfit(logK, Q, 1)[0])
    K_star0 = 2.0
    try:
        popt, _ = curve_fit(
            model, K.astype(float), Q.astype(float),
            p0=[c0, K_star0],
            bounds=([-np.inf, 2.0], [np.inf, 30.0]),
            maxfev=5000,
        )
        c, K_star = popt
        y_hat = model(K.astype(float), c, K_star)
        r2 = _r2(Q, y_hat)
        return {"c": float(c), "K_star": float(K_star), "R2": float(r2)}
    except Exception as exc:
        return {"c": float("nan"), "K_star": float("nan"), "R2": float("nan"),
                "error": str(exc)}


def main() -> None:
    if not AGGREGATED_PARQUET.exists():
        print(f"Missing {AGGREGATED_PARQUET}; run compute_Q.py first.")
        return

    df = pd.read_parquet(AGGREGATED_PARQUET)
    df = df.dropna(subset=["Q"])

    fits: Dict[str, Dict] = {}

    # Group by η (with some tolerance since floats).
    for eta, grp in df.groupby("eta"):
        # Optionally average over seeds per K.
        per_k = grp.groupby("k")["Q"].mean().reset_index().sort_values("k")
        K = per_k["k"].to_numpy(dtype=float)
        Q = per_k["Q"].to_numpy(dtype=float)
        key = f"{eta:g}"

        entry: Dict = {
            "n_valid_K": int(len(K)),
            "K_values": K.tolist(),
            "Q_values": Q.tolist(),
        }
        if len(K) < 3:
            entry["affine"] = {"error": "insufficient data"}
            entry["regime"] = {"error": "insufficient data"}
            fits[key] = entry
            continue

        logK = np.log(K)
        entry["affine"] = _fit_affine(logK, Q)
        entry["regime"] = _fit_regime(K, Q)
        fits[key] = entry

    with open(PER_ETA_FITS_JSON, "w") as f:
        json.dump(fits, f, indent=2)

    print(f"Wrote per-η fits → {PER_ETA_FITS_JSON}")
    for key, entry in fits.items():
        a = entry.get("affine", {})
        r = entry.get("regime", {})
        print(f"  η={key}:  n={entry['n_valid_K']}  "
              f"affine c={a.get('c', float('nan')):.3g} (R²={a.get('R2', float('nan')):.3g})  "
              f"regime c={r.get('c', float('nan')):.3g} K*={r.get('K_star', float('nan')):.3g} "
              f"(R²={r.get('R2', float('nan')):.3g})")


if __name__ == "__main__":
    main()
