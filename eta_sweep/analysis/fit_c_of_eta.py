#!/usr/bin/env python
"""
Meta-fit for c(η) across the six η values.

Attempts three functional forms and selects the winner by BIC:
  (1) Power law:   log c = β · log η + log A
  (2) Quadratic:   log c = a · (log η)² + b · log η + d
  (3) Constant:    c = mean across η

We use the regime-aware c unless it is nan, in which case we fall back to
the affine c.

Writes eta_sweep/results/meta_fit.json with the winning functional form
and its parameters (plus all three fits for reference).

Usage:
    python eta_sweep/analysis/fit_c_of_eta.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ETA_SWEEP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ETA_SWEEP_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from eta_sweep.config import META_FIT_JSON, PER_ETA_FITS_JSON  # noqa: E402


def _bic(rss: float, n: int, k_params: int) -> float:
    if rss <= 0 or n <= 0:
        return float("inf")
    return float(n * math.log(rss / n) + k_params * math.log(n))


def _fit_power_law(log_eta: np.ndarray, log_c: np.ndarray) -> Dict:
    # log c = β log η + log A
    A = np.vstack([log_eta, np.ones_like(log_eta)]).T
    (beta, log_A), _, _, _ = np.linalg.lstsq(A, log_c, rcond=None)
    y_hat = beta * log_eta + log_A
    rss = float(np.sum((log_c - y_hat) ** 2))
    ss_tot = float(np.sum((log_c - np.mean(log_c)) ** 2))
    r2 = 1.0 - rss / ss_tot if ss_tot > 0 else float("nan")
    return {
        "form": "power_law",
        "beta": float(beta),
        "log_A": float(log_A),
        "A": float(math.exp(log_A)),
        "R2_log": float(r2),
        "BIC": _bic(rss, len(log_eta), 2),
    }


def _fit_quadratic(log_eta: np.ndarray, log_c: np.ndarray) -> Dict:
    # log c = a (log η)² + b log η + d
    A = np.vstack([log_eta ** 2, log_eta, np.ones_like(log_eta)]).T
    (a, b, d), _, _, _ = np.linalg.lstsq(A, log_c, rcond=None)
    y_hat = a * log_eta ** 2 + b * log_eta + d
    rss = float(np.sum((log_c - y_hat) ** 2))
    ss_tot = float(np.sum((log_c - np.mean(log_c)) ** 2))
    r2 = 1.0 - rss / ss_tot if ss_tot > 0 else float("nan")
    extremum_log_eta = float(-b / (2 * a)) if abs(a) > 1e-30 else float("nan")
    return {
        "form": "quadratic",
        "a": float(a),
        "b": float(b),
        "d": float(d),
        "R2_log": float(r2),
        "BIC": _bic(rss, len(log_eta), 3),
        "extremum_log_eta": extremum_log_eta,
        "extremum_eta": (
            float(math.exp(extremum_log_eta))
            if not math.isnan(extremum_log_eta) else float("nan")
        ),
        "extremum_type": "min" if a > 0 else "max",
    }


def _fit_constant(log_eta: np.ndarray, log_c: np.ndarray) -> Dict:
    mean = float(np.mean(log_c))
    rss = float(np.sum((log_c - mean) ** 2))
    ss_tot = float(np.sum((log_c - mean) ** 2))
    return {
        "form": "constant",
        "log_c_mean": mean,
        "c_mean": float(math.exp(mean)),
        "R2_log": 0.0,
        "BIC": _bic(rss, len(log_eta), 1),
    }


def main() -> None:
    if not PER_ETA_FITS_JSON.exists():
        print(f"Missing {PER_ETA_FITS_JSON}; run fit_per_eta.py first.")
        return

    with open(PER_ETA_FITS_JSON) as f:
        per_eta = json.load(f)

    etas: List[float] = []
    cs: List[float] = []
    source: List[str] = []
    for key, entry in per_eta.items():
        regime = entry.get("regime", {})
        affine = entry.get("affine", {})
        c = regime.get("c", None)
        src = "regime"
        if c is None or (isinstance(c, float) and math.isnan(c)):
            c = affine.get("c", None)
            src = "affine"
        if c is None or (isinstance(c, float) and math.isnan(c)):
            continue
        if c <= 0:
            # log-space fit requires positive c; flag and skip.
            continue
        etas.append(float(key))
        cs.append(float(c))
        source.append(src)

    out: Dict = {
        "n_points": len(etas),
        "eta_values": etas,
        "c_values": cs,
        "c_source": source,
    }

    if len(etas) < 3:
        out["status"] = "insufficient_data"
        with open(META_FIT_JSON, "w") as f:
            json.dump(out, f, indent=2)
        print("Not enough η points with positive c; wrote insufficient_data stub.")
        return

    log_eta = np.log(np.asarray(etas, dtype=float))
    log_c = np.log(np.asarray(cs, dtype=float))

    fits = {
        "power_law": _fit_power_law(log_eta, log_c),
        "quadratic": _fit_quadratic(log_eta, log_c),
        "constant": _fit_constant(log_eta, log_c),
    }
    out["fits"] = fits

    # Winner = lowest BIC.
    winner = min(fits.items(), key=lambda kv: kv[1]["BIC"])
    out["winner"] = winner[0]
    out["winner_params"] = winner[1]
    out["status"] = "ok"

    with open(META_FIT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote meta-fit → {META_FIT_JSON}")
    print(f"  winner = {winner[0]}")
    for name, fit in fits.items():
        print(f"  {name:<10}  BIC={fit['BIC']:>9.3f}  R²(log)={fit['R2_log']:.3f}")


if __name__ == "__main__":
    main()
