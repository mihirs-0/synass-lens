#!/usr/bin/env python
"""
Stable-regime power-law fit with bootstrap CI.

For η ∈ STABLE_ETAS (default: those classified "stable" in the sweep),
aggregate Q across seeds within each (η, K) cell, fit Q = c · log(K / K*)
per η, then fit the log-log power law c = A · η^(-β).

Bootstrap 95% CIs on (β, A) by resampling seeds within each (η, K) cell
(nested two-level resample: η-level resampling not applied; we treat the
η axis as fixed and the seed axis as the sampling unit).

Output: eta_sweep/results/stable_regime_fit.json
  {
    "method": {
      "per_eta_cell_model": "Q = c · log(K / K*)",
      "meta_model": "c = A · η^(-β)",
      "n_bootstrap": int,
      "stable_etas_used": [...]
    },
    "point_estimate": {
      "A": float, "beta": float, "R2_log": float,
      "per_eta": {<eta>: {"c": ..., "K_star": ..., "R2": ...}}
    },
    "bootstrap_ci_95": {
      "A": [lo, hi], "beta": [lo, hi],
      "per_eta_c": {<eta>: [lo, hi]}
    }
  }

Usage:
    python eta_sweep/analysis/stable_regime_fit.py [--n-bootstrap 1000]
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

OUT_PATH = RESULTS_DIR / "stable_regime_fit.json"

# Default stable-regime η values — can be overridden via CLI.
DEFAULT_STABLE_ETAS = [1e-4, 3e-4, 5e-4, 7e-4, 1e-3]


# ---------------------------------------------------------------------------
# Inner fits
# ---------------------------------------------------------------------------

def _fit_regime(K: np.ndarray, Q: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Q = c · log(K / K*).  Returns (c, K_star, R²).  None on failure."""
    if len(K) < 3:
        return None, None, None

    def model(K_, c, K_star):
        return c * np.log(K_ / K_star)

    logK = np.log(K.astype(float))
    c0 = float(np.polyfit(logK, Q, 1)[0])
    try:
        popt, _ = curve_fit(
            model, K.astype(float), Q.astype(float),
            p0=[c0, 2.0],
            bounds=([-np.inf, 2.0], [np.inf, 30.0]),
            maxfev=5000,
        )
        c, K_star = popt
        yhat = model(K.astype(float), c, K_star)
        ss_res = float(np.sum((Q - yhat) ** 2))
        ss_tot = float(np.sum((Q - Q.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
        return float(c), float(K_star), r2
    except Exception:
        return None, None, None


def _fit_meta(etas: np.ndarray, cs: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """log c = log A - β log η. Returns (A, β, R²_log)."""
    valid = (cs > 0) & ~np.isnan(cs)
    if valid.sum() < 3:
        return None, None, None
    log_eta = np.log(etas[valid])
    log_c = np.log(cs[valid])
    # log_c = -β * log_eta + log_A
    X = np.vstack([log_eta, np.ones_like(log_eta)]).T
    (slope, log_A), *_ = np.linalg.lstsq(X, log_c, rcond=None)
    yhat = slope * log_eta + log_A
    ss_res = float(np.sum((log_c - yhat) ** 2))
    ss_tot = float(np.sum((log_c - log_c.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    return float(math.exp(log_A)), float(-slope), r2


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_fit(
    df: pd.DataFrame,
    stable_etas: List[float],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> Dict:
    """Bootstrap by resampling seeds within each (η, K) cell.

    For each bootstrap iteration:
      1. For each (η, K) cell, resample seeds with replacement → compute mean Q.
      2. Fit per-η Q vs log K → get c_η.
      3. Fit meta-model log c = log A - β log η on the resulting c_η set.
      4. Record (A, β).

    Returns distribution of (A, β) plus 95% CIs.
    """
    cells: List[Tuple[float, int, np.ndarray]] = []  # (η, K, array_of_Q_by_seed)
    for eta in stable_etas:
        sub_eta = df[np.isclose(df["eta"], eta, rtol=1e-6)]
        for k in sorted(sub_eta["k"].unique()):
            sub = sub_eta[sub_eta["k"] == k].dropna(subset=["Q"])
            if len(sub) == 0:
                continue
            cells.append((eta, int(k), sub["Q"].to_numpy(dtype=float)))

    As: List[float] = []
    betas: List[float] = []
    per_eta_cs: Dict[float, List[float]] = {eta: [] for eta in stable_etas}

    for _ in range(n_bootstrap):
        per_eta_means: Dict[float, Dict[int, float]] = {}
        for eta, k, qs in cells:
            resampled = rng.choice(qs, size=len(qs), replace=True)
            per_eta_means.setdefault(eta, {})[k] = float(np.mean(resampled))
        # Per-η regime fit
        c_by_eta: Dict[float, Optional[float]] = {}
        for eta in stable_etas:
            if eta not in per_eta_means:
                continue
            d = per_eta_means[eta]
            K_arr = np.array(sorted(d.keys()), dtype=float)
            Q_arr = np.array([d[int(k)] for k in K_arr], dtype=float)
            c, _, _ = _fit_regime(K_arr, Q_arr)
            c_by_eta[eta] = c
            if c is not None:
                per_eta_cs[eta].append(c)
        # Meta-fit
        etas_arr = np.array([e for e in c_by_eta if c_by_eta[e] is not None], dtype=float)
        cs_arr = np.array([c_by_eta[e] for e in etas_arr], dtype=float)
        A, beta, _ = _fit_meta(etas_arr, cs_arr)
        if A is not None and beta is not None:
            As.append(A)
            betas.append(beta)

    def _ci(xs: List[float]) -> List[Optional[float]]:
        if len(xs) < 10:
            return [None, None]
        return [float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))]

    return {
        "n_bootstrap_success": len(As),
        "A_ci": _ci(As),
        "beta_ci": _ci(betas),
        "per_eta_c_ci": {f"{eta:g}": _ci(per_eta_cs[eta]) for eta in stable_etas},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--stable-etas", type=float, nargs="*", default=DEFAULT_STABLE_ETAS)
    p.add_argument("--rng-seed", type=int, default=42)
    args = p.parse_args()

    if not AGGREGATED_PARQUET.exists():
        print(f"Missing {AGGREGATED_PARQUET}; run compute_Q.py first.")
        return
    df_all = pd.read_parquet(AGGREGATED_PARQUET)
    # K ≤ 36 only — K-extension is a separate analysis.
    df = df_all[df_all["k"].isin({10, 15, 20, 25, 36})].copy()

    # Point-estimate fit: per-(η, K) mean Q across seeds, then regime fit per η.
    point: Dict[str, Dict] = {"per_eta": {}}
    etas_pt, cs_pt = [], []
    for eta in args.stable_etas:
        sub_eta = df[np.isclose(df["eta"], eta, rtol=1e-6)]
        per_k = sub_eta.groupby("k")["Q"].mean().dropna().sort_index()
        K_arr = per_k.index.to_numpy(dtype=float)
        Q_arr = per_k.to_numpy(dtype=float)
        c, k_star, r2 = _fit_regime(K_arr, Q_arr)
        n_seeds_per_k = (
            sub_eta.dropna(subset=["Q"])
            .groupby("k")["seed"].nunique().to_dict()
        )
        point["per_eta"][f"{eta:g}"] = {
            "c": c, "K_star": k_star, "R2": r2,
            "n_K": int(len(K_arr)),
            "K_values": K_arr.tolist(),
            "Q_mean_per_K": Q_arr.tolist(),
            "n_seeds_per_K": {int(k): int(n) for k, n in n_seeds_per_k.items()},
        }
        if c is not None and c > 0:
            etas_pt.append(eta)
            cs_pt.append(c)
    A_pt, beta_pt, r2_pt = _fit_meta(np.array(etas_pt), np.array(cs_pt))
    point["A"] = A_pt
    point["beta"] = beta_pt
    point["R2_log"] = r2_pt

    # Bootstrap CIs
    rng = np.random.default_rng(args.rng_seed)
    bs = bootstrap_fit(df, args.stable_etas, args.n_bootstrap, rng)

    out = {
        "method": {
            "per_eta_cell_model": "Q = c · log(K / K*)",
            "meta_model": "c = A · η^(-β)",
            "n_bootstrap": args.n_bootstrap,
            "stable_etas_used": args.stable_etas,
            "K_range": [10, 36],
            "resample_unit": "seed within (η, K) cell",
        },
        "point_estimate": point,
        "bootstrap_ci_95": bs,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, default=lambda o: None if isinstance(o, float) and (math.isnan(o) or math.isinf(o)) else o)
    print(f"Wrote {OUT_PATH}")
    if A_pt is not None:
        print(f"Point estimate: c = {A_pt:.3g} · η^(-{beta_pt:.3g})  R²(log) = {r2_pt:.3g}")
        print(f"Bootstrap β 95% CI: {bs['beta_ci']}")
        print(f"Bootstrap A 95% CI: {bs['A_ci']}")
    for eta in args.stable_etas:
        key = f"{eta:g}"
        pe = point["per_eta"].get(key, {})
        c = pe.get("c")
        ci = bs["per_eta_c_ci"].get(key, [None, None])
        n_seeds = pe.get("n_seeds_per_K", {})
        seeds_str = "/".join(str(v) for v in n_seeds.values()) or "—"
        c_str = f"{c:.3f}" if c is not None else "—"
        print(f"  η={eta:g}  n_seeds_per_K={seeds_str}  c={c_str}  "
              f"95% CI = [{ci[0]}, {ci[1]}]")


if __name__ == "__main__":
    main()
