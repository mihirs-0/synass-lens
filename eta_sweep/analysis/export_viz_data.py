#!/usr/bin/env python
"""
Export a structured viz_data.json for external visualization.

Reads the aggregated parquet (from compute_Q.py) and the per-η fits
(from fit_per_eta.py), computes the stable-regime power-law meta-fit,
classifies each η into a regime, and writes a compact JSON.

Usage:
    python eta_sweep/analysis/export_viz_data.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ETA_SWEEP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ETA_SWEEP_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from eta_sweep.config import (  # noqa: E402
    AGGREGATED_PARQUET,
    PER_ETA_FITS_JSON,
    RESULTS_DIR,
)

OUT_PATH = RESULTS_DIR / "viz_data.json"
OUT_KEXT_PATH = RESULTS_DIR / "viz_data_k_extension.json"

# Main sweep convention: n_b = 1000 fixed, K ∈ {10, 15, 20, 25, 36} →
# D = n_b · K scales linearly with K.  Cells at K > 36 are the
# K-extension side experiment — isolated into their own JSON so that the
# main c(η) fit isn't mixed with data on a larger K axis.
MAIN_K_VALUES = {10, 15, 20, 25, 36}

# --- Helpers ----------------------------------------------------------------

def _r4(x) -> Optional[float]:
    """Round to 4 significant figures; None stays None; NaN → None."""
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(xf) or math.isinf(xf):
        return None
    return float(f"{xf:.4g}")


def _classify_regime(n_transitioned: int, n_total: int,
                     c: Optional[float]) -> str:
    if n_total == 0:
        return "failure"
    if n_transitioned == 0:
        return "failure"
    if n_transitioned < n_total:
        return "boundary"
    # All transitioned.  Split by whether c is consistent with the stable
    # power law.  Use c ≤ 15 as the cutoff — values above 15 mean the Q
    # vs log K slope has already diverged above the stable trend.
    if c is None or math.isnan(c):
        return "stable"
    return "stable" if c <= 15.0 else "near_boundary"


def _last_delta_z(eta: float, k: int, seed: int = 0) -> Optional[float]:
    """Read the last Δz value from the cell's log.jsonl."""
    # Directory naming uses f"{eta:g}"
    p = RESULTS_DIR / f"eta_{eta:g}_K_{k}_seed_{seed}" / "log.jsonl"
    if not p.exists():
        return None
    last = None
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                last = line
    if last is None:
        return None
    try:
        rec = json.loads(last)
        dz = rec.get("delta_z")
        if dz is None:
            return None
        return float(dz)
    except Exception:
        return None


# --- Main -------------------------------------------------------------------

def _fit_affine(K: np.ndarray, Q: np.ndarray) -> Dict[str, Optional[float]]:
    if len(K) < 3:
        return {"c": None, "q0": None, "R2": None, "n": int(len(K))}
    logK = np.log(K.astype(float))
    A = np.vstack([logK, np.ones_like(logK)]).T
    (c, q0), *_ = np.linalg.lstsq(A, Q.astype(float), rcond=None)
    yhat = c * logK + q0
    ss_res = float(np.sum((Q - yhat) ** 2))
    ss_tot = float(np.sum((Q - Q.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    return {"c": float(c), "q0": float(q0), "R2": r2, "n": int(len(K))}


def _fit_regime(K: np.ndarray, Q: np.ndarray) -> Dict[str, Optional[float]]:
    if len(K) < 3:
        return {"c": None, "K_star": None, "R2": None}
    from scipy.optimize import curve_fit
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
        return {"c": float(c), "K_star": float(K_star), "R2": r2}
    except Exception:
        return {"c": None, "K_star": None, "R2": None}


def main() -> None:
    if not AGGREGATED_PARQUET.exists():
        print(f"Missing {AGGREGATED_PARQUET}; run compute_Q.py first.")
        return
    df_all = pd.read_parquet(AGGREGATED_PARQUET)

    # ---- Split into main (K ≤ 36) and K-extension (K > 36) ----
    df = df_all[df_all["k"].isin(MAIN_K_VALUES)].copy()
    df_kext = df_all[~df_all["k"].isin(MAIN_K_VALUES)].copy()

    # ---- Per-η summary table for the main sweep ----
    eta_values = sorted(df["eta"].unique())
    per_eta_entries: List[Dict] = []

    for eta in eta_values:
        sub = df[df["eta"] == eta].sort_values("k")
        n_total = len(sub)
        n_transitioned = int((sub["status"] == "transitioned").sum())
        trans_frac = n_transitioned / n_total if n_total > 0 else 0.0

        # Re-fit per-η on K ≤ 36 only (ignore any K-extension per_eta_fits)
        sub_valid = sub.dropna(subset=["Q"])
        K = sub_valid["k"].to_numpy(dtype=float)
        Q = sub_valid["Q"].to_numpy(dtype=float)
        affine = _fit_affine(K, Q)
        regime_fit = _fit_regime(K, Q)
        c = regime_fit.get("c")
        c_r2 = regime_fit.get("R2")
        k_star = regime_fit.get("K_star")
        n_k_valid = int(len(K))
        if c is None or (isinstance(c, float) and math.isnan(c)):
            c = affine.get("c")
            c_r2 = affine.get("R2")

        # Classify regime (string)
        regime = _classify_regime(n_transitioned, n_total, c)

        # Valid-Q aggregates
        sub_valid = sub.dropna(subset=["Q"])
        mean_t1 = float(sub_valid["t1"].astype(float).mean()) if len(sub_valid) else None
        mean_sw = float(sub_valid["snap_width_steps"].astype(float).mean()) if len(sub_valid) else None
        fe = sub_valid["frac_excess"].astype(float)
        mean_fe = float(fe.mean()) if len(fe) else None
        std_fe = float(fe.std()) if len(fe) > 1 else None

        # per_K rows
        per_K: List[Dict] = []
        for _, row in sub.iterrows():
            transitioned = (row["status"] == "transitioned")
            final_dz = _last_delta_z(row["eta"], int(row["k"]))
            per_K.append({
                "K": int(row["k"]),
                "transitioned": bool(transitioned),
                "Q": _r4(row.get("Q")),
                "delta_L": _r4(row.get("delta_L")),
                "excess_fraction": _r4(row.get("frac_excess")),
                "t1": int(row["t1"]) if not pd.isna(row.get("t1")) else None,
                "snap_width": (int(row["snap_width_steps"])
                               if not pd.isna(row.get("snap_width_steps"))
                               else None),
                "final_step": int(row.get("final_step", 0) or 0),
                "final_delta_z": _r4(final_dz),
            })

        per_eta_entries.append({
            "eta": _r4(eta),
            "regime": regime,
            "n_transitioned": n_transitioned,
            "n_total": n_total,
            "transition_fraction": _r4(trans_frac),
            "c": _r4(c),
            "c_R2": _r4(c_r2),
            "K_star": _r4(k_star),
            "n_K_valid": n_k_valid,
            "mean_t1": _r4(mean_t1),
            "mean_snap_width": _r4(mean_sw),
            "mean_excess_fraction": _r4(mean_fe),
            "std_excess_fraction": _r4(std_fe),
            "per_K": per_K,
        })

    # ---- Summary counts ----
    n_cells_total = len(df)
    n_transitioned = int((df["status"] == "transitioned").sum())
    n_stuck = int((df["status"] == "stuck").sum())

    # ---- η_c bracket & regime boundaries ----
    # eta_c bracket: [largest η with fraction ≥ 0.5 AND n_total=5,
    #                smallest η with fraction < 0.5 AND n_total=5]
    stable_etas = [e["eta"] for e in per_eta_entries if e["regime"] == "stable"]
    near_etas = [e["eta"] for e in per_eta_entries if e["regime"] == "near_boundary"]
    failure_etas = [e["eta"] for e in per_eta_entries if e["regime"] == "failure"]

    # find the bracket from ordered per_eta_entries
    lo, hi = None, None
    for e in per_eta_entries:
        if e["n_total"] < 5:
            continue  # skip cells that aren't fully complete
        if e["transition_fraction"] is not None and e["transition_fraction"] >= 0.5:
            lo = e["eta"]
        elif (hi is None
              and e["transition_fraction"] is not None
              and e["transition_fraction"] < 0.5):
            hi = e["eta"]

    stable_max_eta = max(stable_etas) if stable_etas else None
    failure_min_eta = min(failure_etas) if failure_etas else None

    summary = {
        "n_cells_total": n_cells_total,
        "n_transitioned": n_transitioned,
        "n_stuck": n_stuck,
        "eta_c_bracket": [_r4(lo), _r4(hi)],
        "regime_boundaries": {
            "stable_max_eta": _r4(stable_max_eta),
            "near_boundary_eta_values": [_r4(x) for x in near_etas],
            "failure_min_eta": _r4(failure_min_eta),
        },
    }

    # ---- Stable-regime power-law meta-fit ----
    # Use η values classified "stable" AND n_total=5 AND c is not None
    meta_stable_rows = [
        e for e in per_eta_entries
        if e["regime"] == "stable" and e["c"] is not None and e["n_total"] == 5
    ]
    meta_stable = None
    if len(meta_stable_rows) >= 3:
        etas_arr = np.array([r["eta"] for r in meta_stable_rows], dtype=float)
        cs_arr = np.array([r["c"] for r in meta_stable_rows], dtype=float)
        log_eta = np.log(etas_arr)
        log_c = np.log(cs_arr)
        A = np.vstack([log_eta, np.ones_like(log_eta)]).T
        (beta, log_A), *_ = np.linalg.lstsq(A, log_c, rcond=None)
        yhat = beta * log_eta + log_A
        r2 = 1.0 - float(np.sum((log_c - yhat) ** 2)) / float(np.sum((log_c - log_c.mean()) ** 2))
        meta_stable = {
            "eta_values_used": [_r4(x) for x in etas_arr.tolist()],
            "A": _r4(float(math.exp(log_A))),
            "beta": _r4(float(-beta)),  # convention: c ∝ η^(-β), so report positive β
            "R2_log": _r4(r2),
            "fit_range_eta": [_r4(float(etas_arr.min())), _r4(float(etas_arr.max()))],
        }

    meta_fit = {"stable_regime": meta_stable}

    # ---- Divergence signals ----
    # η_c = midpoint of bracket (fall back to 2.5e-3 if bracket incomplete)
    if lo is not None and hi is not None:
        eta_c = 0.5 * (lo + hi)
    else:
        eta_c = 2.5e-3

    divergence_points: List[Dict] = []
    for e in per_eta_entries:
        if e["mean_t1"] is None and e["mean_snap_width"] is None \
           and e["mean_excess_fraction"] is None:
            continue
        distance = eta_c - e["eta"]
        divergence_points.append({
            "eta": e["eta"],
            "distance_from_eta_c": _r4(distance),
            "t1": e["mean_t1"],
            "snap_width": e["mean_snap_width"],
            "excess_fraction": e["mean_excess_fraction"],
        })

    divergence_signals = {
        "description": "Critical-slowing signatures approaching eta_c",
        "observables": ["t1", "snap_width", "excess_fraction"],
        "eta_c_used": _r4(eta_c),
        "data_points": divergence_points,
    }

    # ---- Assemble & write ----
    out = {
        "summary": summary,
        "per_eta": per_eta_entries,
        "meta_fit": meta_fit,
        "divergence_signals": divergence_signals,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUT_PATH}")
    print()

    # ---- Compact stdout summary ----
    print(f"{'η':>8}  {'regime':<14}  {'tr/total':>8}  {'c':>8}  {'R²':>6}  "
          f"{'<t1>':>7}  {'<snap>':>7}  {'<frac_ex>':>9}")
    print("-" * 78)
    for e in per_eta_entries:
        eta_str = f"{e['eta']:g}"
        tr_str = f"{e['n_transitioned']}/{e['n_total']}"
        c_str = f"{e['c']:.3g}" if e["c"] is not None else "—"
        r2_str = f"{e['c_R2']:.3f}" if e["c_R2"] is not None else "—"
        t1_str = f"{e['mean_t1']:.0f}" if e["mean_t1"] is not None else "—"
        sw_str = f"{e['mean_snap_width']:.0f}" if e["mean_snap_width"] is not None else "—"
        fe_str = f"{e['mean_excess_fraction']:.3f}" if e["mean_excess_fraction"] is not None else "—"
        print(f"{eta_str:>8}  {e['regime']:<14}  {tr_str:>8}  "
              f"{c_str:>8}  {r2_str:>6}  {t1_str:>7}  {sw_str:>7}  {fe_str:>9}")

    # Summary
    print()
    print(f"n_cells_total = {summary['n_cells_total']}  "
          f"transitioned = {summary['n_transitioned']}  "
          f"stuck = {summary['n_stuck']}")
    print(f"η_c bracket: {summary['eta_c_bracket']}")
    if meta_stable:
        print(f"Stable-regime power law (n={len(meta_stable['eta_values_used'])}): "
              f"c = {meta_stable['A']:.3g} · η^(-{meta_stable['beta']:.3g})  "
              f"R²(log) = {meta_stable['R2_log']:.3g}")

    # ========================================================================
    # K-extension side experiment
    # ========================================================================
    # For each η that has K > 36 data, report the main-sweep 5-K fit (from the
    # main viz_data) plus the extended K points.  This lets a viewer compare
    # what the K≤36 power-law predicts to what's actually observed at higher K.

    kext_entries: List[Dict] = []
    for eta in sorted(df_kext["eta"].unique()):
        ext = df_kext[df_kext["eta"] == eta].sort_values("k")
        main = df[(df["eta"] == eta)].dropna(subset=["Q"]).sort_values("k")

        per_K_main: List[Dict] = []
        for _, row in main.iterrows():
            per_K_main.append({
                "K": int(row["k"]),
                "Q": _r4(row.get("Q")),
                "delta_L": _r4(row.get("delta_L")),
                "transitioned": bool(row["status"] == "transitioned"),
            })

        per_K_ext: List[Dict] = []
        for _, row in ext.iterrows():
            per_K_ext.append({
                "K": int(row["k"]),
                "n_unique_b": 1000,
                "D_examples": 1000 * int(row["k"]),
                "transitioned": bool(row["status"] == "transitioned"),
                "Q": _r4(row.get("Q")),
                "delta_L": _r4(row.get("delta_L")),
                "excess_fraction": _r4(row.get("frac_excess")),
                "t1": int(row["t1"]) if not pd.isna(row.get("t1")) else None,
                "snap_width": (int(row["snap_width_steps"])
                               if not pd.isna(row.get("snap_width_steps"))
                               else None),
                "final_step": int(row.get("final_step", 0) or 0),
            })

        # Re-fit main-sweep (K ≤ 36) for this η
        K_m = main["k"].to_numpy(dtype=float)
        Q_m = main["Q"].to_numpy(dtype=float)
        main_fit = _fit_affine(K_m, Q_m)

        # Re-fit including K-extension
        combined = pd.concat([main, ext]).dropna(subset=["Q"]).sort_values("k")
        K_c = combined["k"].to_numpy(dtype=float)
        Q_c = combined["Q"].to_numpy(dtype=float)
        combined_fit = _fit_affine(K_c, Q_c)

        # Extrapolated Q from the 5-K fit at each extended K
        extrapolated: List[Dict] = []
        if main_fit["c"] is not None:
            for _, row in ext.iterrows():
                if pd.isna(row.get("Q")):
                    continue
                K_i = float(row["k"])
                Q_pred = main_fit["c"] * math.log(K_i) + main_fit["q0"]
                Q_obs = float(row["Q"])
                extrapolated.append({
                    "K": int(K_i),
                    "Q_observed": _r4(Q_obs),
                    "Q_predicted_from_5K": _r4(Q_pred),
                    "residual": _r4(Q_obs - Q_pred),
                    "ratio_obs_pred": _r4(Q_obs / Q_pred) if Q_pred > 0 else None,
                })

        kext_entries.append({
            "eta": _r4(eta),
            "per_K_main_sweep": per_K_main,
            "per_K_extension": per_K_ext,
            "fit_5K_main_only": {
                "c": _r4(main_fit["c"]),
                "q0": _r4(main_fit["q0"]),
                "R2": _r4(main_fit["R2"]),
                "n": main_fit["n"],
            },
            "fit_combined_main_plus_extension": {
                "c": _r4(combined_fit["c"]),
                "q0": _r4(combined_fit["q0"]),
                "R2": _r4(combined_fit["R2"]),
                "n": combined_fit["n"],
            },
            "extrapolation_check": extrapolated,
        })

    kext_out = {
        "description": (
            "K-extension side experiment.  Cells at K > 36 with n_b = 1000 "
            "(D = K·n_b proportional to K, same convention as main sweep).  "
            "Compare per-η power-law fits from the main 5-K range (K ≤ 36) "
            "against extended-range observations to test whether Q ∝ log K "
            "continues, saturates, or breaks at large K."
        ),
        "convention": {
            "n_unique_b": 1000,
            "disambiguation_prefix_length": 2,
            "main_K_values": sorted(MAIN_K_VALUES),
            "extension_K_values": sorted({int(k) for k in df_kext["k"].unique()}),
        },
        "summary": {
            "n_cells_kext": int(len(df_kext)),
            "n_transitioned_kext": int((df_kext["status"] == "transitioned").sum()),
            "n_stuck_kext": int((df_kext["status"] == "stuck").sum()),
        },
        "per_eta": kext_entries,
    }
    with open(OUT_KEXT_PATH, "w") as f:
        json.dump(kext_out, f, indent=2)
    print()
    print(f"Wrote {OUT_KEXT_PATH}")
    print()
    # Compact K-ext table
    print("K-extension per-η:")
    for e in kext_entries:
        print(f"  η={e['eta']:g}  main-fit c={e['fit_5K_main_only']['c']}  "
              f"R²={e['fit_5K_main_only']['R2']}")
        for row in e["extrapolation_check"]:
            print(f"    K={row['K']:>3}  Q_obs={row['Q_observed']:>7}  "
                  f"Q_pred={row['Q_predicted_from_5K']:>7}  "
                  f"ratio={row['ratio_obs_pred']}")


if __name__ == "__main__":
    main()
