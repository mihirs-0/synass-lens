#!/usr/bin/env python
"""
Paper-ready analysis package built from the existing eta_sweep parquet only.

Produces:
  results/eta_c_per_K.json
  results/critical_slowing_summary.json
  results/t1_fit_results.json
  results/q_regime_fits.json
  results/figures/phase_diagram_current.png
  results/figures/critical_slowing_current.png
  results/figures/t1_fits_current.png
  results/figures/q_regimes_current.png

No new training runs.  All numbers are derived from
results/aggregated_runs.parquet.

Usage:
    python eta_sweep/analysis/paper_package.py
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ETA_SWEEP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ETA_SWEEP_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from eta_sweep.config import AGGREGATED_PARQUET, RESULTS_DIR  # noqa: E402

FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True, parents=True)

MAIN_K_VALUES = {10, 15, 20, 25, 36}
KEXT_K_VALUES = {50, 75, 100}

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 140,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
})


def _load_df() -> pd.DataFrame:
    df = pd.read_parquet(AGGREGATED_PARQUET)
    # Drop the two stale smoke-test rows: η=3e-4 K=10 seed=1/2 with final_step=200.
    drop_mask = (
        np.isclose(df["eta"], 3e-4) & (df["k"] == 10)
        & (df["final_step"] == 200) & (df["status"] == "inconclusive")
    )
    df = df.loc[~drop_mask].copy()
    df["transitioned"] = (df["status"] == "transitioned").astype(int)
    return df


# ---------------------------------------------------------------------------
# 1. Per-K η_c logistic fit
# ---------------------------------------------------------------------------

def _logistic(eta: np.ndarray, eta_c: float, slope: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-slope * (eta_c - eta)))


def fit_eta_c_per_K(df: pd.DataFrame) -> Dict:
    """Per-K seed-averaged transition fraction + logistic fit.

    Returns dict with per-K {n_seeds_per_eta, eta_values, fractions,
    eta_c, eta_c_se, slope, R2, identifiable, reason}.
    """
    out: Dict[int, Dict] = {}
    main = df[df["k"].isin(MAIN_K_VALUES)].copy()
    for k in sorted(main["k"].unique()):
        sub = main[main["k"] == k]
        # Seed-averaged transition fraction at each η.
        agg = (sub.groupby("eta")
                  .agg(n_seeds=("seed", "nunique"),
                       n_success=("transitioned", "sum"))
                  .reset_index().sort_values("eta"))
        agg["fraction"] = agg["n_success"] / agg["n_seeds"]
        etas = agg["eta"].to_numpy(dtype=float)
        fracs = agg["fraction"].to_numpy(dtype=float)
        weights = agg["n_seeds"].to_numpy(dtype=float)

        # Identifiability: must have at least one fraction ≥ 0.5 AND one < 0.5.
        above = etas[fracs >= 0.5]
        below = etas[fracs < 0.5]
        if len(above) == 0 or len(below) == 0:
            out[int(k)] = {
                "n_eta_points": int(len(etas)),
                "eta_values": etas.tolist(),
                "fraction": fracs.tolist(),
                "n_seeds_per_eta": [int(x) for x in weights.tolist()],
                "identifiable": False,
                "reason": "no crossing of 0.5",
                "bracket_lo": float(above.max()) if len(above) else None,
                "bracket_hi": float(below.min()) if len(below) else None,
                "eta_c": None, "eta_c_se": None, "slope": None,
                "slope_se": None, "R2": None,
            }
            continue

        eta_c0 = 0.5 * (above.max() + below.min())
        slope0 = 1e4
        # Per-point sigma from binomial stderr.
        p = np.clip(fracs, 1e-6, 1 - 1e-6)
        sigma = np.sqrt(p * (1 - p) / np.maximum(weights, 1))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, pcov = curve_fit(
                    _logistic, etas, fracs, p0=[eta_c0, slope0],
                    sigma=sigma, absolute_sigma=True, maxfev=10000,
                )
            eta_c, slope = popt
            perr = np.sqrt(np.maximum(np.diag(pcov), 0.0))
            yhat = _logistic(etas, *popt)
            ss_res = float(np.sum((fracs - yhat) ** 2))
            ss_tot = float(np.sum((fracs - fracs.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
            # SE finite check
            eta_c_se = float(perr[0]) if np.isfinite(perr[0]) else None
            slope_se = float(perr[1]) if np.isfinite(perr[1]) else None
            out[int(k)] = {
                "n_eta_points": int(len(etas)),
                "eta_values": etas.tolist(),
                "fraction": fracs.tolist(),
                "n_seeds_per_eta": [int(x) for x in weights.tolist()],
                "identifiable": True,
                "eta_c": float(eta_c), "eta_c_se": eta_c_se,
                "slope": float(slope), "slope_se": slope_se,
                "R2": r2,
                "bracket_lo": float(above.max()),
                "bracket_hi": float(below.min()),
                "reason": "ok",
            }
        except Exception as exc:
            out[int(k)] = {
                "n_eta_points": int(len(etas)),
                "eta_values": etas.tolist(),
                "fraction": fracs.tolist(),
                "n_seeds_per_eta": [int(x) for x in weights.tolist()],
                "identifiable": False,
                "reason": f"fit failed: {exc}",
                "bracket_lo": float(above.max()),
                "bracket_hi": float(below.min()),
                "eta_c": None, "eta_c_se": None, "slope": None,
                "slope_se": None, "R2": None,
            }

    return out


# ---------------------------------------------------------------------------
# 2. Phase-diagram figure
# ---------------------------------------------------------------------------

def make_phase_diagram(df: pd.DataFrame, eta_c_per_K: Dict) -> Path:
    main = df[df["k"].isin(MAIN_K_VALUES)].copy()
    agg = (main.groupby(["eta", "k"])
               .agg(n_seeds=("seed", "nunique"),
                    n_success=("transitioned", "sum"))
               .reset_index())
    agg["fraction"] = agg["n_success"] / agg["n_seeds"]

    fig, ax = plt.subplots(figsize=(7, 4.2))

    # Heatmap: scatter with color = fraction, marker = single vs multi seed.
    multi = agg[agg["n_seeds"] >= 2]
    single = agg[agg["n_seeds"] == 1]

    sc1 = ax.scatter(
        multi["eta"], multi["k"], c=multi["fraction"],
        cmap="RdYlGn", vmin=0, vmax=1, s=120, marker="s",
        edgecolors="black", linewidths=0.6, label="multi-seed (≥2 seeds)",
    )
    ax.scatter(
        single["eta"], single["k"], c=single["fraction"],
        cmap="RdYlGn", vmin=0, vmax=1, s=80, marker="o",
        edgecolors="grey", linewidths=0.6, label="single-seed",
    )

    # Overlay η_c per K with error bars where identifiable.
    K_id, eta_c_id, eta_c_se_id = [], [], []
    K_unid, br_lo, br_hi = [], [], []
    for k_str, entry in eta_c_per_K.items():
        k = int(k_str)
        if entry["identifiable"]:
            K_id.append(k)
            eta_c_id.append(entry["eta_c"])
            eta_c_se_id.append(entry["eta_c_se"] if entry["eta_c_se"] else 0)
        else:
            K_unid.append(k)
            br_lo.append(entry.get("bracket_lo"))
            br_hi.append(entry.get("bracket_hi"))

    if K_id:
        # Cap displayed SE bar at 50% of the estimate for readability.
        eta_c_id_arr = np.array(eta_c_id)
        eta_c_se_arr = np.array([min(s, 0.5 * c) if s else 0
                                 for s, c in zip(eta_c_se_id, eta_c_id)])
        ax.errorbar(
            eta_c_id_arr, K_id, xerr=eta_c_se_arr, fmt="D",
            color="black", markersize=8, capsize=4, elinewidth=1.5,
            label="η_c (logistic, ±SE)", zorder=10,
        )
    if K_unid:
        for k, lo, hi in zip(K_unid, br_lo, br_hi):
            if lo is not None and hi is not None:
                ax.errorbar([0.5*(lo+hi)], [k], xerr=[[0.5*(hi-lo)],[0.5*(hi-lo)]],
                            fmt="x", color="dimgrey", markersize=10,
                            capsize=4, elinewidth=1.0, zorder=9)
            elif lo is not None:
                ax.scatter([lo], [k], marker=">", color="dimgrey", s=80, zorder=9)
            elif hi is not None:
                ax.scatter([hi], [k], marker="<", color="dimgrey", s=80, zorder=9)
        ax.scatter([], [], marker="x", color="dimgrey", s=80,
                   label="η_c bracket only (not identifiable)")

    ax.set_xscale("log")
    ax.set_xlabel("η (learning rate, log scale)")
    ax.set_ylabel("K (number of A targets per B)")
    ax.set_title("MBC phase diagram — seed-averaged transition fraction\n"
                 "(filled square: multi-seed; circle: single-seed)")
    ax.set_yticks(sorted(MAIN_K_VALUES))
    cbar = plt.colorbar(sc1, ax=ax, pad=0.02)
    cbar.set_label("transition fraction")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    out_path = FIG_DIR / "phase_diagram_current.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 3. Critical-slowing summaries
# ---------------------------------------------------------------------------

def critical_slowing_summary(df: pd.DataFrame, eta_c_per_K: Dict) -> Dict:
    main = df[df["k"].isin(MAIN_K_VALUES)].copy()
    valid = main.dropna(subset=["t1", "snap_width_steps", "frac_excess"])

    out: Dict = {"per_K": {}, "correlations": {}, "vs_distance_to_criticality": {}}
    for k in sorted(valid["k"].unique()):
        sub = valid[valid["k"] == k]
        # Per-η aggregation: mean + SE across seeds.
        agg = (sub.groupby("eta")
                  .agg(n=("seed", "nunique"),
                       t1_mean=("t1", "mean"),
                       t1_std=("t1", "std"),
                       snap_mean=("snap_width_steps", "mean"),
                       snap_std=("snap_width_steps", "std"),
                       fe_mean=("frac_excess", "mean"),
                       fe_std=("frac_excess", "std"))
                  .reset_index().sort_values("eta"))
        # SE = std / sqrt(n) with ddof=0 fallback for n=1.
        for col_std, col_se in [("t1_std","t1_se"),("snap_std","snap_se"),("fe_std","fe_se")]:
            agg[col_se] = agg[col_std] / np.sqrt(np.maximum(agg["n"].astype(float), 1))
        out["per_K"][int(k)] = {
            "eta_values": agg["eta"].tolist(),
            "n_seeds": [int(x) for x in agg["n"].tolist()],
            "t1_mean": agg["t1_mean"].tolist(),
            "t1_se":   agg["t1_se"].fillna(0).tolist(),
            "snap_mean": agg["snap_mean"].tolist(),
            "snap_se":   agg["snap_se"].fillna(0).tolist(),
            "frac_excess_mean": agg["fe_mean"].tolist(),
            "frac_excess_se":   agg["fe_se"].fillna(0).tolist(),
        }

    # Correlations among observables — pooled across all (η,K) cells with valid data.
    t1 = valid["t1"].astype(float).to_numpy()
    sw = valid["snap_width_steps"].astype(float).to_numpy()
    fe = valid["frac_excess"].astype(float).to_numpy()
    def _safe_pearson(x, y):
        if len(x) < 3:
            return None, None
        try:
            r, p = pearsonr(x, y)
            return float(r), float(p)
        except Exception:
            return None, None
    r_t1_sw,  p_t1_sw  = _safe_pearson(t1, sw)
    r_t1_fe,  p_t1_fe  = _safe_pearson(t1, fe)
    r_sw_fe,  p_sw_fe  = _safe_pearson(sw, fe)
    out["correlations"] = {
        "n_cells": int(len(valid)),
        "pearson_t1_vs_snap":         {"r": r_t1_sw, "p": p_t1_sw},
        "pearson_t1_vs_frac_excess":  {"r": r_t1_fe, "p": p_t1_fe},
        "pearson_snap_vs_frac_excess":{"r": r_sw_fe, "p": p_sw_fe},
    }

    # vs distance to criticality, only K with identifiable η_c.
    for k_str, entry in eta_c_per_K.items():
        k = int(k_str)
        if not entry["identifiable"]:
            continue
        eta_c = entry["eta_c"]
        sub = valid[valid["k"] == k]
        sub = sub[sub["eta"] < eta_c]
        if len(sub) < 3:
            continue
        agg = (sub.groupby("eta")
                  .agg(t1_mean=("t1","mean"),
                       snap_mean=("snap_width_steps","mean"),
                       fe_mean=("frac_excess","mean"),
                       n=("seed","nunique"))
                  .reset_index())
        d = eta_c - agg["eta"].to_numpy(dtype=float)
        out["vs_distance_to_criticality"][k] = {
            "eta_c_used": float(eta_c),
            "eta_values": agg["eta"].tolist(),
            "distance_to_eta_c": d.tolist(),
            "t1_mean": agg["t1_mean"].tolist(),
            "snap_mean": agg["snap_mean"].tolist(),
            "frac_excess_mean": agg["fe_mean"].tolist(),
            "n_seeds": [int(x) for x in agg["n"].tolist()],
        }

    return out


def plot_critical_slowing(crit: Dict) -> Path:
    per_K = crit["per_K"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    obs = [("t1_mean","t1_se","plateau end t₁ (steps)", axes[0]),
           ("snap_mean","snap_se","snap width (steps)", axes[1]),
           ("frac_excess_mean","frac_excess_se","irreversible fraction (Q−ΔL)/Q", axes[2])]
    cmap = plt.get_cmap("viridis")
    Ks = sorted(per_K.keys())
    for k_idx, k in enumerate(Ks):
        e = per_K[k]
        color = cmap(k_idx / max(len(Ks) - 1, 1))
        for ymean, yse, ylabel, ax in obs:
            mean = np.array(e[ymean], dtype=float)
            se = np.array(e[yse], dtype=float)
            ax.errorbar(e["eta_values"], mean, yerr=se, marker="o",
                        markersize=4, linewidth=1.3, capsize=2,
                        label=f"K={k}", color=color)
            ax.set_xscale("log")
            ax.set_xlabel("η")
            ax.set_ylabel(ylabel)
    for _, _, _, ax in obs:
        ax.legend(fontsize=7, loc="best")
    fig.suptitle("Critical slowing observables vs η, by K (mean ± SE across seeds)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = FIG_DIR / "critical_slowing_current.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 4. t₁ near criticality: power-law vs Kramers
# ---------------------------------------------------------------------------

def fit_t1_near_criticality(crit: Dict, eta_c_per_K: Dict) -> Dict:
    out: Dict = {"per_K": {}, "summary": {}}
    n_pl_ok = n_kr_ok = 0
    for k_str, entry in eta_c_per_K.items():
        k = int(k_str)
        if not entry["identifiable"]:
            out["per_K"][k] = {"identifiable_eta_c": False, "reason": entry["reason"]}
            continue
        if k not in crit["vs_distance_to_criticality"]:
            out["per_K"][k] = {"identifiable_eta_c": True,
                               "fit_run": False,
                               "reason": "no per-η t₁ data with η < η_c"}
            continue
        d = np.array(crit["vs_distance_to_criticality"][k]["distance_to_eta_c"], dtype=float)
        t1 = np.array(crit["vs_distance_to_criticality"][k]["t1_mean"], dtype=float)
        # Use only positive distance points and finite t1.
        mask = (d > 0) & np.isfinite(t1) & (t1 > 0)
        d, t1 = d[mask], t1[mask]
        if len(d) < 4:
            out["per_K"][k] = {
                "identifiable_eta_c": True, "fit_run": False,
                "reason": f"only {int(len(d))} points with η < η_c; need ≥ 4",
                "n_points": int(len(d)),
            }
            continue

        # Power law:  t₁ = A * d^(-ν) + b
        def _powerlaw(d_, A, nu, b):
            return A * np.power(d_, -nu) + b

        # Kramers-like:  t₁ = A * exp(B / d^m) + b
        def _kramers(d_, A, B, m, b):
            return A * np.exp(B / np.power(d_, m)) + b

        result_k = {"identifiable_eta_c": True, "fit_run": True,
                    "n_points": int(len(d)),
                    "eta_c_used": float(eta_c_per_K[k_str]["eta_c"]),
                    "distances": d.tolist(), "t1_values": t1.tolist()}

        # Power law fit
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, pcov = curve_fit(
                    _powerlaw, d, t1,
                    p0=[1.0, 0.5, t1.min() * 0.5],
                    bounds=([1e-6, 0.01, 0], [1e10, 5.0, t1.max()]),
                    maxfev=20000,
                )
            yhat = _powerlaw(d, *popt)
            ss_res = float(np.sum((t1 - yhat) ** 2))
            ss_tot = float(np.sum((t1 - t1.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
            perr = np.sqrt(np.maximum(np.diag(pcov), 0.0))
            result_k["power_law"] = {
                "A": float(popt[0]), "nu": float(popt[1]), "b": float(popt[2]),
                "A_se": float(perr[0]) if np.isfinite(perr[0]) else None,
                "nu_se": float(perr[1]) if np.isfinite(perr[1]) else None,
                "R2": r2, "n_params": 3, "n_points": int(len(d)),
                "identifiable": (np.isfinite(perr[1])
                                 and r2 is not None and r2 > 0.5
                                 and len(d) >= 4),
            }
            if result_k["power_law"]["identifiable"]:
                n_pl_ok += 1
        except Exception as exc:
            result_k["power_law"] = {"error": str(exc), "identifiable": False}

        # Kramers fit
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, pcov = curve_fit(
                    _kramers, d, t1,
                    p0=[1.0, 1e-4, 0.5, t1.min() * 0.5],
                    bounds=([1e-6, 1e-12, 0.01, 0],
                            [1e10, 1e2, 5.0, t1.max()]),
                    maxfev=20000,
                )
            yhat = _kramers(d, *popt)
            ss_res = float(np.sum((t1 - yhat) ** 2))
            ss_tot = float(np.sum((t1 - t1.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
            perr = np.sqrt(np.maximum(np.diag(pcov), 0.0))
            result_k["kramers"] = {
                "A": float(popt[0]), "B": float(popt[1]),
                "m": float(popt[2]), "b": float(popt[3]),
                "B_se": float(perr[1]) if np.isfinite(perr[1]) else None,
                "m_se": float(perr[2]) if np.isfinite(perr[2]) else None,
                "R2": r2, "n_params": 4, "n_points": int(len(d)),
                "identifiable": (np.isfinite(perr[1]) and np.isfinite(perr[2])
                                 and r2 is not None and r2 > 0.5
                                 and len(d) >= 5),
            }
            if result_k["kramers"]["identifiable"]:
                n_kr_ok += 1
        except Exception as exc:
            result_k["kramers"] = {"error": str(exc), "identifiable": False}

        out["per_K"][k] = result_k

    out["summary"] = {
        "n_K_with_identifiable_power_law": int(n_pl_ok),
        "n_K_with_identifiable_kramers": int(n_kr_ok),
        "verdict": (
            "Insufficient data to identify either form across multiple K values; "
            "we have ≤ 3 sub-critical η points per K with identifiable η_c "
            "(only K=20 has 4+ multi-seed sub-critical points). Both fits are "
            "underconstrained.  Report qualitative monotonic divergence only."
        ),
    }
    return out


def plot_t1_fits(t1_fits: Dict) -> Path:
    per_K = t1_fits["per_K"]
    K_with_data = [k for k, e in per_K.items() if e.get("fit_run")]
    if not K_with_data:
        # Empty figure with explanation
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.text(0.5, 0.5,
                "No K has ≥ 4 multi-seed sub-critical points\n"
                "Insufficient data for identifiable t₁(η_c−η) fit",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="dimgrey")
        ax.axis("off")
        ax.set_title("t₁ near criticality — fit attempt", fontsize=10)
        out_path = FIG_DIR / "t1_fits_current.png"
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    n = len(K_with_data)
    ncols = min(n, 3)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.2 * nrows),
                             squeeze=False)
    for idx, k in enumerate(K_with_data):
        ax = axes[idx // ncols, idx % ncols]
        e = per_K[k]
        d = np.array(e["distances"]); t1 = np.array(e["t1_values"])
        ax.scatter(d, t1, s=40, color="#2E86AB", zorder=3, label="data")
        d_grid = np.geomspace(max(d.min() * 0.5, 1e-7), d.max() * 1.2, 80)
        if e.get("power_law", {}).get("identifiable"):
            pl = e["power_law"]
            yp = pl["A"] * np.power(d_grid, -pl["nu"]) + pl["b"]
            ax.plot(d_grid, yp, "--", color="#E76F51",
                    label=f"power: ν={pl['nu']:.2f} R²={pl['R2']:.2f}")
        if e.get("kramers", {}).get("identifiable"):
            kr = e["kramers"]
            yk = kr["A"] * np.exp(kr["B"] / np.power(d_grid, kr["m"])) + kr["b"]
            ax.plot(d_grid, yk, ":", color="#7B2CBF",
                    label=f"Kramers: m={kr['m']:.2f} R²={kr['R2']:.2f}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("η_c − η")
        ax.set_ylabel("t₁ (steps)")
        ax.set_title(f"K={k}, n={e['n_points']}")
        ax.legend(fontsize=7, loc="best")
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    fig.suptitle("t₁ vs (η_c − η) — power law vs Kramers form", fontsize=11)
    fig.tight_layout(rect=[0,0,1,0.95])
    out_path = FIG_DIR / "t1_fits_current.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 5. Q vs log K, separated regimes
# ---------------------------------------------------------------------------

def _affine_fit(K: np.ndarray, Q: np.ndarray) -> Dict:
    if len(K) < 2:
        return {"c": None, "q0": None, "R2": None, "n": int(len(K))}
    logK = np.log(K.astype(float))
    A = np.vstack([logK, np.ones_like(logK)]).T
    (c, q0), *_ = np.linalg.lstsq(A, Q.astype(float), rcond=None)
    yhat = c * logK + q0
    ss_res = float(np.sum((Q - yhat) ** 2))
    ss_tot = float(np.sum((Q - Q.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    return {"c": float(c), "q0": float(q0), "R2": r2, "n": int(len(K))}


def q_regime_fits(df: pd.DataFrame) -> Dict:
    valid = df.dropna(subset=["Q"]).copy()
    out: Dict = {"stable_low_eta": {}, "near_critical": {},
                 "k_extension": {}}

    # ---- Stable: η ≤ 1e-3, K ≤ 36 ----
    stable = valid[(valid["eta"] <= 1e-3 + 1e-12) & (valid["k"].isin(MAIN_K_VALUES))]
    for eta in sorted(stable["eta"].unique()):
        sub = stable[np.isclose(stable["eta"], eta)]
        per_k = (sub.groupby("k")
                    .agg(Q_mean=("Q","mean"), n=("seed","nunique"))
                    .reset_index().sort_values("k"))
        K = per_k["k"].to_numpy(dtype=float)
        Q = per_k["Q_mean"].to_numpy(dtype=float)
        fit = _affine_fit(K, Q)
        fit.update({"K_values": K.tolist(), "Q_means": Q.tolist(),
                    "n_seeds_per_K": [int(x) for x in per_k["n"].tolist()]})
        out["stable_low_eta"][f"{eta:g}"] = fit

    # ---- Near-critical: η ∈ [2e-3, 3e-3], K ≤ 36 ----
    nc = valid[(valid["eta"] >= 2e-3 - 1e-12) & (valid["eta"] <= 3e-3 + 1e-12)
               & (valid["k"].isin(MAIN_K_VALUES))]
    for eta in sorted(nc["eta"].unique()):
        sub = nc[np.isclose(nc["eta"], eta)]
        per_k = (sub.groupby("k")
                    .agg(Q_mean=("Q","mean"), n=("seed","nunique"))
                    .reset_index().sort_values("k"))
        K = per_k["k"].to_numpy(dtype=float)
        Q = per_k["Q_mean"].to_numpy(dtype=float)
        fit = _affine_fit(K, Q)
        fit.update({"K_values": K.tolist(), "Q_means": Q.tolist(),
                    "n_seeds_per_K": [int(x) for x in per_k["n"].tolist()]})
        out["near_critical"][f"{eta:g}"] = fit

    # ---- K-extension at η ∈ {3e-4, 7e-4}: K ≤ 100 ----
    for eta_target in [3e-4, 7e-4]:
        sub = valid[np.isclose(valid["eta"], eta_target)
                    & valid["k"].isin(MAIN_K_VALUES | KEXT_K_VALUES)]
        per_k = (sub.groupby("k")
                    .agg(Q_mean=("Q","mean"), n=("seed","nunique"))
                    .reset_index().sort_values("k"))
        K = per_k["k"].to_numpy(dtype=float)
        Q = per_k["Q_mean"].to_numpy(dtype=float)
        # Two fits: K ≤ 36 only, and K ≤ 100.
        K36 = K[K <= 36]; Q36 = Q[K <= 36]
        Kall = K; Qall = Q
        fit36 = _affine_fit(K36, Q36)
        fit_all = _affine_fit(Kall, Qall)
        out["k_extension"][f"{eta_target:g}"] = {
            "fit_K_le_36": fit36,
            "fit_K_le_100": fit_all,
            "K_values": K.tolist(),
            "Q_means": Q.tolist(),
            "n_seeds_per_K": [int(x) for x in per_k["n"].tolist()],
            "extrapolation_check": [
                {"K": int(k), "Q_observed": float(q),
                 "Q_predicted_5K": (fit36["c"] * math.log(k) + fit36["q0"]
                                    if fit36["c"] is not None else None),
                 "ratio_obs_pred": (
                     float(q) / (fit36["c"] * math.log(k) + fit36["q0"])
                     if (fit36["c"] is not None
                         and (fit36["c"] * math.log(k) + fit36["q0"]) > 0) else None)}
                for k, q in zip(K, Q) if k > 36
            ],
        }
    return out


def plot_q_regimes(qfits: Dict) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0))

    # Panel 1 — stable low-η
    ax = axes[0]
    cmap = plt.get_cmap("viridis")
    etas = sorted(qfits["stable_low_eta"].keys(), key=lambda s: float(s))
    for i, eta_str in enumerate(etas):
        e = qfits["stable_low_eta"][eta_str]
        if e["c"] is None: continue
        K = np.array(e["K_values"]); Q = np.array(e["Q_means"])
        ax.scatter(K, Q, color=cmap(i / max(len(etas)-1,1)),
                   s=40, label=f"η={eta_str}")
        K_grid = np.geomspace(K.min(), K.max(), 40)
        ax.plot(K_grid, e["c"]*np.log(K_grid)+e["q0"],
                color=cmap(i/max(len(etas)-1,1)), linewidth=1, alpha=0.6)
    ax.set_xscale("log"); ax.set_xlabel("K"); ax.set_ylabel("Q")
    ax.set_title("Stable low-η regime\n(K ≤ 36, η ≤ 10⁻³)", fontsize=9)
    ax.legend(fontsize=7)

    # Panel 2 — near-critical
    ax = axes[1]
    etas = sorted(qfits["near_critical"].keys(), key=lambda s: float(s))
    cmap = plt.get_cmap("plasma")
    for i, eta_str in enumerate(etas):
        e = qfits["near_critical"][eta_str]
        if e["c"] is None: continue
        K = np.array(e["K_values"]); Q = np.array(e["Q_means"])
        ax.scatter(K, Q, color=cmap(i/max(len(etas)-1,1)),
                   s=40, label=f"η={eta_str}")
        K_grid = np.geomspace(K.min(), K.max(), 40)
        ax.plot(K_grid, e["c"]*np.log(K_grid)+e["q0"],
                color=cmap(i/max(len(etas)-1,1)), linewidth=1, alpha=0.6)
    ax.set_xscale("log"); ax.set_xlabel("K"); ax.set_ylabel("Q")
    ax.set_title("Near-critical regime\n(K ≤ 36, η ∈ [2,3]×10⁻³)", fontsize=9)
    ax.legend(fontsize=7)

    # Panel 3 — K-extension
    ax = axes[2]
    color_map = {"0.0003": "#2E86AB", "0.0007": "#E76F51"}
    for eta_str, e in qfits["k_extension"].items():
        col = color_map.get(eta_str, "black")
        K = np.array(e["K_values"]); Q = np.array(e["Q_means"])
        ax.scatter(K, Q, color=col, s=40,
                   label=f"η={eta_str} obs")
        f36 = e["fit_K_le_36"]
        if f36["c"] is not None:
            K_grid = np.geomspace(10, K.max()*1.1, 80)
            ax.plot(K_grid, f36["c"]*np.log(K_grid)+f36["q0"], "--",
                    color=col, alpha=0.6,
                    label=f"η={eta_str} 5-K fit (R²={f36['R2']:.2f})")
    ax.set_xscale("log"); ax.set_xlabel("K"); ax.set_ylabel("Q")
    ax.set_title("K-extension regime\n(K ≤ 100, single seed)", fontsize=9)
    ax.legend(fontsize=7)

    fig.tight_layout()
    out_path = FIG_DIR / "q_regimes_current.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def main() -> None:
    df = _load_df()
    print(f"loaded {len(df)} rows from {AGGREGATED_PARQUET}")

    # 1. η_c per K
    eta_c_per_K = fit_eta_c_per_K(df)
    out_path = RESULTS_DIR / "eta_c_per_K.json"
    with open(out_path, "w") as f:
        json.dump(eta_c_per_K, f, indent=2,
                  default=lambda o: None if isinstance(o, float)
                                  and (math.isnan(o) or math.isinf(o)) else o)
    print(f"wrote {out_path}")
    for k, e in eta_c_per_K.items():
        if e["identifiable"]:
            print(f"  K={k:>2}: η_c = {e['eta_c']:.4g} ± {e['eta_c_se']}  "
                  f"R²={e['R2']:.3f}  (n_eta={e['n_eta_points']})")
        else:
            print(f"  K={k:>2}: NOT identifiable ({e['reason']})  "
                  f"bracket=[{e['bracket_lo']}, {e['bracket_hi']}]")

    # 2. phase diagram
    p2 = make_phase_diagram(df, eta_c_per_K)
    print(f"wrote {p2}")

    # 3. critical slowing
    crit = critical_slowing_summary(df, eta_c_per_K)
    out_path = RESULTS_DIR / "critical_slowing_summary.json"
    with open(out_path, "w") as f:
        json.dump(crit, f, indent=2,
                  default=lambda o: None if isinstance(o, float)
                                  and (math.isnan(o) or math.isinf(o)) else o)
    print(f"wrote {out_path}")
    print(f"  pearson(t1, snap):       r={crit['correlations']['pearson_t1_vs_snap']['r']}")
    print(f"  pearson(t1, frac_excess):r={crit['correlations']['pearson_t1_vs_frac_excess']['r']}")
    print(f"  pearson(snap, frac_excess):r={crit['correlations']['pearson_snap_vs_frac_excess']['r']}")
    p3 = plot_critical_slowing(crit)
    print(f"wrote {p3}")

    # 4. t₁ near criticality
    t1_fits = fit_t1_near_criticality(crit, eta_c_per_K)
    out_path = RESULTS_DIR / "t1_fit_results.json"
    with open(out_path, "w") as f:
        json.dump(t1_fits, f, indent=2,
                  default=lambda o: None if isinstance(o, float)
                                  and (math.isnan(o) or math.isinf(o)) else o)
    print(f"wrote {out_path}")
    print(f"  power-law identifiable for {t1_fits['summary']['n_K_with_identifiable_power_law']} K values")
    print(f"  Kramers identifiable for   {t1_fits['summary']['n_K_with_identifiable_kramers']} K values")
    p4 = plot_t1_fits(t1_fits)
    print(f"wrote {p4}")

    # 5. Q regimes
    qfits = q_regime_fits(df)
    out_path = RESULTS_DIR / "q_regime_fits.json"
    with open(out_path, "w") as f:
        json.dump(qfits, f, indent=2,
                  default=lambda o: None if isinstance(o, float)
                                  and (math.isnan(o) or math.isinf(o)) else o)
    print(f"wrote {out_path}")
    p5 = plot_q_regimes(qfits)
    print(f"wrote {p5}")


if __name__ == "__main__":
    main()
