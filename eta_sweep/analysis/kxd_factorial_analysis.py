#!/usr/bin/env python
"""
K × D factorial analysis at the 30%-of-log-K threshold.

Four-part analysis per user instructions:
  1. Per-K τ vs D fits with bootstrap CI (4 D values per K, 7 K values)
  2. Bimodality test (Wilcoxon: low-K {5,10,15} vs high-K {20,25,30,36})
  3. Per-D τ vs K fits with bootstrap CI (test K-independence at multi-seed)
  4. Global pooled τ vs D fit with cluster-bootstrap by K

Reports:
  - Per-K exponents, per-D exponents, global exponent
  - Decision: clean two-regime / no regime / messy intermediate
  - Output JSON for downstream paper text update

Plus loads existing multi-seed cells:
  - K=5 D=5000 (multiseed_d_replication)
  - K=10 D=10000 (v_tracking)
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ETA_SWEEP_ROOT = Path(__file__).resolve().parent.parent
RESULTS = ETA_SWEEP_ROOT / "results"


def load_kxd_factorial() -> list[dict]:
    """Load all 78 K×D factorial cells."""
    rows = []
    for d_dir in (RESULTS / "kxd_factorial").glob("D*"):
        D = int(d_dir.name[1:])
        for cell_dir in d_dir.glob("eta_0.001_K_*_seed_*"):
            status_f = cell_dir / "status.json"
            if not status_f.exists():
                continue
            s = json.loads(status_f.read_text())
            if s["status"] != "transitioned" or s["transition_detected_step"] is None:
                continue
            rows.append({
                "K": s["k"], "D": D, "seed": s["seed"],
                "tau": s["transition_detected_step"],
            })
    return rows


def load_existing_multiseed_cells() -> list[dict]:
    """Cells we already have at multi-seed at the 30% threshold."""
    rows = []
    # K=5 D=5000 from multiseed_d_replication (3 seeds)
    for seed in (0, 1, 2):
        f = RESULTS / "multiseed_d_replication" / f"eta_0.001_K_5_seed_{seed}" / "status.json"
        if f.exists():
            s = json.loads(f.read_text())
            if s["status"] == "transitioned":
                rows.append({"K": 5, "D": 5000, "seed": seed, "tau": s["transition_detected_step"]})
    # K=10 D=10000 from v_tracking (3 seeds)
    for seed in (0, 1, 2):
        f = RESULTS / "v_tracking" / f"eta_0.001_K_10_seed_{seed}" / "status.json"
        if f.exists():
            s = json.loads(f.read_text())
            if s["status"] == "transitioned":
                rows.append({"K": 10, "D": 10000, "seed": seed, "tau": s["transition_detected_step"]})
    return rows


def bootstrap_loglog_fit(logD, logT, B=5000, seed=42):
    """Bootstrap CI on log-log slope."""
    if len(logD) < 3:
        return {"slope": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n": len(logD)}
    rng = np.random.default_rng(seed)
    n = len(logD)
    slopes = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        try:
            s, _ = np.polyfit(logD[idx], logT[idx], 1)
            slopes.append(s)
        except Exception:
            pass
    slopes = np.array(slopes)
    point, _ = np.polyfit(logD, logT, 1)
    return {
        "slope": float(point),
        "ci_lo": float(np.percentile(slopes, 2.5)),
        "ci_hi": float(np.percentile(slopes, 97.5)),
        "n": int(n),
    }


def cluster_bootstrap_by_K(rows, B=5000, seed=42):
    """Cluster bootstrap: resample K values with replacement, then all (seed, D) within."""
    Ks = sorted({r["K"] for r in rows})
    by_K = defaultdict(list)
    for r in rows:
        by_K[r["K"]].append(r)
    rng = np.random.default_rng(seed)
    slopes = []
    for _ in range(B):
        idx = rng.integers(0, len(Ks), len(Ks))
        sampled = []
        for i in idx:
            sampled.extend(by_K[Ks[i]])
        logD = np.array([np.log10(r["D"]) for r in sampled])
        logT = np.array([np.log10(r["tau"]) for r in sampled])
        try:
            s, _ = np.polyfit(logD, logT, 1)
            slopes.append(s)
        except Exception:
            pass
    slopes = np.array(slopes)
    logD = np.array([np.log10(r["D"]) for r in rows])
    logT = np.array([np.log10(r["tau"]) for r in rows])
    point, _ = np.polyfit(logD, logT, 1)
    return {
        "slope": float(point),
        "ci_lo": float(np.percentile(slopes, 2.5)),
        "ci_hi": float(np.percentile(slopes, 97.5)),
        "n": len(rows),
        "n_K_clusters": len(Ks),
    }


def main():
    rows = load_kxd_factorial() + load_existing_multiseed_cells()
    print(f"Loaded {len(rows)} cells across {len({r['K'] for r in rows})} K values, "
          f"{len({r['D'] for r in rows})} D values")
    Ks = sorted({r["K"] for r in rows})
    Ds = sorted({r["D"] for r in rows})
    print(f"K values: {Ks}")
    print(f"D values: {Ds}\n")

    # ---- Per-K fits: τ vs D at fixed K ----
    print("=== (1) Per-K τ vs D fits ===")
    per_K_fits = {}
    for K in Ks:
        sub = [r for r in rows if r["K"] == K]
        logD = np.array([np.log10(r["D"]) for r in sub])
        logT = np.array([np.log10(r["tau"]) for r in sub])
        fit = bootstrap_loglog_fit(logD, logT)
        per_K_fits[K] = fit
        print(f"  K={K:2d}: δ={fit['slope']:.3f}  CI=[{fit['ci_lo']:.3f}, {fit['ci_hi']:.3f}]  n={fit['n']}")

    # ---- Bimodality test ----
    print("\n=== (2) Bimodality test (Wilcoxon, low-K vs high-K) ===")
    low_K = [5, 10, 15]
    high_K = [20, 25, 30, 36]
    low_slopes = [per_K_fits[K]["slope"] for K in low_K if K in per_K_fits]
    high_slopes = [per_K_fits[K]["slope"] for K in high_K if K in per_K_fits]
    print(f"  Low-K  ({low_K}): slopes = {[f'{s:.3f}' for s in low_slopes]}")
    print(f"  High-K ({high_K}): slopes = {[f'{s:.3f}' for s in high_slopes]}")
    print(f"  Low-K mean:  {np.mean(low_slopes):.3f} ± {np.std(low_slopes):.3f}")
    print(f"  High-K mean: {np.mean(high_slopes):.3f} ± {np.std(high_slopes):.3f}")
    try:
        from scipy.stats import mannwhitneyu, ranksums
        u_stat, u_p = mannwhitneyu(low_slopes, high_slopes, alternative="two-sided")
        rs_stat, rs_p = ranksums(low_slopes, high_slopes)
        print(f"  Mann-Whitney U: stat={u_stat:.2f}, p={u_p:.4f}")
        print(f"  Wilcoxon rank-sum: stat={rs_stat:.2f}, p={rs_p:.4f}")
    except ImportError:
        u_p = rs_p = float("nan")
        print("  scipy not available; skipping formal test")

    # ---- Per-D fits: τ vs K at fixed D ----
    print("\n=== (3) Per-D τ vs K fits (test K-independence at multi-seed) ===")
    per_D_fits = {}
    for D in Ds:
        sub = [r for r in rows if r["D"] == D]
        logK = np.array([np.log10(r["K"]) for r in sub])
        logT = np.array([np.log10(r["tau"]) for r in sub])
        fit = bootstrap_loglog_fit(logK, logT)
        per_D_fits[D] = fit
        ind_test = "K-INDEPENDENT" if fit["ci_lo"] < 0 < fit["ci_hi"] else "K-DEPENDENT"
        print(f"  D={D:5d}: τ vs K slope={fit['slope']:.3f}  CI=[{fit['ci_lo']:.3f}, {fit['ci_hi']:.3f}]  {ind_test}")

    # ---- Global pooled fit ----
    print("\n=== (4) Global pooled τ vs D fit (cluster bootstrap by K) ===")
    global_fit = cluster_bootstrap_by_K(rows)
    print(f"  Pooled δ={global_fit['slope']:.3f}  CI=[{global_fit['ci_lo']:.3f}, {global_fit['ci_hi']:.3f}]  "
          f"n={global_fit['n']}  K-clusters={global_fit['n_K_clusters']}")

    # ---- Decision ----
    print("\n=== Decision ===")
    # Two-regime: low-K cluster near 1.0, high-K cluster near 1.4
    low_mean = float(np.mean(low_slopes))
    high_mean = float(np.mean(high_slopes))
    gap = high_mean - low_mean

    # Criteria for clean two-regime:
    # (a) High-K mean significantly above low-K mean (Wilcoxon p < 0.10 with n=3,4)
    # (b) Low-K cluster within [0.8, 1.2] (~linear)
    # (c) High-K cluster above 1.2
    is_clean_two_regime = (
        rs_p < 0.10
        and 0.8 <= low_mean <= 1.2
        and high_mean >= 1.2
        and gap >= 0.2
    )

    # Criteria for "no regime" (everything similar):
    is_no_regime = (
        all(abs(s - np.mean([s2 for s2 in low_slopes + high_slopes])) < 0.15
            for s in low_slopes + high_slopes)
        and rs_p > 0.30
    )

    if is_clean_two_regime:
        decision = "CLEAN_TWO_REGIME"
    elif is_no_regime:
        decision = "NO_REGIME"
    else:
        decision = "MESSY_INTERMEDIATE"

    print(f"  low-K mean = {low_mean:.3f}, high-K mean = {high_mean:.3f}, gap = {gap:.3f}")
    print(f"  Wilcoxon p = {rs_p:.4f}")
    print(f"  → {decision}")

    # ---- Save full analysis ----
    out = {
        "per_K_fits": {str(K): v for K, v in per_K_fits.items()},
        "per_D_fits": {str(D): v for D, v in per_D_fits.items()},
        "global_fit": global_fit,
        "bimodality": {
            "low_K_slopes": low_slopes,
            "high_K_slopes": high_slopes,
            "low_K_mean": low_mean,
            "high_K_mean": high_mean,
            "gap": gap,
            "wilcoxon_p": float(rs_p),
            "mannwhitney_p": float(u_p),
        },
        "decision": decision,
        "n_cells": len(rows),
    }
    out_path = RESULTS / "kxd_factorial_analysis.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")
    return out


if __name__ == "__main__":
    main()
