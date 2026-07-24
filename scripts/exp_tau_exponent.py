#!/usr/bin/env python
"""
Experiment 2: τ Exponent Reconciliation

Reviewer concern: Table 1 reports δ=1.31±0.14 for Transformers, but Table 2
reports δ=1.70±0.09. Which is right?

Design: Dense K-sweep at η=1e-3 with finer K grid.
- K ∈ {3, 5, 7, 10, 13, 15, 17, 20, 25, 30, 36}
- Check existing runs first, only train missing ones
- Fit power law on full range vs restricted range {10, 20, 36}
- Report δ and 95% CI for each

Expected output: Single δ estimate with CI on full K range, plus
demonstration that restricted range gives inflated δ.
"""

import sys
import json
import math
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.experiment_helpers import (
    make_config, run_parallel, run_exists, load_history, detect_tau,
)
from omegaconf import OmegaConf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ─────────────────────────────────────────────────────────────────
OUTPUT_DIR = "outputs"
FIG_DIR = Path(OUTPUT_DIR) / "paper_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

K_VALUES = [3, 5, 7, 10, 13, 15, 17, 20, 25, 30, 36]
K_RESTRICTED = [10, 20, 36]
SEED = 42
LR = 1e-3
MAX_STEPS = 50000

NAME_PATTERNS = [
    "landauer_dense_k{k}",
    "k{k}_lr1e-3",
    "exponent_K{k}",
]
NEW_NAME = "exponent_K{k}"


def find_existing_run(k):
    for pattern in NAME_PATTERNS:
        name = pattern.format(k=k)
        if run_exists(name, min_steps=MAX_STEPS * 0.5, output_dir=OUTPUT_DIR):
            return name
    return None


def fit_power_law(k_arr, tau_arr, n_bootstrap=2000):
    mask = np.isfinite(tau_arr) & (tau_arr > 0)
    k_arr, tau_arr = k_arr[mask], tau_arr[mask]
    if len(k_arr) < 2:
        return None, None, None
    log_k, log_tau = np.log(k_arr), np.log(tau_arr)
    A = np.vstack([log_k, np.ones(len(log_k))]).T
    delta, _ = np.linalg.lstsq(A, log_tau, rcond=None)[0]
    rng = np.random.RandomState(42)
    n = len(log_k)
    deltas = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        try:
            d_b, _ = np.linalg.lstsq(
                np.vstack([log_k[idx], np.ones(len(idx))]).T,
                log_tau[idx], rcond=None)[0]
            deltas.append(d_b)
        except Exception:
            continue
    deltas = np.array(deltas)
    return delta, np.percentile(deltas, 2.5), np.percentile(deltas, 97.5)


def main():
    print("=" * 70)
    print("EXPERIMENT 2: τ EXPONENT RECONCILIATION")
    print("=" * 70)

    run_names = {}
    jobs = []

    for k in K_VALUES:
        existing = find_existing_run(k)
        if existing:
            run_names[k] = existing
            print(f"  FOUND K={k}: {existing}")
        else:
            name = NEW_NAME.format(k=k)
            run_names[k] = name
            print(f"  MISSING K={k}: will train as {name}")
            cfg = make_config(
                experiment_name=name, k=k, lr=LR, seed=SEED,
                max_steps=MAX_STEPS, early_stop_frac=0.01,
            )
            jobs.append({
                "cfg_dict": OmegaConf.to_container(cfg, resolve=True),
                "mapping_path": None, "output_dir": OUTPUT_DIR, "name": name,
            })

    if jobs:
        print(f"\n  Training {len(jobs)} missing K values...")
        run_parallel(jobs, max_workers=6, label="exponent-sweep")
    else:
        print("\n  All K values already have runs!")

    # ── Collect τ values ───────────────────────────────────────────────────
    print(f"\n{'K':<6} {'Run name':<25} {'τ':>8} {'log K':>8}")
    print("-" * 52)

    tau_data = {}
    for k in K_VALUES:
        name = run_names[k]
        h = load_history(name, OUTPUT_DIR)
        if h is None:
            print(f"  {k:<6} {name:<25} {'MISS':>8}")
            continue
        log_k = math.log(k)
        tau = detect_tau(h, log_k)
        if tau is not None:
            tau_data[k] = tau
        tau_str = str(tau) if tau else "N/A"
        print(f"  {k:<6} {name:<25} {tau_str:>8} {log_k:>8.3f}")

    # ── Power-law fits ─────────────────────────────────────────────────────
    print(f"\n{'Range':<20} {'δ':>8} {'95% CI':>18} {'n':>4}")
    print("-" * 54)

    fit_results = {}

    ks_full = sorted([k for k in K_VALUES if k in tau_data])
    if len(ks_full) >= 2:
        k_arr = np.array(ks_full, dtype=float)
        tau_arr = np.array([tau_data[k] for k in ks_full], dtype=float)
        delta, ci_lo, ci_hi = fit_power_law(k_arr, tau_arr)
        fit_results["full"] = {"delta": delta, "ci_lo": ci_lo, "ci_hi": ci_hi,
                               "k_values": ks_full, "n": len(ks_full)}
        print(f"  {'Full K range':<20} {delta:>8.3f} [{ci_lo:.3f}, {ci_hi:.3f}] {len(ks_full):>4}")

    ks_rest = sorted([k for k in K_RESTRICTED if k in tau_data])
    if len(ks_rest) >= 2:
        k_arr = np.array(ks_rest, dtype=float)
        tau_arr = np.array([tau_data[k] for k in ks_rest], dtype=float)
        delta_r, ci_lo_r, ci_hi_r = fit_power_law(k_arr, tau_arr)
        fit_results["restricted"] = {"delta": delta_r, "ci_lo": ci_lo_r, "ci_hi": ci_hi_r,
                                     "k_values": ks_rest, "n": len(ks_rest)}
        print(f"  {'K∈{10,20,36}':<20} {delta_r:>8.3f} [{ci_lo_r:.3f}, {ci_hi_r:.3f}] {len(ks_rest):>4}")

    if "full" in fit_results and "restricted" in fit_results:
        d_full = fit_results["full"]["delta"]
        d_rest = fit_results["restricted"]["delta"]
        print(f"\n  Full-range δ = {d_full:.3f}, Restricted δ = {d_rest:.3f}")
        print(f"  Difference: {abs(d_full - d_rest):.3f}")
        if d_rest > d_full:
            print(f"  → Restricted range inflates δ by {d_rest - d_full:.3f}")

    # ── Figure ─────────────────────────────────────────────────────────────
    if tau_data:
        fig, ax = plt.subplots(figsize=(6, 4))
        ks = sorted(tau_data.keys())
        k_arr = np.array(ks, dtype=float)
        tau_arr = np.array([tau_data[k] for k in ks], dtype=float)
        ax.loglog(k_arr, tau_arr, "ko", markersize=6, label="Data", zorder=3)

        if "full" in fit_results and fit_results["full"]["delta"] is not None:
            d = fit_results["full"]["delta"]
            k_fit = np.linspace(k_arr.min() * 0.8, k_arr.max() * 1.2, 100)
            log_k = np.log(k_arr)
            log_tau = np.log(tau_arr)
            A = np.vstack([log_k, np.ones(len(log_k))]).T
            _, log_c = np.linalg.lstsq(A, log_tau, rcond=None)[0]
            ax.loglog(k_fit, np.exp(log_c) * k_fit ** d, "--",
                      color="#3498DB", linewidth=1.5, label=f"Full range: δ={d:.2f}")

        if "restricted" in fit_results and fit_results["restricted"]["delta"] is not None:
            d_r = fit_results["restricted"]["delta"]
            ks_r = np.array(fit_results["restricted"]["k_values"], dtype=float)
            taus_r = np.array([tau_data[k] for k in fit_results["restricted"]["k_values"]], dtype=float)
            log_k_r, log_tau_r = np.log(ks_r), np.log(taus_r)
            A_r = np.vstack([log_k_r, np.ones(len(log_k_r))]).T
            _, log_c_r = np.linalg.lstsq(A_r, log_tau_r, rcond=None)[0]
            ax.loglog(k_fit, np.exp(log_c_r) * k_fit ** d_r, ":",
                      color="#E74C3C", linewidth=1.5, label=f"K∈{{10,20,36}}: δ={d_r:.2f}")
            ax.loglog(ks_r, taus_r, "rs", markersize=8, markerfacecolor="none",
                      markeredgewidth=1.5, zorder=4, label="Table 2 K values")

        ax.set_xlabel("K (ambiguity)", fontsize=10)
        ax.set_ylabel("τ (transition time, steps)", fontsize=10)
        ax.set_title("τ exponent: full vs restricted K range", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)
        plt.tight_layout()
        for ext in ("pdf", "png"):
            plt.savefig(FIG_DIR / f"fig_tau_exponent.{ext}", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"\nFigure saved: {FIG_DIR}/fig_tau_exponent")

    # ── Save summary ───────────────────────────────────────────────────────
    summary = {
        "experiment": "tau_exponent",
        "description": "τ exponent reconciliation (Exp 2)",
        "tau_data": {str(k): v for k, v in tau_data.items()},
        "run_names": {str(k): v for k, v in run_names.items()},
        "power_law_fits": fit_results,
    }
    summary_path = Path(OUTPUT_DIR) / "tau_exponent_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
