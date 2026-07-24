#!/usr/bin/env python
"""
Experiment 8: Extended η*(K) Boundary with More K Values

Reviewer concern: Phase boundary η*(K) ∝ K^{-0.83} uses only 4 K values
(< 1 decade). Need more points.

Design: For each new K, sweep η to find the critical learning rate.
- New K values: {7, 13, 25, 50} (filling gaps in existing {10, 20, 36})
- For each K: η sweep over {3e-4, 5e-4, 1e-3, 2e-3, 3e-3, 5e-3, 1e-2}
- Max_steps=50000 with early stopping
- Total: 4 × 7 = 28 runs

η*(K) = largest η that still converges.

Expected output: Updated η*(K) with 7+ K values spanning > 1 decade.
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

NEW_K_VALUES = [7, 13, 25, 50]
EXISTING_BOUNDARY = {10: 3e-3, 20: 1e-3, 36: 5e-4}
ETA_SWEEP = [3e-4, 5e-4, 1e-3, 2e-3, 3e-3, 5e-3, 1e-2]
SEED = 42
MAX_STEPS = 50000


def eta_str(eta):
    return f"{eta:.0e}".replace("+", "").replace("-0", "-")


def fit_power_law(k_arr, y_arr, n_bootstrap=2000):
    mask = np.isfinite(y_arr) & (y_arr > 0)
    k_arr, y_arr = k_arr[mask], y_arr[mask]
    if len(k_arr) < 2:
        return None, None, None
    log_k, log_y = np.log(k_arr), np.log(y_arr)
    A = np.vstack([log_k, np.ones(len(log_k))]).T
    alpha, _ = np.linalg.lstsq(A, log_y, rcond=None)[0]
    rng = np.random.RandomState(42)
    n = len(log_k)
    alphas = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        try:
            a_b, _ = np.linalg.lstsq(
                np.vstack([log_k[idx], np.ones(len(idx))]).T,
                log_y[idx], rcond=None)[0]
            alphas.append(a_b)
        except Exception:
            continue
    alphas = np.array(alphas)
    return alpha, np.percentile(alphas, 2.5), np.percentile(alphas, 97.5)


def main():
    print("=" * 70)
    print("EXPERIMENT 8: EXTENDED η*(K) BOUNDARY")
    print("=" * 70)

    jobs = []
    all_configs = []

    for k in NEW_K_VALUES:
        for eta in ETA_SWEEP:
            name = f"boundary_K{k}_eta{eta_str(eta)}"
            all_configs.append((k, eta, name))
            if run_exists(name, min_steps=MAX_STEPS * 0.3, output_dir=OUTPUT_DIR):
                print(f"  SKIP {name}")
                continue
            print(f"  QUEUE {name}")
            cfg = make_config(
                experiment_name=name, k=k, lr=eta, seed=SEED,
                max_steps=MAX_STEPS, early_stop_frac=0.01,
            )
            jobs.append({
                "cfg_dict": OmegaConf.to_container(cfg, resolve=True),
                "mapping_path": None, "output_dir": OUTPUT_DIR, "name": name,
            })

    if jobs:
        run_parallel(jobs, max_workers=6, label="boundary-extension")
    else:
        print("  All runs already exist!")

    # ── Analyze ────────────────────────────────────────────────────────────
    print(f"\n{'K':<6} {'η':<12} {'τ':>8} {'Converged':>10}")
    print("-" * 40)

    boundary_results = {}
    for k, eta, name in all_configs:
        h = load_history(name, OUTPUT_DIR)
        if h is None:
            continue
        tau = detect_tau(h, math.log(k))
        converged = tau is not None
        if k not in boundary_results:
            boundary_results[k] = {}
        boundary_results[k][eta] = {"tau": tau, "converged": converged, "name": name}
        print(f"  {k:<6} {eta:<12.1e} {str(tau) if tau else 'N/C':>8} {'Yes' if converged else 'No':>10}")

    # Find η*(K)
    print(f"\n{'K':<6} {'η*':>12} {'τ at η*':>10}")
    print("-" * 32)

    eta_star = {}
    for k in sorted(boundary_results.keys()):
        converged_etas = sorted([
            eta for eta, r in boundary_results[k].items() if r["converged"]
        ], reverse=True)
        if converged_etas:
            best_eta = converged_etas[0]
            eta_star[k] = best_eta
            print(f"  {k:<6} {best_eta:>12.1e} {boundary_results[k][best_eta]['tau']:>10}")
        else:
            print(f"  {k:<6} {'N/A':>12}")

    all_boundary = {**EXISTING_BOUNDARY, **eta_star}

    # ── Fit ────────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("PHASE BOUNDARY FIT")
    print(f"{'='*50}")

    ks = sorted(all_boundary.keys())
    k_arr = np.array(ks, dtype=float)
    eta_arr = np.array([all_boundary[k] for k in ks], dtype=float)

    orig_ks = sorted(EXISTING_BOUNDARY.keys())
    alpha_orig = None
    if len(orig_ks) >= 2:
        k_orig = np.array(orig_ks, dtype=float)
        eta_orig = np.array([EXISTING_BOUNDARY[k] for k in orig_ks], dtype=float)
        alpha_orig, ci_lo_orig, ci_hi_orig = fit_power_law(k_orig, eta_orig)
        print(f"  Original ({len(orig_ks)} points): α = {alpha_orig:.3f} "
              f"[{ci_lo_orig:.3f}, {ci_hi_orig:.3f}]")

    alpha, ci_lo, ci_hi = fit_power_law(k_arr, eta_arr)
    if alpha is not None:
        print(f"  Extended ({len(ks)} points):  α = {alpha:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")
        if alpha_orig is not None:
            orig_w = ci_hi_orig - ci_lo_orig
            new_w = ci_hi - ci_lo
            print(f"  CI width improvement: {1 - new_w / orig_w:.1%}")

    # ── Figure ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))

    if orig_ks:
        k_o = np.array(sorted(EXISTING_BOUNDARY.keys()), dtype=float)
        e_o = np.array([EXISTING_BOUNDARY[k] for k in sorted(EXISTING_BOUNDARY.keys())], dtype=float)
        ax.loglog(k_o, e_o, "rs", markersize=8, label="Original data", zorder=3)

    new_ks = sorted(eta_star.keys())
    if new_ks:
        k_n = np.array(new_ks, dtype=float)
        e_n = np.array([eta_star[k] for k in new_ks], dtype=float)
        ax.loglog(k_n, e_n, "bo", markersize=8, label="New data", zorder=3)

    if alpha is not None:
        k_fit = np.linspace(min(ks) * 0.7, max(ks) * 1.3, 100)
        log_k = np.log(k_arr)
        log_eta = np.log(eta_arr)
        A = np.vstack([log_k, np.ones(len(log_k))]).T
        _, log_c = np.linalg.lstsq(A, log_eta, rcond=None)[0]
        ax.loglog(k_fit, np.exp(log_c) * k_fit ** alpha, "k--", linewidth=1.5,
                  label=f"η* ∝ K^{{{alpha:.2f}}} (n={len(ks)})")

    ax.set_xlabel("K (ambiguity)", fontsize=10)
    ax.set_ylabel("η* (critical learning rate)", fontsize=10)
    ax.set_title("Phase boundary η*(K)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    plt.tight_layout()

    for ext in ("pdf", "png"):
        plt.savefig(FIG_DIR / f"fig_boundary_extension.{ext}", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved: {FIG_DIR}/fig_boundary_extension")

    # ── Save summary ───────────────────────────────────────────────────────
    summary = {
        "experiment": "boundary_extension",
        "description": "Extended η*(K) phase boundary (Exp 8)",
        "existing_boundary": {str(k): v for k, v in EXISTING_BOUNDARY.items()},
        "new_eta_star": {str(k): v for k, v in eta_star.items()},
        "combined_boundary": {str(k): v for k, v in all_boundary.items()},
        "fit": {
            "alpha": float(alpha) if alpha is not None else None,
            "ci_lo": float(ci_lo) if ci_lo is not None else None,
            "ci_hi": float(ci_hi) if ci_hi is not None else None,
            "n_points": len(ks),
        },
        "per_k_results": {
            str(k): {str(eta): v for eta, v in kv.items()}
            for k, kv in boundary_results.items()
        },
    }
    summary_path = Path(OUTPUT_DIR) / "boundary_extension_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
