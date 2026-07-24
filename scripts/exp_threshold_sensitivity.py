#!/usr/bin/env python
"""
Experiment 6: τ Threshold Sensitivity Analysis

Reviewer concern: τ is defined as when loss drops below 50% of log K.
Is the scaling sensitive to this threshold?

Design: Pure re-analysis of existing data — no new training needed.
- Re-compute τ at thresholds: 0.3, 0.4, 0.5 (default), 0.6, 0.7 of log K
- For all existing K-sweep runs at η=1e-3
- Fit power law τ ∝ K^δ at each threshold
- Report δ ± CI for each threshold

Expected output: Table showing δ vs threshold_frac. If δ varies < 0.1
across thresholds, the scaling is robust.
"""

import sys
import json
import math
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.experiment_helpers import load_history, detect_tau

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ─────────────────────────────────────────────────────────────────
OUTPUT_DIR = "outputs"
FIG_DIR = Path(OUTPUT_DIR) / "paper_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# K values from the dense sweep at η=1e-3
K_VALUES = [3, 5, 7, 10, 13, 15, 17, 20, 25, 30, 36]
THRESHOLD_FRACS = [0.3, 0.4, 0.5, 0.6, 0.7]

# Experiment naming convention — try both common patterns
NAME_PATTERNS = [
    "landauer_dense_k{k}",
    "k{k}_lr1e-3",
    "landauer_k{k}_lr1e-3",
    "K{k}_eta1e-3",
]


def find_run(k):
    """Try multiple naming conventions to find an existing run."""
    for pattern in NAME_PATTERNS:
        name = pattern.format(k=k)
        h = load_history(name, OUTPUT_DIR)
        if h is not None:
            return name, h
    return None, None


def fit_power_law(k_arr, tau_arr, n_bootstrap=2000):
    """Fit τ = C * K^δ in log-log space with bootstrap CIs."""
    mask = np.isfinite(tau_arr) & (tau_arr > 0)
    k_arr = k_arr[mask]
    tau_arr = tau_arr[mask]

    if len(k_arr) < 2:
        return None, None, None

    log_k = np.log(k_arr)
    log_tau = np.log(tau_arr)

    # OLS fit
    A = np.vstack([log_k, np.ones(len(log_k))]).T
    result = np.linalg.lstsq(A, log_tau, rcond=None)
    delta, log_c = result[0]

    # Bootstrap for CI
    deltas = []
    n = len(log_k)
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        A_b = np.vstack([log_k[idx], np.ones(len(idx))]).T
        try:
            d_b, _ = np.linalg.lstsq(A_b, log_tau[idx], rcond=None)[0]
            deltas.append(d_b)
        except Exception:
            continue

    deltas = np.array(deltas)
    ci_lo = np.percentile(deltas, 2.5)
    ci_hi = np.percentile(deltas, 97.5)

    return delta, ci_lo, ci_hi


# ── Main analysis ──────────────────────────────────────────────────────────

print("=" * 70)
print("EXPERIMENT 6: τ THRESHOLD SENSITIVITY")
print("=" * 70)

# Load all available histories
available = {}
for k in K_VALUES:
    name, h = find_run(k)
    if h is not None:
        available[k] = (name, h)
        print(f"  Found K={k}: {name}")
    else:
        print(f"  Missing K={k}")

if len(available) < 3:
    print("\nERROR: Need at least 3 K values for power-law fit.")
    print("Available:", list(available.keys()))
    sys.exit(1)

k_arr = np.array(sorted(available.keys()), dtype=float)

# Compute τ at each threshold
results = {}
print(f"\n{'Threshold':<12} {'δ':>8} {'95% CI':>16} {'N_points':>10}")
print("-" * 50)

for frac in THRESHOLD_FRACS:
    tau_vals = []
    k_used = []
    for k in sorted(available.keys()):
        name, h = available[k]
        log_k = math.log(k)
        tau = detect_tau(h, log_k, threshold_frac=frac)
        if tau is not None:
            tau_vals.append(tau)
            k_used.append(k)

    k_used = np.array(k_used, dtype=float)
    tau_vals = np.array(tau_vals, dtype=float)

    delta, ci_lo, ci_hi = fit_power_law(k_used, tau_vals)

    results[frac] = {
        "threshold_frac": frac,
        "delta": float(delta) if delta is not None else None,
        "ci_lo": float(ci_lo) if ci_lo is not None else None,
        "ci_hi": float(ci_hi) if ci_hi is not None else None,
        "n_points": len(k_used),
        "k_values": k_used.tolist(),
        "tau_values": tau_vals.tolist(),
    }

    if delta is not None:
        print(f"  {frac:<12.1f} {delta:>8.3f} [{ci_lo:.3f}, {ci_hi:.3f}] {len(k_used):>10}")
    else:
        print(f"  {frac:<12.1f} {'N/A':>8} {'N/A':>16} {len(k_used):>10}")

# Check robustness
valid_deltas = [r["delta"] for r in results.values() if r["delta"] is not None]
if len(valid_deltas) >= 2:
    delta_range = max(valid_deltas) - min(valid_deltas)
    delta_mean = np.mean(valid_deltas)
    delta_std = np.std(valid_deltas)
    print(f"\nδ range across thresholds: {delta_range:.3f}")
    print(f"δ mean ± std: {delta_mean:.3f} ± {delta_std:.3f}")
    if delta_range < 0.1:
        print("→ ROBUST: δ varies < 0.1 across thresholds")
    elif delta_range < 0.2:
        print("→ MODERATELY ROBUST: δ varies < 0.2 across thresholds")
    else:
        print("→ SENSITIVE: δ varies ≥ 0.2 — threshold choice matters")


# ── Figure: δ vs threshold ─────────────────────────────────────────────────

fracs = [r["threshold_frac"] for r in results.values() if r["delta"] is not None]
deltas = [r["delta"] for r in results.values() if r["delta"] is not None]
ci_los = [r["ci_lo"] for r in results.values() if r["delta"] is not None]
ci_his = [r["ci_hi"] for r in results.values() if r["delta"] is not None]

if deltas:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Panel A: δ vs threshold_frac
    ax = axes[0]
    ax.errorbar(fracs, deltas,
                yerr=[np.array(deltas) - np.array(ci_los),
                      np.array(ci_his) - np.array(deltas)],
                fmt="o-", color="#2C3E50", capsize=5, linewidth=1.5, markersize=6)
    ax.axhline(np.mean(deltas), color="gray", linestyle="--", alpha=0.5,
               label=f"mean δ = {np.mean(deltas):.2f}")
    ax.set_xlabel("Threshold fraction (of log K)", fontsize=10)
    ax.set_ylabel("Power-law exponent δ", fontsize=10)
    ax.set_title("Exponent sensitivity to threshold", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

    # Panel B: τ vs K at each threshold (log-log)
    ax = axes[1]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(THRESHOLD_FRACS)))
    for i, frac in enumerate(THRESHOLD_FRACS):
        r = results[frac]
        if r["delta"] is not None:
            ax.loglog(r["k_values"], r["tau_values"], "o-",
                      color=colors[i], label=f"frac={frac:.1f} (δ={r['delta']:.2f})",
                      markersize=4, linewidth=1.2)
    ax.set_xlabel("K", fontsize=10)
    ax.set_ylabel("τ (steps)", fontsize=10)
    ax.set_title("τ(K) at different thresholds", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, loc="upper left")
    ax.tick_params(labelsize=8)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(FIG_DIR / f"fig_threshold_sensitivity.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved: {FIG_DIR}/fig_threshold_sensitivity")

# ── Save summary JSON ──────────────────────────────────────────────────────

summary = {
    "experiment": "threshold_sensitivity",
    "description": "τ threshold sensitivity analysis (Exp 6)",
    "thresholds": results,
    "robustness": {
        "delta_range": float(delta_range) if len(valid_deltas) >= 2 else None,
        "delta_mean": float(delta_mean) if len(valid_deltas) >= 2 else None,
        "delta_std": float(delta_std) if len(valid_deltas) >= 2 else None,
    },
}

summary_path = Path(OUTPUT_DIR) / "threshold_sensitivity_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary saved: {summary_path}")
