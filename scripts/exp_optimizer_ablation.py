#!/usr/bin/env python
"""
Experiment 5: Optimizer Ablation

Reviewer concern: All results use AdamW. τ scaling might be optimizer-specific.

Design:
- SGD + momentum (0.9): K ∈ {5, 10, 20, 36}, η=1e-3
- SGD no momentum:       K ∈ {5, 10, 20, 36}, η=1e-3
- AdamW varying wd:      K ∈ {10, 20}, wd ∈ {0.0, 0.001, 0.01, 0.1}
- Total: 8 + 4 + 8 = 20 runs

Expected output: Table of τ(K) for each optimizer. If δ is similar
across optimizers, the scaling is robust.
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

K_SWEEP = [5, 10, 20, 36]
K_WD = [10, 20]
WD_VALUES = [0.0, 0.001, 0.01, 0.1]
SEED = 42
LR = 1e-3
MAX_STEPS = 80000

EXPERIMENTS = []
for k in K_SWEEP:
    EXPERIMENTS.append({"name": f"optim_sgd_mom0.9_K{k}", "k": k,
                        "optimizer_type": "sgd", "momentum": 0.9,
                        "weight_decay": 0.0, "group": "SGD+mom0.9"})
for k in K_SWEEP:
    EXPERIMENTS.append({"name": f"optim_sgd_nomom_K{k}", "k": k,
                        "optimizer_type": "sgd", "momentum": 0.0,
                        "weight_decay": 0.0, "group": "SGD"})
for k in K_WD:
    for wd in WD_VALUES:
        EXPERIMENTS.append({"name": f"optim_adamw_wd{wd}_K{k}", "k": k,
                            "optimizer_type": "adamw", "momentum": 0.0,
                            "weight_decay": wd, "group": f"AdamW_wd{wd}"})


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
    print("EXPERIMENT 5: OPTIMIZER ABLATION")
    print("=" * 70)

    jobs = []
    for exp in EXPERIMENTS:
        name = exp["name"]
        if run_exists(name, min_steps=MAX_STEPS * 0.5, output_dir=OUTPUT_DIR):
            print(f"  SKIP {name}")
            continue
        print(f"  QUEUE {name}")
        cfg = make_config(
            experiment_name=name, k=exp["k"], lr=LR, seed=SEED,
            max_steps=MAX_STEPS, optimizer_type=exp["optimizer_type"],
            momentum=exp["momentum"], weight_decay=exp["weight_decay"],
            early_stop_frac=0.01,
        )
        jobs.append({
            "cfg_dict": OmegaConf.to_container(cfg, resolve=True),
            "mapping_path": None, "output_dir": OUTPUT_DIR, "name": name,
        })

    if jobs:
        run_parallel(jobs, max_workers=6, label="optimizer-ablation")
    else:
        print("  All runs already exist!")

    # ── Analyze ────────────────────────────────────────────────────────────
    print(f"\n{'Group':<18} {'K':<6} {'τ':>8} {'Converged':>10}")
    print("-" * 46)

    by_group = {}
    for exp in EXPERIMENTS:
        name, group, k = exp["name"], exp["group"], exp["k"]
        h = load_history(name, OUTPUT_DIR)
        if h is None:
            print(f"  {group:<18} {k:<6} {'MISS':>8}")
            continue
        tau = detect_tau(h, math.log(k))
        converged = tau is not None
        print(f"  {group:<18} {k:<6} {str(tau) if tau else 'N/C':>8} {'Yes' if converged else 'No':>10}")
        if group not in by_group:
            by_group[group] = {}
        by_group[group][k] = {"tau": tau, "converged": converged}

    # ── Power-law fits ─────────────────────────────────────────────────────
    print(f"\n{'Group':<18} {'δ':>8} {'95% CI':>18} {'n':>4}")
    print("-" * 52)

    group_fits = {}
    for group in sorted(by_group.keys()):
        ks = sorted([k for k, v in by_group[group].items() if v["tau"] is not None])
        if len(ks) < 2:
            print(f"  {group:<18} {'N/A':>8} (only {len(ks)} converged)")
            continue
        k_arr = np.array(ks, dtype=float)
        tau_arr = np.array([by_group[group][k]["tau"] for k in ks], dtype=float)
        delta, ci_lo, ci_hi = fit_power_law(k_arr, tau_arr)
        group_fits[group] = {"delta": delta, "ci_lo": ci_lo, "ci_hi": ci_hi, "n": len(ks)}
        if delta is not None:
            print(f"  {group:<18} {delta:>8.3f} [{ci_lo:.3f}, {ci_hi:.3f}] {len(ks):>4}")

    # ── Figure ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    colors = {"SGD+mom0.9": "#E74C3C", "SGD": "#E67E22", "AdamW_wd0.01": "#3498DB"}
    markers = {"SGD+mom0.9": "^", "SGD": "v", "AdamW_wd0.01": "o"}

    for group in ["SGD+mom0.9", "SGD"]:
        if group not in by_group:
            continue
        ks = sorted([k for k, v in by_group[group].items() if v["tau"] is not None])
        if not ks:
            continue
        k_arr = np.array(ks, dtype=float)
        tau_arr = np.array([by_group[group][k]["tau"] for k in ks], dtype=float)
        label = group
        if group in group_fits and group_fits[group]["delta"] is not None:
            label += f" (δ={group_fits[group]['delta']:.2f})"
        ax.loglog(k_arr, tau_arr, f"{markers.get(group, 'o')}-",
                  color=colors.get(group, "gray"), label=label, markersize=6, linewidth=1.5)

    wd_baseline = "AdamW_wd0.01"
    if wd_baseline in by_group:
        ks = sorted([k for k, v in by_group[wd_baseline].items() if v["tau"] is not None])
        if ks:
            k_arr = np.array(ks, dtype=float)
            tau_arr = np.array([by_group[wd_baseline][k]["tau"] for k in ks], dtype=float)
            ax.loglog(k_arr, tau_arr, "o--", color="#3498DB",
                      label="AdamW wd=0.01 (baseline)", markersize=6, linewidth=1.5)

    ax.set_xlabel("K", fontsize=10)
    ax.set_ylabel("τ", fontsize=10)
    ax.set_title("τ(K) by optimizer", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

    ax = axes[1]
    for k in K_WD:
        wds, taus = [], []
        for wd in WD_VALUES:
            group = f"AdamW_wd{wd}"
            if group in by_group and k in by_group[group] and by_group[group][k]["tau"]:
                wds.append(wd if wd > 0 else 1e-4)
                taus.append(by_group[group][k]["tau"])
        if wds:
            ax.semilogx(wds, taus, "o-", label=f"K={k}", markersize=6, linewidth=1.5)

    ax.set_xlabel("Weight decay", fontsize=10)
    ax.set_ylabel("τ", fontsize=10)
    ax.set_title("Effect of weight decay (AdamW)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=8)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(FIG_DIR / f"fig_optimizer_ablation.{ext}", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved: {FIG_DIR}/fig_optimizer_ablation")

    # ── Save summary ───────────────────────────────────────────────────────
    summary = {
        "experiment": "optimizer_ablation",
        "description": "Optimizer ablation (Exp 5)",
        "by_group": {g: {str(k): v for k, v in gv.items()} for g, gv in by_group.items()},
        "power_law_fits": group_fits,
    }
    summary_path = Path(OUTPUT_DIR) / "optimizer_ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
