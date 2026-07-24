#!/usr/bin/env python3
"""
Experiment 4: AdamW Timescale Universality (Paper: 2505.13738 "Power Lines")

Tests whether the composite timescale τ_adamw = B / (η · λ · D) predicts
or correlates with our measured τ_transition across K, batch-size, and LR sweeps.

No checkpoint loading — pure arithmetic on existing training configs and measured τ.
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
SAVE_DIR = OUTPUTS / "paper_figures"
SAVE_DIR.mkdir(exist_ok=True)


def load_history(name: str):
    p = OUTPUTS / name / "training_history.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def compute_tau(history, k, threshold_frac=0.5):
    if history is None:
        return None
    log_k = math.log(k)
    threshold = threshold_frac * log_k
    for s, l in zip(history["steps"], history["first_target_loss"]):
        if l < threshold:
            return s
    return None


# ── Collect all runs with measured τ ─────────────────────────────────────────

runs = []

# K-sweep (Transformer): B=128, η=1e-3, λ=0.01
for k in [3, 5, 7, 10, 13, 17, 20, 25, 30, 36]:
    h = load_history(f"landauer_dense_k{k}")
    tau = compute_tau(h, k)
    if tau is not None:
        D = k * 1000
        runs.append({
            "name": f"Transformer K={k}",
            "arch": "Transformer",
            "sweep": "K-sweep",
            "K": k, "B": 128, "eta": 1e-3, "lam": 0.01, "D": D,
            "tau_trans": tau,
        })

# Batch-size sweep: η=1e-3, λ=0.01, K∈{20,36}
for k in [20, 36]:
    D = k * 1000
    for bs in [128, 256, 512]:
        name = f"temp_lr1e3_bs{bs}_k{k}"
        h = load_history(name)
        tau = compute_tau(h, k)
        if tau is None and k == 36 and bs == 128:
            h = load_history("temp_lr1e3_bs128_k36_30k")
            tau = compute_tau(h, k)
        if tau is not None:
            runs.append({
                "name": f"BS={bs} K={k}",
                "arch": "Transformer",
                "sweep": "BS-sweep",
                "K": k, "B": bs, "eta": 1e-3, "lam": 0.01, "D": D,
                "tau_trans": tau,
            })

# LR sweep: B=128, λ=0.01, K=20
for eta, eta_str in [(3e-4, "3e-4"), (5e-4, "5e-4"), (1e-3, "1e-3"), (2e-3, "2e-3")]:
    h = load_history(f"lr_sweep_eta{eta_str}")
    tau = compute_tau(h, 20)
    if tau is not None:
        runs.append({
            "name": f"η={eta_str} K=20",
            "arch": "Transformer",
            "sweep": "LR-sweep",
            "K": 20, "B": 128, "eta": eta, "lam": 0.01, "D": 20000,
            "tau_trans": tau,
        })

# GatedMLP K-sweep: B=128, η=1e-3, λ=0.01
for k in [10, 15, 20, 25, 36, 50]:
    h = load_history(f"gatedmlp_k{k}")
    tau = compute_tau(h, k)
    if tau is not None:
        runs.append({
            "name": f"GatedMLP K={k}",
            "arch": "GatedMLP",
            "sweep": "K-sweep",
            "K": k, "B": 128, "eta": 1e-3, "lam": 0.01, "D": k * 1000,
            "tau_trans": tau,
        })

# RNN K-sweep: B=128, η=1e-3, λ=0.01
for k in [10, 15, 20, 25, 36, 50, 100]:
    h = load_history(f"rnn_k{k}")
    tau = compute_tau(h, k)
    if tau is not None:
        runs.append({
            "name": f"RNN K={k}",
            "arch": "RNN",
            "sweep": "K-sweep",
            "K": k, "B": 128, "eta": 1e-3, "lam": 0.01, "D": k * 1000,
            "tau_trans": tau,
        })

# ── Compute AdamW timescale ──────────────────────────────────────────────────

for r in runs:
    r["tau_adamw"] = r["B"] / (r["eta"] * r["lam"] * r["D"])
    r["ratio"] = r["tau_trans"] / r["tau_adamw"]
    # Also compute weight decay timescale
    r["tau_wd"] = 1.0 / (r["eta"] * r["lam"])

# ── Print results ────────────────────────────────────────────────────────────

print("=" * 100)
print("Experiment 4: AdamW Timescale Universality")
print("=" * 100)
print(f"\n{'Name':<22} {'Arch':<12} {'K':>4} {'B':>4} {'η':>8} {'D':>7} "
      f"{'τ_adamw':>9} {'τ_trans':>9} {'ratio':>8} {'τ_wd':>9}")
print("-" * 100)
for r in runs:
    print(f"{r['name']:<22} {r['arch']:<12} {r['K']:>4} {r['B']:>4} {r['eta']:>8.1e} "
          f"{r['D']:>7} {r['tau_adamw']:>9.1f} {r['tau_trans']:>9} "
          f"{r['ratio']:>8.1f} {r['tau_wd']:>9.0f}")

# ── Correlation analysis ─────────────────────────────────────────────────────

tau_adamw_arr = np.array([r["tau_adamw"] for r in runs])
tau_trans_arr = np.array([r["tau_trans"] for r in runs])

# Log-log regression
log_adamw = np.log(tau_adamw_arr)
log_trans = np.log(tau_trans_arr)
slope, intercept, r_value, p_value, std_err = stats.linregress(log_adamw, log_trans)

print(f"\n{'='*60}")
print(f"LOG-LOG REGRESSION: log(τ_trans) = {slope:.3f} · log(τ_adamw) + {intercept:.3f}")
print(f"  R² = {r_value**2:.4f},  p = {p_value:.2e},  slope SE = {std_err:.3f}")
print(f"  Equivalent: τ_trans = {np.exp(intercept):.1f} · τ_adamw^{slope:.2f}")
print(f"{'='*60}")

# Breakdown by sweep type
print("\nBreakdown by sweep:")
for sweep_type in ["K-sweep", "BS-sweep", "LR-sweep"]:
    subset = [r for r in runs if r["sweep"] == sweep_type and r["arch"] == "Transformer"]
    if len(subset) >= 3:
        x = np.log([r["tau_adamw"] for r in subset])
        y = np.log([r["tau_trans"] for r in subset])
        sl, ic, rv, pv, se = stats.linregress(x, y)
        print(f"  {sweep_type} (Transformer): slope={sl:.2f}, R²={rv**2:.3f}, "
              f"n={len(subset)}")

# Breakdown by architecture
print("\nBreakdown by architecture:")
for arch in ["Transformer", "GatedMLP", "RNN"]:
    subset = [r for r in runs if r["arch"] == arch and r["sweep"] == "K-sweep"]
    if len(subset) >= 3:
        x = np.log([r["tau_adamw"] for r in subset])
        y = np.log([r["tau_trans"] for r in subset])
        sl, ic, rv, pv, se = stats.linregress(x, y)
        print(f"  {arch}: slope={sl:.2f}, R²={rv**2:.3f}, n={len(subset)}")

# ── Figure ───────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(9, 4))

# Panel (a): τ_trans vs τ_adamw log-log
ax = axes[0]
markers = {"K-sweep": "o", "BS-sweep": "s", "LR-sweep": "^"}
colors = {"Transformer": "#3498DB", "GatedMLP": "#E74C3C", "RNN": "#27AE60"}
for r in runs:
    ax.scatter(r["tau_adamw"], r["tau_trans"],
               marker=markers.get(r["sweep"], "o"),
               color=colors.get(r["arch"], "gray"),
               s=40, zorder=3, alpha=0.8)

# Fit line
x_fit = np.linspace(tau_adamw_arr.min() * 0.5, tau_adamw_arr.max() * 2, 100)
y_fit = np.exp(intercept) * x_fit ** slope
ax.plot(x_fit, y_fit, "k--", alpha=0.5, linewidth=1,
        label=rf"$\tau_{{trans}} \propto \tau_{{adamw}}^{{{slope:.2f}}}$ ($R^2$={r_value**2:.2f})")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"AdamW timescale $\tau_{adamw} = B/(\eta \lambda D)$")
ax.set_ylabel(r"Transition time $\tau_{trans}$")
ax.set_title("(a) AdamW timescale vs transition time")
ax.legend(fontsize=7)

# Custom legend for markers/colors
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#3498DB", markersize=6, label="Transformer"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#E74C3C", markersize=6, label="GatedMLP"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#27AE60", markersize=6, label="RNN"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=6, label="K-sweep"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="gray", markersize=6, label="BS-sweep"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor="gray", markersize=6, label="LR-sweep"),
]
ax.legend(handles=legend_elements, fontsize=6, loc="upper left", ncol=2)

# Panel (b): ratio τ_trans/τ_adamw vs K for the K-sweep
ax = axes[1]
for arch, color in colors.items():
    subset = [r for r in runs if r["arch"] == arch and r["sweep"] == "K-sweep"]
    if subset:
        ks = [r["K"] for r in subset]
        ratios = [r["ratio"] for r in subset]
        ax.plot(ks, ratios, "o-", color=color, label=arch, markersize=5, linewidth=1)

ax.set_xlabel("Ambiguity $K$")
ax.set_ylabel(r"$\tau_{trans} / \tau_{adamw}$")
ax.set_title(r"(b) Ratio $\tau_{trans}/\tau_{adamw}$ vs $K$")
ax.legend(fontsize=7)
ax.set_xscale("log")
ax.set_yscale("log")

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(SAVE_DIR / f"fig_adamw_timescale.{ext}", dpi=300, bbox_inches="tight")
plt.close()

print(f"\nFigure saved: {SAVE_DIR / 'fig_adamw_timescale.pdf'}")
