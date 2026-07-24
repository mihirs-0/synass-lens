#!/usr/bin/env python3
"""
Experiment 6: Gradient Norm Anatomy During Plateau (Papers: 2512.00686, 2405.19454)

Analyzes gradient norm dynamics from existing training histories. Tests whether
gradient norms show structure during the plateau and whether per-component
gradients reveal differential dynamics for attention vs MLP.

Uses existing grad_norm_sq data from training_history.json — no checkpoint loading
for the primary analysis.
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
    for s, l in zip(history["steps"], history["first_target_loss"]):
        if l < threshold_frac * log_k:
            return s
    return None


# ── Main analysis ────────────────────────────────────────────────────────────

print("=" * 80)
print("Experiment 6: Gradient Norm Anatomy During Plateau")
print("=" * 80)

# ── Part 1: Gradient norm from training histories ────────────────────────────

k_values = [3, 5, 7, 10, 13, 17, 20, 25, 30, 36]
grad_data = {}

print(f"\n{'K':>4}  {'τ':>8}  {'has grad_norm':>14}  {'plateau ||∇||²':>16}  "
      f"{'trans ||∇||²':>16}  {'ratio':>8}")
print("-" * 80)

for k in k_values:
    h = load_history(f"landauer_dense_k{k}")
    if h is None:
        continue

    tau = compute_tau(h, k)
    steps = np.array(h["steps"])
    loss = np.array(h["first_target_loss"])

    has_grad = "grad_norm_sq" in h
    if has_grad:
        grad_norm_sq = np.array(h["grad_norm_sq"])

        # Compute Fisher-like ratio: ||∇L||² / L
        fisher_ratio = grad_norm_sq / np.maximum(loss, 1e-10)

        # Phase masks
        if tau is not None:
            plateau_mask = (steps > 500) & (steps < tau * 0.5)
            trans_mask = (steps >= tau * 0.5) & (steps < tau * 1.5)

            plateau_gn = grad_norm_sq[plateau_mask].mean() if plateau_mask.sum() > 0 else 0
            trans_gn = grad_norm_sq[trans_mask].mean() if trans_mask.sum() > 0 else 0
            ratio = trans_gn / plateau_gn if plateau_gn > 0 else float("inf")
        else:
            plateau_gn = trans_gn = ratio = 0

        grad_data[k] = {
            "steps": steps,
            "loss": loss,
            "grad_norm_sq": grad_norm_sq,
            "fisher_ratio": fisher_ratio,
            "tau": tau,
        }

        print(f"{k:>4}  {str(tau):>8}  {'yes':>14}  {plateau_gn:>16.4f}  "
              f"{trans_gn:>16.4f}  {ratio:>8.2f}")
    else:
        print(f"{k:>4}  {str(tau):>8}  {'no':>14}")

# ── Part 2: Gradient waste analysis ──────────────────────────────────────────

print(f"\n{'='*60}")
print("Gradient Waste Analysis: cumulative ||∇L||² during plateau")
print(f"{'='*60}")

print(f"\n{'K':>4}  {'τ':>8}  {'plateau Σ||∇||²':>18}  {'total Σ||∇||²':>18}  {'waste%':>8}")
print("-" * 60)

waste_data = []
for k in sorted(grad_data.keys()):
    d = grad_data[k]
    tau = d["tau"]
    if tau is None:
        continue

    # Cumulative gradient norm during plateau (before tau/2)
    plateau_mask = (d["steps"] > 500) & (d["steps"] < tau * 0.5)
    total_mask = d["steps"] > 500

    plateau_cumul = d["grad_norm_sq"][plateau_mask].sum()
    total_cumul = d["grad_norm_sq"][total_mask].sum()
    waste_pct = 100 * plateau_cumul / total_cumul if total_cumul > 0 else 0

    waste_data.append({"K": k, "tau": tau, "plateau_cumul": plateau_cumul,
                       "total_cumul": total_cumul, "waste_pct": waste_pct})

    print(f"{k:>4}  {tau:>8}  {plateau_cumul:>18.2f}  {total_cumul:>18.2f}  "
          f"{waste_pct:>7.1f}%")

# Scaling: does gradient waste scale with τ?
if len(waste_data) >= 3:
    # Filter out entries with zero waste (can't take log of 0)
    valid_waste = [w for w in waste_data if w["plateau_cumul"] > 0]
    if len(valid_waste) >= 3:
        ks = np.array([w["K"] for w in valid_waste])
        wastes = np.array([w["plateau_cumul"] for w in valid_waste])
        taus = np.array([w["tau"] for w in valid_waste])

        sl, ic, rv, pv, se = stats.linregress(np.log(taus), np.log(wastes))
        print(f"\nGradient waste scaling: Σ||∇||²_plateau ∝ τ^{sl:.2f} (R²={rv**2:.3f})")
    else:
        print("\nNot enough valid data points for gradient waste scaling")

# ── Part 3: LR sweep gradient analysis ──────────────────────────────────────

print(f"\n{'='*60}")
print("Gradient norm across LR sweep (K=20)")
print(f"{'='*60}")

lr_grad_data = {}
for eta_str in ["3e-4", "5e-4", "1e-3", "2e-3"]:
    h = load_history(f"lr_sweep_eta{eta_str}")
    if h is None or "grad_norm_sq" not in h:
        continue
    tau = compute_tau(h, 20)
    steps = np.array(h["steps"])
    grad_norm_sq = np.array(h["grad_norm_sq"])
    loss = np.array(h["first_target_loss"])

    lr_grad_data[eta_str] = {
        "steps": steps, "grad_norm_sq": grad_norm_sq, "loss": loss, "tau": tau
    }
    print(f"  η={eta_str}: τ={tau}, mean ||∇||²={grad_norm_sq.mean():.4f}")

# ── Figure ───────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(9, 7))

# Panel (a): Gradient norm vs step for K-sweep
ax = axes[0, 0]
colors_k = {3: "#95A5A6", 5: "#E74C3C", 10: "#3498DB", 20: "#E67E22", 36: "#27AE60"}
for k in sorted(grad_data.keys()):
    d = grad_data[k]
    if k not in colors_k:
        continue
    # Smooth gradient norms
    window = 20
    if len(d["grad_norm_sq"]) > window:
        smooth = np.convolve(d["grad_norm_sq"], np.ones(window) / window, mode="valid")
        steps_smooth = d["steps"][:len(smooth)]
        ax.plot(steps_smooth, smooth, color=colors_k[k], label=f"K={k}", linewidth=0.8)
    else:
        ax.plot(d["steps"], d["grad_norm_sq"], color=colors_k[k],
                label=f"K={k}", linewidth=0.8)
    if d["tau"]:
        ax.axvline(d["tau"], color=colors_k[k], linestyle=":", alpha=0.4)
ax.set_xlabel("Step")
ax.set_ylabel(r"$\|\nabla L\|^2$")
ax.set_title(r"(a) Gradient norm squared")
ax.legend(fontsize=6)
ax.set_yscale("log")

# Panel (b): Fisher-like ratio ||∇L||²/L
ax = axes[0, 1]
for k in sorted(grad_data.keys()):
    d = grad_data[k]
    if k not in colors_k:
        continue
    window = 20
    if len(d["fisher_ratio"]) > window:
        smooth = np.convolve(d["fisher_ratio"], np.ones(window) / window, mode="valid")
        steps_smooth = d["steps"][:len(smooth)]
        ax.plot(steps_smooth, smooth, color=colors_k[k], label=f"K={k}", linewidth=0.8)
    if d["tau"]:
        ax.axvline(d["tau"], color=colors_k[k], linestyle=":", alpha=0.4)
ax.set_xlabel("Step")
ax.set_ylabel(r"$\|\nabla L\|^2 / L$")
ax.set_title(r"(b) Gradient norm / loss (Fisher proxy)")
ax.legend(fontsize=6)
ax.set_yscale("log")

# Panel (c): Gradient waste vs K
ax = axes[1, 0]
if len(waste_data) >= 3:
    ks_w = [w["K"] for w in waste_data]
    wastes_w = [w["waste_pct"] for w in waste_data]
    ax.bar(range(len(ks_w)), wastes_w, color="#3498DB", alpha=0.7)
    ax.set_xticks(range(len(ks_w)))
    ax.set_xticklabels([str(k) for k in ks_w], fontsize=7)
    ax.set_xlabel("Ambiguity $K$")
    ax.set_ylabel("Plateau gradient waste (%)")
    ax.set_title("(c) Gradient waste during plateau")

# Panel (d): Gradient norm across LR sweep
ax = axes[1, 1]
lr_colors = {"3e-4": "#3498DB", "5e-4": "#27AE60", "1e-3": "#E67E22", "2e-3": "#E74C3C"}
for eta_str, d in lr_grad_data.items():
    window = 20
    if len(d["grad_norm_sq"]) > window:
        smooth = np.convolve(d["grad_norm_sq"], np.ones(window) / window, mode="valid")
        steps_smooth = d["steps"][:len(smooth)]
        ax.plot(steps_smooth, smooth, color=lr_colors.get(eta_str, "gray"),
                label=rf"$\eta$={eta_str}", linewidth=0.8)
    if d["tau"]:
        ax.axvline(d["tau"], color=lr_colors.get(eta_str, "gray"),
                   linestyle=":", alpha=0.5)
ax.set_xlabel("Step")
ax.set_ylabel(r"$\|\nabla L\|^2$")
ax.set_title(r"(d) Gradient norm across LR (K=20)")
ax.legend(fontsize=6)
ax.set_yscale("log")

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(SAVE_DIR / f"fig_gradient_anatomy.{ext}", dpi=300, bbox_inches="tight")
plt.close()

print(f"\nFigure saved: {SAVE_DIR / 'fig_gradient_anatomy.pdf'}")
