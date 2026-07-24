#!/usr/bin/env python3
"""
Cross-architecture τ scaling analysis.

Reads existing runs:
  - landauer_dense_k{3..36}   (Transformer, η=1e-3)
  - gatedmlp_k{10..100}       (GatedMLP, η=1e-3)
  - rnn_k{10..100}            (RNN/LSTM, η=1e-3)

Produces:
  - log-log plot of τ vs K for all three architectures with power-law fits
  - outputs/paper_figures/fig_cross_arch_tau.pdf
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
    """First step where loss < threshold_frac * log(K)."""
    if history is None:
        return None
    log_k = math.log(k)
    threshold = threshold_frac * log_k
    for s, l in zip(history["steps"], history["first_target_loss"]):
        if l < threshold:
            return s
    return None


def power_law_fit(k_arr, tau_arr):
    """Fit τ = a * K^α in log-log space. Returns (alpha, a, r_squared)."""
    log_k = np.log(k_arr)
    log_tau = np.log(tau_arr)
    slope, intercept, r_value, _, _ = stats.linregress(log_k, log_tau)
    return slope, np.exp(intercept), r_value ** 2


# ── Collect data ─────────────────────────────────────────────────────────────

architectures = {
    "Transformer": {
        "prefix": "landauer_dense_k",
        "k_candidates": [3, 5, 7, 10, 13, 17, 20, 25, 30, 36],
        "color": "#3498DB",
        "marker": "o",
    },
    "Gated MLP": {
        "prefix": "gatedmlp_k",
        "k_candidates": [10, 15, 20, 25, 36, 50, 75, 100],
        "color": "#E74C3C",
        "marker": "s",
    },
    "RNN (LSTM)": {
        "prefix": "rnn_k",
        "k_candidates": [10, 15, 20, 25, 36, 50, 75, 100],
        "color": "#27AE60",
        "marker": "^",
    },
}

results = {}
for arch_name, cfg in architectures.items():
    k_vals = []
    tau_vals = []
    for k in cfg["k_candidates"]:
        h = load_history(f"{cfg['prefix']}{k}")
        tau = compute_tau(h, k)
        if tau is not None:
            k_vals.append(k)
            tau_vals.append(tau)
    results[arch_name] = {
        "k": np.array(k_vals),
        "tau": np.array(tau_vals),
    }

# ── Print results ────────────────────────────────────────────────────────────

print("=" * 72)
print("Cross-Architecture τ Scaling")
print("=" * 72)

for arch_name in architectures:
    data = results[arch_name]
    print(f"\n{arch_name}:")
    print(f"  K values with transition: {list(data['k'])}")
    print(f"  τ values:                 {list(data['tau'])}")
    if len(data["k"]) >= 3:
        alpha, a, r2 = power_law_fit(data["k"], data["tau"])
        print(f"  Power-law fit: τ = {a:.1f} · K^{alpha:.2f}  (R² = {r2:.3f})")
        results[arch_name]["alpha"] = alpha
        results[arch_name]["a"] = a
        results[arch_name]["r2"] = r2
    else:
        print("  (insufficient data for fit)")

# Summary table
print("\n" + "=" * 72)
print("SUMMARY: Power-law exponents")
print("=" * 72)
print(f"{'Architecture':<20}  {'α':>6}  {'R²':>6}  {'K range':>12}")
print("-" * 52)
for arch_name in architectures:
    data = results[arch_name]
    if "alpha" in data:
        k_range = f"{int(data['k'].min())}–{int(data['k'].max())}"
        print(f"{arch_name:<20}  {data['alpha']:>6.2f}  {data['r2']:>6.3f}  {k_range:>12}")


# ── Figure ───────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(4.5, 3.5))

for arch_name, cfg in architectures.items():
    data = results[arch_name]
    if len(data["k"]) == 0:
        continue
    label = arch_name
    if "alpha" in data:
        label += rf" ($\alpha$={data['alpha']:.2f}, $R^2$={data['r2']:.2f})"
    ax.scatter(data["k"], data["tau"], marker=cfg["marker"], color=cfg["color"],
               s=40, zorder=3, label=label)
    # Fit line
    if "alpha" in data:
        k_fit = np.linspace(data["k"].min() * 0.8, data["k"].max() * 1.2, 100)
        tau_fit = data["a"] * k_fit ** data["alpha"]
        ax.plot(k_fit, tau_fit, "--", color=cfg["color"], alpha=0.6, linewidth=1.2)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Ambiguity $K$")
ax.set_ylabel(r"Plateau duration $\tau$ (steps)")
ax.set_title(r"$\tau \propto K^{\alpha}$ across architectures")
ax.legend(fontsize=7, loc="upper left")

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(SAVE_DIR / f"fig_cross_arch_tau.{ext}", dpi=300, bbox_inches="tight")
plt.close()

print(f"\nFigure saved: {SAVE_DIR / 'fig_cross_arch_tau.pdf'}")
