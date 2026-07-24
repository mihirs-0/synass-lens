#!/usr/bin/env python3
"""
Plot τ vs D on log-log axes for the staged disambiguation paper.
Shows K-sweep data (blue circles) with power-law fit, and fixed-D control
points (red squares) demonstrating K-invariance.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Style ──────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "mathtext.fontset": "cm",        # Computer Modern for LaTeX-like math
    "font.family": "serif",
})

# ── Data ───────────────────────────────────────────────────────────────
# 1. K-sweep recast as D-sweep  (n_b = 1000 fixed, D = K * n_b)
D_sweep = np.array([3000, 5000, 7000, 10000, 13000,
                     17000, 20000, 25000, 30000, 36000], dtype=float)
tau_sweep = np.array([450, 800, 1050, 1850, 2100,
                      3300, 3950, 5250, 6950, 8750], dtype=float)

# 2. Fixed-D control (4 K-values each, all giving ≈ same τ)
D_ctrl_10k = np.array([10000, 10000, 10000, 10000], dtype=float)
tau_ctrl_10k = np.array([1580, 1600, 1620, 1610], dtype=float)

D_ctrl_20k = np.array([20000, 20000, 20000, 20000], dtype=float)
tau_ctrl_20k = np.array([3900, 3950, 3980, 3960], dtype=float)

# ── Power-law fit in log space ─────────────────────────────────────────
log_D = np.log10(D_sweep)
log_tau = np.log10(tau_sweep)
slope, intercept = np.polyfit(log_D, log_tau, 1)
print(f"Fitted exponent: {slope:.3f}  (expected ~1.19)")

D_fit = np.logspace(np.log10(2500), np.log10(40000), 200)
tau_fit = 10**intercept * D_fit**slope

# ── Figure ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 3.5), dpi=300)

# Fit line (behind points)
ax.plot(D_fit, tau_fit, "--", color="steelblue", linewidth=1.3,
        alpha=0.8, zorder=1)

# K-sweep points
ax.scatter(D_sweep, tau_sweep, s=42, marker="o", facecolors="royalblue",
           edgecolors="black", linewidths=0.5, zorder=3,
           label=r"K-sweep ($n_b = 1{,}000$)")

# Fixed-D controls with slight horizontal jitter
rng = np.random.default_rng(42)
jitter_10k = 10**(np.log10(D_ctrl_10k) + rng.uniform(-0.015, 0.015, 4))
jitter_20k = 10**(np.log10(D_ctrl_20k) + rng.uniform(-0.015, 0.015, 4))

ax.scatter(jitter_10k, tau_ctrl_10k, s=42, marker="s",
           facecolors="indianred", edgecolors="black", linewidths=0.5,
           zorder=3, label=r"Fixed-$D$ control")
ax.scatter(jitter_20k, tau_ctrl_20k, s=42, marker="s",
           facecolors="indianred", edgecolors="black", linewidths=0.5,
           zorder=3)

# ── Axes ───────────────────────────────────────────────────────────────
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Dataset size $D$")
ax.set_ylabel(r"Waiting time $\tau$ (steps)")

# Tick formatting — keep plain numbers, not 10^x
from matplotlib.ticker import FuncFormatter
ax.xaxis.set_major_formatter(FuncFormatter(
    lambda x, _: f"{int(x):,}" if x >= 1 else f"{x}"))
ax.yaxis.set_major_formatter(FuncFormatter(
    lambda y, _: f"{int(y):,}" if y >= 1 else f"{y}"))

# ── Annotation ─────────────────────────────────────────────────────────
# Place the exponent annotation near the middle of the fit line
ax.text(0.38, 0.78, r"$\tau \propto D^{1.19}$",
        transform=ax.transAxes, fontsize=11, color="steelblue",
        fontstyle="italic")

# ── Legend ─────────────────────────────────────────────────────────────
ax.legend(loc="lower right", frameon=True, framealpha=0.9,
          edgecolor="0.8")

# ── Save ───────────────────────────────────────────────────────────────
fig.tight_layout()
out_path = "/Users/mihir/synass-lens/synass-lens/outputs/paper_figures/fig_tau_vs_D.pdf"
fig.savefig(out_path, bbox_inches="tight")
print(f"Saved → {out_path}")

# Also save PNG preview
png_path = out_path.replace(".pdf", ".png")
fig.savefig(png_path, bbox_inches="tight", dpi=300)
print(f"Saved → {png_path}")

plt.close(fig)
