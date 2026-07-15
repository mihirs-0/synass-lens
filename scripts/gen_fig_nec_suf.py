#!/usr/bin/env python
"""Generate Figure 1: Necessity vs Sufficiency X-shaped cross."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Data from multi-seed experiments (mean ± std across 4 seeds)
layers = [0, 1, 2, 3]
layer_labels = ["L0", "L1", "L2", "L3"]

# Ablation Δ (necessity) — normalized to [0,1] range for visual comparison
ablation_mean = np.array([15.61, 4.77, 2.24, 1.27])
ablation_std  = np.array([1.57, 0.67, 0.23, 0.60])

# Normalize ablation to 0-1 scale (divide by max)
abl_norm = ablation_mean / ablation_mean[0]
abl_std_norm = ablation_std / ablation_mean[0]

# Target-only flip rate (sufficiency) — from seed 42 experiment
flip_rate = np.array([0.340, 0.488, 0.754, 1.000])

# Logit lens P(correct) — mean across 4 seeds
logit_mean = np.array([0.064, 0.145, 0.406, 0.963])
logit_std  = np.array([0.007, 0.029, 0.159, 0.022])

fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.5))

x = np.array(layers)

# Plot necessity (ablation Δ, normalized) — decreasing
color_nec = "#d62728"  # red
color_suf = "#1f77b4"  # blue

ax.plot(x, abl_norm, 'o-', color=color_nec, linewidth=2.5, markersize=8,
        label='Necessity (ablation $\\Delta$, normalized)', zorder=3)
ax.fill_between(x, abl_norm - abl_std_norm, abl_norm + abl_std_norm,
                color=color_nec, alpha=0.15)

# Plot sufficiency (target-only flip rate) — increasing
ax.plot(x, flip_rate, 's-', color=color_suf, linewidth=2.5, markersize=8,
        label='Sufficiency (target-only flip rate)', zorder=3)

# Also plot logit lens as dashed for convergent evidence
color_lens = "#2ca02c"  # green
ax.plot(x, logit_mean, '^--', color=color_lens, linewidth=1.8, markersize=7,
        label='Logit lens $P(\\mathrm{correct})$', zorder=2, alpha=0.8)
ax.fill_between(x, logit_mean - logit_std, logit_mean + logit_std,
                color=color_lens, alpha=0.10)

# Mark the crossover region
ax.axvspan(1.3, 2.2, alpha=0.08, color='gray')
ax.text(1.75, 0.03, 'crossover', ha='center', fontsize=8, fontstyle='italic',
        color='gray')

# 7.1x gap annotation between L0 and L2 ablation (necessity) values
ax.annotate('', xy=(0.08, abl_norm[2]), xytext=(0.08, abl_norm[0]),
            arrowprops=dict(arrowstyle='|-|', color='#222', lw=1.5,
                            mutation_scale=6))
ax.text(0.20, (abl_norm[0] + abl_norm[2]) / 2,
        r'$7.1{\pm}1.1\!\times$' + '\ngap',
        ha='left', va='center', fontsize=9, fontweight='bold', color='#222',
        bbox=dict(facecolor='white', edgecolor='#555', alpha=0.9, pad=2))

ax.set_xticks(x)
ax.set_xticklabels(layer_labels, fontsize=11)
ax.set_xlabel("Layer", fontsize=11)
ax.set_ylabel("Normalized score", fontsize=11)
ax.set_ylim(-0.05, 1.12)
ax.legend(fontsize=8, loc='center right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig.savefig("workshop_paper/fig_nec_suf.png", dpi=300, bbox_inches="tight")
print("Saved workshop_paper/fig_nec_suf.png")
