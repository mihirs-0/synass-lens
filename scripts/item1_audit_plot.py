#!/usr/bin/env python
"""Item 1 audit — side-by-side raw per-layer curves."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

OUT = Path('/Users/mihir/synass-lens/synass-lens/outputs/followup/item1')
OUT.mkdir(parents=True, exist_ok=True)

# Primary K=10
with open('/Users/mihir/synass-lens/synass-lens/outputs/followup/block2/block2_gap_metrics.json') as f:
    b2 = json.load(f)
p = b2["per_K"]["10"]["profiles"]
primary = {"ablation": p["ablation"], "lens": p["lens"]}

# Control K=1 seeds — hard-coded from completed stdout
controls = {
    42: {"ablation": [7.543, 3.058, 0.705, 0.032],
         "lens": [0.075, 0.348, 0.955, 1.000]},
    123: {"ablation": [9.218, 1.711, 0.248, 0.034],
          "lens": [0.094, 0.452, 0.951, 1.000]},
    456: {"ablation": [7.582, 3.324, 0.713, 0.027],
          "lens": [0.071, 0.350, 0.962, 1.000]},
}

layers = np.arange(4)
fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150, facecolor="white")

# Panel A: ablation
ax = axes[0]
ax.plot(layers, primary["ablation"], "o-", color="#1F4E79", linewidth=2.5,
        markersize=8, label="primary K=10 (seed 42)")
for s, style in zip(sorted(controls.keys()), ["#E89B44", "#C4452D", "#6CAE75"]):
    ax.plot(layers, controls[s]["ablation"], "s--", color=style, linewidth=1.8,
            markersize=7, label=f"K=1 control (seed {s})")
ax.set_xticks(layers); ax.set_xticklabels([f"L{L}" for L in range(4)])
ax.set_ylabel("ablation Δ (nats)", fontsize=11)
ax.set_title("Necessity: full-layer ablation per layer (raw)", fontsize=11)
ax.legend(fontsize=9, frameon=False)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Panel B: lens
ax = axes[1]
ax.plot(layers, primary["lens"], "o-", color="#1F4E79", linewidth=2.5,
        markersize=8, label="primary K=10 (seed 42)")
for s, style in zip(sorted(controls.keys()), ["#E89B44", "#C4452D", "#6CAE75"]):
    ax.plot(layers, controls[s]["lens"], "s--", color=style, linewidth=1.8,
            markersize=7, label=f"K=1 control (seed {s})")
ax.set_xticks(layers); ax.set_xticklabels([f"L{L}" for L in range(4)])
ax.set_ylabel("logit lens P(correct)", fontsize=11)
ax.set_title("Sufficiency: logit lens per layer (raw)", fontsize=11)
ax.set_ylim(-0.05, 1.08)
ax.legend(fontsize=9, frameon=False, loc="lower right")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.suptitle("Item 1 audit: primary K=10 vs control K=1 — raw per-layer curves",
             fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "audit_curves.png", dpi=180, facecolor="white", bbox_inches="tight")
print("wrote", OUT / "audit_curves.png")
