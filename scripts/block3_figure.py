#!/usr/bin/env python
"""Block 3 figure: training trajectories per layer for three diagnostics,
with onset markers under moderate (0.5) and strict (0.75) thresholds."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent.parent / "outputs" / "followup" / "block3"
with open(BASE / "block3_dynamics.json") as f:
    data = json.load(f)

ckpts = data["checkpoints"]
steps = [c["step"] for c in ckpts]
diagnostics = [("logit_lens", "logit lens $P$(correct)"),
               ("linear_probe", "linear probe acc."),
               ("mlp_probe", "MLP probe acc.")]
layer_colors = {0: "#C4452D", 1: "#E89B44", 2: "#6CAE75", 3: "#3B6FB6"}
layer_labels = {L: f"L{L}" for L in range(4)}

fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=150, facecolor="white",
                         sharey=True)
for ax, (key, ylabel) in zip(axes, diagnostics):
    for L in range(4):
        vals = [c[key][L] for c in ckpts]
        ax.plot(steps, vals, marker="o", markersize=3, linewidth=1.6,
                color=layer_colors[L], label=layer_labels[L])
    for thr, style in [(0.5, "dotted"), (0.75, "dashed")]:
        ax.axhline(thr, color="gray", linestyle=style, linewidth=0.7, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("training step (log scale)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(key.replace("_", " "), fontsize=11)
    ax.set_ylim(-0.05, 1.08)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, loc="upper left", frameon=False)

fig.suptitle("Per-layer decodability across training (seed 42, 32 checkpoints)",
             fontsize=12, y=1.02)
fig.tight_layout()
out = BASE / "block3_figure.png"
fig.savefig(out, dpi=180, facecolor="white", bbox_inches="tight")
print(f"Wrote {out}")
