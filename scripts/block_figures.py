#!/usr/bin/env python
"""Compile the Block 0/1/2/4 summary figures in one pass."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent / "outputs" / "followup"

# ── Block 0: variance decomposition ──
with open(ROOT / "block0" / "block0_stability.json") as f:
    b0 = json.load(f)
dec = b0["variance_decomposition"]
fig, ax = plt.subplots(1, 1, figsize=(6, 3.5), dpi=150, facecolor="white")
metrics = list(dec.keys())
xs = np.arange(len(metrics))
eval_stds = [dec[m]["eval_noise_std"] for m in metrics]
seed_stds = [dec[m]["seed_std"] for m in metrics]
w = 0.38
ax.bar(xs - w/2, eval_stds, w, label="eval-noise std (within seed)",
       color="#E89B44", edgecolor="black", linewidth=0.7)
ax.bar(xs + w/2, seed_stds, w, label="seed std (across seeds)",
       color="#3B6FB6", edgecolor="black", linewidth=0.7)
ax.set_xticks(xs)
ax.set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=8)
ax.set_ylabel("std of metric")
ax.set_title("Block 0: variance decomposition (flagship metrics)")
ax.legend(fontsize=8, frameon=False)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(ROOT / "block0" / "block0_figure.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("wrote block0_figure.png")

# ── Block 1: per-readout per-layer decodability (mean across seeds) ──
with open(ROOT / "block1" / "block1_readouts.json") as f:
    b1 = json.load(f)
fig, ax = plt.subplots(1, 1, figsize=(6, 3.5), dpi=150, facecolor="white")
colors = {"logit_lens": "#3B6FB6", "linear_probe": "#6CAE75", "mlp_probe": "#C4452D"}
for key in ["logit_lens", "linear_probe", "mlp_probe"]:
    m = b1["aggregate"][key]["mean_per_layer"]
    s = b1["aggregate"][key]["std_per_layer"]
    layers = np.arange(4)
    ax.errorbar(layers, m, yerr=s, marker="o", markersize=5, linewidth=1.6,
                color=colors[key], label=key.replace("_", " "), capsize=3)
ax.axhline(0.5, color="gray", linestyle="dotted", linewidth=0.8, alpha=0.6)
ax.set_xticks(layers); ax.set_xticklabels([f"L{L}" for L in range(4)])
ax.set_ylabel("decodability"); ax.set_ylim(-0.05, 1.08)
ax.set_title("Block 1: stronger readouts don't rescue early decodability")
ax.legend(fontsize=8, loc="upper left", frameon=False)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(ROOT / "block1" / "block1_figure.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("wrote block1_figure.png")

# ── Block 2: gap metrics across K ──
with open(ROOT / "block2" / "block2_gap_metrics.json") as f:
    b2 = json.load(f)
Ks = sorted(int(k) for k in b2["per_K"].keys())
rhos = [b2["per_K"][str(k)]["primary"]["spearman_abl_lens"] for k in Ks]
areas = [b2["per_K"][str(k)]["primary"]["area_necessity_sufficiency"] for k in Ks]
ratios = [b2["per_K"][str(k)]["supplementary"]["ratio_L0_L2"] for k in Ks]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), dpi=150, facecolor="white")
for ax, data, name, expected in [
    (axes[0], rhos, "Spearman ρ(ablation, lens)", "< 0"),
    (axes[1], areas, "area(necessity, sufficiency)", "> 0"),
    (axes[2], ratios, "L0/L2 ratio", "> 1"),
]:
    ax.plot(Ks, data, "o-", color="#3B6FB6", linewidth=1.8, markersize=6)
    ax.set_xlabel("K (task difficulty)")
    ax.set_ylabel(name, fontsize=10)
    ax.set_title(name, fontsize=10)
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
axes[0].axhline(0, color="black", linewidth=0.5)
axes[2].axhline(1, color="black", linestyle="dotted", linewidth=0.8)
fig.suptitle("Block 2: gap metrics across K (seed 42)", fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(ROOT / "block2" / "block2_figure.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("wrote block2_figure.png")

# ── Block 4: granularity ladder ──
with open(ROOT / "block4" / "block4_granularity.json") as f:
    b4 = json.load(f)
lens = b4["lens_per_layer"]
ab_layer = b4["full_layer_ablation"]
ab_attn = b4["attn_only_ablation"]
ab_mlp = b4["mlp_only_ablation"]
ab_head_max = b4["head_max_per_layer"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150, facecolor="white")
layers = np.arange(4)
# Left: stacked per-granularity ablation
w = 0.2
axes[0].bar(layers - 1.5*w, ab_layer, w, label="full layer", color="#3B6FB6",
            edgecolor="black", linewidth=0.6)
axes[0].bar(layers - 0.5*w, ab_attn, w, label="attn-only", color="#E89B44",
            edgecolor="black", linewidth=0.6)
axes[0].bar(layers + 0.5*w, ab_mlp, w, label="MLP-only", color="#6CAE75",
            edgecolor="black", linewidth=0.6)
axes[0].bar(layers + 1.5*w, ab_head_max, w, label="head max", color="#C4452D",
            edgecolor="black", linewidth=0.6)
axes[0].set_xticks(layers); axes[0].set_xticklabels([f"L{L}" for L in range(4)])
axes[0].set_ylabel("ablation Δ (nats)")
axes[0].set_title("Ablation Δ at four granularities")
axes[0].legend(fontsize=8, frameon=False)
axes[0].spines["top"].set_visible(False); axes[0].spines["right"].set_visible(False)

# Right: gap metrics by granularity
g = b4["metrics_by_granularity"]
labels = ["full-layer", "component\n(attn max)", "component\n(MLP max)",
          "component\n(max)", "head\n(max)"]
keys = ["G1_full_layer", "G2_component_attn", "G2_component_mlp",
        "G2_component_max", "G3_head_max"]
rhos = [g[k]["spearman"] for k in keys]
areas = [g[k]["area"] for k in keys]
x = np.arange(len(labels))
ax2 = axes[1]
ax2b = ax2.twinx()
l1 = ax2.plot(x, rhos, "o-", color="#3B6FB6", label="Spearman ρ", linewidth=1.8)
l2 = ax2b.plot(x, areas, "s--", color="#C4452D", label="area (nec, suf)",
               linewidth=1.8)
ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8)
ax2.set_ylabel("Spearman ρ (ablation vs lens)", color="#3B6FB6")
ax2.tick_params(axis="y", labelcolor="#3B6FB6")
ax2b.set_ylabel("area between curves", color="#C4452D")
ax2b.tick_params(axis="y", labelcolor="#C4452D")
ax2.set_title("Gap metric vs granularity")
ax2.axhline(0, color="black", linewidth=0.5)
ax2.spines["top"].set_visible(False); ax2b.spines["top"].set_visible(False)
fig.tight_layout()
fig.savefig(ROOT / "block4" / "block4_figure.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("wrote block4_figure.png")
