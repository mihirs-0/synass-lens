#!/usr/bin/env python3
"""
Per-head circuit localization analysis.

Reads: outputs/temp_lr1e3_bs128_k20/probe_results/all_probes.json
Produces: heatmap of per-head attention-to-z across training steps
          outputs/paper_figures/fig_head_nucleation.pdf
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
SAVE_DIR = OUTPUTS / "paper_figures"
SAVE_DIR.mkdir(exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────

probe_path = OUTPUTS / "temp_lr1e3_bs128_k20" / "probe_results" / "all_probes.json"
with open(probe_path) as f:
    data = json.load(f)

history_path = OUTPUTS / "temp_lr1e3_bs128_k20" / "training_history.json"
with open(history_path) as f:
    history = json.load(f)

attn_data = data["probe_results"]["attention_to_z"]
probe_steps = sorted(int(s) for s in attn_data.keys())

N_LAYERS = 4
N_HEADS = 4

# Build matrix: (n_heads_total, n_steps) where rows = L0H0, L0H1, ..., L3H3
n_total = N_LAYERS * N_HEADS
n_steps = len(probe_steps)
attn_matrix = np.zeros((n_total, n_steps))

for j, step in enumerate(probe_steps):
    az = attn_data[str(step)]["attention_to_z"]
    for layer in range(N_LAYERS):
        for head in range(N_HEADS):
            idx = layer * N_HEADS + head
            attn_matrix[idx, j] = az[layer][head]

# Compute z-gap onset from training history for annotation
log_k = math.log(20)
train_steps = history["steps"]
train_loss = history["first_target_loss"]
tau_step = None
for s, l in zip(train_steps, train_loss):
    if l < 0.5 * log_k:
        tau_step = s
        break

# ── Print analysis ───────────────────────────────────────────────────────────

print("=" * 72)
print("Per-Head Nucleation Analysis (K=20, η=1e-3, BS=128)")
print("=" * 72)

# Baseline: average over first 5 probe steps (steps 100-500)
baseline = attn_matrix[:, :5].mean(axis=1)
# Peak: average over last 10 probe steps
peak = attn_matrix[:, -10:].mean(axis=1)
change = peak - baseline

print(f"\n{'Head':<8} {'Baseline':>10} {'Post-trans':>10} {'Change':>10} {'Ratio':>8}")
print("-" * 52)
for layer in range(N_LAYERS):
    for head in range(N_HEADS):
        idx = layer * N_HEADS + head
        name = f"L{layer}H{head}"
        ratio = peak[idx] / baseline[idx] if baseline[idx] > 0 else 0
        print(f"{name:<8} {baseline[idx]:>10.4f} {peak[idx]:>10.4f} {change[idx]:>+10.4f} {ratio:>8.2f}x")

lead_idx = np.argmax(peak)
lead_layer, lead_head = divmod(lead_idx, N_HEADS)
print(f"\nLead head: L{lead_layer}H{lead_head} (peak attention-to-z = {peak[lead_idx]:.4f})")
print(f"Loss transition τ ≈ step {tau_step}")

# Detect when L1H3 first exceeds 2× baseline
lead_baseline = baseline[lead_idx]
onset_step = None
for j, step in enumerate(probe_steps):
    if attn_matrix[lead_idx, j] > 2 * lead_baseline:
        onset_step = step
        break
if onset_step:
    print(f"Lead head exceeds 2× baseline at step {onset_step} "
          f"({tau_step - onset_step if tau_step else '?'} steps before τ)")

# ── Figure ───────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(7, 3.5), gridspec_kw={"width_ratios": [3, 1.2]})

# Panel (a): Heatmap
ax = axes[0]
# Normalize each head relative to its baseline for clearer visualization
baseline_broadcast = baseline[:, np.newaxis]
# Use raw values (not normalized) since the absolute differences are informative
im = ax.imshow(
    attn_matrix,
    aspect="auto",
    cmap="YlOrRd",
    interpolation="nearest",
    extent=[probe_steps[0], probe_steps[-1], n_total - 0.5, -0.5],
)
# Y-axis labels
head_labels = [f"L{l}H{h}" for l in range(N_LAYERS) for h in range(N_HEADS)]
ax.set_yticks(range(n_total))
ax.set_yticklabels(head_labels, fontsize=7)
ax.set_xlabel("Training step")
ax.set_title("(a) Per-head attention to $z$")

# Mark the transition
if tau_step:
    ax.axvline(tau_step, color="blue", linestyle="--", linewidth=1, alpha=0.7,
               label=rf"$\tau$ = {tau_step}")
    ax.legend(fontsize=7, loc="lower right")

cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Attention to $z$", fontsize=8)
cbar.ax.tick_params(labelsize=7)

# Panel (b): Summary bar chart of peak attention by head
ax = axes[1]
colors = ["#3498DB"] * N_HEADS + ["#E74C3C"] * N_HEADS + \
         ["#27AE60"] * N_HEADS + ["#9B59B6"] * N_HEADS
bars = ax.barh(range(n_total), peak, color=colors, alpha=0.8, height=0.7)
# Highlight the lead head
bars[lead_idx].set_alpha(1.0)
bars[lead_idx].set_edgecolor("black")
bars[lead_idx].set_linewidth(1.5)
ax.set_yticks(range(n_total))
ax.set_yticklabels(head_labels, fontsize=7)
ax.set_xlabel("Peak attn-to-$z$", fontsize=8)
ax.set_title("(b) Post-transition", fontsize=9)
ax.invert_yaxis()

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(SAVE_DIR / f"fig_head_nucleation.{ext}", dpi=300, bbox_inches="tight")
plt.close()

print(f"\nFigure saved: {SAVE_DIR / 'fig_head_nucleation.pdf'}")
