#!/usr/bin/env python
"""Combined mechanism figure: 9-cell decomposition + v_t pathway asymmetry.

Output: hild_paper/fig_mechanism_hild.{pdf,png}
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]

# --- 9-cell decomposition data ---
nine = json.loads((REPO / "eta_sweep" / "results" / "optimizer_decomposition_9cell.json").read_text())
ROW_ORDER = [
    "adamw_b1_0", "adamw_b1_0_nobc", "adamw_b1_0_rmsprop_eps", "rmsprop_decoupled_bc",
    "rmsprop_decoupled_only", "adam_coupled_b1_0", "rmsprop_bc_only", "rmsprop", "rmsprop_adamw_eps",
]
SHORT_LABELS = {
    "adamw_b1_0":              "AdamW\n(decoupled)",
    "adamw_b1_0_nobc":         "AdamW\nno-BC",
    "adamw_b1_0_rmsprop_eps":  "AdamW\n+RMS-$\\epsilon$",
    "rmsprop_decoupled_bc":    "RMSProp\n+decoupled+BC",
    "rmsprop_decoupled_only":  "RMSProp\n+decoupled",
    "adam_coupled_b1_0":       "Adam\n(L2)",
    "rmsprop_bc_only":         "RMSProp\n+BC",
    "rmsprop":                 "RMSProp\n(baseline)",
    "rmsprop_adamw_eps":       "RMSProp\n+AdamW-$\\epsilon$",
}

rows = {r["config"]: r for r in nine["rows"]}

# --- v_aggregates: z-pipeline vs control mean v at canonical step ---
v_path = REPO / "eta_sweep" / "results" / "v_tracking" / "eta_0.001_K_10_seed_0" / "v_aggregates.jsonl"
v_lines = [json.loads(l) for l in v_path.read_text().strip().splitlines()]
# Pick the first entry near the plateau center (step ~1500)
target = min(v_lines, key=lambda e: abs(e["step"] - 1500))
print(f"v_t snapshot at step {target['step']}")
z_pipe = []
ctrl = []
labels_z = []
labels_c = []
for k, g in target["per_group"].items():
    if g["is_z_pipeline"]:
        z_pipe.append(g["v_mean"])
        labels_z.append(k)
    else:
        ctrl.append(g["v_mean"])
        labels_c.append(k)
print(f"z-pipeline modules ({len(z_pipe)}): mean v = {np.mean(z_pipe):.3e}")
print(f"control modules ({len(ctrl)}): mean v = {np.mean(ctrl):.3e}")
print(f"ratio (control / z-pipeline): {np.mean(ctrl)/np.mean(z_pipe):.2f}x")

# ── Figure ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13.5, 4.0))
gs = fig.add_gridspec(1, 4, width_ratios=[3.4, 1.0, 0.05, 1.6], wspace=0.25)
ax = fig.add_subplot(gs[0, 0])

# Panel a: 9-cell bar chart
xs = np.arange(len(ROW_ORDER))
width = 0.27
seeds = (0, 1, 2)
for s_idx, s in enumerate(seeds):
    taus = []
    for cfg in ROW_ORDER:
        t = rows[cfg]["taus"][s_idx]
        taus.append(6000 if t is None else t)
    offset = (s_idx - 1) * width
    colors = ["#2ca02c" if rows[c]["weight_decay"] == "decoupled" else "#d62728" for c in ROW_ORDER]
    bars = ax.bar(xs + offset, taus, width,
                  color=colors, alpha=0.55 + 0.18 * s_idx, edgecolor="black", lw=0.5)
    for b, t in zip(bars, taus):
        if t == 6000:
            ax.text(b.get_x() + b.get_width() / 2, 6100, "X",
                    ha="center", fontsize=7, color="darkred", weight="bold")
ax.axhline(2667, color="grey", linestyle=":", lw=0.8, alpha=0.7,
           label=r"baseline $\bar\tau \approx 2667$")
ax.axhline(1500, color="black", linestyle="--", lw=0.7, alpha=0.5,
           label="branch step (common ckpt)")
ax.set_xticks(xs)
ax.set_xticklabels([SHORT_LABELS[c] for c in ROW_ORDER], fontsize=7.5, rotation=15, ha="right")
ax.set_ylabel(r"$\tau$ from common pre-plateau checkpoint")
ax.set_title("(a) 9-condition decomposition: decoupled WD necessary and sufficient (3 seeds each)")
ax.set_ylim(0, 6500)
ax.legend(fontsize=8, loc="lower left")
ax.grid(True, axis="y", alpha=0.25, lw=0.4)

# Vertical dividers / shading by WD type
ax.axvspan(-0.5, 4.5, alpha=0.05, color="green")
ax.axvspan(4.5, 8.5, alpha=0.05, color="red")
ax.text(2.0, 5800, "decoupled WD: 4/4 transition", fontsize=8, color="darkgreen",
        ha="center", style="italic")
ax.text(6.5, 5800, "L2 WD: 5/5 stuck", fontsize=8, color="darkred",
        ha="center", style="italic")

# Panel b: v_t asymmetry
ax2 = fig.add_subplot(gs[0, 3])
mean_z = np.mean(z_pipe)
mean_c = np.mean(ctrl)
xs2 = [0, 1]
ax2.bar(xs2, [mean_z, mean_c],
        color=["#1f77b4", "#aaaaaa"], edgecolor="black", lw=0.6, width=0.6)
# Show individual modules
for v in z_pipe:
    ax2.plot(0, v, "o", color="black", alpha=0.4, ms=4)
for v in ctrl:
    ax2.plot(1, v, "o", color="black", alpha=0.4, ms=4)
ax2.set_xticks(xs2)
ax2.set_xticklabels(["z-pipeline\n(disambig.)", "non-pipeline\n(control)"], fontsize=8)
ax2.set_ylabel(r"per-module mean $v_t$")
ax2.set_yscale("log")
ax2.set_title(rf"(b) $v_t$ asymmetry at step {target['step']}")
ratio = mean_c / mean_z if mean_z > 0 else float("nan")
ax2.text(0.5, mean_c * 5, rf"control / pipeline$\approx${ratio:.1f}$\times$",
         ha="center", fontsize=8.5, color="black",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", lw=0.6))
ax2.grid(True, alpha=0.25, lw=0.4)

plt.tight_layout()
out = REPO / "hild_paper" / "fig_mechanism_hild"
fig.savefig(str(out) + ".pdf", dpi=200)
fig.savefig(str(out) + ".png", dpi=160)
print(f"\nFigure: {out}.pdf")
