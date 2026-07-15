#!/usr/bin/env python
"""4-panel saddle probe figure for HiLD paper.

Panels: (a) gradient norm M1, (b) Hessian λ_min M2, (c) gradient–v_min angle M3,
(d) per-example coherence M4 vs random baseline.

Each panel: 3 seeds × 3 plateau checkpoints (steps 1000, 1500, 2350).
Output: hild_paper/fig_saddle_probe_hild.{pdf,png}
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
SADDLE = REPO / "eta_sweep" / "results" / "saddle_probe"

seeds = [0, 1, 2]
data = {}
for s in seeds:
    f = SADDLE / f"saddle_probe_results_seed{s}.json"
    data[s] = json.loads(f.read_text())

steps = sorted({c["actual_step"] for s in seeds for c in data[s]["checkpoints"]})
# Build per-seed series by checkpoint
def col(seed, key):
    return [c[key] if "M3_angle_deg" not in key else c["M3_angle_deg"]
            for c in data[seed]["checkpoints"]]

fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.2))
seed_colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}

# ── (a) gradient norm M1 ─────────────────────────────────────────
ax = axes[0]
for s in seeds:
    xs = [c["actual_step"] for c in data[s]["checkpoints"]]
    ys = [c["M1_grad_norm"] for c in data[s]["checkpoints"]]
    ax.plot(xs, ys, "o-", color=seed_colors[s], lw=1.6, ms=7, label=f"seed {s}")
ax.set_xlabel("training step")
ax.set_ylabel(r"$\|\nabla L\|$  (full-batch)")
ax.set_title(r"(a) M1: gradient norm")
ax.axhline(0, color="grey", lw=0.7, alpha=0.4)
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.25, lw=0.4)
ax.legend(fontsize=8, loc="lower right")

# ── (b) Hessian λ_min M2 ─────────────────────────────────────────
ax = axes[1]
for s in seeds:
    xs = [c["actual_step"] for c in data[s]["checkpoints"]]
    ys = [c["M2_lambda_min"] for c in data[s]["checkpoints"]]
    ax.plot(xs, ys, "o-", color=seed_colors[s], lw=1.6, ms=7)
ax.axhline(0, color="black", lw=0.8, alpha=0.6, linestyle=":")
ax.set_xlabel("training step")
ax.set_ylabel(r"$\lambda_{\min}(H)$")
ax.set_title(r"(b) M2: $\lambda_{\min}(H)$ (Lanczos)")
ax.grid(True, alpha=0.25, lw=0.4)

# ── (c) gradient–v_min angle M3 ──────────────────────────────────
ax = axes[2]
for s in seeds:
    xs = [c["actual_step"] for c in data[s]["checkpoints"]]
    ys = [c["M3_angle_deg"] for c in data[s]["checkpoints"]]
    ax.plot(xs, ys, "o-", color=seed_colors[s], lw=1.6, ms=7)
ax.axhline(90, color="black", lw=0.8, alpha=0.6, linestyle=":", label=r"$90^\circ$")
ax.set_xlabel("training step")
ax.set_ylabel(r"angle$(\nabla L, v_{\min})$  [deg]")
ax.set_title(r"(c) M3: $\angle(\nabla L, v_{\min})$")
ax.set_ylim(80, 100)
ax.grid(True, alpha=0.25, lw=0.4)
ax.legend(fontsize=8, loc="lower right")

# ── (d) per-example coherence M4 ─────────────────────────────────
ax = axes[3]
for s in seeds:
    xs = [c["actual_step"] for c in data[s]["checkpoints"]]
    ys = [c["M4"]["coherence"] for c in data[s]["checkpoints"]]
    ax.plot(xs, ys, "o-", color=seed_colors[s], lw=1.6, ms=7)
# Random baseline (1/n_examples)
random_baseline = data[0]["checkpoints"][0]["M4"]["random_baseline"]
ax.axhline(random_baseline, color="grey", linestyle="--", lw=1.0, alpha=0.7,
           label=f"random baseline = {random_baseline:.4f}")
ax.set_xlabel("training step")
ax.set_ylabel("per-example coherence")
ax.set_title(r"(d) M4: per-example gradient coherence")
ax.set_ylim(0, 0.005)
ax.grid(True, alpha=0.25, lw=0.4)
ax.legend(fontsize=8, loc="upper right")

plt.tight_layout()
out = REPO / "hild_paper" / "fig_saddle_probe_hild"
fig.savefig(str(out) + ".pdf", dpi=200)
fig.savefig(str(out) + ".png", dpi=160)
print(f"Figure: {out}.pdf")
