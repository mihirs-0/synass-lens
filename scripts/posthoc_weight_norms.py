#!/usr/bin/env python3
"""
Experiment 1: Weight Norm Dynamics (Papers: 2405.19454, 2505.13738)

Tests whether total weight norm exhibits non-monotonic behavior at the transition,
and whether per-component norms reveal reorganization dynamics.

Weights only — no forward passes.
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
SAVE_DIR = OUTPUTS / "paper_figures"
SAVE_DIR.mkdir(exist_ok=True)


def bucket_param(name: str) -> str:
    n = name.lower()
    if "unembed" in n:
        return "unembed"
    if "embed" in n:
        return "embed"
    if "attn" in n:
        return "attention"
    if "mlp" in n:
        return "mlp"
    if "ln" in n:
        return "layernorm"
    return "other"


def compute_tau(run_name, k, threshold_frac=0.5):
    p = OUTPUTS / run_name / "training_history.json"
    if not p.exists():
        return None
    with open(p) as f:
        h = json.load(f)
    log_k = math.log(k)
    threshold = threshold_frac * log_k
    for s, l in zip(h["steps"], h["first_target_loss"]):
        if l < threshold:
            return s
    return None


# ── Configuration ────────────────────────────────────────────────────────────

k_configs = {
    3: "landauer_dense_k3",
    5: "landauer_dense_k5",
    10: "landauer_dense_k10",
    20: "landauer_dense_k20",
    36: "landauer_dense_k36",
}

SAMPLE_EVERY = 500  # load every 500th step checkpoint

# ── Compute weight norms ─────────────────────────────────────────────────────

print("=" * 80)
print("Experiment 1: Weight Norm Dynamics")
print("=" * 80)

all_results = {}

for k, run_name in k_configs.items():
    ckpt_dir = OUTPUTS / run_name / "checkpoints"
    if not ckpt_dir.exists():
        print(f"  K={k}: no checkpoints dir")
        continue

    tau = compute_tau(run_name, k)
    step_dirs = sorted(ckpt_dir.iterdir())
    step_dirs = [d for d in step_dirs if d.is_dir() and d.name.startswith("step_")]

    steps = []
    total_norms = []
    component_norms = {b: [] for b in ["embed", "attention", "mlp", "layernorm", "unembed"]}
    layer_norms = {i: [] for i in range(4)}

    for d in step_dirs:
        step = int(d.name.split("_")[1])
        if step % SAMPLE_EVERY != 0 and step != 100:
            continue

        model_path = d / "model.pt"
        if not model_path.exists():
            continue

        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)

        total_sq = 0.0
        comp_sq = {b: 0.0 for b in component_norms}
        lay_sq = {i: 0.0 for i in range(4)}

        for name, param in state_dict.items():
            if "mask" in name or "IGNORE" in name:
                continue
            norm_sq = param.float().norm().item() ** 2
            total_sq += norm_sq

            bucket = bucket_param(name)
            if bucket in comp_sq:
                comp_sq[bucket] += norm_sq

            # Layer assignment
            for i in range(4):
                if f"blocks.{i}." in name:
                    lay_sq[i] += norm_sq
                    break

        steps.append(step)
        total_norms.append(math.sqrt(total_sq))
        for b in component_norms:
            component_norms[b].append(math.sqrt(comp_sq[b]))
        for i in range(4):
            layer_norms[i].append(math.sqrt(lay_sq[i]))

    all_results[k] = {
        "steps": np.array(steps),
        "total": np.array(total_norms),
        "components": {b: np.array(v) for b, v in component_norms.items()},
        "layers": {i: np.array(v) for i, v in layer_norms.items()},
        "tau": tau,
    }
    print(f"  K={k}: {len(steps)} checkpoints loaded, τ={tau}")

# ── Analysis ─────────────────────────────────────────────────────────────────

print(f"\n{'K':>4}  {'τ':>8}  {'||W|| at τ':>12}  {'||W|| final':>12}  "
      f"{'||W|| init':>12}  {'peak/init':>10}")
print("-" * 70)
for k in sorted(all_results.keys()):
    r = all_results[k]
    init_norm = r["total"][0]
    final_norm = r["total"][-1]
    peak_norm = r["total"].max()

    if r["tau"] is not None:
        idx_tau = np.argmin(np.abs(r["steps"] - r["tau"]))
        tau_norm = r["total"][idx_tau]
        tau_str = f"{tau_norm:.2f}"
    else:
        tau_str = "---"

    print(f"{k:>4}  {str(r['tau']):>8}  {tau_str:>12}  {final_norm:>12.2f}  "
          f"{init_norm:>12.2f}  {peak_norm/init_norm:>10.3f}")

# ── Figure ───────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

# Panel (a): Total weight norm vs step for all K
ax = axes[0]
colors = {3: "#95A5A6", 5: "#E74C3C", 10: "#3498DB", 20: "#E67E22", 36: "#27AE60"}
for k in sorted(all_results.keys()):
    r = all_results[k]
    ax.plot(r["steps"], r["total"], color=colors.get(k, "gray"),
            label=f"K={k}", linewidth=1.2)
    if r["tau"]:
        ax.axvline(r["tau"], color=colors.get(k, "gray"), linestyle=":", alpha=0.4)
ax.set_xlabel("Training step")
ax.set_ylabel(r"$\|W\|_F$")
ax.set_title("(a) Total weight norm")
ax.legend(fontsize=7)

# Panel (b): Per-component norms for K=20
ax = axes[1]
r20 = all_results.get(20)
if r20 is not None:
    comp_colors = {"embed": "#3498DB", "attention": "#E74C3C", "mlp": "#27AE60",
                   "layernorm": "#9B59B6", "unembed": "#E67E22"}
    for comp, vals in r20["components"].items():
        ax.plot(r20["steps"], vals, color=comp_colors.get(comp, "gray"),
                label=comp, linewidth=1.2)
    if r20["tau"]:
        ax.axvline(r20["tau"], color="black", linestyle="--", alpha=0.5,
                   label=rf"$\tau$={r20['tau']}")
    ax.set_xlabel("Training step")
    ax.set_ylabel(r"$\|W_{comp}\|_F$")
    ax.set_title("(b) Component norms (K=20)")
    ax.legend(fontsize=6)

# Panel (c): Per-layer norms for K=20
ax = axes[2]
if r20 is not None:
    layer_colors = {0: "#3498DB", 1: "#E74C3C", 2: "#27AE60", 3: "#9B59B6"}
    for i, vals in r20["layers"].items():
        ax.plot(r20["steps"], vals, color=layer_colors[i],
                label=f"Layer {i}", linewidth=1.2)
    if r20["tau"]:
        ax.axvline(r20["tau"], color="black", linestyle="--", alpha=0.5,
                   label=rf"$\tau$={r20['tau']}")
    ax.set_xlabel("Training step")
    ax.set_ylabel(r"$\|W_{layer}\|_F$")
    ax.set_title("(c) Per-layer norms (K=20)")
    ax.legend(fontsize=7)

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(SAVE_DIR / f"fig_weight_norms.{ext}", dpi=300, bbox_inches="tight")
plt.close()

print(f"\nFigure saved: {SAVE_DIR / 'fig_weight_norms.pdf'}")
