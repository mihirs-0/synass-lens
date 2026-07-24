#!/usr/bin/env python3
"""
Experiment 3: Weight Displacement & Hidden Progress (Papers: 2405.19454, 2512.00686)

Tests whether weight displacement per step ||W(t+Δ) - W(t)|| is nonzero during
the loss plateau (hidden progress), and whether displacement accelerates near τ.

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


def compute_tau(run_name, k, threshold_frac=0.5):
    p = OUTPUTS / run_name / "training_history.json"
    if not p.exists():
        return None
    with open(p) as f:
        h = json.load(f)
    log_k = math.log(k)
    for s, l in zip(h["steps"], h["first_target_loss"]):
        if l < threshold_frac * log_k:
            return s
    return None


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
    return "other"


# ── Main analysis ────────────────────────────────────────────────────────────

print("=" * 80)
print("Experiment 3: Weight Displacement & Hidden Progress")
print("=" * 80)

k_runs = {
    10: "landauer_dense_k10",
    20: "landauer_dense_k20",
    36: "landauer_dense_k36",
}

SAMPLE_EVERY = 200  # use every 200th step for displacement

all_data = {}

for k, run_name in k_runs.items():
    ckpt_dir = OUTPUTS / run_name / "checkpoints"
    if not ckpt_dir.exists():
        continue

    tau = compute_tau(run_name, k)
    step_dirs = sorted([d for d in ckpt_dir.iterdir()
                        if d.is_dir() and d.name.startswith("step_")])

    # Filter to sampled steps
    filtered = []
    for d in step_dirs:
        step = int(d.name.split("_")[1])
        if step % SAMPLE_EVERY == 0:
            filtered.append((step, d))

    steps = []
    total_disp = []
    relative_disp = []
    comp_disp = {b: [] for b in ["embed", "attention", "mlp", "unembed"]}
    cosine_sims = []

    prev_state = None
    prev_delta = None

    for step, d in filtered:
        model_path = d / "model.pt"
        if not model_path.exists():
            continue

        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)

        if prev_state is not None:
            # Compute displacement
            total_sq = 0.0
            norm_sq = 0.0
            comp_sq = {b: 0.0 for b in comp_disp}

            # Current delta vector (flattened)
            delta_parts = []

            for name in state_dict:
                if "mask" in name or "IGNORE" in name:
                    continue
                curr = state_dict[name].float()
                prev = prev_state[name].float()
                diff = curr - prev

                diff_sq = diff.norm().item() ** 2
                total_sq += diff_sq
                norm_sq += prev.norm().item() ** 2

                bucket = bucket_param(name)
                if bucket in comp_sq:
                    comp_sq[bucket] += diff_sq

                delta_parts.append(diff.flatten())

            steps.append(step)
            total_disp.append(math.sqrt(total_sq))
            relative_disp.append(math.sqrt(total_sq / norm_sq) if norm_sq > 0 else 0)

            for b in comp_disp:
                comp_disp[b].append(math.sqrt(comp_sq[b]))

            # Direction consistency: cosine similarity with previous delta
            current_delta = torch.cat(delta_parts)
            if prev_delta is not None:
                cos_sim = torch.nn.functional.cosine_similarity(
                    current_delta.unsqueeze(0), prev_delta.unsqueeze(0)
                ).item()
                cosine_sims.append(cos_sim)
            else:
                cosine_sims.append(float("nan"))

            prev_delta = current_delta

        prev_state = {name: state_dict[name].clone() for name in state_dict}

    all_data[k] = {
        "steps": np.array(steps),
        "total_disp": np.array(total_disp),
        "relative_disp": np.array(relative_disp),
        "components": {b: np.array(v) for b, v in comp_disp.items()},
        "cosine_sim": np.array(cosine_sims),
        "tau": tau,
    }
    print(f"K={k}: {len(steps)} displacement measurements, τ={tau}")

# ── Print analysis ────────────────────────────────────────────────────────────

for k in sorted(all_data.keys()):
    d = all_data[k]
    tau = d["tau"]
    if tau is None:
        continue

    print(f"\nK={k} (τ={tau}):")

    # Plateau phase: steps < tau/2
    # Pre-transition: tau/2 < steps < tau
    # Post-transition: steps > tau
    plateau_mask = d["steps"] < tau * 0.5
    pretrans_mask = (d["steps"] >= tau * 0.5) & (d["steps"] < tau)
    posttrans_mask = d["steps"] >= tau

    for label, mask in [("Plateau", plateau_mask), ("Pre-transition", pretrans_mask),
                         ("Post-transition", posttrans_mask)]:
        if mask.sum() == 0:
            continue
        disp = d["total_disp"][mask]
        rel = d["relative_disp"][mask]
        cos = d["cosine_sim"][mask]
        cos_valid = cos[~np.isnan(cos)]
        print(f"  {label}: mean disp={disp.mean():.4f} (±{disp.std():.4f}), "
              f"relative={rel.mean():.6f}, "
              f"cos_sim={cos_valid.mean():.3f}" if len(cos_valid) > 0 else
              f"  {label}: mean disp={disp.mean():.4f} (±{disp.std():.4f}), "
              f"relative={rel.mean():.6f}")

# ── Figure ───────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(9, 7))

colors_k = {10: "#3498DB", 20: "#E67E22", 36: "#27AE60"}

# Panel (a): Total displacement vs step
ax = axes[0, 0]
for k in sorted(all_data.keys()):
    d = all_data[k]
    ax.plot(d["steps"], d["total_disp"], color=colors_k[k],
            label=f"K={k}", linewidth=1)
    if d["tau"]:
        ax.axvline(d["tau"], color=colors_k[k], linestyle=":", alpha=0.5)
ax.set_xlabel("Step")
ax.set_ylabel(r"$\|W(t+\Delta) - W(t)\|_F$")
ax.set_title("(a) Weight displacement per step")
ax.legend(fontsize=7)
ax.set_yscale("log")

# Panel (b): Relative displacement
ax = axes[0, 1]
for k in sorted(all_data.keys()):
    d = all_data[k]
    ax.plot(d["steps"], d["relative_disp"], color=colors_k[k],
            label=f"K={k}", linewidth=1)
    if d["tau"]:
        ax.axvline(d["tau"], color=colors_k[k], linestyle=":", alpha=0.5)
ax.set_xlabel("Step")
ax.set_ylabel(r"$\|W(t+\Delta)-W(t)\|/\|W(t)\|$")
ax.set_title("(b) Relative displacement")
ax.legend(fontsize=7)
ax.set_yscale("log")

# Panel (c): Per-component displacement for K=20
ax = axes[1, 0]
d20 = all_data.get(20)
if d20 is not None:
    comp_colors = {"embed": "#3498DB", "attention": "#E74C3C",
                   "mlp": "#27AE60", "unembed": "#E67E22"}
    for comp, vals in d20["components"].items():
        ax.plot(d20["steps"], vals, color=comp_colors.get(comp, "gray"),
                label=comp, linewidth=1)
    if d20["tau"]:
        ax.axvline(d20["tau"], color="black", linestyle="--", alpha=0.5,
                   label=rf"$\tau$={d20['tau']}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Component displacement")
    ax.set_title("(c) Per-component displacement (K=20)")
    ax.legend(fontsize=6)
    ax.set_yscale("log")

# Panel (d): Direction consistency (cosine similarity)
ax = axes[1, 1]
for k in sorted(all_data.keys()):
    d = all_data[k]
    valid = ~np.isnan(d["cosine_sim"])
    if valid.sum() > 0:
        # Smooth with rolling window
        window = 5
        if valid.sum() > window:
            cos_smooth = np.convolve(d["cosine_sim"][valid],
                                      np.ones(window) / window, mode="valid")
            steps_smooth = d["steps"][valid][:len(cos_smooth)]
            ax.plot(steps_smooth, cos_smooth, color=colors_k[k],
                    label=f"K={k}", linewidth=1)
        else:
            ax.plot(d["steps"][valid], d["cosine_sim"][valid],
                    color=colors_k[k], label=f"K={k}", linewidth=1)
    if d["tau"]:
        ax.axvline(d["tau"], color=colors_k[k], linestyle=":", alpha=0.5)
ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
ax.set_xlabel("Step")
ax.set_ylabel("Cosine similarity of consecutive deltas")
ax.set_title("(d) Direction consistency")
ax.legend(fontsize=7)
ax.set_ylim(-0.5, 1.0)

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(SAVE_DIR / f"fig_weight_displacement.{ext}", dpi=300, bbox_inches="tight")
plt.close()

print(f"\nFigure saved: {SAVE_DIR / 'fig_weight_displacement.pdf'}")
