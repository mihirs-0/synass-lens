#!/usr/bin/env python3
"""Extract individual panels from composite figures for the paper."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
SAVE_DIR = OUTPUTS / "paper_figures"

# ── Panel (a): Synergy onset vs loss onset (from fig_synergy_proxy panel b) ──

def load_history(name):
    p = OUTPUTS / name / "training_history.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None

def find_onset(values, steps, threshold):
    for s, v in zip(steps, values):
        if v > threshold:
            return s
    return None

k_values = [3, 5, 7, 10, 13, 17, 20, 25, 30, 36]
results = []

for k in k_values:
    h = load_history(f"landauer_dense_k{k}")
    if h is None:
        continue
    steps = np.array(h["steps"])
    loss = np.array(h["first_target_loss"])
    z_shuf = np.array(h["loss_z_shuffled"])
    log_k = math.log(k)
    z_gap = z_shuf - loss

    loss_onset = None
    for s, l in zip(steps, loss):
        if l < 0.9 * log_k:
            loss_onset = int(s)
            break
    synergy_onset = find_onset(z_gap, steps, 0.1)

    if loss_onset and synergy_onset:
        results.append({"K": k, "loss_onset": loss_onset, "synergy_onset": synergy_onset})

fig, ax = plt.subplots(figsize=(3.5, 3.2))
loss_ons = [r["loss_onset"] for r in results]
syn_ons = [r["synergy_onset"] for r in results]
ks_plot = [r["K"] for r in results]
sc = ax.scatter(loss_ons, syn_ons, c=[math.log(k) for k in ks_plot],
                cmap="viridis", s=50, zorder=3, edgecolors="k", linewidths=0.3)
max_val = max(max(loss_ons), max(syn_ons)) * 1.1
ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3, label=r"$t_\mathrm{syn}=t_\mathrm{loss}$")
for r in results:
    ax.annotate(f"K={r['K']}", (r["loss_onset"], r["synergy_onset"]),
                fontsize=5.5, ha="left", va="bottom")
ax.set_xlabel("Loss onset step", fontsize=9)
ax.set_ylabel("Synergy onset step", fontsize=9)
ax.legend(fontsize=7, loc="upper left")
ax.tick_params(labelsize=7)
plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(SAVE_DIR / f"fig_synergy_proxy_panel_b.{ext}", dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: fig_synergy_proxy_panel_b.pdf")

# ── Panel (b): Direction consistency (from fig_weight_displacement panel d) ──

import torch

k_runs = {10: "landauer_dense_k10", 20: "landauer_dense_k20", 36: "landauer_dense_k36"}
SAMPLE_EVERY = 200
colors_k = {10: "#3498DB", 20: "#E67E22", 36: "#27AE60"}

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

def bucket_param(name):
    n = name.lower()
    if "unembed" in n: return "unembed"
    if "embed" in n: return "embed"
    if "attn" in n: return "attention"
    if "mlp" in n: return "mlp"
    return "other"

fig, ax = plt.subplots(figsize=(3.5, 3.2))

for k, run_name in k_runs.items():
    ckpt_dir = OUTPUTS / run_name / "checkpoints"
    if not ckpt_dir.exists():
        continue
    tau = compute_tau(run_name, k)
    step_dirs = sorted([d for d in ckpt_dir.iterdir()
                        if d.is_dir() and d.name.startswith("step_")])
    filtered = [(int(d.name.split("_")[1]), d) for d in step_dirs
                if int(d.name.split("_")[1]) % SAMPLE_EVERY == 0]

    steps, cosine_sims = [], []
    prev_state, prev_delta = None, None

    for step, d in filtered:
        model_path = d / "model.pt"
        if not model_path.exists():
            continue
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)

        if prev_state is not None:
            delta_parts = []
            for name in state_dict:
                if "mask" in name or "IGNORE" in name:
                    continue
                diff = state_dict[name].float() - prev_state[name].float()
                delta_parts.append(diff.flatten())
            current_delta = torch.cat(delta_parts)
            if prev_delta is not None:
                cos_sim = torch.nn.functional.cosine_similarity(
                    current_delta.unsqueeze(0), prev_delta.unsqueeze(0)).item()
                steps.append(step)
                cosine_sims.append(cos_sim)
            prev_delta = current_delta
        prev_state = {name: state_dict[name].clone() for name in state_dict}

    steps = np.array(steps)
    cosine_sims = np.array(cosine_sims)

    # Smooth
    window = 5
    if len(cosine_sims) > window:
        cos_smooth = np.convolve(cosine_sims, np.ones(window)/window, mode="valid")
        steps_smooth = steps[:len(cos_smooth)]
        ax.plot(steps_smooth, cos_smooth, color=colors_k[k],
                label=f"K={k}", linewidth=1.2)
    if tau:
        ax.axvline(tau, color=colors_k[k], linestyle=":", alpha=0.5)

ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
ax.set_xlabel("Step", fontsize=9)
ax.set_ylabel(r"$\cos(\Delta W_t, \Delta W_{t-1})$", fontsize=9)
ax.legend(fontsize=7)
ax.set_ylim(-0.5, 1.0)
ax.tick_params(labelsize=7)
plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(SAVE_DIR / f"fig_displacement_panel_d.{ext}", dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: fig_displacement_panel_d.pdf")
