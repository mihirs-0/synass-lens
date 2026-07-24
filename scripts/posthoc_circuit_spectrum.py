#!/usr/bin/env python3
"""
Experiment 2: OV/QK Circuit Spectrum & Effective Rank (Papers: 2405.19454, 2407.00886)

For each attention head, compute the singular value spectrum of W_OV = W_V @ W_O
and track effective rank across training. Tests whether the nucleating head (L1H3)
shows distinctive spectral changes before the loss transition.

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

N_LAYERS = 4
N_HEADS = 4
D_MODEL = 128
D_HEAD = 32


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


def effective_rank(singular_values):
    """Participation ratio: (Σσ)² / Σσ²."""
    s = np.array(singular_values)
    s = s[s > 1e-10]
    if len(s) == 0:
        return 0.0
    return (s.sum() ** 2) / (s ** 2).sum()


def spectral_analysis(state_dict):
    """Compute OV and QK spectral properties for all heads."""
    results = {}
    for layer in range(N_LAYERS):
        W_Q = state_dict[f"blocks.{layer}.attn.W_Q"].float()  # [n_heads, d_model, d_head]
        W_K = state_dict[f"blocks.{layer}.attn.W_K"].float()
        W_V = state_dict[f"blocks.{layer}.attn.W_V"].float()
        W_O = state_dict[f"blocks.{layer}.attn.W_O"].float()  # [n_heads, d_head, d_model]

        for head in range(N_HEADS):
            # W_OV = W_V @ W_O: [d_model, d_head] @ [d_head, d_model] = [d_model, d_model]
            W_OV = W_V[head] @ W_O[head]
            sv_ov = torch.linalg.svdvals(W_OV).numpy()

            # W_QK = W_Q @ W_K^T: [d_model, d_head] @ [d_head, d_model] = [d_model, d_model]
            W_QK = W_Q[head] @ W_K[head].T
            sv_qk = torch.linalg.svdvals(W_QK).numpy()

            key = (layer, head)
            results[key] = {
                "ov_sv": sv_ov,
                "qk_sv": sv_qk,
                "ov_eff_rank": effective_rank(sv_ov),
                "qk_eff_rank": effective_rank(sv_qk),
                "ov_top_ratio": sv_ov[0] / sv_ov[1] if sv_ov[1] > 1e-10 else float("inf"),
                "ov_nuclear": sv_ov.sum(),
                "ov_spectral": sv_ov[0],
            }
    return results


# ── Main analysis ────────────────────────────────────────────────────────────

print("=" * 80)
print("Experiment 2: OV/QK Circuit Spectrum")
print("=" * 80)

SAMPLE_EVERY = 200
k_runs = {
    10: "landauer_dense_k10",
    20: "landauer_dense_k20",
    36: "landauer_dense_k36",
}

all_data = {}

for k, run_name in k_runs.items():
    ckpt_dir = OUTPUTS / run_name / "checkpoints"
    if not ckpt_dir.exists():
        continue

    tau = compute_tau(run_name, k)
    step_dirs = sorted([d for d in ckpt_dir.iterdir()
                        if d.is_dir() and d.name.startswith("step_")])

    steps = []
    head_data = {(l, h): {"ov_eff_rank": [], "qk_eff_rank": [],
                           "ov_top_ratio": [], "ov_nuclear": [], "ov_spectral": []}
                 for l in range(N_LAYERS) for h in range(N_HEADS)}

    for d in step_dirs:
        step = int(d.name.split("_")[1])
        if step % SAMPLE_EVERY != 0 and step != 100:
            continue

        model_path = d / "model.pt"
        if not model_path.exists():
            continue

        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
        spec = spectral_analysis(state_dict)

        steps.append(step)
        for key in head_data:
            for metric in head_data[key]:
                head_data[key][metric].append(spec[key][metric])

    all_data[k] = {
        "steps": np.array(steps),
        "heads": {key: {m: np.array(v) for m, v in metrics.items()}
                  for key, metrics in head_data.items()},
        "tau": tau,
    }
    print(f"K={k}: {len(steps)} checkpoints, τ={tau}")

# ── Print L1H3 analysis ─────────────────────────────────────────────────────

print(f"\n{'='*60}")
print("L1H3 (nucleating head) spectral evolution at K=20:")
print(f"{'='*60}")
d20 = all_data.get(20)
if d20:
    l1h3 = d20["heads"][(1, 3)]
    tau = d20["tau"]

    # Find values at key timepoints
    for label, target_step in [("Init", 100), ("Mid-plateau", tau // 2 if tau else 5000),
                                ("At τ", tau), ("Final", d20["steps"][-1])]:
        if target_step is None:
            continue
        idx = np.argmin(np.abs(d20["steps"] - target_step))
        print(f"  {label} (step {d20['steps'][idx]}): "
              f"OV eff_rank={l1h3['ov_eff_rank'][idx]:.2f}, "
              f"OV σ₁/σ₂={l1h3['ov_top_ratio'][idx]:.2f}, "
              f"OV nuclear={l1h3['ov_nuclear'][idx]:.3f}")

    # Compare L1H3 to average of other heads
    print(f"\nComparison at final step:")
    for l in range(N_LAYERS):
        for h in range(N_HEADS):
            hd = d20["heads"][(l, h)]
            print(f"  L{l}H{h}: OV eff_rank={hd['ov_eff_rank'][-1]:.2f}, "
                  f"σ₁/σ₂={hd['ov_top_ratio'][-1]:.2f}")

# ── Figure ───────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(9, 7))

d20 = all_data.get(20)
if d20:
    # Panel (a): OV effective rank heatmap
    ax = axes[0, 0]
    n_total = N_LAYERS * N_HEADS
    n_steps = len(d20["steps"])
    rank_matrix = np.zeros((n_total, n_steps))
    for l in range(N_LAYERS):
        for h in range(N_HEADS):
            idx = l * N_HEADS + h
            rank_matrix[idx] = d20["heads"][(l, h)]["ov_eff_rank"]

    im = ax.imshow(rank_matrix, aspect="auto", cmap="viridis",
                   extent=[d20["steps"][0], d20["steps"][-1], n_total - 0.5, -0.5])
    head_labels = [f"L{l}H{h}" for l in range(N_LAYERS) for h in range(N_HEADS)]
    ax.set_yticks(range(n_total))
    ax.set_yticklabels(head_labels, fontsize=6)
    ax.set_xlabel("Step")
    ax.set_title("(a) OV effective rank (K=20)")
    if d20["tau"]:
        ax.axvline(d20["tau"], color="red", linestyle="--", linewidth=1, alpha=0.7)
    plt.colorbar(im, ax=ax, fraction=0.03)

    # Panel (b): σ₁/σ₂ for selected heads
    ax = axes[0, 1]
    highlight = [(1, 3, "L1H3 (lead)", "#E74C3C"),
                 (0, 0, "L0H0", "#3498DB"),
                 (2, 0, "L2H0", "#27AE60"),
                 (3, 0, "L3H0", "#9B59B6")]
    for l, h, label, color in highlight:
        ax.plot(d20["steps"], d20["heads"][(l, h)]["ov_top_ratio"],
                color=color, label=label, linewidth=1.2)
    if d20["tau"]:
        ax.axvline(d20["tau"], color="black", linestyle="--", alpha=0.5,
                   label=rf"$\tau$={d20['tau']}")
    ax.set_xlabel("Step")
    ax.set_ylabel(r"$\sigma_1 / \sigma_2$")
    ax.set_title(r"(b) OV dominant mode ratio (K=20)")
    ax.legend(fontsize=6)

    # Panel (c): OV nuclear norm for all heads at K=20
    ax = axes[1, 0]
    for l in range(N_LAYERS):
        for h in range(N_HEADS):
            alpha = 0.8 if (l, h) == (1, 3) else 0.2
            lw = 2 if (l, h) == (1, 3) else 0.5
            color = "#E74C3C" if (l, h) == (1, 3) else "#3498DB"
            label = "L1H3" if (l, h) == (1, 3) else (None if (l, h) != (0, 0) else "other heads")
            ax.plot(d20["steps"], d20["heads"][(l, h)]["ov_nuclear"],
                    color=color, alpha=alpha, linewidth=lw, label=label)
    if d20["tau"]:
        ax.axvline(d20["tau"], color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("OV nuclear norm")
    ax.set_title("(c) OV nuclear norm (K=20)")
    ax.legend(fontsize=7)

    # Panel (d): Effective rank comparison across K
    ax = axes[1, 1]
    colors_k = {10: "#3498DB", 20: "#E67E22", 36: "#27AE60"}
    for k_val, dk in all_data.items():
        # Plot L1H3 effective rank
        ax.plot(dk["steps"], dk["heads"][(1, 3)]["ov_eff_rank"],
                color=colors_k.get(k_val, "gray"), label=f"K={k_val}", linewidth=1.2)
        if dk["tau"]:
            ax.axvline(dk["tau"], color=colors_k.get(k_val, "gray"),
                       linestyle=":", alpha=0.4)
    ax.set_xlabel("Step")
    ax.set_ylabel("OV effective rank")
    ax.set_title("(d) L1H3 OV rank across K")
    ax.legend(fontsize=7)

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(SAVE_DIR / f"fig_circuit_spectrum.{ext}", dpi=300, bbox_inches="tight")
plt.close()

print(f"\nFigure saved: {SAVE_DIR / 'fig_circuit_spectrum.pdf'}")
