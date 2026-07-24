#!/usr/bin/env python
"""
Plot results for Experiment 5 — Multihead Specialization.

Produces:
  1. Heatmap of α^z_{l,h} over training steps.
  2. Per-head logit advantage over training steps.
  3. MLP vs attention logit contribution over training.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = Path("outputs")
FIG_DIR = OUTPUT_DIR / "figures" / "exp5"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_probe_results():
    path = OUTPUT_DIR / "exp5_multihead" / "multihead_probe_results.json"
    if not path.exists():
        print(f"[Exp 5] No probe results found. Run run_exp5_multihead.py first.")
        return None
    with open(path) as f:
        return json.load(f)


def plot_all():
    data = load_probe_results()
    if data is None:
        return

    steps = [d["step"] for d in data]
    n_layers = len(data[0]["attention_to_z"])
    n_heads = len(data[0]["attention_to_z"][0])

    # --- Heatmap: attention to z per (layer, head) over time ---
    attn_z = np.array([d["attention_to_z"] for d in data])  # (T, L, H)
    T = len(steps)
    n_units = n_layers * n_heads

    fig, ax = plt.subplots(figsize=(12, 4))
    # Reshape to (T, L*H) and transpose for heatmap
    mat = attn_z.reshape(T, n_units).T
    labels = [f"L{l}H{h}" for l in range(n_layers) for h in range(n_heads)]
    im = ax.imshow(mat, aspect="auto", origin="lower",
                   extent=[steps[0], steps[-1], -0.5, n_units - 0.5])
    ax.set_yticks(range(n_units))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("Training step")
    ax.set_title("Attention to z per layer×head")
    plt.colorbar(im, ax=ax, label="α^z")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "exp5_attn_z_heatmap.png", dpi=200)
    fig.savefig(FIG_DIR / "exp5_attn_z_heatmap.pdf", dpi=200)
    plt.close(fig)
    print(f"Saved {FIG_DIR / 'exp5_attn_z_heatmap.png'}")

    # --- Per-head logit advantage over time ---
    advantage = np.array([d["head_logit_advantage"] for d in data])  # (T, L, H)
    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 4), sharey=True)
    if n_layers == 1:
        axes = [axes]
    for l in range(n_layers):
        for h in range(n_heads):
            axes[l].plot(steps, advantage[:, l, h], label=f"H{h}")
        axes[l].set_xlabel("Step")
        axes[l].set_title(f"Layer {l}")
        axes[l].legend(fontsize=7)
    axes[0].set_ylabel("Head logit advantage")
    plt.suptitle("Per-head logit advantage for correct answer via z", fontsize=11)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "exp5_head_advantage.png", dpi=200)
    fig.savefig(FIG_DIR / "exp5_head_advantage.pdf", dpi=200)
    plt.close(fig)
    print(f"Saved {FIG_DIR / 'exp5_head_advantage.png'}")

    # --- MLP vs Attention logit contribution ---
    mlp_adv = np.array([d["mlp_logit_advantage"] for d in data])  # (T, L)
    attn_adv = np.array([d["attn_layer_logit_advantage"] for d in data])  # (T, L)

    fig, ax = plt.subplots(figsize=(10, 5))
    for l in range(n_layers):
        ax.plot(steps, attn_adv[:, l], label=f"Attn L{l}", ls="-")
        ax.plot(steps, mlp_adv[:, l], label=f"MLP L{l}", ls="--")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Logit advantage (correct − mean)")
    ax.set_title("Attention vs MLP logit contribution per layer")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "exp5_mlp_vs_attn.png", dpi=200)
    fig.savefig(FIG_DIR / "exp5_mlp_vs_attn.pdf", dpi=200)
    plt.close(fig)
    print(f"Saved {FIG_DIR / 'exp5_mlp_vs_attn.png'}")

    # --- Head activation timing (5C) ---
    print("\n" + "="*50)
    print("Experiment 5: Head Activation Timing")
    print("="*50)
    # For each head, compute plateau-mean α^z and find first step > 2× that
    n_plateau = max(1, T // 3)
    plateau_mean = attn_z[:n_plateau].mean(axis=0)  # (L, H)
    for l in range(n_layers):
        for h in range(n_heads):
            threshold = 2 * plateau_mean[l, h]
            activation_step = None
            for t_idx, s in enumerate(steps):
                if attn_z[t_idx, l, h] > threshold:
                    activation_step = s
                    break
            act_str = str(activation_step) if activation_step is not None else "never"
            print(f"  L{l}H{h}: plateau_mean={plateau_mean[l,h]:.4f}  "
                  f"activation_step={act_str}")


if __name__ == "__main__":
    plot_all()
