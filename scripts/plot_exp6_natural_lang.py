#!/usr/bin/env python
"""
Plot results for Experiment 6 — Natural Language Demonstration.

Produces:
  - Loss curves for K ∈ {2, 5, 10, 20} showing two-phase dynamics.
  - z-gap curves showing the transition.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = Path("outputs")
FIG_DIR = OUTPUT_DIR / "figures" / "exp6"
FIG_DIR.mkdir(parents=True, exist_ok=True)

K_VALUES = [2, 5, 10, 20]
SEEDS = [42, 43, 44]
COLORS = {2: "tab:blue", 5: "tab:orange", 10: "tab:green", 20: "tab:red"}


def load_history(name):
    p = OUTPUT_DIR / name / "training_history.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def mean_std(prefix, seeds, key):
    all_vals = []
    steps = None
    for s in seeds:
        h = load_history(f"{prefix}_seed{s}")
        if h is None:
            continue
        if steps is None:
            steps = np.array(h["steps"])
        all_vals.append(np.array(h[key]))
    if not all_vals:
        return None, None, None
    min_len = min(len(v) for v in all_vals)
    arr = np.array([v[:min_len] for v in all_vals])
    return steps[:min_len], arr.mean(0), arr.std(0)


def plot_all():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for k in K_VALUES:
        log_k = math.log(k) if k > 1 else 0.0
        color = COLORS[k]
        prefix = f"exp6_natlang_k{k}"

        # Loss
        steps, mean_loss, std_loss = mean_std(prefix, SEEDS, "first_target_loss")
        if steps is None:
            continue
        ax1.plot(steps, mean_loss, label=f"K={k}", color=color)
        ax1.fill_between(steps, mean_loss - std_loss, mean_loss + std_loss, alpha=0.15, color=color)
        if log_k > 0:
            ax1.axhline(log_k, ls=":", color=color, alpha=0.4, lw=0.8)

        # z-gap
        all_gaps = []
        for s in SEEDS:
            h = load_history(f"{prefix}_seed{s}")
            if h is None:
                continue
            zs = np.array(h["loss_z_shuffled"])
            fl = np.array(h["first_target_loss"])
            min_len = min(len(zs), len(fl))
            all_gaps.append(zs[:min_len] - fl[:min_len])
        if all_gaps:
            min_len = min(len(g) for g in all_gaps)
            gap_arr = np.array([g[:min_len] for g in all_gaps])
            ax2.plot(steps[:min_len], gap_arr.mean(0), label=f"K={k}", color=color)
            ax2.fill_between(steps[:min_len],
                             gap_arr.mean(0) - gap_arr.std(0),
                             gap_arr.mean(0) + gap_arr.std(0),
                             alpha=0.15, color=color)

    ax1.set_ylabel("First-target loss")
    ax1.legend()
    ax1.set_title("Experiment 6: Natural Language Entity Disambiguation")

    ax2.set_xlabel("Training step")
    ax2.set_ylabel("z-gap")
    ax2.legend()

    plt.tight_layout()
    fig.savefig(FIG_DIR / "exp6_natural_lang.pdf", dpi=200)
    fig.savefig(FIG_DIR / "exp6_natural_lang.png", dpi=200)
    plt.close(fig)
    print(f"Saved {FIG_DIR / 'exp6_natural_lang.png'}")


if __name__ == "__main__":
    plot_all()
