#!/usr/bin/env python
"""
Architecture ablation: compare dissipation scaling across architectures.

Produces a 3-panel figure:
  A: Q_transition vs log(K) for Transformer, Gated MLP, RNN
  B: Landauer constant c (slope of Q vs log K) bar chart
  C: Plateau duration vs K for all architectures

Usage:
    python scripts/plot_architecture_comparison.py [--output-dir outputs]
"""

import sys
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.compute_landauer_cost import process_experiment


# ------------------------------------------------------------------
# Experiment groups
# ------------------------------------------------------------------
ARCH_GROUPS = {
    "Transformer": {
        "experiments": [
            "landauer_k10", "landauer_k15", "landauer_k20",
            "landauer_k25", "landauer_k36",
            "landauer_k50", "landauer_k75", "landauer_k100", "landauer_k150",
        ],
        "color": "#2196F3",
        "marker": "o",
    },
    "Gated MLP": {
        "experiments": [
            "gatedmlp_k10", "gatedmlp_k15", "gatedmlp_k20",
            "gatedmlp_k25", "gatedmlp_k36",
            "gatedmlp_k50", "gatedmlp_k75", "gatedmlp_k100", "gatedmlp_k150",
        ],
        "color": "#FF5722",
        "marker": "s",
    },
    "RNN (LSTM)": {
        "experiments": [
            "rnn_k10", "rnn_k15", "rnn_k20",
            "rnn_k25", "rnn_k36",
            "rnn_k50", "rnn_k75", "rnn_k100", "rnn_k150",
        ],
        "color": "#4CAF50",
        "marker": "D",
    },
}


def gather_results(output_dir: str):
    """Process all experiments and group by architecture."""
    all_results = {}
    for arch_name, info in ARCH_GROUPS.items():
        results = []
        for exp_name in info["experiments"]:
            exp_dir = Path(output_dir) / exp_name
            if not (exp_dir / "config.yaml").exists():
                continue
            result = process_experiment(exp_name, output_dir)
            if result is not None:
                results.append(result)
        results.sort(key=lambda r: r["K"])
        all_results[arch_name] = results
        n_found = len(results)
        n_total = len(info["experiments"])
        print(f"  {arch_name}: {n_found}/{n_total} experiments found")
    return all_results


def linear_fit(log_ks, q_trans):
    """OLS fit + R²."""
    if len(log_ks) < 2:
        return None, None, None, None
    coeffs = np.polyfit(log_ks, q_trans, 1)
    slope, intercept = coeffs
    predicted = np.polyval(coeffs, log_ks)
    ss_res = np.sum((q_trans - predicted) ** 2)
    ss_tot = np.sum((q_trans - np.mean(q_trans)) ** 2)
    r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return slope, intercept, r_sq, coeffs


# ------------------------------------------------------------------
# Panels
# ------------------------------------------------------------------
def panel_a(ax, all_results):
    """Q_transition vs log(K) with per-architecture linear fits."""
    fit_info = {}

    for arch_name, info in ARCH_GROUPS.items():
        results = all_results.get(arch_name, [])
        valid = [r for r in results if r["Q_transition"] is not None]
        if not valid:
            continue

        log_ks = np.array([r["log_K"] for r in valid])
        q_trans = np.array([r["Q_transition"] for r in valid])
        color = info["color"]
        marker = info["marker"]

        # Data points
        ax.scatter(log_ks, q_trans, color=color, marker=marker, s=80,
                   zorder=5, edgecolors="black", linewidths=0.7,
                   label=arch_name)

        # Linear fit
        slope, intercept, r_sq, coeffs = linear_fit(log_ks, q_trans)
        if slope is not None:
            x_fit = np.linspace(min(log_ks) - 0.1, max(log_ks) + 0.1, 100)
            y_fit = np.polyval(coeffs, x_fit)
            ax.plot(x_fit, y_fit, color=color, linewidth=1.5, alpha=0.7,
                    linestyle="--")
            fit_info[arch_name] = (slope, intercept, r_sq)

    # Annotate fits
    y_text = 0.97
    for arch_name, (slope, intercept, r_sq) in fit_info.items():
        color = ARCH_GROUPS[arch_name]["color"]
        ax.text(0.03, y_text,
                f"{arch_name}: c={slope:.3f}, R²={r_sq:.3f}",
                transform=ax.transAxes, fontsize=8, color=color,
                verticalalignment="top",
                fontweight="bold")
        y_text -= 0.06

    ax.set_xlabel("log(K)", fontsize=10)
    ax.set_ylabel(r"$Q_{\mathrm{transition}}$", fontsize=10)
    ax.set_title("A.  Transition Dissipation vs log(K)", fontsize=11,
                 fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)


def panel_b(ax, all_results):
    """Landauer constant c (slope) as bar chart."""
    arch_names = []
    slopes = []
    r_sqs = []
    colors = []

    for arch_name, info in ARCH_GROUPS.items():
        results = all_results.get(arch_name, [])
        valid = [r for r in results if r["Q_transition"] is not None]
        if len(valid) < 2:
            continue

        log_ks = np.array([r["log_K"] for r in valid])
        q_trans = np.array([r["Q_transition"] for r in valid])
        slope, _, r_sq, _ = linear_fit(log_ks, q_trans)

        arch_names.append(arch_name)
        slopes.append(slope)
        r_sqs.append(r_sq)
        colors.append(info["color"])

    if not slopes:
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                ha="center", fontsize=12)
        ax.set_title("B.  Landauer Constant c", fontsize=11, fontweight="bold")
        return

    x = np.arange(len(arch_names))
    bars = ax.bar(x, slopes, color=colors, edgecolor="black", linewidth=0.8,
                  width=0.5)

    for i, (bar, r_sq) in enumerate(zip(bars, r_sqs)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"R²={r_sq:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(arch_names, fontsize=9)
    ax.set_ylabel("Slope c  (Q / log K)", fontsize=10)
    ax.set_title("B.  Landauer Constant c  (slope of Q vs log K)",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(bottom=0)


def panel_c(ax, all_results):
    """Metastable plateau duration vs K.

    Candidate_loss is normalized over the K-candidate set, so it starts at
    approximately log(K) for all K values even at random init.  This means
    the model enters the metastable plateau essentially immediately (by the
    first checkpoint), and the "learning the marginal" transient that would
    bias raw cross-entropy measurements is absent from candidate_loss.

    Duration = transition_start − plateau_onset, where plateau_onset is
    the first step where candidate_loss < 1.1·log(K) and transition_start
    is the last step where candidate_loss > 0.9·log(K).  In practice
    plateau_onset ≈ first checkpoint for all experiments, so this is
    effectively transition_start minus a small constant.
    """
    for arch_name, info in ARCH_GROUPS.items():
        results = all_results.get(arch_name, [])
        valid = [r for r in results
                 if r.get("plateau_duration") is not None
                 and r["plateau_duration"] > 0]
        if not valid:
            continue

        ks = [r["K"] for r in valid]
        durations = [r["plateau_duration"] for r in valid]
        color = info["color"]
        marker = info["marker"]

        ax.plot(ks, durations, color=color, marker=marker, markersize=7,
                linewidth=1.5, label=arch_name, markeredgecolor="black",
                markeredgewidth=0.6)

    ax.set_xlabel("K", fontsize=10)
    ax.set_ylabel("Metastable Plateau Duration (steps)", fontsize=9)
    ax.set_title("C.  Plateau Duration vs K", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Architecture ablation comparison plot"
    )
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--save-path", type=str,
                        default="outputs/figures/architecture_comparison.png")
    args = parser.parse_args()

    print("Gathering Landauer results for all architectures...")
    all_results = gather_results(args.output_dir)

    # Print summary table
    print("\n" + "=" * 70)
    print("Architecture Ablation — Dissipation Scaling Summary")
    print("=" * 70)
    for arch_name, results in all_results.items():
        valid = [r for r in results if r["Q_transition"] is not None]
        if not valid:
            print(f"\n{arch_name}: no transition data")
            continue
        log_ks = np.array([r["log_K"] for r in valid])
        q_trans = np.array([r["Q_transition"] for r in valid])
        slope, intercept, r_sq, _ = linear_fit(log_ks, q_trans)
        print(f"\n{arch_name}:")
        print(f"  K values with data: {[r['K'] for r in valid]}")
        if slope is not None:
            print(f"  Linear fit: Q = {slope:.4f} · log(K) + {intercept:.4f}")
            print(f"  R² = {r_sq:.4f}")
            ratios = q_trans / log_ks
            print(f"  Q/log(K) ratio: mean={np.mean(ratios):.4f}, "
                  f"std={np.std(ratios):.4f}, "
                  f"CV={np.std(ratios)/np.mean(ratios):.3f}")
    print("=" * 70)

    # Create figure
    fig = plt.figure(figsize=(16, 5.5))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.32)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    panel_a(ax_a, all_results)
    panel_b(ax_b, all_results)
    panel_c(ax_c, all_results)

    fig.suptitle(
        "Architecture Ablation: Is Q ∝ log(K) Universal or Transformer-Specific?",
        fontsize=13, fontweight="bold", y=1.02,
    )

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved figure to: {save_path}")
    plt.close()


if __name__ == "__main__":
    main()
