#!/usr/bin/env python
"""
Per-module distribution of Adam's v_t.

Reviewer 2 Q7: "could you provide per-module distributions to complement means?"

Builds histograms of per-parameter `v_mean` values within each module group
at two representative checkpoints (mid-plateau ≈ step 1500 and post-τ ≈ step 3500).
Output: outputs/paper_figures/fig_v_per_module.{pdf,png}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ETA_SWEEP_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ETA_SWEEP_ROOT.parent
OUT_DIR = REPO_ROOT / "outputs" / "paper_figures"

sys.path.insert(0, str(REPO_ROOT))
from eta_sweep.run_v_tracking import MODULE_GROUPS, Z_PIPELINE_GROUPS, group_for_param  # noqa: E402


def load_step(seed: int, target_step: int) -> dict:
    """Load the v_log entry closest to target_step."""
    path = ETA_SWEEP_ROOT / "results" / "v_tracking" / f"eta_0.001_K_10_seed_{seed}" / "v_log.jsonl"
    best = None
    best_diff = float("inf")
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            d = abs(r["step"] - target_step)
            if d < best_diff:
                best_diff = d
                best = r
    return best


def collect_group_values(snap: dict, key: str = "v_mean") -> dict:
    """Bin per-parameter values by module group."""
    out: dict[str, list[float]] = {g: [] for g in MODULE_GROUPS}
    for pname, stats in snap["per_param"].items():
        g = group_for_param(pname)
        if g is None:
            continue
        out[g].append(stats[key])
    return out


def main():
    # Average over 3 seeds at the same step
    plateau_step = 1500
    post_tau_step = 3500

    plateau = {g: [] for g in MODULE_GROUPS}
    post = {g: [] for g in MODULE_GROUPS}
    for seed in (0, 1, 2):
        p_snap = load_step(seed, plateau_step)
        post_snap = load_step(seed, post_tau_step)
        for g, vals in collect_group_values(p_snap).items():
            plateau[g].extend(vals)
        for g, vals in collect_group_values(post_snap).items():
            post[g].extend(vals)

    # Order groups: z-pipeline first, then control, then bridge
    z_order = ["L0_attn", "L0_mlp", "L2_mlp", "L3_attn"]
    c_order = ["L1_attn", "L1_mlp", "embed", "unembed"]
    other = [g for g in MODULE_GROUPS if g not in z_order + c_order]
    ordered = z_order + c_order

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True)

    for ax, snapshot, title in [
        (axes[0], plateau, f"Plateau (step ≈ {plateau_step})"),
        (axes[1], post, f"Post-$\\tau$ (step ≈ {post_tau_step})"),
    ]:
        positions = []
        labels = []
        colors = []
        data = []
        for i, g in enumerate(ordered):
            vals = np.array(snapshot[g])
            if len(vals) == 0:
                continue
            data.append(np.log10(np.maximum(vals, 1e-12)))
            positions.append(i)
            labels.append(g)
            colors.append("#d62728" if g in Z_PIPELINE_GROUPS else "#1f77b4")

        bp = ax.boxplot(
            data, positions=positions, widths=0.6, patch_artist=True,
            medianprops=dict(color="black", lw=1.4),
            flierprops=dict(marker="o", markersize=2.2, alpha=0.5),
        )
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.grid(True, axis="y", alpha=0.3, linewidth=0.4)
        ax.set_ylabel(r"$\log_{10}(v_t)$ per parameter")

    # Legend
    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor="#d62728", alpha=0.55, label="z-pipeline groups"),
        Patch(facecolor="#1f77b4", alpha=0.55, label="control groups"),
    ]
    axes[0].legend(handles=legend, loc="upper left", fontsize=9, framealpha=0.95)

    fig.suptitle(
        "Per-module distribution of Adam's $v_t$ (3 seeds aggregated, $K=10$, $\\eta=10^{-3}$)",
        fontsize=12,
    )
    plt.tight_layout()
    pdf_path = OUT_DIR / "fig_v_per_module.pdf"
    png_path = OUT_DIR / "fig_v_per_module.png"
    fig.savefig(pdf_path, dpi=200)
    fig.savefig(png_path, dpi=160)
    print(f"  → {pdf_path}")
    print(f"  → {png_path}")

    # Also report numerical summary
    print("\nPer-group v_mean medians (across params, all 3 seeds):")
    for g in ordered:
        if g in Z_PIPELINE_GROUPS:
            tag = "[z-pipe]"
        else:
            tag = "[ctrl]  "
        if len(plateau[g]) and len(post[g]):
            p_med = np.median(plateau[g])
            post_med = np.median(post[g])
            ratio = post_med / p_med if p_med > 0 else float("inf")
            print(f"  {tag} {g:10s}  plateau v={p_med:.2e}  post v={post_med:.2e}  ratio={ratio:.2f}x")


if __name__ == "__main__":
    main()
