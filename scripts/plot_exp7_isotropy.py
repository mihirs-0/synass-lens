#!/usr/bin/env python
"""
Plot results for Experiment 7 — Isotropy Verification.

Produces a 4-panel figure:
  (A) Across-group cosine sim distribution at init / mid-plateau / post-conv.
  (B) Within-group mean similarity over training steps.
  (C) Effective rank over training steps.
  (D) Selector-index mean similarity over training steps.
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
FIG_DIR = OUTPUT_DIR / "figures" / "exp7"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_trajectory():
    path = OUTPUT_DIR / "exp7_isotropy" / "isotropy_trajectory.json"
    if not path.exists():
        print("[Exp 7] No data found.  Run run_exp7_isotropy.py first.")
        return None
    with open(path) as f:
        return json.load(f)


def plot_all():
    traj = load_trajectory()
    if traj is None:
        return

    steps = [d["step"] for d in traj]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax_hist, ax_within, ax_rank, ax_sel = axes.flat

    # --- (A) Across-group cosine sim histograms ---
    # Pick init, ~mid-plateau, post-convergence
    idx_init = 0
    idx_mid = len(traj) // 3
    idx_end = len(traj) - 1

    for idx, label, color in [
        (idx_init, f"Init (step {steps[idx_init]})", "tab:blue"),
        (idx_mid, f"Mid-plateau (step {steps[idx_mid]})", "tab:orange"),
        (idx_end, f"Post-conv (step {steps[idx_end]})", "tab:green"),
    ]:
        sample = traj[idx].get("across_group_cosine", {}).get("sample", [])
        if sample:
            ax_hist.hist(sample, bins=40, alpha=0.5, label=label, color=color, density=True)
    ax_hist.set_xlabel("Cosine similarity")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("(A) Across-group centroid cosine similarity")
    ax_hist.legend(fontsize=8)

    # --- (B) Within-group mean similarity ---
    within_mean = [d.get("within_group_cosine", {}).get("mean", 0.0) for d in traj]
    ax_within.plot(steps, within_mean, "o-", markersize=2)
    ax_within.set_xlabel("Training step")
    ax_within.set_ylabel("Mean within-group cosine sim")
    ax_within.set_title("(B) Within-group spread")

    # --- (C) Effective rank ---
    eff_rank = [d.get("effective_rank", 0.0) for d in traj]
    d_model = traj[0].get("d_model", 128)
    ax_rank.plot(steps, eff_rank, "s-", markersize=2, color="tab:purple")
    ax_rank.axhline(d_model, ls="--", color="gray", lw=0.8, label=f"d_model={d_model}")
    ax_rank.set_xlabel("Training step")
    ax_rank.set_ylabel("Effective rank")
    ax_rank.set_title("(C) Centroid matrix effective rank")
    ax_rank.legend()

    # --- (D) Selector-index alignment ---
    sel_mean = [d.get("selector_index_cosine", {}).get("mean", 0.0) for d in traj]
    ax_sel.plot(steps, sel_mean, "^-", markersize=2, color="tab:red")
    ax_sel.axhline(0, ls="--", color="gray", lw=0.8)
    ax_sel.set_xlabel("Training step")
    ax_sel.set_ylabel("Mean selector-index cosine sim")
    ax_sel.set_title("(D) Selector-index alignment across groups")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "exp7_isotropy.pdf", dpi=200)
    fig.savefig(FIG_DIR / "exp7_isotropy.png", dpi=200)
    plt.close(fig)
    print(f"Saved {FIG_DIR / 'exp7_isotropy.png'}")

    # Summary
    print("\n" + "="*50)
    print("Experiment 7: Isotropy Summary")
    print("="*50)
    for label, idx in [("Init", idx_init), ("Mid-plateau", idx_mid), ("Post-conv", idx_end)]:
        ac = traj[idx].get("across_group_cosine", {})
        wg = traj[idx].get("within_group_cosine", {})
        er = traj[idx].get("effective_rank", 0)
        si = traj[idx].get("selector_index_cosine", {})
        print(f"  {label} (step {steps[idx]}):")
        print(f"    Across-group cos: mean={ac.get('mean', 0):.4f}, std={ac.get('std', 0):.4f}")
        print(f"    Within-group cos: mean={wg.get('mean', 0):.4f}")
        print(f"    Effective rank: {er:.1f}")
        print(f"    Selector-index cos: {si.get('mean', 0):.4f}")


if __name__ == "__main__":
    plot_all()
