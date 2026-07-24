#!/usr/bin/env python
"""
Plot results for Experiment 2 — |B| Sweep + SNR.

Produces:
  (A) Loss curves for all |B| values.
  (B) Log-log: plateau duration vs |B|.
  (C) Log-log: SNR_plateau vs |B|.
  (D) Cosine similarity histograms at three checkpoints.
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
FIG_DIR = OUTPUT_DIR / "figures" / "exp2"
FIG_DIR.mkdir(parents=True, exist_ok=True)

B_VALUES = [50, 100, 250, 500, 1000, 2000]
SEEDS = [42, 43, 44, 45, 46]
K = 20
LOG_K = math.log(K)


def load_json(path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def detect_plateau_duration(steps, first_loss, z_gap):
    onset = None
    for i, fl in enumerate(first_loss):
        if fl < 1.1 * LOG_K:
            onset = steps[i]
            break
    end = None
    for i, zg in enumerate(z_gap):
        if zg > 0.5:
            end = steps[i]
            break
    if onset is not None and end is not None and end > onset:
        return int(end - onset)
    return None


def plot_all():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_loss, ax_plat, ax_snr, ax_iso = axes.flat

    plateau_data = {}  # n_b -> list of durations
    snr_data = {}      # n_b -> list of mean SNR

    for n_b in B_VALUES:
        all_losses = []
        for seed in SEEDS:
            name = f"exp2_B{n_b}_seed{seed}"
            h = load_json(OUTPUT_DIR / name / "training_history.json")
            if h is None:
                continue

            steps = np.array(h["steps"])
            fl = np.array(h["first_target_loss"])
            all_losses.append((steps, fl))

            zs = np.array(h["loss_z_shuffled"])
            min_len = min(len(fl), len(zs))
            z_gap = zs[:min_len] - fl[:min_len]
            d = detect_plateau_duration(steps[:min_len], fl[:min_len], z_gap)
            if d is not None:
                plateau_data.setdefault(n_b, []).append(d)

            snr_res = load_json(OUTPUT_DIR / name / "snr_results.json")
            if snr_res and len(snr_res) > 0:
                mean_snr = np.mean([r["snr_mean"] for r in snr_res])
                snr_data.setdefault(n_b, []).append(mean_snr)

        # (A) Loss curves
        if all_losses:
            min_len = min(len(s) for s, _ in all_losses)
            arr = np.array([fl[:min_len] for _, fl in all_losses])
            steps = all_losses[0][0][:min_len]
            ax_loss.plot(steps, arr.mean(axis=0), label=f"|B|={n_b}")
            ax_loss.fill_between(steps, arr.mean(0) - arr.std(0), arr.mean(0) + arr.std(0), alpha=0.1)

    ax_loss.axhline(LOG_K, ls="--", color="gray", lw=0.8)
    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("First-target loss")
    ax_loss.set_title("(A) Loss curves")
    ax_loss.legend(fontsize=7)

    # (B) Plateau duration vs |B| (log-log)
    if plateau_data:
        xs, ys, errs = [], [], []
        for n_b in sorted(plateau_data.keys()):
            xs.append(n_b)
            ys.append(np.mean(plateau_data[n_b]))
            errs.append(np.std(plateau_data[n_b]))
        xs, ys, errs = np.array(xs, float), np.array(ys), np.array(errs)
        ax_plat.errorbar(xs, ys, yerr=errs, fmt="o-", capsize=3)
        if len(xs) >= 2:
            coeffs = np.polyfit(np.log(xs), np.log(ys + 1), 1)
            ax_plat.plot(xs, np.exp(np.polyval(coeffs, np.log(xs))),
                         "r--", label=f"slope={coeffs[0]:.2f}")
            ax_plat.legend()
    ax_plat.set_xscale("log")
    ax_plat.set_yscale("log")
    ax_plat.set_xlabel("|B|")
    ax_plat.set_ylabel("Plateau duration (steps)")
    ax_plat.set_title("(B) Plateau duration vs |B|")

    # (C) SNR vs |B| (log-log)
    if snr_data:
        xs, ys, errs = [], [], []
        for n_b in sorted(snr_data.keys()):
            xs.append(n_b)
            ys.append(np.mean(snr_data[n_b]))
            errs.append(np.std(snr_data[n_b]))
        xs, ys, errs = np.array(xs, float), np.array(ys), np.array(errs)
        ax_snr.errorbar(xs, ys, yerr=errs, fmt="s-", capsize=3, color="tab:green")
        if len(xs) >= 2:
            coeffs = np.polyfit(np.log(xs), np.log(ys + 1e-12), 1)
            ax_snr.plot(xs, np.exp(np.polyval(coeffs, np.log(xs))),
                        "r--", label=f"slope={coeffs[0]:.2f} (pred: -0.5)")
            ax_snr.legend()
    ax_snr.set_xscale("log")
    ax_snr.set_yscale("log")
    ax_snr.set_xlabel("|B|")
    ax_snr.set_ylabel("SNR (plateau mean)")
    ax_snr.set_title("(C) Gradient SNR vs |B|")

    # (D) Cosine similarity histograms (from first seed of |B|=1000)
    iso_path = OUTPUT_DIR / "exp2_B1000_seed42" / "isotropy_results.json"
    iso = load_json(iso_path)
    if iso:
        for phase, color in [("init", "blue"), ("mid_plateau", "orange"), ("post_convergence", "green")]:
            if phase in iso and "across_group_cosine" in iso[phase]:
                sample = iso[phase]["across_group_cosine"]["sample"]
                ax_iso.hist(sample, bins=50, alpha=0.5, label=phase, color=color, density=True)
        ax_iso.set_xlabel("Cosine similarity")
        ax_iso.set_ylabel("Density")
        ax_iso.set_title("(D) Across-group cosine sim")
        ax_iso.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "exp2_b_sweep.pdf", dpi=200)
    fig.savefig(FIG_DIR / "exp2_b_sweep.png", dpi=200)
    plt.close(fig)
    print(f"Saved {FIG_DIR / 'exp2_b_sweep.png'}")

    # Summary table
    print("\n" + "="*70)
    print("Experiment 2: |B| Sweep Summary")
    print("="*70)
    print(f"{'|B|':>6}  {'Plat dur (mean±std)':>22}  {'SNR (mean±std)':>20}")
    print("-"*52)
    for n_b in B_VALUES:
        pd = plateau_data.get(n_b, [])
        sd = snr_data.get(n_b, [])
        pd_str = f"{np.mean(pd):.0f}±{np.std(pd):.0f}" if pd else "N/A"
        sd_str = f"{np.mean(sd):.4f}±{np.std(sd):.4f}" if sd else "N/A"
        print(f"{n_b:>6}  {pd_str:>22}  {sd_str:>20}")


if __name__ == "__main__":
    plot_all()
