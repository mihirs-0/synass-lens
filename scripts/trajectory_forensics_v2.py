#!/usr/bin/env python3
"""
Trajectory Forensics v2: Smoothed analysis to separate signal from noise.

The raw eval-every-50-step trajectories are extremely noisy near the boundary.
This script overlays a 2000-step rolling mean to reveal the true trend.
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
SAVE_DIR = OUTPUTS / "paper_figures"
SAVE_DIR.mkdir(exist_ok=True)

SEEDS = [42, 123, 7]
SMOOTH_WINDOW = 40  # 40 eval points × 50 steps = 2000 steps


def load_history(k, lr_str, seed):
    name = f"pb_K{k}_lr{lr_str}_s{seed}"
    p = OUTPUTS / name / "training_history.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def smooth(arr, w):
    return np.convolve(arr, np.ones(w) / w, mode="valid")


def classify_smoothed(steps, loss, log_k, w=SMOOTH_WINDOW):
    """Classify based on smoothed trajectory."""
    sm = smooth(loss, w)
    sm_steps = steps[:len(sm)]

    sm_min = sm.min()
    sm_min_step = sm_steps[sm.argmin()]
    sm_final = sm[-1]

    # End slope on smoothed (last 50 smoothed points ≈ 2500 steps)
    n_end = min(50, len(sm))
    end_slope = np.polyfit(np.arange(n_end), sm[-n_end:], 1)[0]

    # Thresholds on smoothed
    below_50 = sm < 0.5 * log_k
    below_80 = sm < 0.8 * log_k

    # Did smoothed ever breach 50%?
    ever_breached_50 = below_50.any()

    # Did smoothed ever go below 50% then back above 80%?
    smoothed_revert = False
    if ever_breached_50:
        first_50_idx = np.argmax(below_50)
        post = sm[first_50_idx:]
        if (post > 0.8 * log_k).any():
            smoothed_revert = True

    # Count 50% crossings on smoothed
    crossings_50 = np.diff(below_50.astype(int))
    n_up_50 = int(np.sum(crossings_50 == -1))

    # Classification
    if sm_final < 0.05 * log_k:
        cls = "CONVERGED"
    elif smoothed_revert:
        cls = "SMOOTHED_REVERSION"
    elif ever_breached_50 and sm_final < 0.5 * log_k and end_slope < 0:
        cls = "ACTIVE_DESCENT"
    elif ever_breached_50 and sm_final > 0.5 * log_k and end_slope > 0:
        cls = "PARTIAL_REVERSION"
    elif ever_breached_50 and end_slope > 0:
        cls = "STALLED_REVERTING"
    elif ever_breached_50:
        cls = "NOISY_DESCENT"
    elif below_80.any():
        cls = "SHALLOW_DIP"
    else:
        cls = "NEVER_BREACHED"

    return {
        "sm_min": float(sm_min),
        "sm_min_step": int(sm_min_step),
        "sm_final": float(sm_final),
        "end_slope": float(end_slope),
        "ever_breached_50": ever_breached_50,
        "smoothed_revert": smoothed_revert,
        "n_up_50": n_up_50,
        "classification": cls,
        "smoothed": sm,
        "sm_steps": sm_steps,
    }


# ── All near-boundary runs ────────────────────────────────────────────────────

RUNS = [
    (10, "1e-2"),
    (20, "5e-3"), (20, "7e-3"),
    (36, "2e-3"), (36, "3e-3"), (36, "5e-3"),
    (5, "1.5e-2"), (5, "2e-2"),
]

print("=" * 100)
print("TRAJECTORY FORENSICS v2: Smoothed Classification (2000-step rolling mean)")
print("=" * 100)

all_runs = []

print(f"\n{'K':>3} {'η':>8} {'seed':>5}  {'smooth_class':>20}  {'sm min':>8} {'(%lgK)':>6}  "
      f"{'sm final':>8} {'(%lgK)':>6}  {'end slope':>10}  {'↑50%':>4}  {'sm revert':>9}")
print("-" * 110)

for k, lr_str in RUNS:
    log_k = math.log(k)
    for seed in SEEDS:
        h = load_history(k, lr_str, seed)
        if h is None:
            continue

        steps = np.array(h["steps"])
        loss = np.array(h["candidate_loss"])

        info = classify_smoothed(steps, loss, log_k)
        info["K"] = k
        info["lr"] = lr_str
        info["seed"] = seed
        info["log_k"] = log_k
        info["steps"] = steps
        info["loss"] = loss
        all_runs.append(info)

        min_pct = 100 * info["sm_min"] / log_k
        final_pct = 100 * info["sm_final"] / log_k
        slope_dir = "↘" if info["end_slope"] < -1e-5 else ("↗" if info["end_slope"] > 1e-5 else "→")

        print(f"{k:>3} {lr_str:>8} {seed:>5}  {info['classification']:>20}  "
              f"{info['sm_min']:>8.4f} {min_pct:>5.1f}%  "
              f"{info['sm_final']:>8.4f} {final_pct:>5.1f}%  "
              f"{info['end_slope']:>10.6f}{slope_dir}  "
              f"{info['n_up_50']:>4}  "
              f"{'YES' if info['smoothed_revert'] else 'no':>9}")

# ── Summary ──────────────────────────────────────────────────────────────────

cats = {}
for r in all_runs:
    c = r["classification"]
    cats.setdefault(c, []).append(r)

print(f"\n{'='*100}")
print("SMOOTHED CLASSIFICATION SUMMARY")
print(f"{'='*100}")
for cat in ["SMOOTHED_REVERSION", "PARTIAL_REVERSION", "STALLED_REVERTING",
            "ACTIVE_DESCENT", "NOISY_DESCENT", "CONVERGED", "SHALLOW_DIP", "NEVER_BREACHED"]:
    runs = cats.get(cat, [])
    if not runs:
        continue
    print(f"\n{cat} ({len(runs)}):")
    for r in runs:
        mp = 100 * r["sm_min"] / r["log_k"]
        fp = 100 * r["sm_final"] / r["log_k"]
        print(f"  K={r['K']}, η={r['lr']}, s={r['seed']}: "
              f"smooth min {mp:.0f}% logK, smooth final {fp:.0f}% logK, "
              f"↑50%={r['n_up_50']}, revert={r['smoothed_revert']}")

# Final verdict
has_reversion = len(cats.get("SMOOTHED_REVERSION", [])) > 0
has_partial = len(cats.get("PARTIAL_REVERSION", [])) > 0
has_stalled = len(cats.get("STALLED_REVERTING", [])) > 0

print(f"\n{'='*100}")
print("VERDICT")
print(f"{'='*100}")
if has_reversion:
    print("GENUINE SMOOTHED REVERSION EXISTS: loss drops below 50% of logK then")
    print("rises back above 80% of logK even after 2000-step smoothing.")
    for r in cats["SMOOTHED_REVERSION"]:
        print(f"  → K={r['K']}, η={r['lr']}, s={r['seed']}")
else:
    print("NO smoothed reversion to plateau (>80% logK) after breaching 50% logK.")

if has_partial or has_stalled:
    print(f"\nHowever, {len(cats.get('PARTIAL_REVERSION', [])) + len(cats.get('STALLED_REVERTING', []))} "
          f"runs show PARTIAL reversion or stalling:")
    for cat in ["PARTIAL_REVERSION", "STALLED_REVERTING"]:
        for r in cats.get(cat, []):
            mp = 100 * r["sm_min"] / r["log_k"]
            fp = 100 * r["sm_final"] / r["log_k"]
            print(f"  → K={r['K']}, η={r['lr']}, s={r['seed']}: "
                  f"smoothed min {mp:.0f}%→final {fp:.0f}% logK, "
                  f"end slope {'positive (reverting)' if r['end_slope'] > 0 else 'negative (descending)'}")

if not has_reversion and not has_partial and not has_stalled:
    print("\nAll near-boundary runs that breach 50% logK are either:")
    print("  - Still actively descending (budget-limited)")
    print("  - Already converged (slow success)")
    print("The 'nucleation zone' = 'slow convergence zone'.")

# ── Figure: Raw + smoothed overlay for ALL mixed-outcome runs ─────────────────

mixed_runs = [r for r in all_runs if (r["K"], r["lr"]) in
              [(k, lr) for k, lr, in [(10, "1e-2"), (20, "5e-3"), (20, "7e-3"),
                                       (36, "2e-3"), (36, "3e-3"), (36, "5e-3")]]]

n = len(mixed_runs)
ncols = 3
nrows = math.ceil(n / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)

for idx, r in enumerate(mixed_runs):
    row, col = divmod(idx, ncols)
    ax = axes[row][col]
    log_k = r["log_k"]

    # Raw trajectory (very light)
    ax.plot(r["steps"], r["loss"], linewidth=0.15, color="#AAAAAA", alpha=0.5)

    # Smoothed trajectory (bold)
    ax.plot(r["sm_steps"], r["smoothed"], linewidth=1.5, color="#2C3E50")

    # Thresholds
    ax.axhline(log_k, color="red", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.axhline(0.5 * log_k, color="orange", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.axhline(0.8 * log_k, color="purple", linestyle=":", linewidth=0.7, alpha=0.4)
    ax.axhline(0.05 * log_k, color="green", linestyle="--", linewidth=0.7, alpha=0.5)

    # Background color
    bg = {"SMOOTHED_REVERSION": "#FFCCCC", "PARTIAL_REVERSION": "#FFE0CC",
          "STALLED_REVERTING": "#FFE0CC", "ACTIVE_DESCENT": "#FFFFCC",
          "NOISY_DESCENT": "#FFFFCC", "CONVERGED": "#CCFFCC",
          "SHALLOW_DIP": "#E0E0E0", "NEVER_BREACHED": "#E0E0E0"}
    ax.set_facecolor(bg.get(r["classification"], "white"))

    mp = 100 * r["sm_min"] / log_k
    fp = 100 * r["sm_final"] / log_k
    slope_dir = "↘" if r["end_slope"] < -1e-5 else ("↗" if r["end_slope"] > 1e-5 else "→")
    ax.set_title(f"K={r['K']}, η={r['lr']}, s={r['seed']}\n"
                 f"[{r['classification']}] sm_min={mp:.0f}% sm_final={fp:.0f}% {slope_dir}",
                 fontsize=7, fontweight="bold")
    ax.set_xlabel("Step", fontsize=6)
    ax.set_ylabel("Loss", fontsize=6)
    ax.tick_params(labelsize=6)
    ax.set_ylim(-0.1, log_k * 1.15)

for idx in range(n, nrows * ncols):
    row, col = divmod(idx, ncols)
    axes[row][col].set_visible(False)

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(SAVE_DIR / f"fig_trajectory_forensics_smoothed.{ext}", dpi=300, bbox_inches="tight")
plt.close()
print(f"\nFigure saved: fig_trajectory_forensics_smoothed")

# ── Zoom figure for the two most interesting runs ─────────────────────────────

focus_runs = [r for r in all_runs
              if (r["K"], r["lr"], r["seed"]) in [(20, "5e-3", 42), (36, "3e-3", 42),
                                                    (36, "5e-3", 42), (36, "2e-3", 7)]]

if focus_runs:
    ncols_z = min(len(focus_runs), 2)
    nrows_z = math.ceil(len(focus_runs) / ncols_z)
    fig, axes = plt.subplots(nrows_z, ncols_z, figsize=(6 * ncols_z, 4 * nrows_z), squeeze=False)

    for idx, r in enumerate(focus_runs):
        row, col = divmod(idx, ncols_z)
        ax = axes[row][col]
        log_k = r["log_k"]

        # Show last 60% of training
        n_total = len(r["steps"])
        start_idx = n_total // 3

        ax.plot(r["steps"][start_idx:], r["loss"][start_idx:],
                linewidth=0.2, color="#BBBBBB", alpha=0.6, label="Raw (eval/50 steps)")

        # Smoothed
        sm_start = max(0, start_idx - SMOOTH_WINDOW)
        ax.plot(r["sm_steps"][sm_start:], r["smoothed"][sm_start:],
                linewidth=2, color="#E74C3C", label="2000-step rolling mean")

        ax.axhline(log_k, color="red", linestyle="--", linewidth=0.8, alpha=0.4, label=f"log K = {log_k:.2f}")
        ax.axhline(0.5 * log_k, color="orange", linestyle="--", linewidth=0.8, alpha=0.4, label="50% log K")
        ax.axhline(0.8 * log_k, color="purple", linestyle=":", linewidth=0.8, alpha=0.4, label="80% log K")

        mp = 100 * r["sm_min"] / log_k
        fp = 100 * r["sm_final"] / log_k
        slope_dir = "descending" if r["end_slope"] < -1e-5 else ("REVERTING" if r["end_slope"] > 1e-5 else "flat")
        ax.set_title(f"K={r['K']}, η={r['lr']}, s={r['seed']} [{r['classification']}]\n"
                     f"Smoothed: min={mp:.0f}% logK → final={fp:.0f}% logK, end: {slope_dir}",
                     fontsize=9)
        ax.set_xlabel("Step", fontsize=9)
        ax.set_ylabel("Candidate loss", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_ylim(-0.1, log_k * 1.1)

    for idx in range(len(focus_runs), nrows_z * ncols_z):
        row, col = divmod(idx, ncols_z)
        axes[row][col].set_visible(False)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(SAVE_DIR / f"fig_trajectory_zooms_smoothed.{ext}", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: fig_trajectory_zooms_smoothed")
