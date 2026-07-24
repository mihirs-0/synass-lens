#!/usr/bin/env python
"""
Plot results for Experiment 1 — Group-Specific Selector Control.

Produces:
  1. Loss curves for Conditions A, B, and representative G values (overlay).
  2. z-gap curves (secondary panel).
  3. Table: plateau duration vs. sharing factor |B|/G with power-law fit.
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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("outputs")
FIG_DIR = OUTPUT_DIR / "figures" / "exp1"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44, 45, 46]
G_VALUES = [10, 50, 100, 500, 1000]
N_B = 1000
K = 20
LOG_K = math.log(K)


def load_history(name: str):
    path = OUTPUT_DIR / name / "training_history.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def mean_std_across_seeds(prefix: str, seeds, key: str):
    """Load *key* from training histories across seeds, return (steps, mean, std)."""
    all_vals = []
    steps = None
    for s in seeds:
        h = load_history(f"{prefix}_seed{s}")
        if h is None:
            continue
        if steps is None:
            steps = np.array(h["steps"])
        vals = np.array(h[key])
        min_len = min(len(steps), len(vals))
        all_vals.append(vals[:min_len])
    if not all_vals:
        return None, None, None
    min_len = min(len(v) for v in all_vals)
    arr = np.array([v[:min_len] for v in all_vals])
    return steps[:min_len], arr.mean(axis=0), arr.std(axis=0)


def compute_z_gap(h):
    """z-gap = loss_z_shuffled - first_target_loss (secondary diagnostic)."""
    s = np.array(h["loss_z_shuffled"])
    c = np.array(h["first_target_loss"])
    min_len = min(len(s), len(c))
    return s[:min_len] - c[:min_len]


def has_candidate_loss(h):
    """Check if training history includes candidate_loss."""
    return "candidate_loss" in h and len(h["candidate_loss"]) > 0


def detect_plateau_duration(steps, candidate_loss):
    """Plateau duration based on candidate loss.

    Onset: candidate_loss first drops below log(K) (random-guess baseline).
    End: candidate_loss drops below 0.5 (confident disambiguation).
    """
    onset = None
    for i, cl in enumerate(candidate_loss):
        if cl < LOG_K:
            onset = steps[i]
            break
    end = None
    for i, cl in enumerate(candidate_loss):
        if cl < 0.5:
            end = steps[i]
            break
    if onset is not None and end is not None and end > onset:
        return int(end - onset)
    return None


# ---------------------------------------------------------------------------
# Figure 1: Loss curves + z-gap
# ---------------------------------------------------------------------------

def plot_loss_and_zgap():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    conditions = [
        ("exp1_shared_z", "Shared z (Cond A)", "tab:blue"),
        ("exp1_private_z", "Private z (Cond B)", "tab:orange"),
    ]
    for G in [10, 100, 1000]:
        conditions.append((f"exp1_supergroup_G{G}", f"G={G} (|B|/G={N_B//G})", None))

    # Detect whether candidate_loss is available from any run
    _use_candidate = False
    for prefix, _, _ in conditions:
        for s in SEEDS:
            h = load_history(f"{prefix}_seed{s}")
            if h is not None and has_candidate_loss(h):
                _use_candidate = True
                break
        if _use_candidate:
            break

    loss_key = "candidate_loss" if _use_candidate else "first_target_loss"
    loss_label = "Candidate loss" if _use_candidate else "First-target loss"

    for prefix, label, color in conditions:
        steps, mean_loss, std_loss = mean_std_across_seeds(prefix, SEEDS, loss_key)
        if steps is None:
            continue
        kwargs = {"color": color} if color else {}
        ax1.plot(steps, mean_loss, label=label, **kwargs)
        ax1.fill_between(steps, mean_loss - std_loss, mean_loss + std_loss, alpha=0.15, **kwargs)

        # z-gap (secondary diagnostic, still based on first-target loss)
        all_gaps = []
        for s in SEEDS:
            h = load_history(f"{prefix}_seed{s}")
            if h is None:
                continue
            all_gaps.append(compute_z_gap(h))
        if all_gaps:
            min_len = min(len(g) for g in all_gaps)
            gap_arr = np.array([g[:min_len] for g in all_gaps])
            ax2.plot(steps[:min_len], gap_arr.mean(axis=0), label=label, **kwargs)
            ax2.fill_between(steps[:min_len],
                             gap_arr.mean(axis=0) - gap_arr.std(axis=0),
                             gap_arr.mean(axis=0) + gap_arr.std(axis=0),
                             alpha=0.15, **kwargs)

    ax1.axhline(LOG_K, ls="--", color="gray", lw=0.8, label=f"log({K}) [chance]")
    ax1.set_ylabel(loss_label)
    ax1.legend(fontsize=8)
    ax1.set_title("Experiment 1: Group-Specific Selector Control")

    ax2.set_xlabel("Training step")
    ax2.set_ylabel("z-gap (Δ_z)")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "exp1_loss_zgap.pdf", dpi=200)
    fig.savefig(FIG_DIR / "exp1_loss_zgap.png", dpi=200)
    plt.close(fig)
    print(f"Saved {FIG_DIR / 'exp1_loss_zgap.png'}")


# ---------------------------------------------------------------------------
# Table: plateau duration vs sharing factor
# ---------------------------------------------------------------------------

def _get_plateau_for_run(h):
    """Extract plateau duration from a single run's history."""
    steps = np.array(h["steps"])
    if has_candidate_loss(h):
        cl = np.array(h["candidate_loss"])
        return detect_plateau_duration(steps, cl)
    return None


def plateau_duration_table():
    rows = []
    # Condition A: sharing factor = |B| = 1000
    durations = []
    for s in SEEDS:
        h = load_history(f"exp1_shared_z_seed{s}")
        if h is None:
            continue
        d = _get_plateau_for_run(h)
        if d is not None:
            durations.append(d)
    if durations:
        rows.append((N_B, np.mean(durations), np.std(durations)))

    # Condition C: varying G
    for G in G_VALUES:
        sharing = N_B // G
        durations = []
        for s in SEEDS:
            h = load_history(f"exp1_supergroup_G{G}_seed{s}")
            if h is None:
                continue
            d = _get_plateau_for_run(h)
            if d is not None:
                durations.append(d)
        if durations:
            rows.append((sharing, np.mean(durations), np.std(durations)))

    # Condition B: sharing factor = 1
    durations = []
    for s in SEEDS:
        h = load_history(f"exp1_private_z_seed{s}")
        if h is None:
            continue
        d = _get_plateau_for_run(h)
        if d is not None:
            durations.append(d)
    if durations:
        rows.append((1, np.mean(durations), np.std(durations)))

    if not rows:
        print("[Exp 1] No plateau duration data available yet.")
        return

    # Print table
    print("\n" + "="*60)
    print("Experiment 1: Plateau Duration vs Sharing Factor")
    print("="*60)
    print(f"{'|B|/G':>8}  {'Duration (mean)':>16}  {'Duration (std)':>15}")
    print("-"*45)
    for sf, m, sd in sorted(rows, key=lambda x: -x[0]):
        print(f"{sf:>8}  {m:>16.1f}  {sd:>15.1f}")

    # Power-law fit: duration ~ alpha * (|B|/G)^beta
    sfs = np.array([r[0] for r in rows if r[0] > 1], dtype=float)
    durs = np.array([r[1] for r in rows if r[0] > 1])
    if len(sfs) >= 2:
        log_sf = np.log(sfs)
        log_dur = np.log(durs + 1)
        beta, log_alpha = np.polyfit(log_sf, log_dur, 1)
        print(f"\nPower-law fit: plateau_duration ~ {math.exp(log_alpha):.1f} × (|B|/G)^{beta:.2f}")
        print(f"Proposition 2 predicts beta = 1.0")

    # Save
    result_path = FIG_DIR / "exp1_plateau_table.json"
    with open(result_path, "w") as f:
        json.dump([{"sharing_factor": int(sf), "mean": m, "std": sd}
                    for sf, m, sd in rows], f, indent=2)
    print(f"Saved to {result_path}")


# ---------------------------------------------------------------------------

def main():
    plot_loss_and_zgap()
    plateau_duration_table()


if __name__ == "__main__":
    main()
