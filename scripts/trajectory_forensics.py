#!/usr/bin/env python3
"""
Trajectory Forensics: What actually happens near the phase boundary?

Examines every near-boundary run to determine whether:
  A) Runs are budget-limited (loss descends monotonically, just slowly)
  B) Runs show failed nucleation (loss drops then REVERTS to plateau)
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

# ── Step 1: Define near-boundary runs ────────────────────────────────────────

# Mixed-outcome (K, η) pairs from the phase boundary outcome matrix
MIXED_OUTCOMES = [
    # (K, lr_str, expected_outcomes)  — outcomes per seed from the paper
    (10, "1e-2",  {"42": "?", "123": "?", "7": "?"}),   # 1/3 succeed
    (20, "5e-3",  {"42": "?", "123": "?", "7": "?"}),   # 2/3 succeed
    (20, "7e-3",  {"42": "?", "123": "?", "7": "?"}),   # 1/3 succeed
    (36, "2e-3",  {"42": "?", "123": "?", "7": "?"}),   # 2/3 succeed
    (36, "3e-3",  {"42": "?", "123": "?", "7": "?"}),   # 2/3 succeed
    (36, "5e-3",  {"42": "?", "123": "?", "7": "?"}),   # 0/3 succeed (but breach?)
]

# Also include the adjacent all-fail runs for comparison
ALL_FAIL = [
    (5,  "1.5e-2", {"42": "?", "123": "?", "7": "?"}),
    (5,  "2e-2",   {"42": "?", "123": "?", "7": "?"}),
]

SEEDS = [42, 123, 7]


def load_history(k, lr_str, seed):
    name = f"pb_K{k}_lr{lr_str}_s{seed}"
    p = OUTPUTS / name / "training_history.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


# ── Step 2 & 3: Load, plot, and classify every trajectory ────────────────────

print("=" * 90)
print("TRAJECTORY FORENSICS: Near-boundary run classification")
print("=" * 90)

all_runs = []

for k, lr_str, _ in MIXED_OUTCOMES + ALL_FAIL:
    log_k = math.log(k)
    for seed in SEEDS:
        h = load_history(k, lr_str, seed)
        if h is None:
            continue

        steps = np.array(h["steps"])
        # Use candidate_loss (evaluates on held-out candidates)
        if "candidate_loss" in h:
            cand_loss = np.array(h["candidate_loss"])
        else:
            cand_loss = np.array(h["first_target_loss"])

        # ── Classification ──
        min_loss = cand_loss.min()
        min_step = steps[cand_loss.argmin()]
        final_loss = cand_loss[-1]

        # Check for breach below 50% of log(K)
        breach_mask = cand_loss < 0.5 * log_k
        breach_idx = np.where(breach_mask)[0]

        # Check for convergence (below 5% of log(K))
        converged = final_loss < 0.05 * log_k

        classification = "NEVER_BREACHED"
        sustained_reversions = []
        n_up_crossings = 0
        n_down_crossings = 0

        if len(breach_idx) > 0:
            first_breach = breach_idx[0]
            post_breach = cand_loss[first_breach:]

            # Check reversion: loss goes back above 80% of log(K) for 500+ steps
            above_80 = post_breach > 0.8 * log_k

            # Find runs of consecutive True values
            reversion_runs = []
            current_run = 0
            for val in above_80:
                if val:
                    current_run += 1
                else:
                    if current_run > 0:
                        reversion_runs.append(current_run)
                    current_run = 0
            if current_run > 0:
                reversion_runs.append(current_run)

            # eval_every=50, so 10 consecutive points = 500 steps
            sustained_reversions = [r for r in reversion_runs if r >= 10]

            if len(sustained_reversions) > 0:
                classification = "FAILED_NUCLEATION"
            elif converged:
                classification = "SLOW_SUCCESS"
            elif final_loss > 0.5 * log_k:
                classification = "INCOMPLETE_DESCENT"
            else:
                classification = "SLOW_SUCCESS"

            # Count crossings of 80% threshold
            below_80 = cand_loss < 0.8 * log_k
            crossings = np.diff(below_80.astype(int))
            n_down_crossings = int(np.sum(crossings == 1))
            n_up_crossings = int(np.sum(crossings == -1))
        else:
            classification = "NEVER_BREACHED"

        # Volatility near minimum
        min_idx = int(cand_loss.argmin())
        window = 20  # 20 eval points = 1000 steps
        start = max(0, min_idx - window)
        end = min(len(cand_loss), min_idx + window)
        volatility = float(cand_loss[start:end].std())

        # End slope (last 5000 steps = last 100 eval points at eval_every=50)
        n_final = min(100, len(cand_loss))
        final_window = cand_loss[-n_final:]
        if len(final_window) > 1:
            slope = np.polyfit(np.arange(len(final_window)), final_window, 1)[0]
        else:
            slope = 0.0

        run_data = {
            "K": k, "lr": lr_str, "seed": seed,
            "log_k": log_k,
            "steps": steps, "cand_loss": cand_loss,
            "min_loss": min_loss, "min_step": min_step,
            "final_loss": final_loss,
            "classification": classification,
            "n_down_crossings": n_down_crossings,
            "n_up_crossings": n_up_crossings,
            "sustained_reversions": len(sustained_reversions),
            "volatility": volatility,
            "end_slope": slope,
            "converged": converged,
        }
        all_runs.append(run_data)

# ── Print classification table ────────────────────────────────────────────────

print(f"\n{'K':>3} {'η':>8} {'seed':>5}  {'class':>22}  {'min loss':>9} {'(% logK)':>8}  "
      f"{'final':>9} {'(% logK)':>8}  {'↓80%':>4} {'↑80%':>4}  {'reverts':>7}  {'end slope':>10}  {'verdict':>10}")
print("-" * 130)

for r in all_runs:
    min_pct = 100 * r["min_loss"] / r["log_k"]
    final_pct = 100 * r["final_loss"] / r["log_k"]
    slope_label = "descending" if r["end_slope"] < -1e-5 else ("REVERTING" if r["end_slope"] > 1e-5 else "flat")
    print(f"{r['K']:>3} {r['lr']:>8} {r['seed']:>5}  {r['classification']:>22}  "
          f"{r['min_loss']:>9.4f} {min_pct:>7.1f}%  "
          f"{r['final_loss']:>9.4f} {final_pct:>7.1f}%  "
          f"{r['n_down_crossings']:>4} {r['n_up_crossings']:>4}  "
          f"{r['sustained_reversions']:>7}  "
          f"{r['end_slope']:>10.6f}  {slope_label:>10}")

# ── Step 4: Summary verdict ──────────────────────────────────────────────────

categories = {}
for r in all_runs:
    c = r["classification"]
    if c not in categories:
        categories[c] = []
    categories[c].append(r)

print(f"\n{'='*90}")
print("TRAJECTORY FORENSICS SUMMARY")
print(f"{'='*90}")

for cat in ["FAILED_NUCLEATION", "INCOMPLETE_DESCENT", "SLOW_SUCCESS", "NEVER_BREACHED"]:
    runs = categories.get(cat, [])
    print(f"\n{cat} ({len(runs)} runs):")
    if runs:
        for r in runs:
            min_pct = 100 * r["min_loss"] / r["log_k"]
            final_pct = 100 * r["final_loss"] / r["log_k"]
            print(f"  K={r['K']}, η={r['lr']}, seed={r['seed']}: "
                  f"min={min_pct:.0f}% logK, final={final_pct:.0f}% logK, "
                  f"↑crossings={r['n_up_crossings']}, reversions={r['sustained_reversions']}")
    else:
        print("  (none)")

n_failed = len(categories.get("FAILED_NUCLEATION", []))
print(f"\n{'='*90}")
print(f"VERDICT: Do genuine FAILED_NUCLEATION events exist?  {'YES' if n_failed > 0 else 'NO'}")
if n_failed > 0:
    print(f"  Count: {n_failed} runs")
    for r in categories["FAILED_NUCLEATION"]:
        print(f"  → K={r['K']}, η={r['lr']}, seed={r['seed']} "
              f"(min {100*r['min_loss']/r['log_k']:.0f}% logK at step {r['min_step']}, "
              f"final {100*r['final_loss']/r['log_k']:.0f}% logK, "
              f"{r['sustained_reversions']} sustained reversions)")
else:
    print("  The 'nucleation zone' framing is NOT supported by trajectory evidence.")
    print("  Reclassify as 'slow convergence zone' or 'budget-limited zone'.")
print(f"{'='*90}")

# ── Step 2 (figures): Plot all near-boundary trajectories ─────────────────────

# Only plot the mixed-outcome runs (not all-fail)
mixed_runs = [r for r in all_runs if (r["K"], r["lr"]) in
              [(k, lr) for k, lr, _ in MIXED_OUTCOMES]]

n_mixed = len(mixed_runs)
ncols = 3
nrows = math.ceil(n_mixed / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows), squeeze=False)

for idx, r in enumerate(mixed_runs):
    row, col = divmod(idx, ncols)
    ax = axes[row][col]

    log_k = r["log_k"]
    ax.plot(r["steps"], r["cand_loss"], linewidth=0.5, color="#2C3E50", alpha=0.8)

    # Threshold lines
    ax.axhline(log_k, color="red", linestyle="--", linewidth=0.8, alpha=0.6, label=f"log K = {log_k:.2f}")
    ax.axhline(0.5 * log_k, color="orange", linestyle="--", linewidth=0.8, alpha=0.6, label="50% log K")
    ax.axhline(0.05 * log_k, color="green", linestyle="--", linewidth=0.8, alpha=0.6, label="5% log K")

    # Color the background by classification
    bg_colors = {
        "FAILED_NUCLEATION": "#FFCCCC",
        "INCOMPLETE_DESCENT": "#FFF3CC",
        "SLOW_SUCCESS": "#CCFFCC",
        "NEVER_BREACHED": "#CCCCCC",
    }
    ax.set_facecolor(bg_colors.get(r["classification"], "white"))

    outcome = "SUCCESS" if r["converged"] else "FAIL"
    ax.set_title(f"K={r['K']}, η={r['lr']}, s={r['seed']} — {outcome}\n"
                 f"[{r['classification']}]", fontsize=7, fontweight="bold")
    ax.set_xlabel("Step", fontsize=6)
    ax.set_ylabel("Candidate loss", fontsize=6)
    ax.tick_params(labelsize=6)
    ax.set_ylim(-0.1, log_k * 1.15)

    if idx == 0:
        ax.legend(fontsize=5, loc="upper right")

# Hide unused subplots
for idx in range(n_mixed, nrows * ncols):
    row, col = divmod(idx, ncols)
    axes[row][col].set_visible(False)

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(SAVE_DIR / f"fig_trajectory_forensics.{ext}", dpi=300, bbox_inches="tight")
plt.close()
print(f"\nFigure saved: {SAVE_DIR / 'fig_trajectory_forensics.pdf'}")

# ── Step 5: Zoom plots for interesting trajectories ──────────────────────────

interesting = [r for r in all_runs
               if r["classification"] in ("FAILED_NUCLEATION", "INCOMPLETE_DESCENT")]

if interesting:
    n_int = len(interesting)
    ncols_z = min(3, n_int)
    nrows_z = math.ceil(n_int / ncols_z)
    fig, axes = plt.subplots(nrows_z, ncols_z, figsize=(5 * ncols_z, 3.5 * nrows_z), squeeze=False)

    for idx, r in enumerate(interesting):
        row, col = divmod(idx, ncols_z)
        ax = axes[row][col]
        log_k = r["log_k"]

        # Find the interesting region: around the minimum
        min_idx = int(np.argmin(r["cand_loss"]))
        # Show from 2000 steps before breach to end, or from halfway to the min
        breach_idx = np.where(r["cand_loss"] < 0.5 * log_k)[0]
        if len(breach_idx) > 0:
            zoom_start = max(0, breach_idx[0] - 40)  # 40 eval points = 2000 steps before breach
        else:
            zoom_start = max(0, min_idx - 100)
        zoom_end = len(r["steps"])

        ax.plot(r["steps"][zoom_start:zoom_end], r["cand_loss"][zoom_start:zoom_end],
                linewidth=0.8, color="#2C3E50")
        ax.axhline(log_k, color="red", linestyle="--", linewidth=0.7, alpha=0.5)
        ax.axhline(0.5 * log_k, color="orange", linestyle="--", linewidth=0.7, alpha=0.5)
        ax.axhline(0.8 * log_k, color="purple", linestyle=":", linewidth=0.7, alpha=0.5, label="80% log K")
        ax.axhline(0.05 * log_k, color="green", linestyle="--", linewidth=0.7, alpha=0.5)

        # Mark the minimum
        ax.axvline(r["min_step"], color="blue", linestyle=":", alpha=0.5, linewidth=0.7)
        ax.annotate(f"min={100*r['min_loss']/log_k:.0f}%",
                    (r["min_step"], r["min_loss"]),
                    fontsize=6, color="blue", ha="left")

        bg = "#FFCCCC" if r["classification"] == "FAILED_NUCLEATION" else "#FFF3CC"
        ax.set_facecolor(bg)

        slope_label = "↘" if r["end_slope"] < -1e-5 else ("↗ REVERT" if r["end_slope"] > 1e-5 else "→ flat")
        ax.set_title(f"K={r['K']}, η={r['lr']}, s={r['seed']} [{r['classification']}]\n"
                     f"min={100*r['min_loss']/log_k:.0f}% logK, final={100*r['final_loss']/log_k:.0f}% logK, "
                     f"end {slope_label}",
                     fontsize=7)
        ax.set_xlabel("Step", fontsize=7)
        ax.set_ylabel("Candidate loss", fontsize=7)
        ax.tick_params(labelsize=6)
        if idx == 0:
            ax.legend(fontsize=5)

    for idx in range(n_int, nrows_z * ncols_z):
        row, col = divmod(idx, ncols_z)
        axes[row][col].set_visible(False)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(SAVE_DIR / f"fig_trajectory_zooms.{ext}", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {SAVE_DIR / 'fig_trajectory_zooms.pdf'}")
else:
    print("No FAILED_NUCLEATION or INCOMPLETE_DESCENT runs to zoom.")

# ── Step 6: End-of-training derivative for all non-converged runs ────────────

print(f"\n{'='*90}")
print("END-OF-TRAINING LOSS DERIVATIVE (non-converged runs)")
print(f"{'='*90}")
print(f"\n{'K':>3} {'η':>8} {'seed':>5}  {'final loss':>11} {'(% logK)':>8}  {'slope':>12}  {'verdict':>12}")
print("-" * 70)

non_converged = [r for r in all_runs if not r["converged"]]
for r in non_converged:
    final_pct = 100 * r["final_loss"] / r["log_k"]
    if r["end_slope"] < -1e-5:
        verdict = "descending"
    elif r["end_slope"] > 1e-5:
        verdict = "REVERTING"
    else:
        verdict = "flat"
    print(f"{r['K']:>3} {r['lr']:>8} {r['seed']:>5}  {r['final_loss']:>11.4f} {final_pct:>7.1f}%  "
          f"{r['end_slope']:>12.6f}  {verdict:>12}")
