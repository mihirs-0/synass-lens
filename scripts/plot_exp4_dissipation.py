#!/usr/bin/env python
"""
Plot results for Experiment 4 — Optimizer-Aware Dissipation.

Produces:
  - Q_work vs log(K) for AdamW, overlaid with Q_grad vs log(K).
  - Same plot for SGD.
  - R² table for each Q measure vs log(K).
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
FIG_DIR = OUTPUT_DIR / "figures" / "exp4"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = ["adamw", "sgd", "adamw_noclip"]
K_VALUES = [10, 15, 20, 25, 36]
Q_MEASURES = ["Q_grad", "Q_work", "Q_update"]
SEEDS = [42, 43, 44, 45, 46]


def load_results():
    path = OUTPUT_DIR / "exp4_dissipation_results.json"
    if not path.exists():
        print(f"[Exp 4] Run analyze_exp4_dissipation.py first.")
        return []
    with open(path) as f:
        return json.load(f)


def r_squared(x, y):
    if len(x) < 2:
        return float("nan")
    coeffs = np.polyfit(x, y, 1)
    pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


def plot_all():
    results = load_results()
    if not results:
        return

    fig, axes = plt.subplots(1, len(CONFIGS), figsize=(6 * len(CONFIGS), 5), sharey=False)
    if len(CONFIGS) == 1:
        axes = [axes]

    r2_table = []

    for ci, config_name in enumerate(CONFIGS):
        ax = axes[ci]
        runs = [r for r in results if r["config"] == config_name]

        for q_key, marker, color in zip(Q_MEASURES, ["o", "s", "^"], ["tab:blue", "tab:orange", "tab:green"]):
            xs, ys, errs = [], [], []
            for k in K_VALUES:
                k_runs = [r for r in runs if r["k"] == k and r[q_key] is not None]
                if not k_runs:
                    continue
                vals = [r[q_key] for r in k_runs]
                xs.append(math.log(k))
                ys.append(np.mean(vals))
                errs.append(np.std(vals))

            if xs:
                xs, ys, errs = np.array(xs), np.array(ys), np.array(errs)
                ax.errorbar(xs, ys, yerr=errs, fmt=f"{marker}-", capsize=3,
                            color=color, label=q_key)
                r2 = r_squared(xs, ys)
                r2_table.append({"config": config_name, "measure": q_key, "R2": round(r2, 4)})

        ax.set_xlabel("log(K)")
        ax.set_ylabel("Q (transition)")
        ax.set_title(config_name)
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "exp4_dissipation.pdf", dpi=200)
    fig.savefig(FIG_DIR / "exp4_dissipation.png", dpi=200)
    plt.close(fig)
    print(f"Saved {FIG_DIR / 'exp4_dissipation.png'}")

    # R² table
    print("\n" + "="*50)
    print("Experiment 4: R² for Q vs log(K)")
    print("="*50)
    print(f"{'Config':>15} {'Measure':>12} {'R²':>8}")
    print("-"*38)
    for row in r2_table:
        print(f"{row['config']:>15} {row['measure']:>12} {row['R2']:>8.4f}")

    with open(FIG_DIR / "exp4_r2_table.json", "w") as f:
        json.dump(r2_table, f, indent=2)


if __name__ == "__main__":
    plot_all()
