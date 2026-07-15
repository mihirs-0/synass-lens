#!/usr/bin/env python
"""
Composite main-text figure: pre-loss diagnostics during the plateau.

Per Reviewer 7 / 8 / 9 ask: the z-shuffle gap, gradient log-slope, and
attention-to-z all in one figure, with τ marked.

Reuses existing v_tracking trajectories (3 seeds, K=10, η=10⁻³).
Output: outputs/paper_figures/fig_plateau_diagnostics.{pdf,png}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = REPO_ROOT / "eta_sweep" / "results"
OUT = REPO_ROOT / "outputs" / "paper_figures"


def load_run(seed: int):
    """Load v_tracking trajectory for one seed."""
    base = RESULTS / "v_tracking" / f"eta_0.001_K_10_seed_{seed}"
    log = [json.loads(l) for l in (base / "log.jsonl").read_text().strip().splitlines()]
    aggs = [json.loads(l) for l in (base / "v_aggregates.jsonl").read_text().strip().splitlines()]
    status = json.loads((base / "status.json").read_text())
    return log, aggs, status


def gradient_log_slope(steps, gnorms, window=5):
    """Estimate d log(|g|) / d step using a rolling window of window pts."""
    log_g = np.log10(np.maximum(gnorms, 1e-12))
    slopes = np.zeros_like(log_g)
    for i in range(len(steps)):
        lo = max(0, i - window)
        hi = min(len(steps), i + window + 1)
        if hi - lo < 2:
            slopes[i] = 0.0
            continue
        s, _ = np.polyfit(steps[lo:hi], log_g[lo:hi], 1)
        slopes[i] = s
    return slopes


def main():
    seeds = (0, 1, 2)
    runs = {s: load_run(s) for s in seeds}
    taus = {s: r[2]["transition_detected_step"] for s, r in runs.items()}
    log_K = np.log(10)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7), sharex=False)

    # === Panel (a): cand_loss/log K and Δz overlay ===
    ax = axes[0, 0]
    for s, (log, _, _) in runs.items():
        st = np.array([r["step"] for r in log])
        cl = np.array([r["candidate_loss"] / log_K for r in log])
        ax.plot(st, cl, alpha=0.7, lw=1.4, label=f"seed {s}")
        ax.axvline(taus[s], color="grey", linestyle=":", alpha=0.35)
    ax.axhline(1.0, color="black", linestyle="--", lw=0.7, alpha=0.4, label="log K floor")
    ax.axhline(0.5, color="green", linestyle="--", lw=0.7, alpha=0.4, label="50% transition cut")
    ax.set_xlabel("step")
    ax.set_ylabel("candidate loss / log K")
    ax.set_title("(a) Loss trajectory")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_xlim(0, 5000)
    ax.set_ylim(-0.05, 1.2)

    # === Panel (b): Δz onset (selector sensitivity) ===
    ax = axes[0, 1]
    for s, (log, _, _) in runs.items():
        st = np.array([r["step"] for r in log])
        dz = np.array([r["delta_z"] for r in log])
        ax.plot(st, dz, alpha=0.75, lw=1.4, label=f"seed {s}")
        ax.axvline(taus[s], color="grey", linestyle=":", alpha=0.35)
    ax.axhline(0.1, color="orange", linestyle="--", lw=0.7, alpha=0.5, label="Δz onset threshold")
    ax.set_xlabel("step")
    ax.set_ylabel(r"$\Delta_z$ (z-shuffle gap, nats)")
    ax.set_title("(b) Selector sensitivity onset")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(0, 5000)
    ax.set_yscale("symlog", linthresh=0.01)

    # === Panel (c): gradient norm and its log-slope ===
    ax = axes[1, 0]
    for s, (log, _, _) in runs.items():
        st = np.array([r["step"] for r in log])
        g = np.array([r["grad_norm_sq_held_out"] for r in log])
        ax.plot(st, g, alpha=0.75, lw=1.4, label=f"seed {s}")
        ax.axvline(taus[s], color="grey", linestyle=":", alpha=0.35)
    ax.set_xlabel("step")
    ax.set_ylabel(r"$\|g\|^2$ (held-out batch)")
    ax.set_yscale("log")
    ax.set_title("(c) Gradient norm")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0, 5000)

    # === Panel (d): gradient log-slope (sign change) ===
    ax = axes[1, 1]
    for s, (log, _, _) in runs.items():
        st = np.array([r["step"] for r in log])
        g = np.array([r["grad_norm_sq_held_out"] for r in log])
        slopes = gradient_log_slope(st, g, window=4)
        ax.plot(st, slopes, alpha=0.75, lw=1.4, label=f"seed {s}")
        ax.axvline(taus[s], color="grey", linestyle=":", alpha=0.35)
    ax.axhline(0.0, color="black", linestyle="--", lw=0.7, alpha=0.5, label="zero (slope flips)")
    ax.set_xlabel("step")
    ax.set_ylabel(r"$d \log_{10}\|g\|^2 / d t$ (rolling window)")
    ax.set_title("(d) Gradient log-slope (sign change at $\\sim 0.4\\tau$)")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0, 5000)

    fig.suptitle("Plateau-internal diagnostics — three measurements move during the apparent flat phase",
                 fontsize=12)
    plt.tight_layout()
    fig.savefig(OUT / "fig_plateau_diagnostics.pdf", dpi=200)
    fig.savefig(OUT / "fig_plateau_diagnostics.png", dpi=160)
    print(f"  → {OUT / 'fig_plateau_diagnostics.pdf'}")


if __name__ == "__main__":
    main()
