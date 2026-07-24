#!/usr/bin/env python
"""
Four-arm optimizer overlay for paper §6.

Plots candidate_loss / log K vs step for all four optimizer arms on a
single axis.  Vanilla SGD is shown as a representative K=5 trace (the
scout did not include K=10, but the stuck signature is K-independent;
caveat in caption).  AdamW, RMSProp, and SGD+momentum are all K=10.

Output: outputs/paper_figures/fig_optimizer_decomposition.{pdf,png}
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ETA_SWEEP_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ETA_SWEEP_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = REPO_ROOT / "outputs" / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_log(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (steps, candidate_loss, delta_z) arrays."""
    rows = [json.loads(l) for l in path.read_text().strip().splitlines()]
    steps = np.array([r["step"] for r in rows])
    cand = np.array([r["candidate_loss"] for r in rows])
    dz = np.array([r["delta_z"] for r in rows])
    return steps, cand, dz


def mean_over_seeds(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average candidate_loss/log K across seeds.  Aligns to the shortest run."""
    series = [load_log(p) for p in paths]
    n_min = min(len(s[0]) for s in series)
    steps = series[0][0][:n_min]
    cand = np.stack([s[1][:n_min] for s in series])
    return steps, cand.mean(axis=0), cand.std(axis=0)


def main():
    log_k_10 = math.log(10)
    log_k_5 = math.log(5)

    arms = {}

    # AdamW K=10, 3 seeds
    adamw_paths = [
        ETA_SWEEP_ROOT / "results" / "v_tracking" / f"eta_0.001_K_10_seed_{s}" / "log.jsonl"
        for s in (0, 1, 2)
    ]
    adamw_steps, adamw_mean, adamw_std = mean_over_seeds(adamw_paths)
    arms["AdamW (K=10, η=10⁻³)"] = (adamw_steps, adamw_mean / log_k_10, adamw_std / log_k_10, "C2")

    # RMSProp K=10, 3 seeds at η=1e-3
    rms_paths = [
        ETA_SWEEP_ROOT / "results" / "manual_precond" / f"eta_0.001_K_10_seed_{s}" / "log.jsonl"
        for s in (0, 1, 2)
    ]
    rms_steps, rms_mean, rms_std = mean_over_seeds(rms_paths)
    arms["RMSProp (K=10, η=10⁻³)"] = (rms_steps, rms_mean / log_k_10, rms_std / log_k_10, "C0")

    # SGD+momentum=0.9 K=10, 3 seeds at η=0.01
    sgdm_paths = [
        ETA_SWEEP_ROOT / "results" / "sgd_momentum_control" / f"eta_0.01_K_10_seed_{s}" / "log.jsonl"
        for s in (0, 1, 2)
    ]
    sgdm_steps, sgdm_mean, sgdm_std = mean_over_seeds(sgdm_paths)
    arms["SGD+mom=0.9 (K=10, η=10⁻²)"] = (sgdm_steps, sgdm_mean / log_k_10, sgdm_std / log_k_10, "C3")

    # Vanilla SGD: K=5, η=0.01 (scout did not include K=10; stuck signature is K-independent)
    sgd_path = ETA_SWEEP_ROOT / "results" / "sgd_control" / "eta_0.01_K_5_seed_0" / "log.jsonl"
    sgd_steps, sgd_cand, sgd_dz = load_log(sgd_path)
    arms["SGD vanilla (K=5*, η=10⁻²)"] = (sgd_steps, sgd_cand / log_k_5, np.zeros_like(sgd_cand), "C1")

    # ---- Plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 4.5))

    for label, (steps, mean, std, color) in arms.items():
        ax.plot(steps, mean, label=label, color=color, lw=1.6)
        if std.max() > 0:
            ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.18)

    ax.axhline(1.0, color="grey", linestyle="--", lw=0.8, alpha=0.55,
               label="marginal floor (cand_loss = log K)")
    ax.axhline(0.5, color="black", linestyle=":", lw=0.8, alpha=0.55,
               label="transition cut (50% of log K)")

    ax.set_xlabel("training step")
    ax.set_ylabel("candidate loss / log K")
    ax.set_ylim(-0.05, 1.25)
    ax.set_xlim(0, 7000)
    ax.legend(loc="center right", fontsize=8.5, framealpha=0.92)
    ax.grid(True, alpha=0.25, linewidth=0.4)

    ax.set_title("Four-arm optimizer comparison", fontsize=11)

    plt.tight_layout()
    pdf_path = OUT_DIR / "fig_optimizer_decomposition.pdf"
    png_path = OUT_DIR / "fig_optimizer_decomposition.png"
    fig.savefig(pdf_path, dpi=200)
    fig.savefig(png_path, dpi=160)
    print(f"  → {pdf_path}")
    print(f"  → {png_path}")


if __name__ == "__main__":
    main()
