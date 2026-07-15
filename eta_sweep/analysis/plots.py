#!/usr/bin/env python
"""
Plots for the η-sweep.

Produces four figures under eta_sweep/results/figures/:
  01_per_run_trajectories.png  — faceted candidate-loss + Δz per (η, K)
  02_Q_vs_logK_per_eta.png     — six panels with affine + regime fits
  03_c_of_eta.png              — log-log c(η) with the meta-fit overlay
  04_diagnostics.png           — (a) frac excess, (b) snap width / τ, (c) success fraction

Usage:
    python eta_sweep/analysis/plots.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ETA_SWEEP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ETA_SWEEP_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from eta_sweep.config import (  # noqa: E402
    AGGREGATED_PARQUET,
    ETA_VALUES,
    K_VALUES,
    META_FIT_JSON,
    PER_ETA_FITS_JSON,
    RESULTS_DIR,
    run_dir,
)

FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 140,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})


# ---------------------------------------------------------------------------
# Figure 1: per-run trajectories
# ---------------------------------------------------------------------------

def _load_log(eta: float, k: int, seed: int) -> pd.DataFrame:
    path = run_dir(eta, k, seed) / "log.jsonl"
    if not path.exists():
        return pd.DataFrame()
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def plot_per_run(df: pd.DataFrame, seed: int = 0) -> None:
    etas = sorted(df["eta"].unique())
    ks = sorted(df["k"].unique())
    n_eta, n_k = len(etas), len(ks)
    fig, axes = plt.subplots(n_eta, n_k, figsize=(2.2 * n_k, 2.0 * n_eta),
                             squeeze=False)

    for i, eta in enumerate(etas):
        for j, k in enumerate(ks):
            ax = axes[i, j]
            log = _load_log(eta, k, seed)
            if log.empty:
                ax.set_title(f"η={eta:g}  K={k}\n[no data]",
                             fontsize=8)
                ax.axis("off")
                continue
            ax2 = ax.twinx()
            ax.plot(log["step"], log["candidate_loss"], color="#2E86AB",
                    label="cand loss", linewidth=1.3)
            ax.axhline(math.log(k), color="grey", linestyle=":",
                       linewidth=0.8)
            ax2.plot(log["step"], log["delta_z"], color="#E76F51",
                     label="Δz", linewidth=1.0, alpha=0.7)
            ax.set_title(f"η={eta:g}  K={k}", fontsize=8)
            ax.set_ylim(bottom=0)
            ax2.set_ylim(bottom=0)
            if i == n_eta - 1:
                ax.set_xlabel("step")
            if j == 0:
                ax.set_ylabel("cand loss")
            if j == n_k - 1:
                ax2.set_ylabel("Δz", color="#E76F51")
            ax.tick_params(labelsize=7)
            ax2.tick_params(labelsize=7)
    fig.suptitle(f"Per-run candidate loss (blue) and Δz (red)  seed={seed}",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG_DIR / "01_per_run_trajectories.png")
    plt.close(fig)
    print(f"wrote {FIG_DIR / '01_per_run_trajectories.png'}")


# ---------------------------------------------------------------------------
# Figure 2: Q vs log K per η
# ---------------------------------------------------------------------------

def plot_Q_vs_logK(agg: pd.DataFrame, fits: Dict) -> None:
    etas = sorted(agg["eta"].unique())
    n = len(etas)
    ncols = 3
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows),
                             squeeze=False)

    for idx, eta in enumerate(etas):
        ax = axes[idx // ncols, idx % ncols]
        sub = agg[agg["eta"] == eta].dropna(subset=["Q"])
        by_k = sub.groupby("k")["Q"].mean().reset_index().sort_values("k")
        K = by_k["k"].to_numpy(dtype=float)
        Q = by_k["Q"].to_numpy(dtype=float)
        if len(K) == 0:
            ax.set_title(f"η={eta:g}  [no data]")
            continue
        ax.scatter(K, Q, color="#2E86AB", s=40, zorder=3)
        ax.set_xscale("log")
        ax.set_xlabel("K")
        ax.set_ylabel("Q")
        key = f"{eta:g}"
        entry = fits.get(key, {})
        a = entry.get("affine", {})
        r = entry.get("regime", {})
        K_grid = np.geomspace(max(K.min(), 2), K.max() * 1.1, 64)
        if "c" in a and not math.isnan(a.get("c", float("nan"))):
            ax.plot(K_grid, a["c"] * np.log(K_grid) + a["q0"], "--",
                    color="#888", linewidth=1,
                    label=f"affine c={a['c']:.2f} R²={a['R2']:.2f}")
        if "c" in r and not math.isnan(r.get("c", float("nan"))):
            ax.plot(K_grid, r["c"] * np.log(K_grid / r["K_star"]), "-",
                    color="#E76F51", linewidth=1.2,
                    label=f"regime c={r['c']:.2f}  K*={r['K_star']:.2f}  R²={r['R2']:.2f}")
        ax.legend(loc="best", fontsize=7)
        ax.set_title(f"η={eta:g}")

    # Hide any unused panels
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_Q_vs_logK_per_eta.png")
    plt.close(fig)
    print(f"wrote {FIG_DIR / '02_Q_vs_logK_per_eta.png'}")


# ---------------------------------------------------------------------------
# Figure 3: c(η) with meta-fit overlay
# ---------------------------------------------------------------------------

def plot_c_of_eta(meta: Dict) -> None:
    if meta.get("status") != "ok":
        print("meta_fit not ok; skipping Figure 3")
        return
    etas = np.asarray(meta["eta_values"], dtype=float)
    cs = np.asarray(meta["c_values"], dtype=float)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(etas, cs, s=70, color="#2E86AB", zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("η (learning rate)")
    ax.set_ylabel("c  (slope of Q vs log K)")
    ax.set_title(f"c(η) — winner = {meta['winner']}")

    # Overlay the winning fit
    eta_grid = np.geomspace(etas.min() * 0.5, etas.max() * 2.0, 120)
    log_eta = np.log(eta_grid)
    winner = meta["winner_params"]
    form = winner["form"]
    if form == "power_law":
        log_c = winner["beta"] * log_eta + winner["log_A"]
        label = f"c ∝ η^{winner['beta']:.2f}  R²={winner['R2_log']:.2f}"
    elif form == "quadratic":
        log_c = winner["a"] * log_eta ** 2 + winner["b"] * log_eta + winner["d"]
        label = f"quadratic  R²={winner['R2_log']:.2f}"
    else:
        log_c = np.full_like(log_eta, winner["log_c_mean"])
        label = f"constant c = {winner['c_mean']:.2f}"
    ax.plot(eta_grid, np.exp(log_c), "-", color="#E76F51", linewidth=1.5,
            label=label)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_c_of_eta.png")
    plt.close(fig)
    print(f"wrote {FIG_DIR / '03_c_of_eta.png'}")


# ---------------------------------------------------------------------------
# Figure 4: diagnostic panels
# ---------------------------------------------------------------------------

def plot_diagnostics(agg: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    etas = sorted(agg["eta"].unique())
    valid = agg.dropna(subset=["Q"])

    # (a) (Q − ΔL)/Q vs η, one point per (η, K)
    ax = axes[0]
    for eta in etas:
        sub = valid[valid["eta"] == eta]
        if sub.empty:
            continue
        ax.scatter([eta] * len(sub), sub["frac_excess"], alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("η")
    ax.set_ylabel("(Q − ΔL) / Q")
    ax.set_title("Irreversible excess fraction")

    # (b) Snap width (steps) vs η, coloured by K
    ax = axes[1]
    for k in sorted(valid["k"].unique()):
        sub = valid[valid["k"] == k]
        ax.scatter(sub["eta"], sub["snap_width_steps"], label=f"K={k}", alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("η")
    ax.set_ylabel("snap width (steps)")
    ax.set_title("Snap width")
    ax.legend(fontsize=7)

    # (c) Fraction of successful transitions per η.
    ax = axes[2]
    per_eta_success = []
    per_eta_total = []
    for eta in etas:
        sub = agg[agg["eta"] == eta]
        success = int(((sub["status"] == "transitioned")).sum())
        total = int(len(sub))
        per_eta_success.append(success / total if total else 0.0)
        per_eta_total.append(total)
    ax.bar(range(len(etas)), per_eta_success, color="#2E86AB")
    ax.set_xticks(range(len(etas)))
    ax.set_xticklabels([f"{e:g}" for e in etas], rotation=30)
    ax.set_ylabel("fraction transitioned")
    ax.set_title("Success fraction per η")
    for i, total in enumerate(per_eta_total):
        ax.text(i, 0.02, f"n={total}", ha="center", fontsize=7,
                color="white" if per_eta_success[i] > 0.2 else "black")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_diagnostics.png")
    plt.close(fig)
    print(f"wrote {FIG_DIR / '04_diagnostics.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not AGGREGATED_PARQUET.exists():
        print(f"Missing {AGGREGATED_PARQUET}; run compute_Q.py first.")
        return
    agg = pd.read_parquet(AGGREGATED_PARQUET)

    fits: Dict = {}
    if PER_ETA_FITS_JSON.exists():
        with open(PER_ETA_FITS_JSON) as f:
            fits = json.load(f)

    meta: Dict = {}
    if META_FIT_JSON.exists():
        with open(META_FIT_JSON) as f:
            meta = json.load(f)

    plot_per_run(agg, seed=0)
    plot_Q_vs_logK(agg, fits)
    if meta:
        plot_c_of_eta(meta)
    plot_diagnostics(agg)


if __name__ == "__main__":
    main()
