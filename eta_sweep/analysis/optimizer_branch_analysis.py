#!/usr/bin/env python
"""
Q-branch analysis: τ from common pre-plateau checkpoint under different
per-step update rules with FRESH optimizer state.

Tests the hypothesis "AdamW's success vs RMSProp/SGD+mom is about
first-moment denoising at the per-step level, not long-term accumulation."

Predictions per the hypothesis:
  - AdamW β₁=0       : fails (no first-moment averaging)
  - AdamW β₁=0.5     : maybe slow
  - AdamW β₁=0.9     : control, transitions
  - AdamW β₁=0.99    : transitions
  - RMSProp          : fails (no first-moment averaging)
  - SGD+mom (η=0.01) : fails (no preconditioning)

If β₁=0 transitions and RMSProp doesn't, the hypothesis is falsified;
the AdamW vs RMSProp difference is somewhere narrower (bias correction,
ε placement, decoupled vs L2 weight decay).
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


def load_branch(config: str, seed: int):
    d = RESULTS / "optimizer_branch" / f"{config}_seed{seed}"
    sf = d / "status.json"
    lf = d / "log.jsonl"
    if not sf.exists():
        return None
    s = json.loads(sf.read_text())
    log = [json.loads(l) for l in lf.read_text().strip().splitlines()] if lf.exists() else []
    return {"status": s, "log": log}


def main():
    configs = ["adamw_b1_0", "adamw_b1_0.5", "adamw_b1_0.9", "adamw_b1_0.99",
               "rmsprop", "sgd_mom"]
    config_labels = {
        "adamw_b1_0":   r"AdamW $\beta_1{=}0$",
        "adamw_b1_0.5": r"AdamW $\beta_1{=}0.5$",
        "adamw_b1_0.9": r"AdamW $\beta_1{=}0.9$ (control)",
        "adamw_b1_0.99":r"AdamW $\beta_1{=}0.99$",
        "rmsprop":      r"RMSProp $\alpha{=}0.999$",
        "sgd_mom":      r"SGD+mom $\beta{=}0.9$ ($\eta{=}10^{-2}$)",
    }
    seeds = (0, 1)

    print("=== Branch results (branched from common pre-plateau checkpoint at step 1500) ===\n")
    results = {}
    for cfg in configs:
        taus, statuses = [], []
        for s in seeds:
            r = load_branch(cfg, s)
            if r is None:
                continue
            taus.append(r["status"]["transition_detected_step"])
            statuses.append(r["status"]["status"])
        results[cfg] = {"taus": taus, "statuses": statuses}
        clean = [t for t in taus if t]
        mean_t = np.mean(clean) if clean else None
        print(f"  {cfg:18s} τ={taus} status={statuses} mean={mean_t}")

    # ---- Figure ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    log_K = np.log(10)

    # Panel (a): trajectories
    ax = axes[0]
    colors = {"adamw_b1_0":   "C0", "adamw_b1_0.5": "C9", "adamw_b1_0.9":"C2",
              "adamw_b1_0.99":"C5", "rmsprop":      "C3", "sgd_mom":     "C1"}
    for cfg in configs:
        for s in seeds:
            r = load_branch(cfg, s)
            if r is None or not r["log"]:
                continue
            st = np.array([row["step"] for row in r["log"]])
            cl = np.array([row["candidate_loss"] for row in r["log"]]) / log_K
            label = config_labels[cfg] if s == 0 else None
            ax.plot(st, cl, color=colors[cfg], alpha=0.7, lw=1.4, label=label)
    ax.axvline(1500, color="black", linestyle="--", lw=0.8, alpha=0.6, label="branch @ step 1500")
    ax.axhline(1.0, color="grey", linestyle="--", lw=0.5, alpha=0.4)
    ax.axhline(0.5, color="black", linestyle=":", lw=0.5, alpha=0.4)
    ax.set_xlabel("step")
    ax.set_ylabel("candidate loss / log K")
    ax.set_title("(a) Branch trajectories from common pre-plateau checkpoint")
    ax.legend(fontsize=7.5, loc="lower left", framealpha=0.92)
    ax.set_xlim(0, 6000)
    ax.set_ylim(-0.05, 1.25)

    # Panel (b): τ bar chart
    ax = axes[1]
    x = np.arange(len(configs))
    width = 0.35
    s0_taus = [results[c]["taus"][0] if len(results[c]["taus"]) >= 1 and results[c]["taus"][0] else 6000 for c in configs]
    s1_taus = [results[c]["taus"][1] if len(results[c]["taus"]) >= 2 and results[c]["taus"][1] else 6000 for c in configs]
    s0_stuck = [(t == 6000) for t in s0_taus]
    s1_stuck = [(t == 6000) for t in s1_taus]

    bars0 = ax.bar(x - width/2, s0_taus, width, color=[colors[c] for c in configs],
                    alpha=0.85, edgecolor="black", lw=0.6, label="seed 0")
    bars1 = ax.bar(x + width/2, s1_taus, width, color=[colors[c] for c in configs],
                    alpha=0.5, edgecolor="black", lw=0.6, label="seed 1")
    # Mark stuck cells
    for i, (b, stuck) in enumerate(zip(bars0, s0_stuck)):
        if stuck:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 50, "STUCK",
                    ha="center", fontsize=7, color="red", weight="bold")
    for i, (b, stuck) in enumerate(zip(bars1, s1_stuck)):
        if stuck:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 50, "STUCK",
                    ha="center", fontsize=7, color="red", weight="bold")
    ax.axhline(1500, color="black", linestyle="--", lw=0.7, alpha=0.6, label="branch @ 1500")
    ax.axhline(2500, color="grey", linestyle=":", lw=0.7, alpha=0.55, label="baseline τ=2500")
    ax.set_xticks(x)
    ax.set_xticklabels([config_labels[c] for c in configs], rotation=30, ha="right", fontsize=8.5)
    ax.set_ylabel(r"$\tau$ (transition step) — bar at 6000 if stuck")
    ax.set_title("(b) τ per config (2 seeds each)")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.92)
    ax.grid(True, axis="y", alpha=0.3, lw=0.4)

    fig.suptitle("Optimizer-rule branching from common pre-plateau weights "
                 "(step 1500, fresh optimizer state)", fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT / "fig_optimizer_branch.pdf", dpi=200)
    fig.savefig(OUT / "fig_optimizer_branch.png", dpi=160)
    print(f"\n→ {OUT / 'fig_optimizer_branch.pdf'}")

    # ---- Conclusion ----
    print("\n=== Conclusion ===")
    adamw_taus = []
    for c in ["adamw_b1_0", "adamw_b1_0.5", "adamw_b1_0.9", "adamw_b1_0.99"]:
        adamw_taus.extend([t for t in results[c]["taus"] if t])
    rms_stuck = all(s == "stuck" for s in results["rmsprop"]["statuses"])
    sgdm_stuck = all(s == "stuck" for s in results["sgd_mom"]["statuses"])
    b1_0_trans = all(s == "transitioned" for s in results["adamw_b1_0"]["statuses"])

    print(f"  AdamW β₁∈{{0, 0.5, 0.9, 0.99}}: all transition. mean τ = {np.mean(adamw_taus):.0f}")
    print(f"  AdamW β₁=0 transitions: {b1_0_trans}")
    print(f"  RMSProp stuck:          {rms_stuck}")
    print(f"  SGD+mom stuck:          {sgdm_stuck}")
    if b1_0_trans and rms_stuck:
        print("  → 'first-moment smoothing' hypothesis FALSIFIED: β₁=0 still transitions,")
        print("     but RMSProp doesn't. The AdamW-vs-RMSProp gap is narrower than first-moment.")

    out = {c: {"taus": [int(t) if t else None for t in results[c]["taus"]],
               "statuses": results[c]["statuses"]} for c in configs}
    (RESULTS / "optimizer_branch_analysis.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
