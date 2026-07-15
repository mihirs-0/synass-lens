#!/usr/bin/env python
"""
Eight-cell decomposition: which AdamW-vs-RMSProp implementation detail
explains the gap?

Conditions (3 seeds each, branched from common pre-plateau checkpoint):
  - adamw_b1_0           : positive control
  - adamw_b1_0_nobc      : AdamW β₁=0 with bias correction REMOVED
  - rmsprop              : negative control
  - rmsprop_decoupled_bc : RMSProp + decoupled WD + Adam-style BC

Plus the previous four configs (β₁=0.5, 0.9, 0.99, sgd_mom) for context.

Decision matrix:
  AdamW(b1=0,no-bc) | RMSProp(decoupled+bc) | conclusion
  STUCK             | TRANSITIONS           | both BC and decoupled WD matter
  TRANSITIONS       | TRANSITIONS           | decoupled WD sufficient; BC is speedup
  STUCK             | STUCK                 | something else gates the transition
  TRANSITIONS       | STUCK                 | neither closes the gap
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
    # 8-cell focus: bias-correction × decoupled-weight-decay decomposition
    focus_configs = ["adamw_b1_0", "adamw_b1_0_nobc", "rmsprop", "rmsprop_decoupled_bc"]
    focus_labels = {
        "adamw_b1_0":           r"AdamW $\beta_1{=}0$  (BC + decoupled WD)",
        "adamw_b1_0_nobc":      r"AdamW $\beta_1{=}0$, no BC  (decoupled WD only)",
        "rmsprop":              r"RMSProp  (no BC, L2 WD)",
        "rmsprop_decoupled_bc": r"RMSProp + decoupled WD + BC",
    }
    seeds = (0, 1, 2)

    print("=== 8-cell decomposition (3 seeds each) ===\n")
    results = {}
    for cfg in focus_configs:
        taus, statuses = [], []
        for s in seeds:
            r = load_branch(cfg, s)
            if r is None:
                continue
            taus.append(r["status"]["transition_detected_step"])
            statuses.append(r["status"]["status"])
        results[cfg] = {"taus": taus, "statuses": statuses}
        clean = [t for t in taus if t is not None]
        mean_t = np.mean(clean) if clean else None
        print(f"  {cfg:25s} τ={taus} status={statuses} mean={mean_t}")

    # ---- Decision ----
    nobc_transitions = any(s == "transitioned" for s in results["adamw_b1_0_nobc"]["statuses"])
    rms_dec_bc_transitions = all(s == "transitioned" for s in results["rmsprop_decoupled_bc"]["statuses"])

    print("\n=== Decision ===")
    print(f"  AdamW(β₁=0, no BC) transitions: {nobc_transitions}")
    print(f"  RMSProp + decoupled WD + BC transitions: {rms_dec_bc_transitions}")
    if nobc_transitions and rms_dec_bc_transitions:
        # Compute speed factor: τ_no_bc / τ_with_bc
        nobc_taus = [t for t in results["adamw_b1_0_nobc"]["taus"] if t is not None]
        baseline_taus = [t for t in results["adamw_b1_0"]["taus"] if t is not None]
        speed_factor = np.mean(nobc_taus) / np.mean(baseline_taus) if baseline_taus else None
        print(f"  → DECOUPLED WD IS SUFFICIENT to close the AdamW-RMSProp gap.")
        print(f"  → Bias correction is a speedup modulator: removing it slows τ by ~{speed_factor:.1f}×.")
        print(f"  → The geometric application of weight decay is the dominant mechanism.")

    # ---- Figure ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    log_K = np.log(10)

    # Panel (a): trajectories for the 4 focus configs
    ax = axes[0]
    colors = {"adamw_b1_0": "C2", "adamw_b1_0_nobc": "C5",
              "rmsprop": "C3", "rmsprop_decoupled_bc": "C0"}
    for cfg in focus_configs:
        for s in seeds:
            r = load_branch(cfg, s)
            if r is None or not r["log"]:
                continue
            st = np.array([row["step"] for row in r["log"]])
            cl = np.array([row["candidate_loss"] for row in r["log"]]) / log_K
            label = focus_labels[cfg] if s == 0 else None
            ax.plot(st, cl, color=colors[cfg], alpha=0.7, lw=1.4, label=label)
    ax.axvline(1500, color="black", linestyle="--", lw=0.8, alpha=0.6, label="branch @ 1500")
    ax.axhline(1.0, color="grey", linestyle="--", lw=0.5, alpha=0.4)
    ax.axhline(0.5, color="black", linestyle=":", lw=0.5, alpha=0.4)
    ax.set_xlabel("step")
    ax.set_ylabel("candidate loss / log K")
    ax.set_title("(a) Trajectories from common pre-plateau checkpoint")
    ax.legend(fontsize=8, loc="lower left", framealpha=0.92)
    ax.set_xlim(0, 6000)
    ax.set_ylim(-0.05, 1.25)

    # Panel (b): τ bar chart (3 seeds, 4 configs)
    ax = axes[1]
    x = np.arange(len(focus_configs))
    width = 0.25
    for s_idx, s in enumerate(seeds):
        taus = []
        for cfg in focus_configs:
            r = load_branch(cfg, s)
            if r is None or r["status"]["transition_detected_step"] is None:
                taus.append(6000)
            else:
                taus.append(r["status"]["transition_detected_step"])
        offset = (s_idx - 1) * width
        bars = ax.bar(x + offset, taus, width,
                      color=[colors[c] for c in focus_configs],
                      alpha=0.55 + 0.15 * s_idx,
                      edgecolor="black", lw=0.5,
                      label=f"seed {s}")
        # Mark stuck cells with text
        for i, (b, t) in enumerate(zip(bars, taus)):
            if t == 6000:
                ax.text(b.get_x() + b.get_width()/2, 6100, "STUCK",
                        ha="center", fontsize=6.5, color="red", weight="bold")
    ax.axhline(1500, color="black", linestyle="--", lw=0.7, alpha=0.55, label="branch")
    ax.axhline(2500, color="grey", linestyle=":", lw=0.7, alpha=0.55, label="baseline τ=2500")
    ax.set_xticks(x)
    ax.set_xticklabels([focus_labels[c] for c in focus_configs], rotation=18, ha="right", fontsize=8.5)
    ax.set_ylabel(r"$\tau$  (bar at 6000 = stuck)")
    ax.set_title("(b) τ per config across 3 seeds")
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.92)
    ax.grid(True, axis="y", alpha=0.3, lw=0.4)
    ax.set_ylim(0, 6500)

    fig.suptitle("AdamW vs RMSProp gap decomposed: bias correction × decoupled weight decay",
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT / "fig_optimizer_decomposition_8cell.pdf", dpi=200)
    fig.savefig(OUT / "fig_optimizer_decomposition_8cell.png", dpi=160)
    print(f"\n→ {OUT / 'fig_optimizer_decomposition_8cell.pdf'}")

    # Save JSON
    out = {c: {"taus": [int(t) if t else None for t in results[c]["taus"]],
               "statuses": results[c]["statuses"]} for c in focus_configs}
    (RESULTS / "optimizer_decomposition_8cell.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
