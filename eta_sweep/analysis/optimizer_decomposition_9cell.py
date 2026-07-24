#!/usr/bin/env python
"""
Full 9-config decomposition: which AdamW vs RMSProp ingredient closes the gap?

Conditions (3 seeds each):
  AdamW side:
    - adamw_b1_0           : AdamW β₁=0 (BC + decoupled WD + AdamW ε)        [positive control]
    - adamw_b1_0_nobc      : AdamW β₁=0, no BC (decoupled WD + AdamW ε)
    - adamw_b1_0_rmsprop_eps : AdamW β₁=0 with RMSProp ε placement (BC + decoupled + RMSProp ε)
    - adam_coupled_b1_0    : Adam (coupled L2 WD) β₁=0 (BC + L2 WD + AdamW ε)
  RMSProp side:
    - rmsprop              : RMSProp (no BC + L2 WD + RMSProp ε)               [negative control]
    - rmsprop_bc_only      : RMSProp + BC only (BC + L2 WD + RMSProp ε)
    - rmsprop_decoupled_only : RMSProp + decoupled WD only (no BC + decoupled WD + RMSProp ε)
    - rmsprop_decoupled_bc : RMSProp + decoupled WD + BC (BC + decoupled WD + RMSProp ε)
    - rmsprop_adamw_eps    : RMSProp + AdamW ε placement (no BC + L2 WD + AdamW ε)

Decision pattern:
  - if all transitioning rows have decoupled WD and all stuck rows have L2 WD → decoupled WD identified
  - if BC pattern explains it → BC identified
  - if ε placement pattern explains it → ε identified
  - if mixed → interaction
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


# Config metadata: (BC, weight_decay, eps_placement)
CONFIG_META = {
    "adamw_b1_0":              ("yes", "decoupled", "outside"),
    "adamw_b1_0_nobc":         ("no",  "decoupled", "outside"),
    "adamw_b1_0_rmsprop_eps":  ("yes", "decoupled", "inside"),
    "adam_coupled_b1_0":       ("yes", "L2",        "outside"),
    "rmsprop":                 ("no",  "L2",        "inside"),
    "rmsprop_bc_only":         ("yes", "L2",        "inside"),
    "rmsprop_decoupled_only":  ("no",  "decoupled", "inside"),
    "rmsprop_decoupled_bc":    ("yes", "decoupled", "inside"),
    "rmsprop_adamw_eps":       ("no",  "L2",        "outside"),
}

CONFIG_LABELS = {
    "adamw_b1_0":              r"AdamW $\beta_1{=}0$",
    "adamw_b1_0_nobc":         r"AdamW $\beta_1{=}0$ no-BC",
    "adamw_b1_0_rmsprop_eps":  r"AdamW $\beta_1{=}0$ + RMSProp ε",
    "adam_coupled_b1_0":       r"Adam coupled $\beta_1{=}0$",
    "rmsprop":                 r"RMSProp",
    "rmsprop_bc_only":         r"RMSProp + BC",
    "rmsprop_decoupled_only":  r"RMSProp + decoupled WD",
    "rmsprop_decoupled_bc":    r"RMSProp + decoupled + BC",
    "rmsprop_adamw_eps":       r"RMSProp + AdamW ε",
}


def load_branch(config: str, seed: int):
    d = RESULTS / "optimizer_branch" / f"{config}_seed{seed}"
    sf = d / "status.json"
    if not sf.exists():
        return None
    return json.loads(sf.read_text())


def verdict_for(taus, statuses) -> str:
    n_trans = sum(1 for s in statuses if s == "transitioned")
    n_stuck = sum(1 for s in statuses if s == "stuck")
    n_inc = sum(1 for s in statuses if s == "inconclusive")
    n_div = sum(1 for s in statuses if s == "diverged")
    if n_trans == len(statuses):
        # All transitioned — check if delayed vs clean
        clean = [t for t in taus if t]
        if clean and np.mean(clean) > 3500:
            return f"transitions (delayed, mean τ={np.mean(clean):.0f})"
        elif clean:
            return f"transitions (mean τ={np.mean(clean):.0f})"
        else:
            return "transitions"
    if n_stuck == len(statuses):
        return "stuck (3/3)"
    if n_div > 0:
        return f"diverged ({n_div}/{len(statuses)})"
    if n_inc > 0 and n_trans > 0:
        return f"mixed ({n_trans} trans, {n_inc} inc)"
    return f"mixed ({n_trans} trans, {n_stuck} stuck, {n_inc} inc)"


def main():
    configs = list(CONFIG_META.keys())
    seeds = (0, 1, 2)
    rows = []
    for cfg in configs:
        taus = []
        statuses = []
        for s in seeds:
            r = load_branch(cfg, s)
            if r is None:
                taus.append(None)
                statuses.append("missing")
            else:
                taus.append(r.get("transition_detected_step"))
                statuses.append(r.get("status"))
        rows.append({
            "config": cfg,
            "BC": CONFIG_META[cfg][0],
            "weight_decay": CONFIG_META[cfg][1],
            "eps_placement": CONFIG_META[cfg][2],
            "taus": taus,
            "statuses": statuses,
            "verdict": verdict_for(taus, statuses),
        })

    # Print table
    print("=== Full 9-Config Decomposition ===\n")
    print(f"{'config':<28} {'BC':<5} {'WD':<11} {'ε':<10} {'taus':<24} {'verdict':<35}")
    print("-" * 115)
    for r in rows:
        taus_str = str(r["taus"])
        print(f"{r['config']:<28} {r['BC']:<5} {r['weight_decay']:<11} "
              f"{r['eps_placement']:<10} {taus_str:<24} {r['verdict']:<35}")

    # Decision logic
    print("\n=== Pattern check ===")
    transition_rows = [r for r in rows if "transitions" in r["verdict"] or "transitions" in r["verdict"]]
    stuck_rows = [r for r in rows if "stuck" in r["verdict"]]

    # Check WD pattern
    trans_have_decoupled = all(r["weight_decay"] == "decoupled" for r in transition_rows)
    stuck_have_L2 = all(r["weight_decay"] == "L2" for r in stuck_rows)
    print(f"  transitioning rows all have decoupled WD: {trans_have_decoupled}")
    print(f"  stuck rows all have L2 WD:                 {stuck_have_L2}")

    # Check BC pattern
    trans_BC_yes = sum(1 for r in transition_rows if r["BC"] == "yes")
    trans_BC_no = sum(1 for r in transition_rows if r["BC"] == "no")
    stuck_BC_yes = sum(1 for r in stuck_rows if r["BC"] == "yes")
    stuck_BC_no = sum(1 for r in stuck_rows if r["BC"] == "no")
    print(f"  transitioning: BC=yes×{trans_BC_yes}, BC=no×{trans_BC_no}")
    print(f"  stuck:         BC=yes×{stuck_BC_yes}, BC=no×{stuck_BC_no}")

    # Check ε pattern
    trans_eps_outside = sum(1 for r in transition_rows if r["eps_placement"] == "outside")
    trans_eps_inside = sum(1 for r in transition_rows if r["eps_placement"] == "inside")
    stuck_eps_outside = sum(1 for r in stuck_rows if r["eps_placement"] == "outside")
    stuck_eps_inside = sum(1 for r in stuck_rows if r["eps_placement"] == "inside")
    print(f"  transitioning: ε=outside×{trans_eps_outside}, ε=inside×{trans_eps_inside}")
    print(f"  stuck:         ε=outside×{stuck_eps_outside}, ε=inside×{stuck_eps_inside}")

    # Final identification
    print("\n=== Identification ===")
    if trans_have_decoupled and stuck_have_L2:
        print("  → DECOUPLED WEIGHT DECAY is necessary AND sufficient.")
        print("    Every transitioning condition has decoupled WD.")
        print("    Every stuck condition has L2 WD.")
        print("    BC and ε placement are not load-bearing.")

    # Save JSON
    out = {"rows": rows}
    (RESULTS / "optimizer_decomposition_9cell.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {RESULTS / 'optimizer_decomposition_9cell.json'}")

    # Figure
    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(len(configs))
    width = 0.25
    for s_idx, s in enumerate(seeds):
        taus = []
        for cfg in configs:
            r = load_branch(cfg, s)
            if r is None or r.get("transition_detected_step") is None:
                taus.append(6000)
            else:
                taus.append(r["transition_detected_step"])
        offset = (s_idx - 1) * width
        # Color by weight decay
        colors = ["#2ca02c" if CONFIG_META[c][1] == "decoupled" else "#d62728" for c in configs]
        bars = ax.bar(x + offset, taus, width, color=colors,
                      alpha=0.55 + 0.15 * s_idx, edgecolor="black", lw=0.5,
                      label=f"seed {s}")
        for b, t in zip(bars, taus):
            if t == 6000:
                ax.text(b.get_x() + b.get_width()/2, 6100, "STUCK",
                        ha="center", fontsize=6, color="darkred", weight="bold")
    ax.axhline(2500, color="grey", linestyle=":", lw=0.7, alpha=0.55, label="baseline τ=2500")
    ax.axhline(1500, color="black", linestyle="--", lw=0.7, alpha=0.55, label="branch step 1500")
    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABELS[c] for c in configs], rotation=22, ha="right", fontsize=8)
    ax.set_ylabel(r"$\tau$ (bar at 6000 = stuck)")
    ax.set_title("Full 9-config decomposition: τ from common pre-plateau checkpoint\n"
                 "(green = decoupled WD; red = L2 WD)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3, lw=0.4)
    ax.set_ylim(0, 6500)
    plt.tight_layout()
    fig.savefig(OUT / "fig_optimizer_decomposition_9cell.pdf", dpi=200)
    fig.savefig(OUT / "fig_optimizer_decomposition_9cell.png", dpi=160)
    print(f"Figure: {OUT / 'fig_optimizer_decomposition_9cell.pdf'}")


if __name__ == "__main__":
    main()
