"""Cleaner v8 figure. 3 panels, each with one clear story.

A: behavior — cand CE/logK + top-1 vs step/τ.
B: within-fiber swap — differs vs swap accuracy, with Phase-2 region
   annotated (the gap where z matters but is not yet correct).
C: K=20 replicated — individual seed dots + mean line, no big error bars.

No Δz inset; Δz is a supporting metric mentioned only in text.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

WS = REPO / "workshop_paper_calibrated"
RES = REPO / "eta_sweep" / "results"


def main():
    dk10 = json.load(open(RES / "dense_k10_trajectory.json"))
    pat = json.load(open(RES / "multi_seed_patching_K20.json"))
    wfs = json.load(open(RES / "within_fiber_swap.json"))

    log_k = dk10["log_k"]
    tau = dk10["tau_estimate"]
    ckpts = dk10["checkpoints"]
    steps = np.array([c["step"] for c in ckpts])
    nstep = steps / tau

    beh = dk10["dense_behavioral"]
    dense_steps = sorted(int(s) for s in beh.keys())
    dense_steps = np.array([s for s in dense_steps if s <= 6000])
    dense_cand = np.array([beh[str(s)]["candidate_loss"] for s in dense_steps])
    dense_top1 = np.array([beh[str(s)]["candidate_accuracy"] for s in dense_steps])
    dense_nstep = dense_steps / tau

    swap_acc = np.array([c["swap_swap_acc"] or np.nan for c in ckpts])
    differs = np.array([c["swap_differs_from_clean"] or np.nan for c in ckpts])

    # ----- 3-panel figure -----
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(7.4, 2.5))
    plt.subplots_adjust(left=0.07, right=0.99, top=0.85, bottom=0.21,
                        wspace=0.32)

    # --- A: behavior ---
    axA.plot(dense_nstep, dense_cand / log_k, color="#1f77b4", lw=1.6,
             label=r"cand. CE / $\log K$")
    axA.plot(dense_nstep, dense_top1, color="#d62728", lw=1.6,
             label="clean top-1")
    axA.axhline(1.0, color="grey", ls=":", lw=0.7)
    axA.axhline(0.1, color="grey", ls=":", lw=0.7)
    axA.set_xlim(0, 2.0)
    axA.set_ylim(-0.05, 1.20)
    axA.set_xlabel(r"step / $\tau$")
    axA.set_ylabel("ratio / accuracy")
    axA.set_title("A. Behavior:\nmarginal plateau then lookup",
                  fontsize=9.5)
    axA.legend(fontsize=7.5, loc="center right")

    # --- B: within-fiber swap with annotated Phase 2 region ---
    axB.plot(nstep, differs, "s--", color="#8c564b", lw=1.6, ms=4.5,
             label="differs from clean")
    axB.plot(nstep, swap_acc, "o-", color="#9467bd", lw=1.6, ms=4.5,
             label="swap accuracy")
    axB.axhline(0.1, color="grey", ls=":", lw=0.7)

    # Find the Phase 2 region: differs > swap_acc + 0.15 and swap_acc < 0.5
    p2_mask = (~np.isnan(differs)) & (~np.isnan(swap_acc)) & \
              (differs - swap_acc > 0.15) & (swap_acc < 0.5)
    if p2_mask.any():
        p2_x = nstep[p2_mask]
        p2_lo = swap_acc[p2_mask]
        p2_hi = differs[p2_mask]
        axB.fill_between(p2_x, p2_lo, p2_hi,
                          color="#ff9933", alpha=0.18,
                          label=r"Phase 2: $z$ matters, not correct")

    axB.set_xlim(0, 2.0)
    axB.set_ylim(-0.05, 1.05)
    axB.set_xlabel(r"step / $\tau$")
    axB.set_ylabel("rate")
    axB.set_title(r"B. Within-fiber swap:" + "\n" +
                  r"$z$ matters before it is correct",
                  fontsize=9.5)
    axB.legend(fontsize=7, loc="lower right")

    for ax in (axA, axB):
        ax.axvspan(0.0, 0.7, alpha=0.05, color="#1f77b4")
        ax.axvspan(0.7, 1.05, alpha=0.08, color="#ff7f0e")

    # --- C: K=20 replication, individual seed dots ---
    cells = ["CellA_seed0", "CellA_seed1", "CellA_seed2"]
    phases = ["plateau", "transition", "post"]
    R_data = {ph: [] for ph in phases}
    top1_data = {ph: [] for ph in phases}
    swap_data = {ph: [] for ph in phases}
    for c in cells:
        for ph in phases:
            patch_key = "early" if ph == "plateau" else ph
            pp = pat["cells"][c]["phase_results"].get(patch_key, {})
            wfs_p = wfs["cells"][c]["phases"].get(ph, {})
            beh = pp.get("behavioral", {})
            rec = (pp.get("patching", {}).get("recovery_per_layer", [None]*4)
                   or [None]*4)
            if rec[0] is not None:
                R_data[ph].append(rec[0])
            top = beh.get("candidate_accuracy")
            if top is not None: top1_data[ph].append(top)
            sa = wfs_p.get("swap_accuracy")
            if sa is not None: swap_data[ph].append(sa)

    x_pos = np.arange(3, dtype=float)
    metrics = [
        ("top-1", top1_data, "#d62728", "s", -0.18),
        ("swap acc", swap_data, "#9467bd", "^", 0.0),
        (r"$L0$ $R_z$", R_data, "#ff7f0e", "o", 0.18),
    ]
    for label, data, col, marker, dx in metrics:
        # Mean line
        means = [np.mean(data[p]) for p in phases]
        axC.plot(x_pos + dx, means, color=col, lw=1.0, ls="-", alpha=0.7,
                 zorder=1)
        # Individual seed dots, jittered
        for j, p in enumerate(phases):
            for v in data[p]:
                axC.plot(x_pos[j] + dx, v, marker=marker, color=col,
                         ms=5, mec="white", mew=0.5, ls="none", zorder=2)
        # Legend proxy
        axC.plot([], [], marker=marker, color=col, ls="-", lw=1.0,
                 ms=5, label=label)

    axC.axhline(0, color="grey", ls=":", lw=0.5)
    axC.axhline(0.05, color="grey", ls=":", lw=0.5, alpha=0.5)
    axC.set_xticks(x_pos)
    axC.set_xticklabels(phases)
    axC.set_ylim(-0.45, 1.20)
    axC.set_ylabel("value")
    axC.set_title(r"C. $K{=}20$ replication:" + "\n" +
                  "rise at transition (3 seed dots)",
                  fontsize=9.5)
    axC.legend(fontsize=7.5, loc="upper left")

    out_path = WS / "fig_evidence.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.savefig(WS / "fig_evidence.png", bbox_inches="tight", dpi=160)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
