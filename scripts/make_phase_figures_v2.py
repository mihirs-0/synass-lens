"""Generate v8 figures: clean 3-panel empirical evidence.

  fig_evidence.pdf
    A. Dense K=10 cand/logK + top-1 over normalized step
    B. Within-fiber swap: differs vs swap-acc over normalized step
    C. K=20 3-seed L0 patch recovery: plateau / transition / post

The TikZ schematic is in the .tex file directly (not built here).
"""
from __future__ import annotations
import json
import math
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
    dense_dz = np.array([
        beh[str(s)]["delta_z"] if beh[str(s)]["delta_z"] is not None else np.nan
        for s in dense_steps
    ])
    dense_nstep = dense_steps / tau

    swap_acc = np.array([c["swap_swap_acc"] or np.nan for c in ckpts])
    differs = np.array([c["swap_differs_from_clean"] or np.nan for c in ckpts])

    # --- 3-panel figure ---
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(7.2, 2.4))
    plt.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.20,
                        wspace=0.32)

    # Panel A: dense K=10 cand/logK + top-1 + Δz inset
    axA.plot(dense_nstep, dense_cand / log_k, color="#1f77b4", lw=1.5,
             label=r"cand. CE / $\log K$")
    axA.plot(dense_nstep, dense_top1, color="#d62728", lw=1.5,
             label=r"clean top-1")
    axA.axhline(1.0, color="grey", ls=":", lw=0.7)
    axA.axhline(0.1, color="grey", ls=":", lw=0.7)
    axA.set_xlim(0, 2.0)
    axA.set_ylim(-0.05, 1.20)
    axA.set_xlabel(r"step / $\tau$")
    axA.set_ylabel("ratio / accuracy")
    axA.set_title(r"A. K=10 dense", fontsize=10)
    axA.legend(fontsize=7, loc="center right")
    # Δz inset
    axAi = axA.inset_axes([0.10, 0.55, 0.35, 0.40])
    axAi.plot(dense_nstep, dense_dz, color="#2ca02c", lw=1.0)
    axAi.set_yscale("symlog", linthresh=0.1)
    axAi.set_title(r"$\Delta_z$", fontsize=7, pad=2)
    axAi.tick_params(axis="both", labelsize=5)
    axAi.set_xlim(0, 2.0)

    # Panel B: within-fiber swap
    axB.plot(nstep, differs, "s--", color="#8c564b", lw=1.4, ms=4,
             label="differs from clean")
    axB.plot(nstep, swap_acc, "o-", color="#9467bd", lw=1.4, ms=4,
             label="swap accuracy")
    axB.axhline(0.1, color="grey", ls=":", lw=0.7)
    axB.set_xlim(0, 2.0)
    axB.set_ylim(-0.05, 1.05)
    axB.set_xlabel(r"step / $\tau$")
    axB.set_ylabel("rate")
    axB.set_title(r"B. K=10 within-fiber swap", fontsize=10)
    axB.legend(fontsize=7, loc="center right")
    # phase shading on A and B
    for ax in (axA, axB):
        ax.axvspan(0.0, 0.7, alpha=0.05, color="#1f77b4")
        ax.axvspan(0.7, 1.05, alpha=0.08, color="#ff7f0e")

    # Panel C: K=20 3-seed L0 patch recovery
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
    x_pos = np.arange(3)
    R_mean = [np.mean(R_data[p]) for p in phases]
    R_std = [np.std(R_data[p]) for p in phases]
    top1_mean = [np.mean(top1_data[p]) for p in phases]
    top1_std = [np.std(top1_data[p]) for p in phases]
    swap_mean = [np.mean(swap_data[p]) for p in phases]
    swap_std = [np.std(swap_data[p]) for p in phases]
    axC.errorbar(x_pos - 0.15, R_mean, yerr=R_std, fmt="o-",
                 color="#ff7f0e", ms=6, lw=1.4, capsize=4,
                 label=r"$L0$ patch $R_z$")
    axC.errorbar(x_pos, top1_mean, yerr=top1_std, fmt="s-",
                 color="#d62728", ms=5, lw=1.2, capsize=4, label="top-1")
    axC.errorbar(x_pos + 0.15, swap_mean, yerr=swap_std, fmt="^-",
                 color="#9467bd", ms=5, lw=1.2, capsize=4,
                 label="swap acc")
    axC.axhline(0, color="grey", ls=":", lw=0.7)
    axC.set_xticks(x_pos)
    axC.set_xticklabels(phases)
    axC.set_ylim(-0.45, 1.15)
    axC.set_ylabel("value")
    axC.set_title(r"C. K=20 (3 seeds, mean $\pm$ std)", fontsize=10)
    axC.legend(fontsize=7, loc="lower right")

    out_path = WS / "fig_evidence.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.savefig(WS / "fig_evidence.png", bbox_inches="tight", dpi=160)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
