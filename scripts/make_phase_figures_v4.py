"""Final claim figure: 3 panels, each one fact.

A. Marginal plateau established (small).
B. The plateau dissociation: z changes predictions, but not correctly.
   GROUPED BARS — plateau vs post — showing differs ≫ swap_acc at plateau,
   both high post.  Phase-2 gap annotated.
C. Task-correct route appears at transition.
   Paired seed lines for L0 R_z (plateau → transition only).
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
    beh = dk10["dense_behavioral"]
    dense_steps = sorted(int(s) for s in beh.keys())
    dense_steps = np.array([s for s in dense_steps if s <= 5000])
    dense_cand = np.array([beh[str(s)]["candidate_loss"] for s in dense_steps])
    dense_top1 = np.array([beh[str(s)]["candidate_accuracy"] for s in dense_steps])
    dense_nstep = dense_steps / tau

    # --- aggregate K=20 3-seed swap data per phase ---
    cells = ["CellA_seed0", "CellA_seed1", "CellA_seed2"]
    swap_phases = ["plateau", "transition", "post"]
    differs_K20 = {p: [] for p in swap_phases}
    swap_K20 = {p: [] for p in swap_phases}
    for c in cells:
        for ph in swap_phases:
            wfs_p = wfs["cells"][c]["phases"].get(ph, {})
            df = wfs_p.get("swap_predicts_differs_from_clean_rate")
            if df is not None: differs_K20[ph].append(df)
            sa = wfs_p.get("swap_accuracy")
            if sa is not None: swap_K20[ph].append(sa)
    # K=10 dense at canonical plateau & post checkpoints
    # plateau: average of normalized step 0.4–0.7 from dense traj
    ckpts = dk10["checkpoints"]
    plateau_ck = [c for c in ckpts if 0.3 <= c["step"] / tau <= 0.7]
    post_ck = [c for c in ckpts if c["step"] / tau >= 1.5]
    differs_K10_plateau = np.mean([c["swap_differs_from_clean"]
                                    for c in plateau_ck])
    swap_K10_plateau = np.mean([c["swap_swap_acc"] for c in plateau_ck])
    differs_K10_post = np.mean([c["swap_differs_from_clean"]
                                 for c in post_ck])
    swap_K10_post = np.mean([c["swap_swap_acc"] for c in post_ck])

    # --- aggregate K=20 patch L0 ---
    R_K20 = {p: [] for p in ("plateau", "transition")}
    for c in cells:
        for ph in ("plateau", "transition"):
            patch_key = "early" if ph == "plateau" else ph
            pp = pat["cells"][c]["phase_results"].get(patch_key, {})
            rec = (pp.get("patching", {}).get("recovery_per_layer", [None]*4)
                   or [None]*4)
            if rec[0] is not None:
                R_K20[ph].append(rec[0])

    # ----- figure -----
    fig = plt.figure(figsize=(7.4, 2.55))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.6, 1.2],
                           left=0.07, right=0.99, top=0.84, bottom=0.20,
                           wspace=0.40)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])

    # === A: marginal plateau established ===
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
    axA.set_title("A. Marginal plateau\nestablished",
                  fontsize=9.5, pad=4)
    axA.legend(fontsize=7, loc="center right")
    axA.axvspan(0.0, 0.7, alpha=0.05, color="#1f77b4")
    axA.axvspan(0.7, 1.05, alpha=0.08, color="#ff7f0e")

    # === B: grouped bar plot (centerpiece) ===
    # Two phase groups: Plateau, Post-transition
    # Two bars each: differs (brown), swap_acc (purple)
    # Use K=20 3-seed for error bars, but also show K=10 dense as second group
    groups = ["Plateau", "Post"]
    differs_means = [np.mean(differs_K20["plateau"]),
                     np.mean(differs_K20["post"])]
    differs_stds = [np.std(differs_K20["plateau"]),
                    np.std(differs_K20["post"])]
    swap_means = [np.mean(swap_K20["plateau"]),
                  np.mean(swap_K20["post"])]
    swap_stds = [np.std(swap_K20["plateau"]),
                 np.std(swap_K20["post"])]

    x = np.arange(len(groups), dtype=float)
    bw = 0.36
    axB.bar(x - bw/2, differs_means, bw, yerr=differs_stds,
             color="#8c564b", alpha=0.92, label="differs from clean",
             capsize=4, edgecolor="white")
    axB.bar(x + bw/2, swap_means, bw, yerr=swap_stds,
             color="#9467bd", alpha=0.92, label="swap accuracy",
             capsize=4, edgecolor="white")
    # Phase-2 gap annotation on plateau
    gap_low = swap_means[0]
    gap_high = differs_means[0]
    axB.annotate("", xy=(0.0, gap_high - 0.02), xytext=(0.0, gap_low + 0.02),
                  arrowprops=dict(arrowstyle="<->", color="#ff6600", lw=2))
    axB.text(0.20, (gap_low + gap_high) / 2,
              "Phase-2\ngap", color="#cc4400", fontsize=8, ha="left",
              va="center", fontweight="bold")
    # chance line
    axB.axhline(0.10, color="grey", ls=":", lw=0.7, alpha=0.7)
    axB.text(1.85, 0.12, "chance ($1/K$)", fontsize=6.5, color="grey",
              ha="right")
    axB.set_xticks(x)
    axB.set_xticklabels([g + "\n(K=20, n=3)" for g in groups], fontsize=8.5)
    axB.set_ylim(-0.05, 1.18)
    axB.set_ylabel("rate")
    axB.set_title("B. Plateau: $z$ changes predictions,\nbut not correctly",
                   fontsize=9.5, pad=4)
    axB.legend(fontsize=7.5, loc="upper left")

    # === C: paired seed lines, plateau → transition ===
    R_p = R_K20["plateau"]
    R_t = R_K20["transition"]
    seed_colors = ["#ff7f0e", "#2ca02c", "#1f77b4"]
    for i, (rp, rt, c) in enumerate(zip(R_p, R_t, seed_colors)):
        axC.plot([0, 1], [rp, rt], "o-", color=c, lw=1.4, ms=6,
                  label=f"seed {i}")
    # mean line in black
    mean_p, mean_t = np.mean(R_p), np.mean(R_t)
    axC.plot([0, 1], [mean_p, mean_t], "s--", color="black", lw=1.5, ms=5,
              alpha=0.6, label="mean")
    axC.axhline(0, color="grey", ls=":", lw=0.7)
    axC.set_xticks([0, 1])
    axC.set_xticklabels(["plateau", "transition"])
    axC.set_xlim(-0.18, 1.18)
    axC.set_ylim(-0.45, 1.20)
    axC.set_ylabel(r"$L0$ patch recovery $R_z$")
    axC.set_title(r"C. Task-correct route" + "\n" +
                   r"appears at transition",
                   fontsize=9.5, pad=4)
    axC.legend(fontsize=7, loc="upper left")

    out_path = WS / "fig_evidence.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.savefig(WS / "fig_evidence.png", bbox_inches="tight", dpi=160)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
