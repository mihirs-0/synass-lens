"""Final figure: all 3 panels at K=20, three seeds.

A. Marginal plateau (K=20, 3 seeds, dense behavioral from log.jsonl).
B. Plateau dissociation: differs vs swap_acc bar plot (K=20, 3 seeds).
C. Task-correct route at transition: L0 R_z paired seed lines (K=20).

Dense K=10 trajectory is now appendix-only (separate figure).
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


def load_k20_dense_behavioral():
    """Load 3-seed K=20 dense behavioral logs."""
    log_k = math.log(20)
    seeds = {}
    for s in (0, 1, 2):
        fn = (REPO / "eta_sweep" / "results" /
              f"eta_0.001_K_20_seed_{s}" / "log.jsonl")
        rows = [json.loads(l) for l in open(fn)]
        steps = np.array([r["step"] for r in rows])
        cand = np.array([r.get("candidate_loss") or np.nan for r in rows])
        top1 = np.array([r.get("candidate_accuracy") or np.nan for r in rows])
        # Find τ as first step where cand < 0.5 log K
        tau = None
        for i, r in enumerate(rows):
            if r.get("candidate_loss") is not None and \
               r["candidate_loss"] < 0.5 * log_k:
                tau = r["step"]; break
        if tau is None: tau = 4000
        seeds[s] = {"steps": steps, "cand": cand, "top1": top1, "tau": tau}
    return seeds, log_k


def main():
    pat = json.load(open(RES / "multi_seed_patching_K20.json"))
    wfs = json.load(open(RES / "within_fiber_swap.json"))
    seeds_data, log_k = load_k20_dense_behavioral()

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

    # === A: K=20 3-seed dense candidate-CE trajectory ===
    # Also overlay candidate_accuracy at 4 phase checkpoints (from patching run)
    for s, color in zip([0, 1, 2],
                         ["#1f77b4", "#3a8fcc", "#5fa7d8"]):
        d = seeds_data[s]
        nstep = d["steps"] / d["tau"]
        axA.plot(nstep, d["cand"] / log_k, color=color, lw=1.4, alpha=0.85,
                  zorder=1)
    # top-1 from patching JSON (sparse — 4 checkpoints per seed)
    for s, c in enumerate(cells):
        cell_pat = pat["cells"][c]["phase_results"]
        seed_color = ["#d62728", "#e35e5e", "#ec8888"][s]
        seed_tau = seeds_data[s]["tau"]
        steps_top, vals_top = [], []
        for ph_key in ("early", "transition", "post"):
            pp = cell_pat.get(ph_key, {})
            beh = pp.get("behavioral", {}) if pp else {}
            t1 = beh.get("candidate_accuracy")
            if t1 is None: continue
            stp = pp.get("step")
            if stp is None: continue
            steps_top.append(stp); vals_top.append(t1)
        if steps_top:
            axA.plot(np.array(steps_top) / seed_tau, vals_top, "o-",
                     color=seed_color, lw=1.2, ms=5, zorder=3, alpha=0.95)
    axA.plot([], [], color="#1f77b4", lw=1.4, label=r"cand. CE / $\log K$")
    axA.plot([], [], "o-", color="#d62728", lw=1.4, ms=5,
             label="clean top-1")
    axA.axhline(1.0, color="grey", ls=":", lw=0.7)
    axA.axhline(1/20, color="grey", ls=":", lw=0.7)
    axA.set_xlim(0, 2.0)
    axA.set_ylim(-0.05, 1.20)
    axA.set_xlabel(r"step / $\tau$")
    axA.set_ylabel("ratio / accuracy")
    axA.set_title("A. Marginal plateau\n($K{=}20$, 3 seeds)",
                  fontsize=9.5, pad=4)
    axA.legend(fontsize=7, loc="center right")
    axA.axvspan(0.0, 0.7, alpha=0.05, color="#1f77b4")
    axA.axvspan(0.7, 1.05, alpha=0.08, color="#ff7f0e")

    # === B: grouped bars (centerpiece) ===
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
    gap_low = swap_means[0]; gap_high = differs_means[0]
    axB.annotate("", xy=(0.0, gap_high - 0.02), xytext=(0.0, gap_low + 0.02),
                  arrowprops=dict(arrowstyle="<->", color="#ff6600", lw=2))
    axB.text(0.20, (gap_low + gap_high) / 2,
              "Phase-2\ngap", color="#cc4400", fontsize=8, ha="left",
              va="center", fontweight="bold")
    axB.axhline(1/20, color="grey", ls=":", lw=0.7, alpha=0.7)
    axB.text(1.85, 1/20 + 0.02, r"chance ($1/K$)", fontsize=6.5,
              color="grey", ha="right")
    axB.set_xticks(x)
    axB.set_xticklabels([g + r"\n($K{=}20$, $n{=}3$)" for g in groups],
                        fontsize=8.5)
    axB.set_xticks(x)
    axB.set_xticklabels([f"{g}\n($K{{=}}20$, $n{{=}}3$)" for g in groups],
                        fontsize=8)
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
    mean_p, mean_t = np.mean(R_p), np.mean(R_t)
    axC.plot([0, 1], [mean_p, mean_t], "s--", color="black", lw=1.5, ms=5,
              alpha=0.6, label="mean")
    axC.axhline(0, color="grey", ls=":", lw=0.7)
    axC.set_xticks([0, 1])
    axC.set_xticklabels(["plateau", "transition"])
    axC.set_xlim(-0.18, 1.18)
    axC.set_ylim(-0.45, 1.20)
    axC.set_ylabel(r"$L0$ patch recovery $R_z$")
    axC.set_title("C. Routing/lookup\nrises at transition",
                   fontsize=9.5, pad=4)
    axC.legend(fontsize=7, loc="upper left")

    out_path = WS / "fig_evidence.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.savefig(WS / "fig_evidence.png", bbox_inches="tight", dpi=160)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
