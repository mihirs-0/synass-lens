"""Final claim figure with swap-vs-matched-random control as Panel B.

A. Marginal plateau (K=20, 3 seeds, dense cand CE + top-1 dots).
B. Plateau swap specificity: within-fiber swap vs matched-norm random
   perturbation. Bar groups: differs rate, KL.
C. Routing/lookup transition: L0 patch recovery paired seed lines.
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
    log_k = math.log(20)
    seeds = {}
    for s in (0, 1, 2):
        fn = (REPO / "eta_sweep" / "results" /
              f"eta_0.001_K_20_seed_{s}" / "log.jsonl")
        rows = [json.loads(l) for l in open(fn)]
        steps = np.array([r["step"] for r in rows])
        cand = np.array([r.get("candidate_loss") or np.nan for r in rows])
        tau = None
        for i, r in enumerate(rows):
            if r.get("candidate_loss") is not None and \
               r["candidate_loss"] < 0.5 * log_k:
                tau = r["step"]; break
        if tau is None: tau = 4000
        seeds[s] = {"steps": steps, "cand": cand, "tau": tau}
    return seeds, log_k


def main():
    pat = json.load(open(RES / "multi_seed_patching_K20.json"))
    ctrl = json.load(open(RES / "within_fiber_swap_controls.json"))
    seeds_data, log_k = load_k20_dense_behavioral()

    cells = ["CellA_seed0", "CellA_seed1", "CellA_seed2"]

    # --- gather plateau swap-vs-random per seed ---
    sw_diff = []; rd_diff = []; kl_sw = []; kl_rd = []
    for c in cells:
        s = ctrl["cells"][c]["plateau"]["summary"]
        sw_diff.append(s["swap_differs_rate"])
        rd_diff.append(s["random_differs_rate"])
        kl_sw.append(s["kl_sw_mean"])
        kl_rd.append(s["kl_random_mean"])

    # --- patching paired seed lines ---
    R_p = []; R_t = []
    for c in cells:
        for ph_key, lst in [("early", R_p), ("transition", R_t)]:
            pp = pat["cells"][c]["phase_results"].get(ph_key, {})
            rec = (pp.get("patching", {}).get("recovery_per_layer", [None]*4)
                   or [None]*4)
            if rec[0] is not None:
                lst.append(rec[0])

    # --- figure ---
    fig = plt.figure(figsize=(7.6, 2.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.55, 1.15],
                           left=0.07, right=0.99, top=0.83, bottom=0.20,
                           wspace=0.42)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])

    # === A: marginal plateau ===
    for s, color in zip([0, 1, 2],
                         ["#1f77b4", "#3a8fcc", "#5fa7d8"]):
        d = seeds_data[s]
        nstep = d["steps"] / d["tau"]
        axA.plot(nstep, d["cand"] / log_k, color=color, lw=1.4, alpha=0.85,
                  zorder=1)
    for s_i, c in enumerate(cells):
        cell_pat = pat["cells"][c]["phase_results"]
        seed_color = ["#d62728", "#e35e5e", "#ec8888"][s_i]
        seed_tau = seeds_data[s_i]["tau"]
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
    axA.set_xlim(0, 2.0); axA.set_ylim(-0.05, 1.20)
    axA.set_xlabel(r"step / $\tau$")
    axA.set_ylabel("ratio / accuracy")
    axA.set_title("A. Marginal plateau", fontsize=9.5, pad=4)
    axA.legend(fontsize=7, loc="center right")
    axA.axvspan(0.0, 0.7, alpha=0.05, color="#1f77b4")
    axA.axvspan(0.7, 1.05, alpha=0.08, color="#ff7f0e")

    # === B: swap vs matched random at plateau ===
    # Two grouped bars: (left) differs rate, (right) KL
    x = np.array([0.0, 1.0])
    bw = 0.34
    sw_diff_m = np.mean(sw_diff); sw_diff_s = np.std(sw_diff)
    rd_diff_m = np.mean(rd_diff); rd_diff_s = np.std(rd_diff)

    # Left axis: differs rate
    axB.bar(x[0] - bw/2, sw_diff_m, bw, yerr=sw_diff_s,
             color="#9467bd", alpha=0.92, capsize=4, edgecolor="white",
             label="$z$-swap")
    axB.bar(x[0] + bw/2, rd_diff_m, bw, yerr=rd_diff_s,
             color="#bdbdbd", alpha=0.92, capsize=4, edgecolor="white",
             label="matched random")
    axB.set_xticks(x); axB.set_xticklabels(
        [f"differs rate", f"KL from clean"], fontsize=8.5)
    axB.set_ylabel("differs rate (left scale)", fontsize=8)
    axB.set_ylim(0, 0.85)
    axB.legend(fontsize=7.5, loc="upper right")

    # Right axis: KL
    axB2 = axB.twinx()
    kl_sw_m = np.mean(kl_sw); kl_sw_s = np.std(kl_sw)
    kl_rd_m = np.mean(kl_rd); kl_rd_s = np.std(kl_rd)
    axB2.bar(x[1] - bw/2, kl_sw_m, bw, yerr=kl_sw_s,
              color="#9467bd", alpha=0.92, capsize=4, edgecolor="white")
    axB2.bar(x[1] + bw/2, kl_rd_m, bw, yerr=kl_rd_s,
              color="#bdbdbd", alpha=0.92, capsize=4, edgecolor="white")
    axB2.set_ylabel("KL (right scale, nats)", fontsize=8, color="#666666")
    axB2.set_ylim(0, 0.18)
    axB2.tick_params(axis="y", labelsize=7, colors="#666666")

    # ratio annotations
    axB.text(x[0], sw_diff_m + sw_diff_s + 0.04, f"{sw_diff_m/rd_diff_m:.1f}×",
             ha="center", fontsize=8, color="#cc4400", fontweight="bold")
    axB2.text(x[1], kl_sw_m + kl_sw_s + 0.012, f"{kl_sw_m/kl_rd_m:.1f}×",
              ha="center", fontsize=8, color="#cc4400", fontweight="bold")

    axB.set_title("B. Plateau: $z$-swap vs matched random\n($K{=}20$, 3 seeds)",
                   fontsize=9.5, pad=4)

    # === C: patching paired seeds ===
    seed_colors = ["#ff7f0e", "#2ca02c", "#1f77b4"]
    for i, (rp, rt, c) in enumerate(zip(R_p, R_t, seed_colors)):
        axC.plot([0, 1], [rp, rt], "o-", color=c, lw=1.4, ms=6,
                  label=f"seed {i}")
    mean_p, mean_t = np.mean(R_p), np.mean(R_t)
    axC.plot([0, 1], [mean_p, mean_t], "s--", color="black", lw=1.5, ms=5,
              alpha=0.6, label="mean")
    axC.axhline(0, color="grey", ls=":", lw=0.7)
    axC.set_xticks([0, 1]); axC.set_xticklabels(["plateau", "transition"])
    axC.set_xlim(-0.18, 1.18); axC.set_ylim(-0.45, 1.20)
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
