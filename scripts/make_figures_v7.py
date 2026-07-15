"""Two separate main-text figures.

fig_phase_trajectory.pdf: phase story.
  A: marginal plateau (cand CE / logK + clean top-1, 3 seeds K=20)
  B: within-fiber swap (differs from clean vs swap accuracy)

fig_controls_patching.pdf: controls + causal evidence.
  A: z-swap vs matched random (flip rate + KL bars)
  B: patch recovery at plateau vs transition, 3 seeds
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
        for r in rows:
            if r.get("candidate_loss") is not None and \
               r["candidate_loss"] < 0.5 * log_k:
                tau = r["step"]; break
        if tau is None: tau = 4000
        seeds[s] = {"steps": steps, "cand": cand, "tau": tau}
    return seeds, log_k


# ---------- Figure 2: phase trajectory ----------
def fig_phase_trajectory():
    pat = json.load(open(RES / "multi_seed_patching_K20.json"))
    wfs = json.load(open(RES / "within_fiber_swap.json"))
    seeds_data, log_k = load_k20_dense_behavioral()
    cells = ["CellA_seed0", "CellA_seed1", "CellA_seed2"]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 2.5))
    plt.subplots_adjust(left=0.08, right=0.99, top=0.86, bottom=0.20,
                        wspace=0.32)

    # === Panel A: marginal plateau ===
    for s, color in zip([0, 1, 2], ["#1f77b4", "#3a8fcc", "#5fa7d8"]):
        d = seeds_data[s]
        nstep = d["steps"] / d["tau"]
        axA.plot(nstep, d["cand"] / log_k, color=color, lw=1.4, alpha=0.9)
    # clean top-1 dots (4 phase ckpts/seed)
    for s_i, c in enumerate(cells):
        cell_pat = pat["cells"][c]["phase_results"]
        col = ["#d62728", "#e35e5e", "#ec8888"][s_i]
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
                     color=col, lw=1.0, ms=5, alpha=0.95)
    # legend proxies
    axA.plot([], [], color="#1f77b4", lw=1.4, label=r"cand. CE / $\log K$")
    axA.plot([], [], "o-", color="#d62728", lw=1.4, ms=5,
             label="clean top-1")
    # reference lines
    axA.axhline(1.0, color="grey", ls=":", lw=0.7)
    axA.axhline(1/20, color="grey", ls=":", lw=0.7)
    axA.text(2.02, 1.0, r"$\log K$", fontsize=7, color="grey", va="center")
    axA.text(2.02, 1/20, r"$1/K$", fontsize=7, color="grey", va="center")
    axA.axvspan(0.0, 0.7, alpha=0.05, color="#1f77b4")
    axA.axvspan(0.7, 1.05, alpha=0.08, color="#ff7f0e")
    axA.set_xlim(0, 2.0); axA.set_ylim(-0.05, 1.20)
    axA.set_xlabel(r"step / $\tau$")
    axA.set_ylabel("ratio / accuracy")
    axA.set_title("A. Marginal plateau before lookup", fontsize=9.5, pad=4)
    axA.legend(fontsize=7.5, loc="center right", framealpha=0.9)

    # === Panel B: within-fiber swap, mean ± std across seeds ===
    phases = ["plateau", "transition", "post"]
    differs_K20 = {p: [] for p in phases}
    swap_K20 = {p: [] for p in phases}
    for c in cells:
        for ph in phases:
            wfs_p = wfs["cells"][c]["phases"].get(ph, {})
            df = wfs_p.get("swap_predicts_differs_from_clean_rate")
            if df is not None: differs_K20[ph].append(df)
            sa = wfs_p.get("swap_accuracy")
            if sa is not None: swap_K20[ph].append(sa)

    x = np.arange(len(phases), dtype=float)
    df_mean = [np.mean(differs_K20[p]) for p in phases]
    df_std = [np.std(differs_K20[p]) for p in phases]
    sa_mean = [np.mean(swap_K20[p]) for p in phases]
    sa_std = [np.std(swap_K20[p]) for p in phases]

    axB.errorbar(x - 0.06, df_mean, yerr=df_std, fmt="s-", color="#8c564b",
                  ms=6, lw=1.5, capsize=4, label="differs from clean")
    axB.errorbar(x + 0.06, sa_mean, yerr=sa_std, fmt="o-", color="#9467bd",
                  ms=6, lw=1.5, capsize=4, label="swap accuracy")
    axB.axhline(1/20, color="grey", ls=":", lw=0.7)
    axB.text(2.10, 1/20, r"$1/K$", fontsize=7, color="grey", va="center")
    axB.set_xticks(x); axB.set_xticklabels(phases)
    axB.set_xlim(-0.30, 2.30); axB.set_ylim(-0.05, 1.10)
    axB.set_ylabel("rate")
    axB.set_title(r"B. Within-fiber swap: $z$ matters before it is correct",
                   fontsize=9.5, pad=4)
    axB.legend(fontsize=7.5, loc="center right", framealpha=0.9)

    out = WS / "fig_phase_trajectory.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(WS / "fig_phase_trajectory.png", bbox_inches="tight",
                 dpi=160)
    print(f"wrote {out}")
    plt.close()


# ---------- Figure 3: controls + patching ----------
def fig_controls_patching():
    ctrl = json.load(open(RES / "within_fiber_swap_controls.json"))
    raw = json.load(open(RES / "raw_patching_K20.json"))
    cells = ["CellA_seed0", "CellA_seed1", "CellA_seed2"]

    # plateau control values per seed
    sw_diff, rd_diff, kl_sw, kl_rd = [], [], [], []
    for c in cells:
        s = ctrl["cells"][c]["plateau"]["summary"]
        sw_diff.append(s["swap_differs_rate"])
        rd_diff.append(s["random_differs_rate"])
        kl_sw.append(s["kl_sw_mean"])
        kl_rd.append(s["kl_random_mean"])

    # raw patching: corruption gap and recovered gap, per seed and phase
    corrupt_p, corrupt_t = [], []
    rec_p, rec_t = [], []
    for c in cells:
        for ph_key, gap_lst, rec_lst in [
            ("plateau", corrupt_p, rec_p),
            ("transition", corrupt_t, rec_t),
        ]:
            pp = raw["cells"][c]["phases"][ph_key]
            gap_lst.append(pp["denom_mean (L_corrupt - L_clean)"])
            rec_lst.append(pp["abs_recovery_mean (L_corrupt - L_patch_L0)"])

    fig = plt.figure(figsize=(7.0, 2.55))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 0.95, 1.30],
                           left=0.08, right=0.99, top=0.84, bottom=0.21,
                           wspace=0.50)
    axA1 = fig.add_subplot(gs[0, 0])
    axA2 = fig.add_subplot(gs[0, 1])
    axB = fig.add_subplot(gs[0, 2])

    # === A1: prediction flip rate ===
    bw = 0.45
    x = np.array([0.0, 1.0])
    axA1.bar(x[0], np.mean(sw_diff), bw, yerr=np.std(sw_diff),
              color="#9467bd", capsize=4, edgecolor="white",
              label=r"$z$-swap")
    axA1.bar(x[1], np.mean(rd_diff), bw, yerr=np.std(rd_diff),
              color="#9e9e9e", capsize=4, edgecolor="white",
              label="matched random")
    ratio_d = np.mean(sw_diff) / np.mean(rd_diff)
    ymax = max(np.mean(sw_diff) + np.std(sw_diff),
                np.mean(rd_diff) + np.std(rd_diff))
    axA1.text(0.5, ymax + 0.06, f"{ratio_d:.1f}$\\times$",
               ha="center", fontsize=9, color="#cc4400", fontweight="bold")
    axA1.set_xticks(x); axA1.set_xticklabels([r"$z$-swap", "rand."],
                                              fontsize=8)
    axA1.set_ylim(0, max(1.0, ymax * 1.4))
    axA1.set_ylabel("prediction flip rate", fontsize=8.5)
    axA1.set_title("A1. Flip rate", fontsize=9, pad=4)

    # === A2: KL from clean ===
    axA2.bar(x[0], np.mean(kl_sw), bw, yerr=np.std(kl_sw),
              color="#9467bd", capsize=4, edgecolor="white",
              label=r"$z$-swap")
    axA2.bar(x[1], np.mean(kl_rd), bw, yerr=np.std(kl_rd),
              color="#9e9e9e", capsize=4, edgecolor="white",
              label="matched random")
    ratio_kl = np.mean(kl_sw) / np.mean(kl_rd)
    ymax2 = max(np.mean(kl_sw) + np.std(kl_sw),
                 np.mean(kl_rd) + np.std(kl_rd))
    axA2.text(0.5, ymax2 + 0.012, f"{ratio_kl:.1f}$\\times$",
               ha="center", fontsize=9, color="#cc4400", fontweight="bold")
    axA2.set_xticks(x); axA2.set_xticklabels([r"$z$-swap", "rand."],
                                              fontsize=8)
    axA2.set_ylim(0, ymax2 * 1.4)
    axA2.set_ylabel("KL from clean (nats)", fontsize=8.5)
    axA2.set_title("A2. KL from clean", fontsize=9, pad=4)

    # === B: raw corruption gap and recovered gap, per seed and phase ===
    # x-axis groups: plateau (x=0,1), transition (x=2,3); each group has
    # corruption gap and recovered gap
    x_corr = np.array([0.0, 2.0])  # corruption gap positions
    x_rec = np.array([1.0, 3.0])   # recovered gap positions

    # individual seed dots
    for s_i in range(3):
        # corruption gap (open circles)
        axB.scatter(x_corr[0], corrupt_p[s_i], s=42,
                     facecolors="none", edgecolors="#666666", lw=1.4,
                     zorder=2)
        axB.scatter(x_corr[1], corrupt_t[s_i], s=42,
                     facecolors="none", edgecolors="#666666", lw=1.4,
                     zorder=2)
        # recovered gap (filled black)
        axB.scatter(x_rec[0], rec_p[s_i], s=42, color="black", zorder=2)
        axB.scatter(x_rec[1], rec_t[s_i], s=42, color="black", zorder=2)

    # small mean markers (short horizontal bars)
    for xi, vals, col in [
        (x_corr[0], corrupt_p, "#666666"),
        (x_corr[1], corrupt_t, "#666666"),
        (x_rec[0], rec_p, "black"),
        (x_rec[1], rec_t, "black"),
    ]:
        m = np.mean(vals)
        axB.plot([xi - 0.20, xi + 0.20], [m, m], "-", color=col, lw=2.5,
                  zorder=3)

    # legend proxies
    axB.scatter([], [], s=42, facecolors="none", edgecolors="#666666",
                 lw=1.4, label="corruption gap\n$L_{\\rm corr} - L_{\\rm clean}$")
    axB.scatter([], [], s=42, color="black",
                 label="recovered gap\n$L_{\\rm corr} - L_{\\rm patch}$")

    axB.axhline(0, color="grey", ls=":", lw=0.7)
    axB.set_xticks([0.5, 2.5])
    axB.set_xticklabels(["plateau", "transition"])
    axB.set_xlim(-0.6, 3.6)
    axB.set_ylabel("nats")
    axB.set_title("B. Raw corruption and recovered gaps",
                   fontsize=9.5, pad=4)
    axB.legend(fontsize=6.5, loc="upper left", framealpha=0.9,
                handletextpad=0.4)

    out = WS / "fig_controls_patching.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(WS / "fig_controls_patching.png", bbox_inches="tight",
                 dpi=160)
    print(f"wrote {out}")
    plt.close()


def main():
    fig_phase_trajectory()
    fig_controls_patching()


if __name__ == "__main__":
    main()
