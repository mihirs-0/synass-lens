"""Generate v7 figures:

  fig_phase_trajectory.pdf      Two-panel phase trajectory.
                                Left: dense K=10 single-seed. Right: K=20 3-seed band.

Output goes into the workshop_paper_calibrated/ dir so the .tex can include it.
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

WS_DIR = REPO_ROOT / "workshop_paper_calibrated"
RES_DIR = REPO_ROOT / "eta_sweep" / "results"


def load_dense_k10():
    return json.load(open(RES_DIR / "dense_k10_trajectory.json"))


def load_k20_3seed():
    pat = json.load(open(RES_DIR / "multi_seed_patching_K20.json"))
    wfs = json.load(open(RES_DIR / "within_fiber_swap.json"))
    return pat, wfs


def plot_dense_k10(ax_top, ax_mid, ax_bot, dk10):
    log_k = dk10["log_k"]
    tau = dk10["tau_estimate"]
    ckpts = dk10["checkpoints"]
    steps = np.array([c["step"] for c in ckpts])
    nstep = steps / tau

    # behavioral (dense from training_history)
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

    # CE/log K and top-1
    ax_top.plot(dense_nstep, dense_cand / log_k,
                color="#1f77b4", lw=1.6, label=r"cand. CE / $\log K$")
    ax_top.plot(dense_nstep, dense_top1, color="#d62728", lw=1.6,
                label=r"clean top-1")
    ax_top.axhline(1.0, color="grey", ls=":", lw=0.8)
    ax_top.axhline(1.0/10, color="grey", ls=":", lw=0.8)
    ax_top.set_ylabel("ratio / accuracy")
    ax_top.set_ylim(-0.05, 1.25)
    ax_top.legend(fontsize=8, loc="center right")
    ax_top.set_title(r"K=10 canonical (single seed): dense trajectory",
                     fontsize=10)

    # Δz on its own axis
    ax_mid.plot(dense_nstep, dense_dz, color="#2ca02c", lw=1.6,
                label=r"$\Delta_z$ (nats)")
    ax_mid.set_ylabel(r"$\Delta_z$ (nats)")
    ax_mid.legend(fontsize=8, loc="center right")
    ax_mid.set_yscale("symlog", linthresh=0.1)

    # Within-fiber swap acc + differs (sparse from our run)
    swap_acc = np.array([c["swap_swap_acc"] or np.nan for c in ckpts])
    differs = np.array([c["swap_differs_from_clean"] or np.nan for c in ckpts])
    ax_bot.plot(nstep, swap_acc, "o-", color="#9467bd", lw=1.4, ms=4,
                label="within-fiber swap acc")
    ax_bot.plot(nstep, differs, "s--", color="#8c564b", lw=1.4, ms=3.5,
                label="differs from clean")
    ax_bot.axhline(1.0/10, color="grey", ls=":", lw=0.8)
    ax_bot.set_ylabel("rate")
    ax_bot.set_ylim(-0.05, 1.05)
    ax_bot.legend(fontsize=8, loc="center right")
    ax_bot.set_xlabel(r"step / $\tau$ (transition $\approx$ step "
                      f"{tau})")

    # phase shading
    for ax in (ax_top, ax_mid, ax_bot):
        ax.axvspan(0.0, 0.7, alpha=0.05, color="#1f77b4")
        ax.axvspan(0.7, 1.05, alpha=0.08, color="#ff7f0e")
        ax.axvspan(1.05, dense_nstep.max(), alpha=0.05, color="#2ca02c")
    # phase labels at top
    ax_top.text(0.35, 1.18, "Phase 1+2", ha="center", fontsize=8,
                color="#1f77b4")
    ax_top.text(0.85, 1.18, "Phase 3+4", ha="center", fontsize=8,
                color="#ff7f0e")
    ax_top.text(min(1.5, dense_nstep.max()*0.9), 1.18, "post",
                ha="center", fontsize=8, color="#2ca02c")
    ax_top.set_xlim(0, dense_nstep.max())


def plot_k20_3seed_split(ax_rates, ax_dz, pat_data, wfs_data):
    """Mean ± seed-band at plateau / transition / post for K=20.
    Top: rate-like metrics on [0,1]. Bottom: Δz on its own axis."""
    log_k = math.log(20)
    cells = ["CellA_seed0", "CellA_seed1", "CellA_seed2"]
    phases = ["plateau", "transition", "post"]

    rows = {ph: {"cand_ratio": [], "top1": [], "dz": [], "R_L0": [],
                 "swap_acc": [], "differs": []} for ph in phases}

    for c in cells:
        pat_cell = pat_data["cells"][c]["phase_results"]
        wfs_cell = wfs_data["cells"][c]["phases"]
        for ph in phases:
            patch_key = "early" if ph == "plateau" else ph
            pp = pat_cell.get(patch_key, {})
            wfs_p = wfs_cell.get(ph, {})
            beh = pp.get("behavioral", {}) if pp else {}
            rec = (pp.get("patching", {}).get("recovery_per_layer", [None]*4)
                   or [None]*4)
            cl = beh.get("candidate_loss")
            if cl is not None: rows[ph]["cand_ratio"].append(cl / log_k)
            top = beh.get("candidate_accuracy")
            if top is not None: rows[ph]["top1"].append(top)
            dz = beh.get("delta_z")
            if dz is not None: rows[ph]["dz"].append(dz)
            if rec[0] is not None: rows[ph]["R_L0"].append(rec[0])
            sa = wfs_p.get("swap_accuracy")
            if sa is not None: rows[ph]["swap_acc"].append(sa)
            df = wfs_p.get("swap_predicts_differs_from_clean_rate")
            if df is not None: rows[ph]["differs"].append(df)

    rate_metrics = [
        ("cand_ratio", r"cand/$\log K$", "#1f77b4"),
        ("top1", r"top-1", "#d62728"),
        ("R_L0", r"$R_{L0}$", "#ff7f0e"),
        ("swap_acc", "swap acc", "#9467bd"),
        ("differs", "differs", "#8c564b"),
    ]
    x_pos = np.arange(len(phases))
    bw = 0.16
    for i, (key, label, col) in enumerate(rate_metrics):
        means = [np.mean(rows[ph][key]) if rows[ph][key] else np.nan
                 for ph in phases]
        stds = [np.std(rows[ph][key]) if rows[ph][key] else 0 for ph in phases]
        ax_rates.errorbar(x_pos + (i - 2) * bw, means, yerr=stds, fmt="o",
                          color=col, ms=5, lw=1.2, label=label, capsize=3)
    ax_rates.set_xticks(x_pos)
    ax_rates.set_xticklabels(phases)
    ax_rates.axhline(0, color="grey", ls=":", lw=0.5)
    ax_rates.axhline(1.0, color="grey", ls=":", lw=0.5)
    ax_rates.set_title(r"K=20 canonical (3 seeds, mean $\pm$ std)",
                       fontsize=10)
    ax_rates.set_ylabel("ratio / accuracy")
    ax_rates.set_ylim(-0.45, 1.20)
    ax_rates.legend(fontsize=7, ncol=2, loc="lower right")

    # Δz panel
    means = [np.mean(rows[ph]["dz"]) if rows[ph]["dz"] else np.nan
             for ph in phases]
    stds = [np.std(rows[ph]["dz"]) if rows[ph]["dz"] else 0 for ph in phases]
    ax_dz.errorbar(x_pos, means, yerr=stds, fmt="o-", color="#2ca02c",
                   ms=6, lw=1.5, label=r"$\Delta_z$ (nats)", capsize=4)
    ax_dz.set_xticks(x_pos); ax_dz.set_xticklabels(phases)
    ax_dz.set_ylabel(r"$\Delta_z$ (nats)")
    ax_dz.legend(fontsize=8, loc="upper left")


def main():
    dk10 = load_dense_k10()
    pat, wfs = load_k20_3seed()

    # Layout: 4 subplots — left column 3 stacked (dense K=10), right one (K=20)
    fig = plt.figure(figsize=(7.0, 5.5))
    gs = fig.add_gridspec(3, 2, width_ratios=[2.4, 1.2], hspace=0.20,
                          wspace=0.34, left=0.10, right=0.985,
                          top=0.93, bottom=0.10)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_mid = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_bot = fig.add_subplot(gs[2, 0], sharex=ax_top)
    ax_k20_rates = fig.add_subplot(gs[:2, 1])
    ax_k20_dz = fig.add_subplot(gs[2, 1])

    plot_dense_k10(ax_top, ax_mid, ax_bot, dk10)
    plot_k20_3seed_split(ax_k20_rates, ax_k20_dz, pat, wfs)

    out_path = WS_DIR / "fig_phase_trajectory.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.savefig(WS_DIR / "fig_phase_trajectory.png", bbox_inches="tight",
                dpi=160)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
