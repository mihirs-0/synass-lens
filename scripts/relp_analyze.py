#!/usr/bin/env python
"""
E3 analysis + deliverable plots for the RelP circuit movie.

Assembly is measured against an ANCHOR checkpoint (default 4000 = first
checkpoint after the loss transition completes), not the final checkpoint:
the circuit keeps silently reorganizing long after the loss converges, which
makes the final checkpoint a moving target (documented as the late-drift
finding).

Reads results/relp/<experiment>/step_*.json, the experiment's
training_history.json, and analysis_outputs/raw_results.json (L0 attention
trajectories from the earlier selector-routing analysis).

Writes results/relp/<experiment>/analysis.json and figs/*.png.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_movie(res_dir: Path):
    steps, data = [], {}
    for f in sorted(res_dir.glob("step_*.json")):
        d = json.load(open(f))
        steps.append(d["step"])
        data[d["step"]] = d
    steps.sort()
    return steps, data


def score_vec(d, key="cond_score"):
    return np.array(d[key]).flatten()


def member_set(v, k=256):
    return set(np.argsort(-np.abs(v))[:k].tolist())


def jaccard(a, b):
    if not a or not b:
        return float("nan")
    return len(a & b) / len(a | b)


def pear(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def crossing(xs, ys, level):
    """First upward crossing of `level`, linear interpolation."""
    for i in range(1, len(xs)):
        if ys[i - 1] < level <= ys[i]:
            f = (level - ys[i - 1]) / (ys[i] - ys[i - 1])
            return xs[i - 1] + f * (xs[i] - xs[i - 1])
    return None


def frac_recovered(curve_dict):
    c, f = curve_dict["clean_m"], curve_dict["floor_m"]
    if abs(c - f) < 1e-6:
        return [float("nan")] * len(curve_dict["faith_m"])
    return [(m - f) / (c - f) for m in curve_dict["faith_m"]]


def size_at(curve_dict, frac):
    fr = frac_recovered(curve_dict)
    for k, v in zip(curve_dict["sizes"], fr):
        if v == v and v >= frac:
            return k
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="landauer_dense_k10")
    ap.add_argument("--res-dir", default="results/relp")
    ap.add_argument("--k-member", type=int, default=256)
    ap.add_argument("--anchor-step", type=int, default=4000)
    ap.add_argument("--final-step", type=int, default=50000)
    ap.add_argument("--no-attn-ref", action="store_true")
    args = ap.parse_args()

    res_dir = Path(args.res_dir) / args.experiment
    fig_dir = res_dir / "figs"
    fig_dir.mkdir(exist_ok=True)
    steps, data = load_movie(res_dir)
    anchor, final = data[args.anchor_step], data[args.final_step]
    K = json.load(open(res_dir / "manifest.json"))["k"]
    logK = math.log(K)

    hist = json.load(open(Path("outputs") / args.experiment / "training_history.json"))
    h_steps = np.array(hist["steps"])
    h_ftl = np.array(hist["first_target_loss"])
    h_cand = np.array(hist["candidate_loss"], dtype=float)
    kernel = np.ones(7)
    cand_sm = np.convolve(h_cand, kernel, mode="same") / np.convolve(
        np.ones_like(h_cand), kernel, mode="same")  # edge-normalized smoothing
    behav_prog = np.clip(1.0 - cand_sm / logK, 0, 1)   # 0 = marginal, 1 = conditional

    # ---- per-checkpoint quantities ----
    kM = args.k_member
    sc_cond = {s: score_vec(data[s], "cond_score") for s in steps}
    sc_marg = {s: score_vec(data[s], "marg_score") for s in steps}
    mem_cond = {s: member_set(sc_cond[s], kM) for s in steps}
    mem_marg = {s: member_set(sc_marg[s], kM) for s in steps}
    anc_cond, anc_marg = mem_cond[args.anchor_step], mem_marg[args.anchor_step]

    jac_anchor = [jaccard(mem_cond[s], anc_cond) for s in steps]
    jac_final = [jaccard(mem_cond[s], mem_cond[args.final_step]) for s in steps]
    jac_prev = [float("nan")] + [jaccard(mem_cond[steps[i]], mem_cond[steps[i - 1]])
                                 for i in range(1, len(steps))]
    r_anchor = [pear(sc_cond[s], sc_cond[args.anchor_step]) for s in steps]
    r_final = [pear(sc_cond[s], sc_cond[args.final_step]) for s in steps]
    r_prev = [float("nan")] + [pear(sc_cond[steps[i]], sc_cond[steps[i - 1]])
                               for i in range(1, len(steps))]
    jac_marg_anchor = [jaccard(mem_marg[s], anc_marg) for s in steps]
    r_marg_anchor = [pear(sc_marg[s], sc_marg[args.anchor_step]) for s in steps]

    m_diff = [data[s]["metrics"]["m_diff"] for s in steps]
    m_marg = [data[s]["metrics"]["m_marg"] for s in steps]
    ce = [data[s]["metrics"]["ce_first"] for s in steps]

    z_mass_cf = [data[s]["pos_mass"]["cond_cf"]["z"] for s in steps]
    ro_mass_cf = [data[s]["pos_mass"]["cond_cf"]["readout"] for s in steps]
    b_mass_marg = [data[s]["pos_mass"]["rel_marg"]["B"] for s in steps]

    head_cond_all = {f"L{l}H{h}": [data[s]["head_relevance"]["cond"][l][h] for s in steps]
                     for l in range(4) for h in range(4)}

    size75c = [size_at(data[s]["curve_cond"], 0.75) for s in steps]
    size50c = [size_at(data[s]["curve_cond"], 0.50) for s in steps]
    size90c = [data[s]["size90_cond"] for s in steps]
    size90m = [data[s]["size90_marg"] for s in steps]

    # L0 attention-to-z trajectories from the earlier analysis (all 4 heads)
    attn_steps, attn_l0 = [], {h: [] for h in range(4)}
    raw = Path("analysis_outputs/raw_results.json")
    if raw.exists() and not args.no_attn_ref:
        r = json.load(open(raw))
        ar = r["attention_results"]
        for s in sorted(ar.keys(), key=int):
            attn_steps.append(int(s))
            for h in range(4):
                attn_l0[h].append(ar[s]["mean_attn_to_z"][0][h])

    # ---- lead-time table: step at which each progress measure crosses level ----
    ia = steps.index(args.anchor_step)

    def prog_from(vals):
        ref = vals[ia]
        if ref == 0:
            return None
        return [v / ref for v in vals]

    trajectories = {
        "behavior(1-cand/logK)": (h_steps.tolist(), behav_prog.tolist()),
        "r_score_vs_anchor": (steps, r_anchor),
        "jaccard_vs_anchor": (steps, jac_anchor),
        "z_mass_cf": (steps, prog_from(z_mass_cf)),
        "b_mass_marg": (steps, prog_from(b_mass_marg)),
        "m_diff": (steps, prog_from(m_diff)),
        "m_marg": (steps, prog_from(m_marg)),
        "r_marg_vs_anchor": (steps, r_marg_anchor),
    }
    if attn_steps:
        # z-attention progress for the head with the largest rise (L0H2 here):
        rises = {h: (max(attn_l0[h]) - attn_l0[h][0]) for h in range(4)}
        h_star = max(rises, key=rises.get)
        v0 = attn_l0[h_star][0]
        vf = np.mean(attn_l0[h_star][-10:])
        prog = [(v - v0) / (vf - v0) for v in attn_l0[h_star]]
        trajectories[f"L0H{h_star}_attn_to_z"] = (attn_steps, prog)

    levels = [0.10, 0.25, 0.50, 0.75]
    lead_table = {}
    for name, (xs, ys) in trajectories.items():
        if ys is None:
            continue
        lead_table[name] = {str(lv): crossing(xs, ys, lv) for lv in levels}

    # ---- H1: plateau checkpoints ----
    plateau_steps = [s for s in steps if s <= 1300]
    h1 = {
        "plateau_m_diff": {s: data[s]["metrics"]["m_diff"] for s in plateau_steps},
        "plateau_m_marg": {s: data[s]["metrics"]["m_marg"] for s in plateau_steps},
        "anchor_m_diff": anchor["metrics"]["m_diff"],
        "anchor_m_marg": anchor["metrics"]["m_marg"],
        "plateau_z_mass_cf": {s: data[s]["pos_mass"]["cond_cf"]["z"] for s in plateau_steps},
        "anchor_z_mass_cf": anchor["pos_mass"]["cond_cf"]["z"],
        "plateau_marg_frac512": {
            s: frac_recovered(data[s]["curve_marg"])[data[s]["curve_marg"]["sizes"].index(512)]
            for s in plateau_steps},
        "plateau_marg_size90": {s: data[s]["size90_marg"] for s in plateau_steps},
        "marg_membership_stability_plateau": [
            jaccard(mem_marg[plateau_steps[i]], mem_marg[plateau_steps[i - 1]])
            for i in range(1, len(plateau_steps))],
        "marg_score_r_plateau_pairs": [
            pear(sc_marg[plateau_steps[i]], sc_marg[plateau_steps[i - 1]])
            for i in range(1, len(plateau_steps))],
    }

    # ---- H3: recruitment vs rewiring (anchor-based) ----
    h3 = {}
    for pl in [1100, 1300, 1500]:
        if pl in data:
            h3[f"jac_marg{pl}_vs_cond_anchor"] = jaccard(mem_marg[pl], anc_cond)
            h3[f"jac_marg{pl}_vs_marg_anchor"] = jaccard(mem_marg[pl], anc_marg)
            h3[f"jac_cond{pl}_vs_cond_anchor"] = jaccard(mem_cond[pl], anc_cond)
            h3[f"r_marg{pl}_vs_cond_anchor"] = pear(sc_marg[pl], sc_cond[args.anchor_step])
    h3["jac_marg_anchor_vs_cond_anchor"] = jaccard(anc_marg, anc_cond)
    h3["r_marg_anchor_vs_cond_anchor"] = pear(sc_marg[args.anchor_step], sc_cond[args.anchor_step])

    # ---- late drift ----
    late = [s for s in steps if s >= args.anchor_step]
    drift = {
        "steps": late,
        "r_vs_anchor": [pear(sc_cond[s], sc_cond[args.anchor_step]) for s in late],
        "jac_vs_anchor": [jaccard(mem_cond[s], anc_cond) for s in late],
        "m_diff": [data[s]["metrics"]["m_diff"] for s in late],
        "ce_first": [data[s]["metrics"]["ce_first"] for s in late],
    }

    analysis = {
        "k_member": kM, "anchor_step": args.anchor_step, "log_K": logK,
        "lead_table": lead_table,
        "H1": h1, "H3": h3, "late_drift": drift,
        "steps": steps,
        "jac_anchor": jac_anchor, "jac_final": jac_final, "jac_prev": jac_prev,
        "r_anchor": r_anchor, "r_final": r_final, "r_prev": r_prev,
        "jac_marg_anchor": jac_marg_anchor, "r_marg_anchor": r_marg_anchor,
        "m_diff": m_diff, "m_marg": m_marg, "ce_first": ce,
        "size90_cond": size90c, "size75_cond": size75c, "size50_cond": size50c,
        "size90_marg": size90m,
        "z_mass_cf": z_mass_cf, "readout_mass_cf": ro_mass_cf, "b_mass_marg": b_mass_marg,
        "head_cond_all": head_cond_all,
    }
    for kk in (128, 512):
        mc = {s: member_set(sc_cond[s], kk) for s in steps}
        analysis[f"jac_anchor_k{kk}"] = [jaccard(mc[s], mc[args.anchor_step]) for s in steps]

    with open(res_dir / "analysis.json", "w") as f:
        json.dump(analysis, f, indent=1)

    # =======================================================================
    # Plots
    # =======================================================================
    t_loss = lead_table["behavior(1-cand/logK)"]["0.5"]
    xmax = 6000

    # (a) loss curve + circuit size at fixed faithfulness
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(h_steps, h_ftl, color="0.4", lw=1, label="first-target loss")
    ax1.plot(h_steps, h_cand, color="tab:red", lw=1, label="candidate loss (conditional)")
    ax1.axhline(logK, color="tab:red", ls=":", lw=0.8)
    ax1.text(50, logK + 0.06, "log K", color="tab:red", fontsize=8)
    if t_loss:
        ax1.axvline(t_loss, color="k", ls="--", lw=0.8)
    ax1.set_xlim(0, xmax); ax1.set_xlabel("step"); ax1.set_ylabel("loss")
    ax2 = ax1.twinx()
    # sizes are only meaningful once the metric is materially nonzero
    ok_c = [md >= 0.25 for md in m_diff]
    ok_m = [mm >= 0.05 for mm in m_marg]
    for series, ok, mk, col, lab in [
            (size50c, ok_c, "o", "tab:blue", "neurons @50% faith (cond.)"),
            (size75c, ok_c, "s", "tab:cyan", "neurons @75% faith (cond.)"),
            (size90m, ok_m, "^", "tab:green", "neurons @90% faith (marginal)")]:
        xs = [s for s, v, o in zip(steps, series, ok) if v is not None and o and s <= xmax]
        ys = [v for s, v, o in zip(steps, series, ok) if v is not None and o and s <= xmax]
        ax2.plot(xs, ys, mk + "-", color=col, alpha=0.8, label=lab, ms=4)
    ax2.set_ylabel("circuit size (neurons)"); ax2.set_yscale("log")
    ln1, lb1 = ax1.get_legend_handles_labels(); ln2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(ln1 + ln2, lb1 + lb2, loc="center right", fontsize=8)
    ax1.set_title(f"{args.experiment}: loss vs circuit size at fixed faithfulness")
    fig.tight_layout(); fig.savefig(fig_dir / "a_loss_circuit_size.png", dpi=150); plt.close(fig)

    # (b) attribution mass by position vs step
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(steps, z_mass_cf, "o-", color="tab:blue", label="z-position mass (cond. counterfactual)")
    ax1.plot(steps, ro_mass_cf, "^-", color="tab:cyan", alpha=0.7, label="readout-position mass (cond.)")
    ax1.plot(steps, b_mass_marg, "s-", color="tab:orange", label="B-position mass (marginal relevance)")
    if t_loss:
        ax1.axvline(t_loss, color="k", ls="--", lw=0.8, label=f"loss transition ({t_loss:.0f})")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("step"); ax1.set_ylabel("|attribution| mass")
    ax1.legend(fontsize=8); ax1.set_title("Attribution mass by input position")
    fig.tight_layout(); fig.savefig(fig_dir / "b_attribution_mass.png", dpi=150); plt.close(fig)

    # (c) membership heatmap sorted by first entry (movie ckpts <= anchor + late)
    all_neurons = sorted(set().union(*mem_cond.values()))
    first_entry = {}
    for n in all_neurons:
        for s in steps:
            if n in mem_cond[s]:
                first_entry[n] = s
                break
    order = sorted(all_neurons, key=lambda n: (first_entry[n], n))
    mat = np.zeros((len(order), len(steps)))
    for j, s in enumerate(steps):
        for i, n in enumerate(order):
            mat[i, j] = 1.0 if n in mem_cond[s] else 0.0
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.imshow(mat, aspect="auto", cmap="Blues", interpolation="nearest")
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels(steps, rotation=90, fontsize=6)
    ax.set_xlabel("checkpoint step")
    ax.set_ylabel(f"neuron (top-{kM} member at any ckpt, sorted by first entry)")
    if t_loss:
        jj = float(np.interp(t_loss, steps, range(len(steps))))
        ax.axvline(jj, color="r", ls="--", lw=1, label=f"loss transition ({t_loss:.0f})")
    jj = steps.index(args.anchor_step)
    ax.axvline(jj, color="g", ls=":", lw=1, label=f"anchor ({args.anchor_step})")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("Conditional-circuit membership across training")
    fig.tight_layout(); fig.savefig(fig_dir / "c_membership_heatmap.png", dpi=150); plt.close(fig)

    # (d) E1 faithfulness curves with baselines
    if "baselines" in final:
        fig, ax = plt.subplots(figsize=(7, 5))
        cc = final["curve_cond"]
        ax.plot(cc["sizes"], frac_recovered(cc), "o-", label="RelP (signed mean)", color="tab:blue")
        bl = final["baselines"]
        ax.plot(bl["ig"]["sizes"], frac_recovered(bl["ig"]), "s-",
                label="Integrated Gradients", color="tab:green")
        ax.plot(bl["relp_absmean"]["sizes"], frac_recovered(bl["relp_absmean"]), "d-",
                label="RelP (abs mean)", color="tab:cyan", alpha=0.7)
        ax.plot(bl["activation"]["sizes"], frac_recovered(bl["activation"]), "^-",
                label="top |activation|", color="tab:orange")
        rnd = np.mean([frac_recovered(bl[f"random_{i}"]) for i in range(3)], axis=0)
        ax.plot(bl["random_0"]["sizes"], rnd, "v-", label="random (3 seeds)", color="0.5")
        if "per_example_curve" in final:
            pe = final["per_example_curve"]
            ax.plot(pe["sizes"], frac_recovered(pe), "*-", label="RelP per-example", color="tab:red")
        ax.axhline(0.9, color="k", ls=":", lw=0.8)
        ax.set_xscale("log"); ax.set_xlabel("circuit size (neurons kept)")
        ax.set_ylabel("fraction of logit-diff recovered")
        ax.legend(fontsize=8)
        ax.set_title(f"Faithfulness at step {args.final_step} (basis = 2048 neurons)")
        fig.tight_layout(); fig.savefig(fig_dir / "d_e1_faithfulness.png", dpi=150); plt.close(fig)

    # (e) assembly vs loss transition (anchor-referenced)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(steps, r_anchor, "o-", color="tab:blue", label="score corr. vs anchor (cond.)")
    ax1.plot(steps, jac_anchor, "s-", color="tab:cyan", alpha=0.8,
             label=f"Jaccard vs anchor (cond., k={kM})")
    ax1.plot(steps, r_marg_anchor, "d-", color="tab:orange", alpha=0.8,
             label="score corr. vs anchor (marginal)")
    hs = h_steps[h_steps <= max(steps)]
    ax1.plot(hs, behav_prog[:len(hs)], "-", color="tab:red", lw=1.2,
             label="behavioral progress 1 - cand/logK")
    if attn_steps:
        name = [k for k in trajectories if k.startswith("L0H")][0]
        xs, prog = trajectories[name]
        ax1.plot(xs, prog, "-", color="tab:green", alpha=0.8,
                 label=f"{name.split('_')[0]} attn-to-z progress")
    if t_loss:
        ax1.axvline(t_loss, color="k", ls="--", lw=0.8, label=f"loss transition ({t_loss:.0f})")
    ax1.axhline(0.5, color="k", ls=":", lw=0.6)
    ax1.set_xscale("log"); ax1.set_xlabel("step"); ax1.set_ylabel("progress / Jaccard / corr")
    ax1.set_ylim(-0.05, 1.15); ax1.set_xlim(80, max(steps) * 1.1)
    ax1.legend(fontsize=7, loc="upper left")
    ax1.set_title("Circuit assembly vs loss transition (anchored at post-transition ckpt)")
    fig.tight_layout(); fig.savefig(fig_dir / "e_jaccard_assembly.png", dpi=150); plt.close(fig)

    # (f) late drift
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(drift["steps"], drift["r_vs_anchor"], "o-", color="tab:blue",
             label="score corr. vs anchor")
    ax1.plot(drift["steps"], drift["jac_vs_anchor"], "s-", color="tab:cyan",
             label=f"Jaccard vs anchor (k={kM})")
    ax1.set_ylim(0, 1.05); ax1.set_ylabel("similarity to anchor circuit")
    ax1.set_xlabel("step")
    ax2 = ax1.twinx()
    ax2.plot(drift["steps"], drift["ce_first"], "^-", color="tab:red", alpha=0.6,
             label="first-target CE")
    ax2.set_ylabel("CE (first target token)"); ax2.set_yscale("log")
    ln1, lb1 = ax1.get_legend_handles_labels(); ln2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(ln1 + ln2, lb1 + lb2, fontsize=8)
    ax1.set_title("Silent circuit drift after loss convergence")
    fig.tight_layout(); fig.savefig(fig_dir / "f_late_drift.png", dpi=150); plt.close(fig)

    print("lead_table:", json.dumps(lead_table, indent=1))
    print("H3:", json.dumps(h3, indent=1))
    print(f"figs -> {fig_dir}")


if __name__ == "__main__":
    main()
