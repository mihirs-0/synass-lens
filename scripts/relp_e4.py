#!/usr/bin/env python
"""
E4: subcritical (trapped) run vs successful run at K=20, same seed/init/data.

lr_sweep_eta1e-3: successful (transition midpoint ~4200);
lr_sweep_eta5e-3: trapped at the log-vocab plateau for all 50k steps.

Question: does the trapped run show partial circuit assembly that stalls?
Measures per trapped checkpoint: similarity (score Pearson / top-k Jaccard)
of its conditional and marginal RelP score vectors to the successful run's
post-transition anchor circuit; its own behavioral metrics; z/B masses;
self-consistency between consecutive trapped checkpoints.
"""

import argparse
import json
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
    return sorted(steps), data


def vec(d, key):
    return np.array(d[key]).flatten()


def member(v, k=256):
    return set(np.argsort(-np.abs(v))[:k].tolist())


def jac(a, b):
    return len(a & b) / len(a | b) if (a or b) else float("nan")


def pear(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--success", default="lr_sweep_eta1e-3")
    ap.add_argument("--trapped", default="lr_sweep_eta5e-3")
    ap.add_argument("--anchor-step", type=int, default=12000)
    ap.add_argument("--res-dir", default="results/relp")
    args = ap.parse_args()

    s_steps, s_data = load_movie(Path(args.res_dir) / args.success)
    t_steps, t_data = load_movie(Path(args.res_dir) / args.trapped)
    anc = s_data[args.anchor_step]
    anc_cond, anc_marg = vec(anc, "cond_score"), vec(anc, "marg_score")
    anc_cond_m, anc_marg_m = member(anc_cond), member(anc_marg)

    out = {"anchor": {"experiment": args.success, "step": args.anchor_step,
                      "m_diff": anc["metrics"]["m_diff"], "m_marg": anc["metrics"]["m_marg"]},
           "trapped": [], "success_self": []}

    prev = None
    for s in t_steps:
        d = t_data[s]
        c, g = vec(d, "cond_score"), vec(d, "marg_score")
        row = {
            "step": s,
            "m_diff": d["metrics"]["m_diff"], "m_marg": d["metrics"]["m_marg"],
            "ce_first": d["metrics"]["ce_first"], "acc_first": d["metrics"]["acc_first"],
            "r_cond_vs_success_anchor": pear(c, anc_cond),
            "r_marg_vs_success_anchor": pear(g, anc_marg),
            "jac_cond_vs_success_anchor": jac(member(c), anc_cond_m),
            "jac_marg_vs_success_anchor": jac(member(g), anc_marg_m),
            "z_mass_cf": d["pos_mass"]["cond_cf"]["z"],
            "b_mass_marg": d["pos_mass"]["rel_marg"]["B"],
            "r_cond_vs_prev": pear(c, prev) if prev is not None else None,
        }
        prev = c
        out["trapped"].append(row)

    # success run's own similarity-to-anchor trajectory, for scale
    for s in s_steps:
        d = s_data[s]
        out["success_self"].append({
            "step": s,
            "m_diff": d["metrics"]["m_diff"], "m_marg": d["metrics"]["m_marg"],
            "ce_first": d["metrics"]["ce_first"],
            "r_cond_vs_anchor": pear(vec(d, "cond_score"), anc_cond),
            "r_marg_vs_anchor": pear(vec(d, "marg_score"), anc_marg),
            "z_mass_cf": d["pos_mass"]["cond_cf"]["z"],
            "b_mass_marg": d["pos_mass"]["rel_marg"]["B"],
        })

    res_dir = Path(args.res_dir) / args.trapped
    with open(res_dir / "e4_comparison.json", "w") as f:
        json.dump(out, f, indent=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    ts = [r["step"] for r in out["trapped"]]
    ss = [r["step"] for r in out["success_self"]]
    ax1.plot(ss, [r["r_cond_vs_anchor"] for r in out["success_self"]], "o-",
             color="tab:blue", label=f"success run (η=1e-3) vs own anchor")
    ax1.plot(ts, [r["r_cond_vs_success_anchor"] for r in out["trapped"]], "s-",
             color="tab:red", label="trapped run (η=5e-3) vs success anchor")
    ax1.plot(ts, [r["r_marg_vs_success_anchor"] for r in out["trapped"]], "d--",
             color="tab:orange", alpha=0.7, label="trapped marginal vs success anchor")
    ax1.set_xscale("log"); ax1.set_ylim(-0.1, 1.05)
    ax1.set_xlabel("step"); ax1.set_ylabel("score correlation to success-run circuit")
    ax1.legend(fontsize=8); ax1.set_title("Circuit assembly: trapped vs successful (K=20)")
    ax2.plot(ss, [r["m_diff"] for r in out["success_self"]], "o-", color="tab:blue",
             label="success m_diff")
    ax2.plot(ts, [r["m_diff"] for r in out["trapped"]], "s-", color="tab:red",
             label="trapped m_diff")
    ax2.plot(ts, [r["m_marg"] for r in out["trapped"]], "d--", color="tab:orange",
             label="trapped m_marg")
    ax2.set_xscale("log"); ax2.set_xlabel("step"); ax2.set_ylabel("logit margin")
    ax2.legend(fontsize=8); ax2.set_title("Behavioral margins")
    fig.tight_layout()
    fig.savefig(res_dir / "g_trapped_vs_success.png", dpi=150)

    for r in out["trapped"]:
        print(f"step {r['step']:6d} m_diff {r['m_diff']:7.3f} m_marg {r['m_marg']:7.3f} "
              f"r_cond {r['r_cond_vs_success_anchor']:6.3f} r_marg {r['r_marg_vs_success_anchor']:6.3f} "
              f"jac_cond {r['jac_cond_vs_success_anchor']:5.3f} z_mass {r['z_mass_cf']:8.3f} "
              f"r_prev {r['r_cond_vs_prev'] if r['r_cond_vs_prev'] is None else round(r['r_cond_vs_prev'],3)}")


if __name__ == "__main__":
    main()
