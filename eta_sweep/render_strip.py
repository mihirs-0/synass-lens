#!/usr/bin/env python
"""Render the two-faced phase strip (the note's lead figure) from gate_sweep results.
Three+1 labels: converged / non-learning / structured-trap / diverged (+ mixed at boundary)."""
import json, glob
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

RES = "eta_sweep/results"
COL = {"converged": "#2ECC71", "nonlearning": "#9aa4ab", "trap": "#E74C3C",
       "diverged": "#222222", "mixed": "#F1C40F", "converging": "#A9DFBF"}


def load():
    c = {}
    for s in glob.glob(f"{RES}/gate_sweep/*/status.json"):
        d = json.load(open(s))
        L = [json.loads(x) for x in open(s.replace("status.json", "log.jsonl")) if x.strip()]
        lf = L[-1]["train_loss"] if L else None
        dz = [r.get("delta_z") for r in L if r.get("delta_z") is not None]
        c[(d["direction"], d["eta"], d["seed"])] = (lf, d.get("final_step"), max(dz) if dz else None)
    return c


def cat_fwd(lf, fs, dz):
    if lf is None or (fs is not None and fs < 100):
        return "diverged"
    if lf < 0.5:          # forward is a 6-char target; <0.5 = learned the map (vs gray ~3.0)
        return "converged"
    if lf > 2.5:
        return "nonlearning"
    return "converging"


def cat_inv(lf, fs, dz):
    if lf is None:
        return None
    if lf < 0.3:
        return "converged"
    if lf > 2.5:
        return "trap"
    return "mixed"


def cell_cat(c, direction, eta, catfn):
    cats = [catfn(*c[(direction, eta, s)]) for s in (0, 1) if (direction, eta, s) in c]
    if not cats:
        return None
    return cats[0] if len(set(cats)) == 1 else "mixed"


def main():
    c = load()
    FET = [1e-3, 3e-3, 6e-3, 1.2e-2, 2.5e-2, 5e-2, 1e-1, 2e-1, 5e-1]
    IET = [1e-3, 3e-3, 6e-3, 1.2e-2, 2.5e-2, 5e-2]
    allet = sorted(set(FET) | set(IET))
    xi = {e: i for i, e in enumerate(allet)}
    fig, ax = plt.subplots(figsize=(11, 2.9))
    for direction, etas, catfn, row in [("forward", FET, cat_fwd, 1), ("inverse", IET, cat_inv, 0)]:
        for e in etas:
            cat = cell_cat(c, direction, e, catfn)
            if not cat:
                continue
            ax.add_patch(Rectangle((xi[e] - 0.46, row - 0.46), 0.92, 0.92,
                                   facecolor=COL.get(cat, "white"), edgecolor="white", lw=2))
    ax.set_xlim(-0.6, len(allet) - 0.4); ax.set_ylim(-0.7, 1.7)
    ax.set_xticks(range(len(allet))); ax.set_xticklabels([f"{e:g}" for e in allet], rotation=45)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["inverse  (B,z)→A\n[entropy-decreasing]", "forward  (A,z)→B\n[entropy-increasing]"])
    ax.set_xlabel("learning rate  η", fontsize=11)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)
    leg = [Patch(facecolor=COL["converged"], label="converged"),
           Patch(facecolor=COL["nonlearning"], label="non-learning"),
           Patch(facecolor=COL["trap"], label="structured trap (Δz≈0, cand=log K)"),
           Patch(facecolor=COL["diverged"], label="diverged"),
           Patch(facecolor=COL["mixed"], label="mixed (boundary)")]
    ax.legend(handles=leg, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.32),
              frameon=False, fontsize=8.5, handlelength=1.1)
    ax.set_title("Two-faced phase strip: the structured ruin boundary is on one face only",
                 pad=34, fontsize=12)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(f"{RES}/two_faced_strip.{ext}", dpi=160, bbox_inches="tight")
    print(f"saved {RES}/two_faced_strip.png / .pdf")


if __name__ == "__main__":
    main()
