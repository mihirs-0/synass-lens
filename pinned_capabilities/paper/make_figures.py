"""Four paper figures from saved artifacts. Publication style: thin marks,
direct labels, colorblind-safe, survival/censoring shown. No new data."""
import json, math
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = Path("/Users/mihir/synass-lens/synass-lens/pinned_capabilities")
FIG = R / "paper/figs"; FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
CLEAN, VONLY, BOTH, MONLY = "#e6550d", "#2171b5", "#08306b", "#9ecae1"


def ce_series(path, key="full_vocab_ce"):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return [(r["step"], r[key]) for r in rows if r.get(key) is not None]


# ---------- Fig 1: the fork ----------
fig, ax = plt.subplots(figsize=(4.3, 3.0))
clean = ce_series(R / "results/gate0e_adam_decomp/C0_stream00/metrics.jsonl")
both = ce_series(R / "results/reacq_2x2/N_collapsed_seed0/metrics.jsonl")
vonly = ce_series(R / "results/v_escape/s1.00_seed3/metrics.jsonl")
ax.plot(*zip(*clean), color=CLEAN, lw=1.5, label="clean full-gradient")
ax.plot(*zip(*vonly), color=VONLY, lw=1.3, label="v-only noise")
ax.plot(*zip(*both), color=BOTH, lw=1.5, label="both-channel (minibatch-magnitude)")
ax.axhline(3.0, color="#bbb", lw=0.8, ls="--")
ax.axhline(math.log(36), color="#ddd", lw=0.8, ls=":")
ax.annotate("collapse floor (ln 36)", (12000, math.log(36) + 0.03), fontsize=6, color="#999")
ax.scatter([both[-1][0]], [both[-1][1]], marker=">", color=BOTH, s=30, zorder=5)
ax.annotate("censored ≥16k", (both[-1][0] - 200, both[-1][1] + 0.12), fontsize=6.5, color=BOTH, ha="right")
ax.annotate("re-solves ~800", (clean[3][0] + 300, 1.0), fontsize=6.5, color=CLEAN)
ax.set_xlabel("continued-training step"); ax.set_ylabel("first-token cross-entropy")
ax.set_title("The fork: identical (θ,m,v); only the noise routing differs", fontsize=8)
ax.legend(fontsize=6.5, frameon=False, loc="upper right"); ax.set_ylim(0, 3.8)
fig.tight_layout(); fig.savefig(FIG / "fig1_fork.png", dpi=150); plt.close(fig)

# ---------- Fig 2: realized-power pinning ----------
fig, ax = plt.subplots(figsize=(4.3, 3.0))
ax.plot([1, 2, 4], [0.98, 0.975, 0.984], "o-", color=VONLY, lw=1.3, ms=5, label="v-only  (Γ≈1)")
ax.plot([0.5, 0.6, 0.7, 1.0], [1.99, 1.86, 2.15, 1.89], "s-", color=BOTH, lw=1.3, ms=5, label="both-channel  (Γ≈2)")
ax.plot([0.02, 0.05, 0.1, 0.25], [251, 2618, 7773, 1.02e7], "^-", color="#d94801", lw=1.3, ms=5, label="m-only, clean v  (Γ≥250)")
ax.axhspan(0.85, 1.15, color=VONLY, alpha=0.08)
ax.set_yscale("log"); ax.set_xscale("log")
ax.set_xlabel("injected noise dial (magnitude / dose)"); ax.set_ylabel("realized update-space power  Γ")
ax.set_title("Routing pins realized power; the dial is nearly invariant", fontsize=8)
ax.legend(fontsize=6.5, frameon=False, loc="center right")
ax.annotate("target band\n[0.85,1.15]", (1.3, 1.0), fontsize=6, color=VONLY)
fig.tight_layout(); fig.savefig(FIG / "fig2_pinning.png", dpi=150); plt.close(fig)

# ---------- Fig 3: the dissociation ----------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.0))
# panel A: drift vs diffusion at fork + trajectory states
states = ["fork", "traj\n1,000", "traj\n3,700"]
mu_v = [0.498, 0.375, 0.223]; mu_b = [0.502, 0.375, 0.223]
vperp_ratio = [3.9e4, 4.3e6, 5.3e7]
x = range(len(states)); w = 0.35
a1b = a1.twinx()
a1.bar([i - w / 2 for i in x], mu_v, w, color=VONLY, label="μ∥ v-only")
a1.bar([i + w / 2 for i in x], mu_b, w, color=BOTH, label="μ∥ both")
a1b.plot(list(x), vperp_ratio, "o--", color="#d94801", lw=1.3, ms=5, label="V⊥ ratio both/v")
a1b.set_yscale("log"); a1b.set_ylabel("V⊥ ratio (both / v-only)", color="#d94801", fontsize=7)
a1.set_xticks(list(x)); a1.set_xticklabels(states, fontsize=7); a1.set_ylabel("escape-drift μ∥")
a1.set_title("Drift matched (μ∥), diffusion 4–7 orders apart", fontsize=8)
a1.legend(fontsize=6, frameon=False, loc="upper right")
# panel B: own-axis P_T growth (amplification) vs trapped flat
c0 = ce_series(R / "results/cond_update/traj_analysis.json") if False else None
pa = json.load(open(R / "results/cond_update/phase_a.json"))
for arm, col, lab in [("C0", CLEAN, "clean (own axis)"), ("V", VONLY, "v-only (own axis)")]:
    ser = pa["A1_own_axis"][arm]["series"]
    a2.plot([x["step"] for x in ser], [x["P_T"] for x in ser], color=col, lw=1.4, label=lab)
# trapped arms on shared axis (near-flat) from traj_analysis
ta = json.load(open(R / "results/cond_update/traj_analysis.json"))
for arm, col, lab in [("N", BOTH, "both-channel (trapped)"), ("N_PM", MONLY, "reduced-amp (trapped)")]:
    ser = ta["phase3"][arm]
    a2.plot([x["step"] for x in ser], [x["P_T"] for x in ser], color=col, lw=1.2, ls="-", label=lab)
a2.set_xlabel("step"); a2.set_ylabel("cumulative projected movement P_T")
a2.set_title("Own-axis amplification vs trapped-flat", fontsize=8)
a2.set_yscale("symlog", linthresh=1); a2.legend(fontsize=6, frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig(FIG / "fig3_dissociation.png", dpi=150); plt.close(fig)

# ---------- Fig 4: the 2x2 + dose-response inset ----------
fig, ax = plt.subplots(figsize=(4.3, 3.0))
cells = {("collapsed", "clean"): 800, ("collapsed", "v-noise"): 4600,
         ("fresh", "clean"): 2200, ("fresh", "v-noise"): 8400}
xs = {"clean": 0, "v-noise": 1}; ys = {"collapsed": 1, "fresh": 0}
for (row, col), val in cells.items():
    ax.scatter(xs[col], ys[row], s=1400, c=[[0.15, 0.35, 0.6]], alpha=0.15, marker="s")
    ax.annotate(f"{val:,}", (xs[col], ys[row]), ha="center", va="center", fontsize=11, weight="bold")
ax.set_xticks([0, 1]); ax.set_xticklabels(["clean", "v-noise"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["fresh", "collapsed"])
ax.set_xlim(-0.6, 1.6); ax.set_ylim(-0.6, 1.6)
ax.set_title("Reacquisition onset (step); original table\nadvantage eliminated (possibly inverted)", fontsize=8)
ins = ax.inset_axes([0.60, 0.60, 0.36, 0.34])
ins.plot([0.25, 0.5, 0.75, 1.0], [3400, 3600, 4600, 4600], "o-", color=VONLY, lw=1, ms=3)
ins.set_title("dose–response", fontsize=6); ins.set_xlabel("s", fontsize=6); ins.set_ylabel("onset", fontsize=6)
ins.tick_params(labelsize=5)
fig.tight_layout(); fig.savefig(FIG / "fig4_2x2.png", dpi=150); plt.close(fig)

print("wrote fig1_fork, fig2_pinning, fig3_dissociation, fig4_2x2 to", FIG)
