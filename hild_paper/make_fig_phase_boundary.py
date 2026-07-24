#!/usr/bin/env python
"""
Regenerate the phase boundary figure for the HiLD paper.

Two panels:
  (a) (eta, K) heatmap classifying each cell as transitioned / mixed / stuck / diverged
  (b) eta_c(K) vs K log-log with power-law fit + contradicted noise-driven prediction overlay

Output: hild_paper/fig_phase_boundary_hild.{pdf,png}
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

REPO = Path(__file__).resolve().parents[1]
df = pd.read_parquet(REPO / "eta_sweep" / "results" / "aggregated_runs.parquet")

agg = df.groupby(["k", "eta"]).agg(
    n_total=("status", "count"),
    n_trans=("status", lambda s: (s == "transitioned").sum()),
    n_stuck=("status", lambda s: (s == "stuck").sum()),
    n_div=("status",  lambda s: (s == "diverged").sum()),
).reset_index()
agg["p_trans"] = agg["n_trans"] / agg["n_total"]


def fit_eta_c(threshold: float):
    eta_c = {}
    for k, sub in agg.groupby("k"):
        sub = sub.sort_values("eta")
        passes = sub[sub["p_trans"] >= threshold]
        fails = sub[sub["p_trans"] < threshold]
        if len(passes) and len(fails) and fails["eta"].min() > passes["eta"].max():
            eta_c[k] = float(np.sqrt(passes["eta"].max() * fails["eta"].min()))
        elif len(passes):
            eta_c[k] = float(passes["eta"].max())
    return eta_c


fits = {}
for thr in (0.3, 0.5, 0.7, 0.9):
    ec = fit_eta_c(thr)
    Ks = np.array(sorted(ec.keys()))
    etas = np.array([ec[k] for k in Ks])
    lk, le = np.log(Ks), np.log(etas)
    slope, intercept = np.polyfit(lk, le, 1)
    ss_res = np.sum((le - (slope * lk + intercept)) ** 2)
    ss_tot = np.sum((le - le.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    fits[thr] = dict(K=Ks, eta_c=etas, alpha=-slope, A=float(np.exp(intercept)), r2=r2)
    print(f"threshold={thr}: alpha={-slope:.3f}, R2={r2:.3f}, A={np.exp(intercept):.4f}")

# ── Figure ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

# Panel a: heatmap
ax = axes[0]
Ks = sorted(agg["k"].unique())
etas = sorted(agg["eta"].unique())
mat = np.full((len(Ks), len(etas)), np.nan)
for _, r in agg.iterrows():
    i = Ks.index(int(r["k"]))
    j = etas.index(float(r["eta"]))
    if r["n_div"] > 0:
        mat[i, j] = -1
    else:
        mat[i, j] = r["p_trans"]
cmap = plt.get_cmap("viridis")
cmap.set_bad("white")
masked = np.ma.masked_invalid(mat)
im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=1, origin="lower")
ax.set_xticks(range(len(etas)))
ax.set_xticklabels([f"{e:g}" for e in etas], rotation=55, ha="right", fontsize=7)
ax.set_yticks(range(len(Ks)))
ax.set_yticklabels(Ks, fontsize=8)
ax.set_xlabel(r"learning rate $\eta$", fontsize=10)
ax.set_ylabel(r"ambiguity $K$", fontsize=10)
ax.set_title(r"(a) Classification across $(\eta, K)$ cells", fontsize=10)
# Annotate counts
for i, k in enumerate(Ks):
    for j, e in enumerate(etas):
        cell = agg[(agg["k"] == k) & (agg["eta"] == e)]
        if len(cell) == 0:
            continue
        c = cell.iloc[0]
        txt = f"{int(c['n_trans'])}/{int(c['n_total'])}"
        # viridis: dark at low, bright at high → text white on dark, black on light
        col = "white" if c["p_trans"] < 0.55 else "black"
        ax.text(j, i, txt, ha="center", va="center", fontsize=6, color=col)
cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cb.set_label("fraction transitioned", fontsize=9)

# Panel b: eta_c vs K with fit + contradicted prediction
ax = axes[1]
colors = plt.cm.viridis(np.linspace(0.15, 0.85, 4))
labels = {0.3: "30% thr.", 0.5: "50%", 0.7: "70%", 0.9: "90%"}
for c, (thr, col) in enumerate(zip([0.3, 0.5, 0.7, 0.9], colors)):
    f = fits[thr]
    ax.plot(f["K"], f["eta_c"], "o-", color=col, alpha=0.85, lw=1.5, ms=6,
            label=fr"{labels[thr]}: $\alpha={f['alpha']:.2f}$ ($R^2={f['r2']:.2f}$)")
# Contradicted noise-driven prediction: eta_c would INCREASE with K (positive slope)
# Plot a representative line with positive slope through the median data point
K_grid = np.array(sorted(fits[0.5]["K"]))
median_eta = np.exp(np.median(np.log(fits[0.5]["eta_c"])))
median_K = np.exp(np.median(np.log(K_grid)))
# slope +0.5 representative; arrow indicates direction
pred_eta = median_eta * (K_grid / median_K) ** (+0.5)
ax.plot(K_grid, pred_eta, "--", color="grey", lw=1.4, alpha=0.85,
        label=r"noise-driven escape: $\eta_c$ would rise with $K$")
ax.annotate("", xy=(K_grid[-1] * 0.95, pred_eta[-1] * 1.0),
            xytext=(K_grid[-1] * 0.5, pred_eta[-1] * 0.55),
            arrowprops=dict(arrowstyle="->", color="grey", lw=1.0))
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$K$", fontsize=11)
ax.set_ylabel(r"$\eta_c(K)$", fontsize=11)
ax.set_title(r"(b) Phase boundary $\eta_c(K)$: data falls; noise-escape rises", fontsize=10)
ax.legend(fontsize=7.5, loc="lower left", framealpha=0.95)
ax.grid(True, which="both", alpha=0.25, lw=0.4)

plt.tight_layout()
out = Path(__file__).parent / "fig_phase_boundary_hild"
fig.savefig(str(out) + ".pdf", dpi=200)
fig.savefig(str(out) + ".png", dpi=160)
print(f"\nFigure: {out}.pdf")
