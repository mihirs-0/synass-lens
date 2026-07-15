#!/usr/bin/env python
"""v_t asymmetry across the plateau (3 seeds at K=10, eta=1e-3).

Shows that the z-pipeline vs control v_t ratio is robustly > 1 across the entire
plateau region, addressing the "one-snapshot artifact" concern.

Output: hild_paper/fig_vt_trajectory.{pdf,png}
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]

seeds = [0, 1, 2]
data = {}
for s in seeds:
    p = REPO / "eta_sweep" / "results" / "v_tracking" / f"eta_0.001_K_10_seed_{s}" / "v_aggregates.jsonl"
    if not p.exists():
        continue
    rows = []
    for line in p.read_text().strip().splitlines():
        ent = json.loads(line)
        z_pipe = [g["v_mean"] for g in ent["per_group"].values() if g.get("is_z_pipeline")]
        ctrl = [g["v_mean"] for g in ent["per_group"].values() if not g.get("is_z_pipeline")]
        if not z_pipe or not ctrl:
            continue
        rows.append((ent["step"], np.mean(z_pipe), np.mean(ctrl), np.mean(ctrl) / np.mean(z_pipe)))
    data[s] = rows

# Get τ for each seed
taus = {}
for s in seeds:
    sf = REPO / "eta_sweep" / "results" / "v_tracking" / f"eta_0.001_K_10_seed_{s}" / "status.json"
    if sf.exists():
        taus[s] = json.loads(sf.read_text()).get("transition_detected_step", 2500)
    else:
        taus[s] = 2500

fig, ax = plt.subplots(1, 1, figsize=(7.5, 3.4))

for s in seeds:
    rows = data[s]
    steps = np.array([r[0] for r in rows])
    ratios = np.array([r[3] for r in rows])
    ax.plot(steps, ratios, marker="o", lw=1.5, alpha=0.85, ms=3.5,
            label=fr"seed {s} ($\tau\!=\!{taus[s]}$)")
    ax.axvline(taus[s], color=f"C{s}", linestyle=":", alpha=0.4, lw=0.8)

ax.axhline(1.0, color="black", lw=0.8, alpha=0.5, linestyle="--",
           label="symmetry (no asymmetry)")
ax.set_xlabel("training step")
ax.set_ylabel(r"$\bar{v}_t^{\text{control}} / \bar{v}_t^{\text{z-pipe}}$")
ax.set_title(r"$v_t$ asymmetry trajectory across the plateau (3 seeds, $K\!=\!10$, $\eta\!=\!10^{-3}$)")
ax.set_yscale("log")
ax.set_xlim(0, 5000)
ax.grid(True, alpha=0.3, lw=0.4, which="both")
ax.legend(fontsize=8, loc="upper right", ncol=2)

plt.tight_layout()
out = REPO / "hild_paper" / "fig_vt_trajectory"
fig.savefig(str(out) + ".pdf", dpi=200)
fig.savefig(str(out) + ".png", dpi=160)
print(f"Figure: {out}.pdf")

# Print summary stats
print("\n=== Plateau-region asymmetry (between step 600 and τ) ===")
for s in seeds:
    rows = data[s]
    plateau = [r for r in rows if 600 <= r[0] <= taus[s]]
    if not plateau:
        continue
    ratios = [r[3] for r in plateau]
    print(f"seed {s}: n={len(plateau)} entries, "
          f"min={min(ratios):.2f}, median={np.median(ratios):.2f}, max={max(ratios):.2f}")
