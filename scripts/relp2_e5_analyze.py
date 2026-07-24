#!/usr/bin/env python
"""E5 analysis figure + summary numbers from e5_step_*.json."""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = sys.argv[1] if len(sys.argv) > 1 else "landauer_dense_k10"
res = Path("results/relp2") / EXP
steps, data = [], {}
for f in sorted(res.glob("e5_step_*.json")):
    d = json.load(open(f))
    steps.append(d["step"])
    data[d["step"]] = d
steps.sort()

# spike hygiene: trend panels only to 24000 (pre-spike); full range shown greyed
pre = [s for s in steps if s <= 24000]

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

ax = axes[0]
for key, col in [("m_z_pos1", "tab:blue"), ("m_B_pos1", "tab:orange"),
                 ("m_both_pos1", "tab:green")]:
    ax.plot(pre, [data[s]["metrics"][key] for s in pre], "o-", color=col,
            ms=3, label=key)
ax.axvline(1871, color="k", ls="--", lw=0.8, label="loss transition (1871)")
ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=0.1)
ax.set_xlabel("step"); ax.set_ylabel("logit margin")
ax.legend(fontsize=8); ax.set_title("behavioral margins by regime (pos 1)")

ax = axes[1]
series = {
    "B-mass (ΔB regime)": [data[s]["pos_mass_B"]["B"] for s in pre],
    "z-mass (Δz regime)": [data[s]["pos_mass_z"]["z"] for s in pre],
    "readout-mass (Δz)": [data[s]["pos_mass_z"]["readout"] for s in pre],
}
for (lab, ys), col in zip(series.items(), ["tab:orange", "tab:blue", "tab:cyan"]):
    ax.plot(pre, ys, "o-", color=col, ms=3, label=lab)
ax.axvline(1871, color="k", ls="--", lw=0.8)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("step"); ax.set_ylabel("|attribution| mass")
ax.legend(fontsize=8); ax.set_title("attribution mass, matched regimes")

ax = axes[2]
ax.plot(steps, [data[s]["cross_regime_jaccard"]["z_vs_B"] for s in steps], "o-",
        color="tab:purple", ms=3, label="Jaccard(z-circuit, B-circuit), k=256")
ax.plot(steps, [data[s]["cross_regime_jaccard"]["B_vs_both"] for s in steps], "s-",
        color="0.5", ms=3, alpha=0.7, label="Jaccard(B, both)")
ax.axhline(0.07, color="k", ls=":", lw=0.7, label="chance")
ax.axvline(1871, color="k", ls="--", lw=0.8)
ax.set_xscale("log"); ax.set_ylim(0, 1)
ax.set_xlabel("step"); ax.set_ylabel("Jaccard")
ax.legend(fontsize=8); ax.set_title("cross-regime circuit overlap")

fig.suptitle(f"{EXP}: E5 B-movie — conditioning regimes across training")
fig.tight_layout()
fig.savefig(res / "e5_bmovie.png", dpi=150)
print(f"-> {res}/e5_bmovie.png")

# onset table: step at which each margin crosses fractions of its step-4000 value
ref = data[4000]["metrics"] if 4000 in data else data[steps[-1]]["metrics"]
def crossing(key, frac):
    target = frac * ref[key]
    for i in range(1, len(pre)):
        a, b = data[pre[i - 1]]["metrics"][key], data[pre[i]]["metrics"][key]
        if a < target <= b:
            t = (target - a) / (b - a)
            return pre[i - 1] + t * (pre[i] - pre[i - 1])
    return None
print("onsets (step at X% of step-4000 value):")
for key in ["m_B_pos1", "m_z_pos1"]:
    print(" ", key, {f"{int(f*100)}%": round(crossing(key, f) or -1)
                     for f in (0.01, 0.05, 0.10, 0.50)})
