#!/usr/bin/env python
"""Item 5 — readout-conditioned area metric.

For each seed × readout combination in Block 1's data, compute the
area-between-normalized-curves using that readout (lens / linear / MLP)
as the decodability term. Paired with per-seed ablation profiles from
Block 0.

Claim to test: the area-based gap is not tied to a specific decoder family.
"""

import json
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent / "outputs" / "followup"
OUT = ROOT / "item5"
OUT.mkdir(exist_ok=True)

# Ablation profiles from Block 0 (per-seed, per-config)
with open(ROOT / "block0" / "block0_stability.json") as f:
    b0 = json.load(f)

# Decodability readouts from Block 1 (per-seed, per-layer, per-readout)
with open(ROOT / "block1" / "block1_readouts.json") as f:
    b1 = json.load(f)

def norm(p):
    return np.array(p) / (np.max(np.abs(p)) + 1e-9)

def area_metric(ablation, decodability):
    a = norm(ablation); d = norm(decodability)
    xs = np.linspace(0, 1, len(a))
    return float(np.trapz(np.abs(a - d), xs))

def spearman(ablation, decodability):
    r, _ = stats.spearmanr(norm(ablation), norm(decodability))
    return float(r)

# For each seed, use the mean across Block 0 configs as that seed's ablation profile
seed_ablation = {}
for run in b0["runs"]:
    s = run["seed"]
    seed_ablation.setdefault(s, []).append(run["ablation_delta_per_layer"])
seed_ablation = {s: np.mean(np.array(v), axis=0).tolist() for s, v in seed_ablation.items()}

results = {"config": {"readouts": ["logit_lens", "linear_probe", "mlp_probe"],
                      "source_ablation": "block0 per-seed mean across 6 configs",
                      "source_decodability": "block1 final checkpoint"},
           "per_seed": {}}

for s_str, d in b1["per_seed"].items():
    s = int(s_str)
    if s not in seed_ablation:
        continue
    abl = seed_ablation[s]
    per_readout = {}
    for readout in ["logit_lens", "linear_probe", "mlp_probe"]:
        dec = d[readout]
        per_readout[readout] = {
            "decodability": dec,
            "area": area_metric(abl, dec),
            "spearman": spearman(abl, dec),
        }
    results["per_seed"][s] = {
        "ablation": abl,
        **per_readout,
    }

# Aggregate across seeds per readout
agg = {}
for readout in ["logit_lens", "linear_probe", "mlp_probe"]:
    areas = [results["per_seed"][s][readout]["area"] for s in results["per_seed"]]
    rhos = [results["per_seed"][s][readout]["spearman"] for s in results["per_seed"]]
    agg[readout] = {
        "area_mean": float(np.mean(areas)),
        "area_std": float(np.std(areas, ddof=0)),
        "spearman_mean": float(np.mean(rhos)),
        "spearman_std": float(np.std(rhos, ddof=0)),
        "per_seed_area": {int(s): v for s, v in zip(results["per_seed"].keys(), areas)},
    }
results["aggregate_per_readout"] = agg

# Verdict: area metric values should be close across readouts if the metric
# is decoder-agnostic. "Close" = max-min spread < 0.15 (≈25% of mean).
area_means = [agg[r]["area_mean"] for r in agg]
spread = float(max(area_means) - min(area_means))
mean_of_means = float(np.mean(area_means))
rel_spread = spread / (mean_of_means + 1e-9)
verdict = ("supports" if rel_spread < 0.25 else
           "partially_supports" if rel_spread < 0.50 else
           "weakens")
results["verdict"] = verdict
results["area_spread"] = {"absolute": spread, "relative": rel_spread,
                          "mean_of_means": mean_of_means}

with open(OUT / "item5_readout_area.json", "w") as f:
    json.dump(results, f, indent=2)

print("=== Item 5: readout-conditioned area metric ===")
print(f"{'readout':<14}  area_mean  area_std  ρ_mean  ρ_std")
for r, v in agg.items():
    print(f"{r:<14}  {v['area_mean']:>9.3f}  {v['area_std']:>8.3f}  "
          f"{v['spearman_mean']:>+6.2f}  {v['spearman_std']:>5.2f}")
print(f"\nArea spread across readouts: abs={spread:.3f}  rel={rel_spread*100:.0f}%")
print(f"Verdict: {verdict}")
print(f"Wrote {OUT/'item5_readout_area.json'}")
