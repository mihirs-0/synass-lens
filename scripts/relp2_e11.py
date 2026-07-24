#!/usr/bin/env python
"""
E11 — Subcritical run, redone on the correct trap.

Target: landauer_k100 (D = N*K = 1000*100 = 100K examples, same eta=1e-3 and
seed protocol as the main run). Behavior: pinned at the C shelf for ~30k
steps, then a *slow partial* descent (first-token CE ~2.0, acc ~36% at 90k) —
subcritical in flight, unlike the K=20 high-eta noise trap (flat forever).

Tested signatures:
  stalled/slow nucleation: nonzero self-consistency between consecutive
    checkpoints; monotone accumulation of similarity toward its own latest
    circuit; rising-then-slow m_z/m_B.
  noise trap (run-1 E4): margins ~ 0, self-consistency ~ 0, no accumulation.

Also reports score correlation to the K=10 run's step-4000 circuit (the
prompt's comparator), with the caveat that the tasks differ (K, prefix
structure), so only the sign/trend is interpretable.

Output: results/relp2/landauer_k100/e11_subcritical.json + figure.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf
from src.data import create_tokenizer_from_config
from scripts.relp_core import load_model, select_device, run_relp
from scripts.relp_movie import neuron_scores, position_mass, group_mass
from scripts.relp2_core import (
    build_pair_batch, to_device, regime_metrics, register_regime_metrics,
    REGIME_BASELINES,
)


def pear(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="landauer_k100")
    ap.add_argument("--steps", type=int, nargs="+", default=[
        200, 1000, 2000, 5000, 10000, 15000, 20000, 25000, 30000, 35000,
        40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000,
        85000, 90000])
    ap.add_argument("--n-b", type=int, default=12)
    ap.add_argument("--out", default="results/relp2")
    ap.add_argument("--k10-ref", default="results/relp/landauer_dense_k10/step_004000.json")
    args = ap.parse_args()

    register_regime_metrics()
    device = select_device()
    exp_dir = Path("outputs") / args.experiment
    cfg = OmegaConf.load(exp_dir / "config.yaml")
    tok = create_tokenizer_from_config(cfg)
    pb = build_pair_batch(cfg, tok, n_b_eval=args.n_b, seed=1234)
    ids, pb = to_device(pb, device)
    n_b, K = pb.n_b, pb.k
    n_layers, d_mlp = int(cfg.model.n_layers), int(cfg.model.d_mlp)

    k10_ref = np.array(json.load(open(args.k10_ref))["cond_score"]).flatten()

    rows = []
    prev = {}
    scores_by_step = {}
    for step in args.steps:
        model = load_model(cfg, tok, exp_dir / "checkpoints", step, device)
        model.requires_grad_(True)
        with torch.no_grad():
            rm = regime_metrics(model(ids), pb)
        row = {"step": step,
               "m_z": float(rm["m_z_pos1"].mean()), "m_B": float(rm["m_B_pos1"].mean()),
               "ce_pos1": float(rm["ce_pos1"].mean()), "acc_pos1": float(rm["acc_pos1"].mean())}
        for regime in ("z", "B"):
            acts, grads, _ = run_relp(model, ids, pb, f"m_{regime}_pos1", linearize=True)
            bfn = REGIME_BASELINES[regime]
            attr = {l: grads[f"mlp_post_{l}"] *
                    (acts[f"mlp_post_{l}"] - bfn(acts[f"mlp_post_{l}"], n_b, K))
                    for l in range(n_layers)}
            signed, _ = neuron_scores(attr, n_layers, d_mlp)
            v = signed.flatten().numpy()
            row[f"r_prev_{regime}"] = pear(v, prev[regime]) if regime in prev else None
            row[f"mass_{regime}"] = group_mass(position_mass(attr, n_layers))
            prev[regime] = v
            scores_by_step.setdefault(regime, {})[step] = v
        row["r_k10ref_z"] = pear(scores_by_step["z"][step], k10_ref)
        rows.append(row)
        print(f"step {step:6d} m_z={row['m_z']:6.3f} m_B={row['m_B']:6.3f} "
              f"ce={row['ce_pos1']:.3f} acc={row['acc_pos1']:.3f} "
              f"r_prev_z={row['r_prev_z'] if row['r_prev_z'] is None else round(row['r_prev_z'], 3)} "
              f"r_k10={row['r_k10ref_z']:.3f}", flush=True)
        del model
        if device == "mps":
            torch.mps.empty_cache()

    # similarity to own latest circuit (step 90000) — accumulation measure
    last = args.steps[-1]
    for row in rows:
        row["r_final_z"] = pear(scores_by_step["z"][row["step"]], scores_by_step["z"][last])
        row["r_final_B"] = pear(scores_by_step["B"][row["step"]], scores_by_step["B"][last])

    out_dir = Path(args.out) / args.experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "e11_subcritical.json", "w") as f:
        json.dump({"experiment": args.experiment, "n_b": args.n_b, "k": K,
                   "rows": rows}, f, indent=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ss = [r["step"] for r in rows]
    ax1.plot(ss, [r["r_final_z"] for r in rows], "o-", label="r to own 90k circuit (Δz)")
    ax1.plot(ss, [r["r_final_B"] for r in rows], "s-", label="r to own 90k circuit (ΔB)")
    ax1.plot(ss[1:], [r["r_prev_z"] for r in rows[1:]], ".-", color="0.6",
             label="self-consistency (consecutive, Δz)")
    ax1.plot(ss, [r["r_k10ref_z"] for r in rows], "d--", color="tab:red", alpha=0.7,
             label="r to K=10 step-4000 circuit")
    ax1.axhline(0, color="k", lw=0.5)
    ax1.set_xlabel("step"); ax1.set_ylabel("score correlation"); ax1.legend(fontsize=8)
    ax1.set_title(f"{args.experiment}: circuit accumulation")
    ax2.plot(ss, [r["m_z"] for r in rows], "o-", label="m_z")
    ax2.plot(ss, [r["m_B"] for r in rows], "s-", label="m_B")
    ax2b = ax2.twinx()
    ax2b.plot(ss, [r["ce_pos1"] for r in rows], "^-", color="tab:red", alpha=0.6, label="CE pos1")
    ax2b.set_ylabel("CE (pos 1)")
    ax2.set_xlabel("step"); ax2.set_ylabel("logit margin"); ax2.legend(fontsize=8, loc="upper left")
    ax2.set_title("behavioral margins")
    fig.tight_layout()
    fig.savefig(out_dir / "e11_subcritical.png", dpi=150)
    print(f"-> {out_dir}/e11_subcritical.json + .png")


if __name__ == "__main__":
    main()
