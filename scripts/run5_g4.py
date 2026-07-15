#!/usr/bin/env python
"""
Run 5 / G4 — Same map, different machines?
Cross-seed geometry (CKA at readout) vs circuitry (RelP cond_score r) on the
phantom_d post-snap checkpoints. Thresholds in predictions_run5.json (G4).
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import create_tokenizer_from_config
from scripts.experiment_helpers import make_config
from scripts.relp_core import (build_eval_batch, run_relp, z_resample_baseline,
                               load_model, select_device)
from scripts.relp_movie import neuron_scores
from scripts.relp2_core import build_pair_batch, to_device

ARMS = ["control", "targeted", "random", "reverse"]
SEEDS = [101, 202, 303]
STEP = 6000
READOUT = 10
N_LAYERS, D_MLP = 4, 512
OUT = Path("results/geometry")


def cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    num = np.linalg.norm(X.T @ Y, "fro") ** 2
    den = np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro")
    return float(num / den)


def main():
    device = select_device()
    cfg = make_config("g4_probe", k=10, max_steps=6000)   # seed 42 mapping
    tok = create_tokenizer_from_config(cfg)
    # geometry grid (1280 rows) and circuit grid (run-2 relp protocol, n_b=64)
    pb = build_pair_batch(cfg, tok, n_b_eval=128, seed=1234)
    ids_geo, pb = to_device(pb, device)
    batch = build_eval_batch(cfg, tok, n_b_eval=64, seed=1234)
    ids_cir = batch.input_ids.to(device)
    for attr in ["b_idx", "z_idx", "cand_tids", "noncand_tids", "correct_tid"]:
        setattr(batch, attr, getattr(batch, attr).to(device))

    resid, score = {}, {}
    for arm in ARMS:
        for seed in SEEDS:
            name = f"phantom_d_{arm}_s{seed}"
            ckpt_dir = Path("outputs") / name / "checkpoints"
            model = load_model(cfg, tok, ckpt_dir, STEP, device)
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    ids_geo, return_type=None,
                    names_filter=lambda n: n.endswith("hook_resid_post"))
            resid[name] = {l: cache["resid_post", l][:, READOUT, :]
                           .cpu().numpy().astype(np.float32)
                           for l in range(N_LAYERS)}
            del cache
            acts, grads, _ = run_relp(model, ids_cir, batch, "m_diff",
                                      linearize=True)
            attr = {l: grads[f"mlp_post_{l}"] *
                       (acts[f"mlp_post_{l}"] -
                        z_resample_baseline(acts[f"mlp_post_{l}"],
                                            batch.n_b, batch.k))
                    for l in range(N_LAYERS)}
            signed, _ = neuron_scores(attr, N_LAYERS, D_MLP)
            score[name] = np.asarray(signed, dtype=np.float32)
            print(f"[done] {name}", flush=True)
            del model
            if device == "mps":
                torch.mps.empty_cache()

    res = {}
    for arm in ARMS:
        pairs = [(SEEDS[i], SEEDS[j]) for i in range(3) for j in range(i + 1, 3)]
        arm_res = {"pairs": []}
        for (a, b) in pairs:
            na, nb = f"phantom_d_{arm}_s{a}", f"phantom_d_{arm}_s{b}"
            geo = {f"L{l}": cka(resid[na][l], resid[nb][l])
                   for l in range(N_LAYERS)}
            cir = float(np.corrcoef(score[na], score[nb])[0, 1])
            arm_res["pairs"].append({"seeds": [a, b], "cka": geo,
                                     "circuit_r": cir})
        for l in range(N_LAYERS):
            arm_res[f"mean_cka_L{l}"] = float(np.mean(
                [p["cka"][f"L{l}"] for p in arm_res["pairs"]]))
        arm_res["mean_circuit_r"] = float(np.mean(
            [p["circuit_r"] for p in arm_res["pairs"]]))
        res[arm] = arm_res
        print(f"{arm}: CKA L0-L3 = " +
              " ".join(f'{arm_res[f"mean_cka_L{l}"]:.3f}' for l in range(4)) +
              f"  circuit_r = {arm_res['mean_circuit_r']:.3f}", flush=True)

    c = res["control"]
    verdict = {f"L{l}": c[f"mean_cka_L{l}"] - c["mean_circuit_r"]
               for l in (1, 2)}
    res["verdict"] = {
        "gap_L1": verdict["L1"], "gap_L2": verdict["L2"],
        "pass": bool(verdict["L1"] >= 0.3 and verdict["L2"] >= 0.3)}
    with open(OUT / "g4_cross_seed.json", "w") as f:
        json.dump(res, f, indent=1)
    print("G4 verdict:", json.dumps(res["verdict"]))


if __name__ == "__main__":
    main()
