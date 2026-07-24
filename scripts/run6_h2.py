#!/usr/bin/env python
"""
Run 6 / H2.1-2.3 — Map nulls, trivial-code removal, factor concentration.
Attack: "CKA is high for boring reasons." Thresholds in predictions_run6.json.
(H2.4 stitching is a separate script, run6_stitch.py.)
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import create_tokenizer_from_config
from src.model import create_model_from_config
from scripts.experiment_helpers import make_config
from scripts.relp_core import load_model, select_device
from scripts.relp2_core import build_pair_batch, to_device
from scripts.run5_g4 import cka
from scripts.run5_g1 import subspace

READOUT = 10
N_LAYERS = 4
OUT = Path("results/adversarial")
SEEDS = [101, 202, 303]


def readout_resid(model, ids, device):
    with torch.no_grad():
        _, cache = model.run_with_cache(
            ids, return_type=None,
            names_filter=lambda n: n.endswith("hook_resid_post"))
    out = {l: cache["resid_post", l][:, READOUT, :].cpu().numpy()
           .astype(np.float32) for l in range(N_LAYERS)}
    del cache
    return out


def main():
    device = select_device()
    cfg = make_config("h2", k=10, max_steps=50000)
    tok = create_tokenizer_from_config(cfg)
    pb = build_pair_batch(cfg, tok, n_b_eval=128, seed=1234)
    ids, pb = to_device(pb, device)
    b_lab = pb.base.b_idx.cpu().numpy()
    z_lab = pb.base.z_idx.cpu().numpy()

    R = {}
    for s in SEEDS:
        R[f"s{s}"] = readout_resid(load_model(
            cfg, tok, Path("outputs") / f"phantom_d_control_s{s}" /
            "checkpoints", 6000, device), ids, device)
        R[f"s{s}_1000"] = readout_resid(load_model(
            cfg, tok, Path("outputs") / f"phantom_d_control_s{s}" /
            "checkpoints", 1000, device), ids, device)
    R["ancestor600"] = readout_resid(load_model(
        cfg, tok, Path("outputs") / "landauer_dense_k10" / "checkpoints",
        600, device), ids, device)
    R["task8c"] = readout_resid(load_model(
        cfg, tok, Path("outputs") / "e8_8c_nostruct" / "checkpoints",
        15000, device), ids, device)
    for i, sd in enumerate((7, 8)):
        torch.manual_seed(sd)
        m = create_model_from_config(cfg, tok).to(device).eval()
        R[f"untrained{i}"] = readout_resid(m, ids, device)
        del m
    print("[residuals collected]", flush=True)

    pairs = [("s101", "s202"), ("s101", "s303"), ("s202", "s303")]
    def mean_cka(keysA, keysB=None):
        vals = {f"L{l}": [] for l in range(N_LAYERS)}
        kp = [(a, b) for a, b in keysA] if keysB is None else \
             [(a, b) for a in keysA for b in keysB]
        for a, b in kp:
            for l in range(N_LAYERS):
                vals[f"L{l}"].append(cka(R[a][l], R[b][l]))
        return {k: float(np.mean(v)) for k, v in vals.items()}

    res = {"trained_cross_seed": mean_cka(pairs)}
    res["null_a_untrained"] = mean_cka([("untrained0", "untrained1")])
    res["null_b_ancestry"] = mean_cka([(f"s{s}", "ancestor600")
                                       for s in SEEDS])
    res["null_c_pre_transition"] = mean_cka(
        [(f"s{a}_1000", f"s{b}_1000") for (a, b) in
         [(101, 202), (101, 303), (202, 303)]])
    res["null_d_other_task"] = mean_cka([f"s{s}" for s in SEEDS],
                                        ["task8c"])

    gaps, worst = {}, {}
    for l in (1, 2):
        t = res["trained_cross_seed"][f"L{l}"]
        nulls = {k: res[k][f"L{l}"] for k in
                 ("null_a_untrained", "null_b_ancestry",
                  "null_c_pre_transition", "null_d_other_task")}
        wk = max(nulls, key=nulls.get)
        gaps[f"L{l}"] = t - nulls[wk]
        worst[f"L{l}"] = wk
    res["verdict_nulls"] = {
        "gap_vs_worst_null": gaps, "worst_null": worst,
        "pass": bool(all(g >= 0.2 for g in gaps.values()))}

    # trivial-code removal: residualize on untrained0 representation
    def residualize(X, Z):
        Zc = np.hstack([Z, np.ones((len(Z), 1), dtype=np.float32)])
        beta, *_ = np.linalg.lstsq(Zc, X, rcond=None)
        return X - Zc @ beta
    resid_cka, anc_cka = {}, {}
    for l in range(N_LAYERS):
        vals, avals = [], []
        for a, b in pairs:
            Xa = residualize(R[a][l], R["untrained0"][l])
            Xb = residualize(R[b][l], R["untrained0"][l])
            vals.append(cka(Xa, Xb))
            Xa2 = residualize(R[a][l], R["ancestor600"][l])
            Xb2 = residualize(R[b][l], R["ancestor600"][l])
            avals.append(cka(Xa2, Xb2))
        resid_cka[f"L{l}"] = float(np.mean(vals))
        anc_cka[f"L{l}"] = float(np.mean(avals))
    res["residualized_on_untrained"] = resid_cka
    res["residualized_on_ancestor"] = anc_cka
    res["verdict_trivial_code"] = {
        "ratio_L1": resid_cka["L1"] / res["trained_cross_seed"]["L1"],
        "ratio_L2": resid_cka["L2"] / res["trained_cross_seed"]["L2"],
        "pass": bool(resid_cka["L1"] >= 0.5 * res["trained_cross_seed"]["L1"]
                     and resid_cka["L2"] >= 0.5 *
                     res["trained_cross_seed"]["L2"])}

    # factor-subspace concentration
    conc = {}
    for l in range(N_LAYERS):
        row = {}
        for tag, lab, rank in (("B", b_lab, 60), ("z", z_lab, 9)):
            v_in, v_out = [], []
            for a, b in pairs:
                Va = subspace(R[a][l], lab, rank)
                Vb = subspace(R[b][l], lab, rank)
                v_in.append(cka(R[a][l] @ Va.T @ Va, R[b][l] @ Vb.T @ Vb))
                v_out.append(cka(R[a][l] - R[a][l] @ Va.T @ Va,
                                 R[b][l] - R[b][l] @ Vb.T @ Vb))
            row[tag] = {"within": float(np.mean(v_in)),
                        "complement": float(np.mean(v_out))}
        conc[f"L{l}"] = row
    res["factor_concentration"] = conc

    with open(OUT / "h2_nulls.json", "w") as f:
        json.dump(res, f, indent=1)
    for k in ("trained_cross_seed", "null_a_untrained", "null_b_ancestry",
              "null_c_pre_transition", "null_d_other_task"):
        print(k, {kk: round(vv, 3) for kk, vv in res[k].items()})
    print("verdict nulls:", json.dumps(res["verdict_nulls"]))
    print("verdict trivial-code:", json.dumps(res["verdict_trivial_code"]))


if __name__ == "__main__":
    main()
