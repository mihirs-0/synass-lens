#!/usr/bin/env python
"""
Run 6 / H1 — Charitable circuit comparison.
Attack: "low neuron overlap is a basis artifact." Match neurons first
(Hungarian on activation profiles), THEN compare attributions; also compare
attribution-weighted activation subspaces; compute analytic chance levels.
Same audit on within-run comparisons (plateau vs final, across spike 1).
Thresholds: results/adversarial/predictions_run6.json (H1).
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import create_tokenizer_from_config
from scripts.experiment_helpers import make_config
from scripts.relp_core import (build_eval_batch, run_relp, z_resample_baseline,
                               b_resample_baseline, load_model, select_device)
from scripts.relp_movie import neuron_scores

N_LAYERS, D_MLP = 4, 512
OUT = Path("results/adversarial")
SEEDS = [101, 202, 303]


def get_model_data(model, ids, batch, device):
    """Activation profiles (per neuron, flattened rows x pos) + cond/marg
    attribution scores, per layer."""
    with torch.no_grad():
        _, cache = model.run_with_cache(
            ids, return_type=None,
            names_filter=lambda n: n.endswith("mlp.hook_post"))
    acts_prof = {l: cache[f"blocks.{l}.mlp.hook_post"]
                 .reshape(-1, D_MLP).cpu().numpy().astype(np.float32)
                 for l in range(N_LAYERS)}
    del cache
    acts, grads_d, _ = run_relp(model, ids, batch, "m_diff", linearize=True)
    _, grads_g, _ = run_relp(model, ids, batch, "m_marg", linearize=True)
    n_b, K = batch.n_b, batch.k
    cond = {l: grads_d[f"mlp_post_{l}"] *
               (acts[f"mlp_post_{l}"] -
                z_resample_baseline(acts[f"mlp_post_{l}"], n_b, K))
            for l in range(N_LAYERS)}
    marg = {l: grads_g[f"mlp_post_{l}"] *
               (acts[f"mlp_post_{l}"] -
                b_resample_baseline(acts[f"mlp_post_{l}"], n_b, K))
            for l in range(N_LAYERS)}
    cond_s, _ = neuron_scores(cond, N_LAYERS, D_MLP)
    marg_s, _ = neuron_scores(marg, N_LAYERS, D_MLP)
    cond_s = np.asarray(cond_s, dtype=np.float32).reshape(N_LAYERS, D_MLP)
    marg_s = np.asarray(marg_s, dtype=np.float32).reshape(N_LAYERS, D_MLP)
    # attribution-weighted activity subspace (input space, comparable):
    # top-10 left singular vectors of acts weighted per-neuron by |cond score|
    sub = {}
    for l in range(N_LAYERS):
        M = acts_prof[l] * np.abs(cond_s[l])[None, :]
        U, _, _ = np.linalg.svd(M, full_matrices=False)
        sub[l] = U[:, :10]
    return acts_prof, cond_s, marg_s, sub


def match_then_corr(profA, profB, sA, sB):
    """Hungarian match neurons by activation-profile correlation; return
    (unmatched corr, matched corr) of attribution scores."""
    out = {}
    for l in range(N_LAYERS):
        A = profA[l] - profA[l].mean(0)
        B = profB[l] - profB[l].mean(0)
        A = A / (np.linalg.norm(A, axis=0) + 1e-8)
        B = B / (np.linalg.norm(B, axis=0) + 1e-8)
        C = A.T @ B                                  # [512, 512] corr
        ri, ci = linear_sum_assignment(-C)
        perm = np.empty(D_MLP, dtype=int)
        perm[ri] = ci
        raw = float(np.corrcoef(sA[l], sB[l])[0, 1])
        matched = float(np.corrcoef(sA[l], sB[l][perm])[0, 1])
        out[f"L{l}"] = {"raw": raw, "matched": matched,
                        "match_quality": float(C[ri, ci].mean())}
    return out


def sub_overlap(UA, UB):
    """Mean cosine of principal angles between 10-dim input-space subspaces."""
    return {f"L{l}": float(np.linalg.svd(UA[l].T @ UB[l],
                                         compute_uv=False).mean())
            for l in range(N_LAYERS)}


def main():
    device = select_device()
    cfg = make_config("h1", k=10, max_steps=50000)
    tok = create_tokenizer_from_config(cfg)
    batch = build_eval_batch(cfg, tok, n_b_eval=64, seed=1234)
    ids = batch.input_ids.to(device)
    for a in ["b_idx", "z_idx", "cand_tids", "noncand_tids", "correct_tid"]:
        setattr(batch, a, getattr(batch, a).to(device))

    data = {}
    jobs = [(f"s{s}", Path("outputs") / f"phantom_d_control_s{s}" /
             "checkpoints", 6000) for s in SEEDS]
    jobs += [(f"main{st}", Path("outputs") / "landauer_dense_k10" /
              "checkpoints", st) for st in (1200, 4000, 25300, 27500)]
    for name, ckdir, step in jobs:
        model = load_model(cfg, tok, ckdir, step, device)
        data[name] = get_model_data(model, ids, batch, device)
        print(f"[done] {name}", flush=True)
        del model
        if device == "mps":
            torch.mps.empty_cache()

    res = {"analytic_chance": {
        "abs_r_95_null_2048": 1.96 / np.sqrt(2048),
        "abs_r_95_null_512": 1.96 / np.sqrt(512),
        "jaccard_top205_of_2048": 205**2 / 2048 / (2 * 205 - 205**2 / 2048)}}

    # cross-seed, control arm
    pairs = [("s101", "s202"), ("s101", "s303"), ("s202", "s303")]
    cross = []
    for a, b in pairs:
        pa, sa, _, ua = data[a]
        pb, sb, _, ub = data[b]
        cross.append({"pair": [a, b],
                      "corr": match_then_corr(pa, pb, sa, sb),
                      "subspace": sub_overlap(ua, ub)})
    res["cross_seed"] = cross
    mm = {f"L{l}": float(np.mean([c["corr"][f"L{l}"]["matched"]
                                  for c in cross])) for l in range(N_LAYERS)}
    ms = {f"L{l}": float(np.mean([c["subspace"][f"L{l}"] for c in cross]))
          for l in range(N_LAYERS)}
    max_matched = max(mm.values())
    # sampled random-subspace null for 10-dim in R^(n_rows)
    n_rows = data["s101"][0][0].shape[0]
    rng = np.random.RandomState(0)
    null_s = []
    for _ in range(20):
        qa, _ = np.linalg.qr(rng.randn(n_rows, 10))
        qb, _ = np.linalg.qr(rng.randn(n_rows, 10))
        null_s.append(np.linalg.svd(qa.T @ qb, compute_uv=False).mean())
    res["subspace_null"] = float(np.mean(null_s))
    res["verdict_cross_seed"] = {
        "mean_matched_by_layer": mm, "max_matched": max_matched,
        "mean_subspace_by_layer": ms,
        "survives_full": bool(max_matched < 0.3),
        "dies_F1": bool(max_matched >= 0.5),
        "subspace_stable": bool(max(ms.values()) - res["subspace_null"] >= 0.3
                                and max_matched < 0.3)}

    # within-run audits
    p12, c12, m12, u12 = data["main1200"]
    p40, c40, m40, u40 = data["main4000"]
    res["demolition_audit"] = {
        "marg1200_vs_cond4000": match_then_corr(p12, p40, m12, c40),
        "cond1200_vs_cond4000": match_then_corr(p12, p40, c12, c40),
        "subspace_1200_vs_4000": sub_overlap(u12, u40)}
    pa, ca, _, ua = data["main25300"]
    pb, cb, _, ub = data["main27500"]
    res["spike_audit"] = {
        "cond_pre_vs_post": match_then_corr(pa, pb, ca, cb),
        "subspace_pre_vs_post": sub_overlap(ua, ub)}

    with open(OUT / "h1_circuit_comparison.json", "w") as f:
        json.dump(res, f, indent=1)
    print("H1 cross-seed matched (by layer):", mm)
    print("H1 subspace overlap:", ms, "null:", res["subspace_null"])
    print("verdict:", json.dumps(res["verdict_cross_seed"]))


if __name__ == "__main__":
    main()
