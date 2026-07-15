#!/usr/bin/env python
"""
Run 5 / G2 + G3 — conservation replication on K=20 (predictions_k20.json)
and crystallization replicate/extend/functional (predictions_run5.json G3).

Stages: extract | g2 | crystal | noise
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import create_tokenizer_from_config
from src.training.checkpoint import list_checkpoints
from scripts.relp_core import load_model, select_device
from scripts.relp2_core import build_pair_batch, to_device, regime_metrics
from scripts.geometry_probes import (load_resid, split_idx, fit_probe,
                                     RESID, OUT, READOUT, LAYERS, MAIN_EXP)

K20 = "lr_sweep_eta1e-3"
PK = json.load(open(OUT / "predictions_k20.json"))["setup_facts_from_history_only"]
ANCHOR = PK["anchor_checkpoint"]
K20_STEPS = sorted({18000, ANCHOR, 24000, 28000, 32000, 36000, 40000,
                    PK["pre_spike"], PK["mid_collapse"], PK["post_recovery"], 50000})
MAIN_EXTEND = [28000, 32000, 36000, 40000]
VCHARS = "abcdefghijklmnopqrstuvwxyz0123456789"


def k20_setup():
    cfg = OmegaConf.load(f"outputs/{K20}/config.yaml")
    tok = create_tokenizer_from_config(cfg)
    pb = build_pair_batch(cfg, tok, n_b_eval=64, seed=1234)
    char_tid = {c: tok.token_to_id[c] for c in VCHARS}
    tid_to_col = {t: i for i, t in enumerate(sorted(char_tid.values()))}
    correct = pb.base.correct_tid.numpy()
    q10 = np.zeros((len(correct), len(tid_to_col)), dtype=np.float32)
    q10[np.arange(len(correct)), [tid_to_col[t] for t in correct]] = 1.0
    labels = {"b_idx": pb.base.b_idx.numpy(), "slot": pb.base.z_idx.numpy(),
              "q10": q10}
    return cfg, tok, pb, labels


def cmd_extract():
    device = select_device()
    # K=20
    cfg, tok, pb, labels = k20_setup()
    ids, pb = to_device(pb, device)
    np.savez_compressed(RESID / f"{K20}_labels.npz",
                        **{k: v for k, v in labels.items()})
    ckpt_dir = Path("outputs") / K20 / "checkpoints"
    for step in K20_STEPS:
        f = RESID / f"{K20}_{step}.npz"
        if f.exists():
            continue
        model = load_model(cfg, tok, ckpt_dir, step, device)
        with torch.no_grad():
            _, cache = model.run_with_cache(
                ids, return_type=None,
                names_filter=lambda n: n.endswith("hook_resid_post"))
        arrs = {f"L{l}": cache["resid_post", l][:, [7, READOUT], :]
                .cpu().numpy().astype(np.float16) for l in LAYERS}
        np.savez_compressed(f, keep_pos=np.array([7, READOUT]), **arrs)
        print(f"[ok] {f.name}", flush=True)
        del model, cache
        if device == "mps":
            torch.mps.empty_cache()
    # main-run extension for G3
    from scripts.geometry_probes import get_setup
    cfg, tok, pb, _ = get_setup(MAIN_EXP)
    ids, pb = to_device(pb, device)
    ckpt_dir = Path("outputs") / MAIN_EXP / "checkpoints"
    for step in MAIN_EXTEND:
        f = RESID / f"{MAIN_EXP}_{step}.npz"
        if f.exists():
            continue
        model = load_model(cfg, tok, ckpt_dir, step, device)
        outs = {}
        for lo in range(0, ids.shape[0], 2048):
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    ids[lo:lo + 2048], return_type=None,
                    names_filter=lambda n: n.endswith("hook_resid_post"))
            for l in LAYERS:
                outs.setdefault(l, []).append(
                    cache["resid_post", l][:, [1, 2, 3, 4, 5, 6, 7, READOUT], :]
                    .cpu().numpy().astype(np.float16))
            del cache
        np.savez_compressed(f, keep_pos=np.array([1, 2, 3, 4, 5, 6, 7, READOUT]),
                            **{f"L{l}": np.concatenate(outs[l]) for l in LAYERS})
        print(f"[ok] {f.name}", flush=True)
        del model
        if device == "mps":
            torch.mps.empty_cache()


def cmd_g2():
    lab = np.load(RESID / f"{K20}_labels.npz")
    b, slot = lab["b_idx"], lab["slot"]
    chanceB, chanceZ = 1 / 64, 1 / 20
    tr, te = split_idx(len(b))
    out = {}
    for l in LAYERS:
        X0 = load_resid(K20, ANCHOR, l, READOUT)
        accB0, mB, sB = fit_probe(X0, b, tr, te)
        accZ0, mZ, sZ = fit_probe(X0, slot, tr, te)
        res = {"anchor": {"B": accB0, "slot": accZ0}}
        for step in K20_STEPS:
            if step == ANCHOR:
                continue
            X = load_resid(K20, step, l, READOUT).astype(np.float32)
            Xs_B = (X[te] - sB[0]) / sB[1]
            Xs_Z = (X[te] - sZ[0]) / sZ[1]
            res[step] = {"B": float(mB.score(Xs_B, b[te])),
                         "slot": float(mZ.score(Xs_Z, slot[te]))}
        out[f"L{l}"] = res
        print(f"L{l}: anchor B={accB0:.3f} z={accZ0:.3f} | " + " ".join(
            f"{s}:B={res[s]['B']:.2f}/z={res[s]['slot']:.2f}"
            for s in K20_STEPS if s != ANCHOR), flush=True)

    def fid(l, step, key, chance):
        r = out[f"L{l}"]
        return (r[step][key] - chance) / (r["anchor"][key] - chance)
    pre, mid, post = PK["pre_spike"], PK["mid_collapse"], PK["post_recovery"]
    T1 = max(fid(l, post, "slot", chanceZ) for l in (1, 2)) >= 0.90
    T2 = max(fid(l, post, "B", chanceB) for l in (1, 2)) >= 0.80
    T3 = all(
        max(fid(l, post, k, c) for l in (1, 2)) >=
        2 * max(fid(l, mid, k, c) for l in (1, 2))
        for k, c in (("B", chanceB), ("slot", chanceZ)))
    # circuit similarity across the spike from existing relp checkpoints
    circ = None
    try:
        s40 = json.load(open(f"results/relp/{K20}/step_040000.json"))["cond_score"]
        s50 = json.load(open(f"results/relp/{K20}/step_050000.json"))["cond_score"]
        circ = float(np.corrcoef(np.array(s40), np.array(s50))[0, 1])
    except FileNotFoundError:
        pass
    verdict = {"T1_z_conservation": bool(T1), "T2_B_rides_through": bool(T2),
               "T3_recovery_restores": bool(T3),
               "fidelities_midlayers": {
                   "slot_post": max(fid(l, post, "slot", chanceZ) for l in (1, 2)),
                   "B_post": max(fid(l, post, "B", chanceB) for l in (1, 2)),
                   "slot_mid": max(fid(l, mid, "slot", chanceZ) for l in (1, 2)),
                   "B_mid": max(fid(l, mid, "B", chanceB) for l in (1, 2)),
                   "slot_pre": max(fid(l, pre, "slot", chanceZ) for l in (1, 2)),
                   "B_pre": max(fid(l, pre, "B", chanceB) for l in (1, 2))},
               "circuit_r_40000_vs_50000": circ,
               "pass_all": bool(T1 and T2 and T3)}
    with open(OUT / "g2_k20_conservation.json", "w") as f:
        json.dump({"transfer": out, "verdict": verdict}, f, indent=1)
    print("G2 verdict:", json.dumps(verdict))


def heldout_B_r2(exp, step, layer, q10, b):
    from sklearn.linear_model import Ridge
    rng = np.random.RandomState(0)
    ub = np.unique(b)
    perm = rng.permutation(len(ub))
    n_tr = int(len(ub) * 0.78)
    tr_b = set(ub[perm[:n_tr]].tolist())
    tr = np.array([i for i in range(len(b)) if b[i] in tr_b])
    te = np.array([i for i in range(len(b)) if b[i] not in tr_b])
    X = load_resid(exp, step, layer, READOUT).astype(np.float32)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    m = Ridge(alpha=1.0)
    m.fit((X[tr] - mu) / sd, q10[tr])
    p = m.predict((X[te] - mu) / sd)
    return float(1 - ((q10[te] - p) ** 2).sum() /
                 ((q10[te] - q10[te].mean(0)) ** 2).sum())


def cmd_crystal():
    from scipy.stats import spearmanr
    # replicate on K=20 drift
    lab = np.load(RESID / f"{K20}_labels.npz")
    drift = [18000, 20000, 24000, 28000, 32000, 36000, 40000]
    rep = {}
    for l in LAYERS:
        rep[f"L{l}"] = [heldout_B_r2(K20, s, l, lab["q10"], lab["b_idx"])
                        for s in drift]
    r2s = rep["L3"]
    rho = float(spearmanr(drift, r2s).statistic)
    verdict_rep = {"steps": drift, "r2_L3": r2s, "spearman_L3": rho,
                   "delta_last_first": r2s[-1] - r2s[0],
                   "pass": bool(rho >= 0.7 and r2s[-1] - r2s[0] >= 0.05)}
    # extend main
    mlab = np.load(RESID / f"{MAIN_EXP}_labels.npz")
    ext_steps = [4000, 8000, 16000, 24000, 28000, 32000, 36000, 40000,
                 42700, 44000, 50000]
    ext = {f"L{l}": {s: heldout_B_r2(MAIN_EXP, s, l, mlab["q10"], mlab["b_idx"])
                     for s in ext_steps} for l in (2, 3)}
    out = {"replicate_k20": {"per_layer": rep, "verdict": verdict_rep},
           "extend_main": ext}
    with open(OUT / "g3_crystallization.json", "w") as f:
        json.dump(out, f, indent=1)
    print("K20 L3 r2 over drift:", [round(x, 3) for x in r2s],
          "rho=", round(rho, 3), "pass:", verdict_rep["pass"])
    print("main L3 extend:", {s: round(ext['L3'][s], 3) for s in ext_steps})


def cmd_noise():
    from scripts.geometry_probes import get_setup
    device = select_device()
    cfg, tok, pb, _ = get_setup(MAIN_EXP)
    ids, pb = to_device(pb, device)
    ckpt_dir = Path("outputs") / MAIN_EXP / "checkpoints"
    out = {}
    for step in (4000, 24000):
        model = load_model(cfg, tok, ckpt_dir, step, device)
        with torch.no_grad():
            clean = float(regime_metrics(model(ids), pb)["ce_pos1"].mean())
        row = {"clean_ce": clean}
        for l in (2, 3):
            norm = float(np.linalg.norm(
                load_resid(MAIN_EXP, step, l, READOUT).astype(np.float32),
                axis=1).mean())
            for rel in (0.25, 0.5, 1.0):
                dces = []
                for seed in range(5):
                    g = torch.Generator(device="cpu").manual_seed(seed)
                    def hook(resid, hook):
                        eps = torch.randn(resid.shape[0], resid.shape[2],
                                          generator=g) * rel * norm / np.sqrt(128)
                        resid[:, READOUT, :] += eps.to(resid.device)
                        return resid
                    with torch.no_grad():
                        rm = regime_metrics(model.run_with_hooks(
                            ids, fwd_hooks=[(f"blocks.{l}.hook_resid_post",
                                             hook)]), pb)
                    dces.append(float(rm["ce_pos1"].mean()) - clean)
                row[f"L{l}_rel{rel}"] = float(np.mean(dces))
        out[step] = row
        print(step, {k: round(v, 4) for k, v in row.items()}, flush=True)
        del model
    passes = all(out[24000][f"L3_rel{r}"] < out[4000][f"L3_rel{r}"]
                 for r in (0.25, 0.5, 1.0))
    out["verdict_functional_i"] = {"pass": bool(passes)}
    with open(OUT / "g3_noise.json", "w") as f:
        json.dump(out, f, indent=1)
    print("G3 functional(i) pass:", passes)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extract", "g2", "crystal", "noise"])
    a = ap.parse_args()
    {"extract": cmd_extract, "g2": cmd_g2,
     "crystal": cmd_crystal, "noise": cmd_noise}[a.stage]()
