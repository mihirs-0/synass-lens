#!/usr/bin/env python
"""
Run 4 — The Map and the Machine. Probing pipeline for the precommitted
predictions in results/geometry/predictions.json (committed BEFORE this
file was written; do not edit thresholds here).

Stages:
  extract    residuals at positions of interest for all pinned checkpoints
  controls   positive/negative control gate (must PASS before anything else)
  probe      per-checkpoint per-layer probes at readout + post-B positions
  transfer   P3: probes trained at 4000, frozen-evaluated at drift/spike ckpts
  analyze    P1-P5 predicted-vs-observed verdicts -> verdicts.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import create_tokenizer_from_config
from scripts.experiment_helpers import make_config
from scripts.relp_core import load_model, select_device
from scripts.relp2_core import build_pair_batch, to_device

OUT = Path("results/geometry")
RESID = OUT / "resid"
PRED = json.load(open(OUT / "predictions.json"))
OPS = PRED["operationalizations"]

MAIN_EXP = "landauer_dense_k10"
PARKED_EXP = "e12_8e_xor"
MAIN_STEPS = OPS["checkpoints_main"]
PARKED_STEPS = OPS["checkpoints_parked"]
READOUT = OPS["readout_position"]       # 10
POST_B = OPS["post_B_position"]         # 7
B_POSITIONS = [1, 2, 3, 4, 5, 6]
LAYERS = [0, 1, 2, 3]
VOCAB = "abcdefghijklmnopqrstuvwxyz0123456789"
SPLIT_SEED = 0
Z1 = "abcdefghij"
Z2 = "klmnopqrst"


def get_setup(exp):
    if exp == MAIN_EXP:
        cfg = make_config(exp, k=10, max_steps=50000)
        tok = create_tokenizer_from_config(cfg)
        pb = build_pair_batch(cfg, tok, n_b_eval=128, seed=1234)
        slot = pb.base.z_idx.numpy()
    else:
        from scripts.relp2_e12 import build_design
        md = build_design("8e_xor")
        cfg = make_config(f"e12_8e_xor", k=md.k, max_steps=25000)
        tok = create_tokenizer_from_config(cfg)
        pb = build_pair_batch(cfg, tok, n_b_eval=64, seed=1234, mapping_data=md)
        zs = pb.base.z_strings
        slot = np.array([(Z1.index(zs[int(z)][0]) + Z2.index(zs[int(z)][1])) % 10
                         for z in pb.base.z_idx.numpy()])
    char_tid = {c: tok.token_to_id[c] for c in VOCAB}
    tid_to_col = {tid: i for i, tid in enumerate(sorted(char_tid.values()))}
    n_b, k = pb.base.n_b, pb.base.k
    b_idx = pb.base.b_idx.numpy()
    correct = pb.base.correct_tid.numpy()
    # belief targets over the 36-char simplex
    V = len(tid_to_col)
    q10 = np.zeros((n_b * k, V), dtype=np.float32)
    q10[np.arange(n_b * k), [tid_to_col[t] for t in correct]] = 1.0
    q7 = np.zeros((n_b * k, V), dtype=np.float32)
    for bi in range(n_b):
        cols = sorted({tid_to_col[int(t)] for t in pb.base.cand_tids[bi]})
        rows = np.where(b_idx == bi)[0]
        q7[np.ix_(rows, cols)] = 1.0 / len(cols)
    labels = {"b_idx": b_idx, "slot": slot, "correct_col":
              np.array([tid_to_col[t] for t in correct]), "q7": q7, "q10": q10,
              "n_b": n_b, "k": k}
    return cfg, tok, pb, labels


def cmd_extract(exp):
    cfg, tok, pb, labels = get_setup(exp)
    device = select_device()
    ids, pb = to_device(pb, device)
    steps = MAIN_STEPS if exp == MAIN_EXP else PARKED_STEPS
    RESID.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RESID / f"{exp}_labels.npz",
                        **{k: v for k, v in labels.items()
                           if isinstance(v, np.ndarray)})
    ckpt_dir = Path("outputs") / cfg.experiment.name / "checkpoints"
    keep_pos = sorted(set(B_POSITIONS + [POST_B, READOUT]))
    for step in steps:
        f = RESID / f"{exp}_{step}.npz"
        if f.exists():
            print(f"[skip] {f.name}")
            continue
        model = load_model(cfg, tok, ckpt_dir, step, device)
        outs = {}
        CH = 2048
        for lo in range(0, ids.shape[0], CH):
            chunk = ids[lo:lo + CH]
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    chunk, return_type=None,
                    names_filter=lambda n: n.endswith("hook_resid_post"))
            for l in LAYERS:
                r = cache["resid_post", l][:, keep_pos, :].cpu().numpy()
                outs.setdefault(l, []).append(r.astype(np.float16))
            del cache
        arrs = {f"L{l}": np.concatenate(outs[l]) for l in LAYERS}
        np.savez_compressed(f, keep_pos=np.array(keep_pos), **arrs)
        print(f"[ok] {f.name}", flush=True)
        del model
        if device == "mps":
            torch.mps.empty_cache()


def load_resid(exp, step, layer, pos):
    d = np.load(RESID / f"{exp}_{step}.npz")
    keep = list(d["keep_pos"])
    r = d[f"L{layer}"]
    if isinstance(pos, (list, tuple)):
        return np.concatenate([r[:, keep.index(p), :] for p in pos], axis=1)
    return r[:, keep.index(pos), :]


def split_idx(n, seed=SPLIT_SEED):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_tr = int(0.8 * n)
    return perm[:n_tr], perm[n_tr:]


def fit_probe(X, y, tr, te, kind="clf"):
    from sklearn.linear_model import LogisticRegression, Ridge
    Xtr, Xte = X[tr].astype(np.float32), X[te].astype(np.float32)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    if kind == "clf":
        m = LogisticRegression(max_iter=2000, C=1.0)
        m.fit(Xtr, y[tr])
        return float(m.score(Xte, y[te])), m, (mu, sd)
    m = Ridge(alpha=1.0)
    m.fit(Xtr, y[tr])
    pred = m.predict(Xte)
    ss_res = ((y[te] - pred) ** 2).sum()
    ss_tot = ((y[te] - y[te].mean(0)) ** 2).sum()
    return float(1 - ss_res / ss_tot), m, (mu, sd)


def cmd_controls():
    exp = MAIN_EXP
    lab = np.load(RESID / f"{exp}_labels.npz")
    b = lab["b_idx"]; slot = lab["slot"]
    n = len(b)
    tr, te = split_idx(n)
    rng = np.random.RandomState(1)
    res = {}
    for step in (600, 4000):
        Xb = load_resid(exp, step, 0, B_POSITIONS)
        acc_pos, _, _ = fit_probe(Xb, b, tr, te)
        Xr = load_resid(exp, step, 2, READOUT)
        acc_shufB, _, _ = fit_probe(Xr, rng.permutation(b), tr, te)
        acc_shufZ, _, _ = fit_probe(Xr, rng.permutation(slot), tr, te)
        res[step] = {"B_at_Bpos_L0concat": acc_pos,
                     "shuffled_B_readout_L2": acc_shufB,
                     "shuffled_slot_readout_L2": acc_shufZ}
        print(f"step {step}: B@Bpos={acc_pos:.4f} (need >=0.99)  "
              f"shufB={acc_shufB:.4f} (need <=0.028)  "
              f"shufZ={acc_shufZ:.4f} (need <=0.13)", flush=True)
    ok = all(v["B_at_Bpos_L0concat"] >= 0.99 and
             v["shuffled_B_readout_L2"] <= 0.028 and
             v["shuffled_slot_readout_L2"] <= 0.13 for v in res.values())
    with open(OUT / "controls.json", "w") as f:
        json.dump({"pass": ok, "detail": res}, f, indent=1)
    print("CONTROLS:", "PASS" if ok else "FAIL — stop condition (a)")


def cmd_probe(exp):
    lab = np.load(RESID / f"{exp}_labels.npz")
    b, slot, q7, q10 = lab["b_idx"], lab["slot"], lab["q7"], lab["q10"]
    n = len(b)
    tr, te = split_idx(n)
    steps = MAIN_STEPS if exp == MAIN_EXP else PARKED_STEPS
    rows = []
    for step in steps:
        row = {"step": step}
        for l in LAYERS:
            Xr = load_resid(exp, step, l, READOUT)
            row[f"B_acc_L{l}"], _, _ = fit_probe(Xr, b, tr, te)
            row[f"slot_acc_L{l}"], _, _ = fit_probe(Xr, slot, tr, te)
            row[f"belief10_r2_L{l}"], _, _ = fit_probe(Xr, q10, tr, te, "reg")
            X7 = load_resid(exp, step, l, POST_B)
            row[f"belief7_r2_L{l}"], _, _ = fit_probe(X7, q7, tr, te, "reg")
        rows.append(row)
        print(f"{exp} {step}: " + " ".join(
            f"L{l} B={row[f'B_acc_L{l}']:.3f} z={row[f'slot_acc_L{l}']:.3f}"
            for l in LAYERS), flush=True)
    with open(OUT / f"probes_{exp}.json", "w") as f:
        json.dump(rows, f, indent=1)


def cmd_transfer():
    exp = MAIN_EXP
    lab = np.load(RESID / f"{exp}_labels.npz")
    b, slot, q10 = lab["b_idx"], lab["slot"], lab["q10"]
    n = len(b)
    tr, te = split_idx(n)
    t0 = PRED["operationalizations"]["probe_train_checkpoint_P3"]
    eval_steps = [8000, 16000, 24000, 25300, 27500, 42700, 44000, 50000]
    out = {}
    for l in LAYERS:
        X0 = load_resid(exp, t0, l, READOUT)
        accB0, mB, sB = fit_probe(X0, b, tr, te)
        accZ0, mZ, sZ = fit_probe(X0, slot, tr, te)
        r20, mQ, sQ = fit_probe(X0, q10, tr, te, "reg")
        res = {"train_ckpt": {"B": accB0, "slot": accZ0, "belief_r2": r20}}
        for step in eval_steps:
            X = load_resid(exp, step, l, READOUT).astype(np.float32)
            def ap(m, ms):
                return (X[te] - ms[0]) / ms[1]
            accB = float(mB.score(ap(mB, sB), b[te]))
            accZ = float(mZ.score(ap(mZ, sZ), slot[te]))
            pred = mQ.predict(ap(mQ, sQ))
            ss_res = ((q10[te] - pred) ** 2).sum()
            ss_tot = ((q10[te] - q10[te].mean(0)) ** 2).sum()
            res[step] = {"B": accB, "slot": accZ,
                         "belief_r2": float(1 - ss_res / ss_tot)}
        out[f"L{l}"] = res
        print(f"L{l}: train B={accB0:.3f} z={accZ0:.3f} r2={r20:.3f}; " +
              " ".join(f"{s}:B={res[s]['B']:.2f}/z={res[s]['slot']:.2f}"
                       for s in eval_steps), flush=True)
    with open(OUT / "transfer_P3.json", "w") as f:
        json.dump(out, f, indent=1)


def crossing(steps, vals, target):
    for i in range(1, len(vals)):
        if vals[i - 1] < target <= vals[i]:
            f = (target - vals[i - 1]) / (vals[i] - vals[i - 1])
            return steps[i - 1] + f * (steps[i] - steps[i - 1])
    return None


def cmd_analyze():
    probes = json.load(open(OUT / f"probes_{MAIN_EXP}.json"))
    parked = json.load(open(OUT / f"probes_{PARKED_EXP}.json"))
    transfer = json.load(open(OUT / "transfer_P3.json"))
    lab = np.load(RESID / f"{MAIN_EXP}_labels.npz")
    chanceB, chanceZ = 1 / 128, 0.10
    verdicts = {}

    # best layer at 4000 by belief10 r2
    row4000 = next(r for r in probes if r["step"] == 4000)
    best_l = max(LAYERS, key=lambda l: row4000[f"belief10_r2_L{l}"])
    verdicts["best_layer_by_belief_r2_at_4000"] = best_l
    per_layer_4000 = {f"L{l}": {k.replace(f"_L{l}", ""): row4000[k]
                     for k in row4000 if k.endswith(f"L{l}")} for l in LAYERS}
    verdicts["layer_heterogeneity_at_4000"] = per_layer_4000

    # P1
    plateau = [r for r in probes if 400 <= r["step"] <= 1200]
    bmax = max(max(r[f"B_acc_L{l}"] for l in LAYERS) for r in plateau)
    p1_main = bmax <= chanceB + 0.05
    steps = [r["step"] for r in probes]
    accB = [max(r[f"B_acc_L{l}"] for l in LAYERS) for r in probes]
    accB4000 = max(row4000[f"B_acc_L{l}"] for l in LAYERS)
    sepB = [(a - chanceB) / (accB4000 - chanceB) for a in accB]
    crossB = crossing(steps, sepB, 0.5)
    p1_time = crossB is not None and 1500 <= crossB <= 2400
    verdicts["P1"] = {"plateau_max_B_readout": bmax, "pass_main": p1_main,
                      "crossing_geoB": crossB, "pass_cotiming": p1_time}

    # P2
    accZ = [max(r[f"slot_acc_L{l}"] for l in LAYERS) for r in probes]
    accZ4000 = max(row4000[f"slot_acc_L{l}"] for l in LAYERS)
    sepZ = [(a - chanceZ) / (accZ4000 - chanceZ) for a in accZ]
    crossZ = crossing(steps, sepZ, 0.5)
    ordering = (crossB is not None and crossZ is not None
                and crossB < crossZ < 2725 < 2796)
    verdicts["P2"] = {"crossing_geoB": crossB, "crossing_geoZ": crossZ,
                      "behavioral_mB": 2725, "behavioral_mZ": 2796,
                      "pass": bool(ordering)}

    # P3 (at best layer; all layers reported in transfer_P3.json)
    tl = transfer[f"L{best_l}"]
    t0 = tl["train_ckpt"]
    def fid(step, key, chance):
        if key == "belief_r2":
            return tl[str(step)][key] / t0[key]
        return (tl[str(step)][key] - chance) / (t0[key] - chance)
    post_spike = {s: {"B": fid(s, "B", chanceB), "slot": fid(s, "slot", chanceZ),
                      "belief": fid(s, "belief_r2", 0)}
                  for s in (27500, 44000)}
    min_fid = min(min(v.values()) for v in post_spike.values())
    verdicts["P3"] = {"layer": best_l, "post_spike_fidelity": post_spike,
                      "min_fidelity": min_fid,
                      "pass": min_fid >= 0.80,
                      "conservation_ratio_vs_circuit_0.30": min_fid / 0.30}

    # P4
    late = next(r for r in parked if r["step"] == 24200)
    mid = next(r for r in parked if r["step"] == 12000)
    bB = max(late[f"B_acc_L{l}"] for l in LAYERS)
    bZ = max(late[f"slot_acc_L{l}"] for l in LAYERS)
    verdicts["P4"] = {"B_readout_late": bB, "slot_readout_late": bZ,
                      "pass": bB >= 0.90 and bZ <= 0.15,
                      "bonus_slot_mid_vs_late": {
                          "mid_12000": max(mid[f"slot_acc_L{l}"] for l in LAYERS),
                          "late_24200": bZ}}

    # P5
    def cloud_ratio(step):
        X = load_resid(MAIN_EXP, step, best_l, READOUT).astype(np.float32)
        y = lab["correct_col"]
        sub = np.random.RandomState(0).choice(len(X), 400, replace=False)
        X, y = X[sub], y[sub]
        D = np.linalg.norm(X[:, None] - X[None], axis=-1)
        same = (y[:, None] == y[None]) & ~np.eye(len(y), dtype=bool)
        diff = ~np.eye(len(y), dtype=bool)
        return float(D[diff].mean() / D[same].mean())
    r600, r4000 = cloud_ratio(600), cloud_ratio(4000)
    verdicts["P5"] = {"ratio_600": r600, "ratio_4000": r4000,
                      "pass": r600 <= 2.0, "guard_4000_gt2": r4000 > 2.0}

    with open(OUT / "verdicts.json", "w") as f:
        json.dump(verdicts, f, indent=1)
    for k in ("P1", "P2", "P3", "P4", "P5"):
        print(k, json.dumps(verdicts[k]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extract", "controls", "probe",
                                      "transfer", "analyze"])
    ap.add_argument("exp", nargs="?", default=MAIN_EXP)
    args = ap.parse_args()
    if args.stage == "extract":
        cmd_extract(args.exp)
    elif args.stage == "controls":
        cmd_controls()
    elif args.stage == "probe":
        cmd_probe(args.exp)
    elif args.stage == "transfer":
        cmd_transfer()
    else:
        cmd_analyze()
