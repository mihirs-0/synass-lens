#!/usr/bin/env python
"""
Run 5 / G1 — The erasure movie (decodability != usage; this is the causal
instrument). Thresholds precommitted in predictions_run5.json (G1).

Per checkpoint, per layer: estimate B- and z-subspaces of the readout-position
residual (top-r singular directions of class-mean deviations, from the run-4
extracted residuals), delete each during a live forward (hook), measure
behavioral damage (delta ce_pos1, delta m_B, delta m_z) vs clean and vs a
random subspace of equal rank.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.geometry_probes import (get_setup, load_resid, split_idx,
                                     fit_probe, MAIN_EXP, PARKED_EXP,
                                     MAIN_STEPS, RESID, OUT, READOUT, LAYERS)
from scripts.relp_core import load_model, select_device
from scripts.relp2_core import to_device, regime_metrics

PRED = json.load(open(OUT / "predictions_run5.json"))["G1_erasure_movie"]
R_Z = 9
R_B = 20
R_UPGRADE = [30, 40, 60, 90]


def subspace(X, labels, r):
    """Top-r singular directions of class-mean deviations."""
    X = X.astype(np.float32)
    classes = np.unique(labels)
    M = np.stack([X[labels == c].mean(0) for c in classes]) - X.mean(0)
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    return Vt[:min(r, Vt.shape[0])]                     # [r, d]


def random_subspace(d, r, seed=0):
    g = np.random.RandomState(seed).randn(d, r)
    Q, _ = np.linalg.qr(g)
    return Q.T[:r]


def erased_forward_metrics(model, ids, pb, layer, V, device):
    """Forward with the subspace V projected out of resid_post[layer] at the
    readout position; returns regime metrics."""
    P = torch.eye(V.shape[1], device=device) - \
        torch.tensor(V.T @ V, device=device, dtype=torch.float32)

    def hook(resid, hook):
        resid[:, READOUT, :] = resid[:, READOUT, :] @ P
        return resid

    with torch.no_grad():
        logits = model.run_with_hooks(
            ids, fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook)])
        return regime_metrics(logits, pb)


def summarize(rm):
    return {"ce_pos1": float(rm["ce_pos1"].mean()),
            "m_B": float(rm["m_B_pos1"].mean()),
            "m_z": float(rm["m_z_pos1"].mean())}


def run_exp(exp, steps, r_b):
    cfg, tok, pb, labels = get_setup(exp)
    device = select_device()
    ids, pb = to_device(pb, device)
    b_lab, z_lab = labels["b_idx"], labels["slot"]
    ckpt_dir = Path("outputs") / cfg.experiment.name / "checkpoints"
    d = 128
    rows = []
    for step in steps:
        model = load_model(cfg, tok, ckpt_dir, step, device)
        with torch.no_grad():
            clean = summarize(regime_metrics(model(ids), pb))
        row = {"step": step, "clean": clean}
        for l in LAYERS:
            X = load_resid(exp, step, l, READOUT)
            Vb = subspace(X, b_lab, r_b)
            Vz = subspace(X, z_lab, R_Z)
            row[f"L{l}"] = {
                "B": summarize(erased_forward_metrics(model, ids, pb, l, Vb, device)),
                "z": summarize(erased_forward_metrics(model, ids, pb, l, Vz, device)),
                "rand_rB": summarize(erased_forward_metrics(
                    model, ids, pb, l, random_subspace(d, r_b), device)),
                "rand_rz": summarize(erased_forward_metrics(
                    model, ids, pb, l, random_subspace(d, R_Z, seed=1), device)),
            }
        rows.append(row)
        print(f"{exp} {step}: clean m_B={clean['m_B']:.2f} m_z={clean['m_z']:.2f} | "
              f"L1 B-erase m_B={row['L1']['B']['m_B']:.2f} "
              f"z-erase m_z={row['L1']['z']['m_z']:.2f}", flush=True)
        del model
        if device == "mps":
            torch.mps.empty_cache()
    return rows, (b_lab, z_lab)


def gates(rows, exp, labels, r_b):
    """Sanity gates at step 4000 (main) — run BEFORE interpretation."""
    b_lab, z_lab = labels
    g4000 = next(r for r in rows if r["step"] == 4000)
    conv = g4000["clean"]
    # (i) positive control: B erasure at best layer
    bestB = min(g4000[f"L{l}"]["B"]["m_B"] for l in LAYERS)
    gi = (conv["m_B"] - bestB) >= 0.5 * conv["m_B"] or \
         max(g4000[f"L{l}"]["B"]["ce_pos1"] for l in LAYERS) - conv["ce_pos1"] >= 1.0
    # (ii) specificity: random erasure costs <= 10% of converged m everywhere
    gii = all(conv["m_B"] - g4000[f"L{l}"][k]["m_B"] <= 0.10 * conv["m_B"]
              and conv["m_z"] - g4000[f"L{l}"][k]["m_z"] <= 0.10 * conv["m_z"]
              for l in LAYERS for k in ("rand_rB", "rand_rz"))
    # (iii) erasure worked: post-erasure decodability at step 4000
    tr, te = split_idx(len(b_lab))
    giii_detail = {}
    for l in LAYERS:
        X = load_resid(exp, 4000, l, READOUT).astype(np.float32)
        Vb = subspace(X, b_lab, r_b)
        Vz = subspace(X, z_lab, R_Z)
        Xb = X - X @ Vb.T @ Vb
        Xz = X - X @ Vz.T @ Vz
        accB, _, _ = fit_probe(Xb, b_lab, tr, te)
        accZ, _, _ = fit_probe(Xz, z_lab, tr, te)
        giii_detail[f"L{l}"] = {"B_after": accB, "z_after": accZ}
    giii = all(v["B_after"] <= 0.15 and v["z_after"] <= 0.15
               for v in giii_detail.values())
    return {"i_positive": gi, "ii_specificity": gii,
            "iii_erasure_worked": giii, "iii_detail": giii_detail,
            "all_pass": bool(gi and gii and giii)}


def main():
    steps = MAIN_STEPS
    r_b = R_B
    rows, labels = run_exp(MAIN_EXP, steps, r_b)
    g = gates(rows, MAIN_EXP, labels, r_b)
    print("GATES:", json.dumps({k: v for k, v in g.items() if k != "iii_detail"}))
    if not g["iii_erasure_worked"]:
        for r_try in R_UPGRADE:
            print(f"[upgrade] gate iii failed at r_B={r_b}; retry r_B={r_try}")
            r_b = r_try
            rows, labels = run_exp(MAIN_EXP, steps, r_b)
            g = gates(rows, MAIN_EXP, labels, r_b)
            if g["iii_erasure_worked"]:
                break

    # first-read times
    conv = next(r for r in rows if r["step"] == 4000)["clean"]
    def cost(row, l, which, m):
        return row["clean"][m] - row[f"L{l}"][which][m]
    def first_read(which, m):
        for r in rows:
            if any(cost(r, l, which, m) >= 0.10 * conv[m] for l in LAYERS):
                return r["step"]
        return None
    fr = {"B": first_read("B", "m_B"), "z": first_read("z", "m_z")}
    plateau_max = {
        "B": max(cost(r, l, "B", "m_B") / conv["m_B"]
                 for r in rows if r["step"] <= 1200 for l in LAYERS),
        "z": max(cost(r, l, "z", "m_z") / conv["m_z"]
                 for r in rows if r["step"] <= 1200 for l in LAYERS)}
    verdict = {
        "rank_B_used": r_b, "gates": {k: v for k, v in g.items()},
        "first_read": fr,
        "pass_ordering": fr["B"] is not None and fr["z"] is not None
                         and fr["B"] < fr["z"],
        "pass_window": all(v is not None and 1400 <= v <= 2800
                           for v in fr.values()),
        "plateau_max_cost_frac": plateau_max,
        "pass_unread_plateau": all(v <= 0.10 for v in plateau_max.values()),
    }

    # bonus: parked run
    prows, plab = run_exp(PARKED_EXP, [24200], r_b)
    p = prows[0]
    dce = max(p[f"L{l}"]["B"]["ce_pos1"] for l in LAYERS) - p["clean"]["ce_pos1"]
    dz = max(abs(p["clean"]["m_z"] - p[f"L{l}"]["z"]["m_z"]) for l in LAYERS)
    verdict["bonus_parked"] = {
        "B_erase_dCE": float(dce), "pass_B_devastates": bool(dce >= 1.0),
        "z_erase_dm_z": float(dz), "clean_m_z": p["clean"]["m_z"]}

    with open(OUT / "g1_erasure.json", "w") as f:
        json.dump({"rows": rows, "parked": prows, "verdict": verdict}, f, indent=1)
    print("G1 verdict:", json.dumps(verdict["first_read"]),
          "ordering:", verdict["pass_ordering"],
          "window:", verdict["pass_window"],
          "unread-plateau:", verdict["pass_unread_plateau"])


if __name__ == "__main__":
    main()
