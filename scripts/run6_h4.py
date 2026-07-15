#!/usr/bin/env python
"""
Run 6 / H4 — Erasure hardened.
(1) control battery: mean-replacement + variance-matched noise vs factor
    erasure at first-read checkpoints and 4000;
(2) counterfactual subspace patching (the decisive test);
(3) cascade as intervals across threshold x rank sweep + phantom_d seeds.
Thresholds: predictions_run6.json (H4).
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.geometry_probes import (get_setup, load_resid, MAIN_EXP, RESID,
                                     READOUT, LAYERS)
from scripts.run5_g1 import subspace, random_subspace
from scripts.relp_core import load_model, select_device
from scripts.relp2_core import to_device, regime_metrics

OUT = Path("results/adversarial")
G1 = json.load(open("results/geometry/g1_erasure.json"))


def hooked_metrics(model, ids, pb, layer, fn):
    def hook(resid, hook):
        resid[:, READOUT, :] = fn(resid[:, READOUT, :])
        return resid
    with torch.no_grad():
        return regime_metrics(model.run_with_hooks(
            ids, fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook)]), pb)


def summ(rm):
    return {"ce_pos1": float(rm["ce_pos1"].mean()),
            "m_B": float(rm["m_B_pos1"].mean()),
            "m_z": float(rm["m_z_pos1"].mean())}


def control_battery(cfg, tok, pb, ids, labels, device):
    b_lab, z_lab = labels["b_idx"], labels["slot"]
    ckdir = Path("outputs") / MAIN_EXP / "checkpoints"
    rows = []
    for step in (2400, 2600, 4000):
        model = load_model(cfg, tok, ckdir, step, device)
        with torch.no_grad():
            clean = summ(regime_metrics(model(ids), pb))
        row = {"step": step, "clean": clean}
        for l in LAYERS:
            X = load_resid(MAIN_EXP, step, l, READOUT).astype(np.float32)
            for tag, lab, r in (("B", b_lab, 60), ("z", z_lab, 9)):
                V = subspace(X, lab, r)
                Vt = torch.tensor(V, device=device, dtype=torch.float32)
                comp = X @ V.T
                comp_mean = torch.tensor(comp.mean(0) @ V, device=device,
                                         dtype=torch.float32)
                comp_var = float((comp - comp.mean(0)).var())
                # factor erasure (projection)
                era = summ(hooked_metrics(model, ids, pb, l,
                          lambda x, Vt=Vt: x - (x @ Vt.T) @ Vt))
                # mean replacement
                mrep = summ(hooked_metrics(model, ids, pb, l,
                           lambda x, Vt=Vt, cm=comp_mean:
                           x - (x @ Vt.T) @ Vt + cm))
                # matched-rank random projection
                Rv = torch.tensor(random_subspace(128, r), device=device,
                                  dtype=torch.float32)
                rnd = summ(hooked_metrics(model, ids, pb, l,
                          lambda x, Rv=Rv: x - (x @ Rv.T) @ Rv))
                # variance-matched noise injection
                g = torch.Generator(device="cpu").manual_seed(0)
                def noise(x, r=r, cv=comp_var, g=g):
                    eps = torch.randn(x.shape[0], r, generator=g) * np.sqrt(cv)
                    return x + eps.to(x.device) @ torch.tensor(
                        random_subspace(128, r, seed=2), device=x.device,
                        dtype=torch.float32)
                nz = summ(hooked_metrics(model, ids, pb, l, noise))
                row[f"L{l}_{tag}"] = {"erase": era, "mean_rep": mrep,
                                      "rand_proj": rnd, "noise": nz}
        rows.append(row)
        print(f"[battery] {step} done", flush=True)
        del model
        if device == "mps":
            torch.mps.empty_cache()
    # verdict: factor damage > every matched control (on own m, at some layer)
    verd = {}
    for row in rows:
        st = row["step"]
        for tag, m in (("B", "m_B"), ("z", "m_z")):
            ok_layers = []
            for l in LAYERS:
                d = row[f"L{l}_{tag}"]
                dmg = row["clean"][m] - d["erase"][m]
                ctrl = max(row["clean"][m] - d["mean_rep"][m],
                           row["clean"][m] - d["rand_proj"][m],
                           row["clean"][m] - d["noise"][m])
                if dmg > ctrl:
                    ok_layers.append(l)
            verd[f"{st}_{tag}"] = ok_layers
    return rows, verd


def patching(cfg, tok, pb, ids, labels, device):
    """Counterfactual subspace patching at 4000, L1/L2."""
    b_lab, z_lab = labels["b_idx"], labels["slot"]
    n = ids.shape[0]
    rng = np.random.RandomState(3)
    model = load_model(cfg, tok, Path("outputs") / MAIN_EXP / "checkpoints",
                       4000, device)
    k = int(labels["k"]); n_b = int(labels["n_b"])
    correct_col = labels["correct_col"]
    res = {}
    for l in (1, 2):
        X = load_resid(MAIN_EXP, 4000, l, READOUT).astype(np.float32)
        Vb = torch.tensor(subspace(X, b_lab, 60), device=device,
                          dtype=torch.float32)
        Vz = torch.tensor(subspace(X, z_lab, 9), device=device,
                          dtype=torch.float32)
        Xt = torch.tensor(X, device=device)
        for factor, V in (("B", Vb), ("z", Vz)):
            # donor differs from recipient in exactly this factor
            recips, donors = [], []
            for _ in range(512):
                i = rng.randint(n)
                bi, zi = i // k, i % k
                if factor == "B":
                    bj = rng.randint(n_b)
                    while bj == bi:
                        bj = rng.randint(n_b)
                    j = bj * k + zi
                else:
                    zj = rng.randint(k)
                    while zj == zi:
                        zj = rng.randint(k)
                    j = bi * k + zj
                recips.append(i)
                donors.append(j)
            recips = np.array(recips)
            donors = np.array(donors)
            don_col = torch.tensor(correct_col[donors], device=device)
            char_tids = sorted({tok.token_to_id[c] for c in
                                "abcdefghijklmnopqrstuvwxyz0123456789"})
            col_to_tid = torch.tensor(char_tids, device=device)

            def run_patch(mode):
                xd = Xt[donors]
                def hook(resid, hook):
                    xr = resid[recips_t, READOUT, :]
                    if mode == "full":
                        resid[recips_t, READOUT, :] = xd
                    else:
                        resid[recips_t, READOUT, :] = \
                            xr + ((xd - xr) @ V.T) @ V
                    return resid
                with torch.no_grad():
                    lg = model.run_with_hooks(
                        ids, fwd_hooks=[(f"blocks.{l}.hook_resid_post",
                                         hook)])
                lo = lg[recips_t, READOUT, :]
                return lo.gather(1, col_to_tid[don_col][:, None]).mean()

            recips_t = torch.tensor(recips, device=device)
            with torch.no_grad():
                lg0 = model(ids)
            base = lg0[recips_t, READOUT, :].gather(
                1, col_to_tid[don_col][:, None]).mean()
            full = run_patch("full")
            sub = run_patch("sub")
            eff = float((sub - base) / (full - base + 1e-9))
            res[f"L{l}_{factor}"] = {
                "base_logit": float(base), "full": float(full),
                "sub": float(sub), "transfer_frac": eff}
            print(f"[patch] L{l} {factor}: transfer={eff:.3f}", flush=True)
    del model
    # verdict per claim layer
    verd = {}
    for l in (1, 2):
        own = min(res[f"L{l}_B"]["transfer_frac"],
                  res[f"L{l}_z"]["transfer_frac"])
        # cross-factor: patch B-subspace on a z-differing pair etc. --
        # approximated by own-vs-other comparison at same layer:
        verd[f"L{l}"] = {"own_min": own}
    return res, verd


def cross_patch(cfg, tok, pb, ids, labels, device):
    """Cross-factor control: patch the OTHER factor's subspace for the same
    donor/recipient pairs; must transfer <= 15%."""
    b_lab, z_lab = labels["b_idx"], labels["slot"]
    k = int(labels["k"]); n_b = int(labels["n_b"])
    n = ids.shape[0]
    correct_col = labels["correct_col"]
    rng = np.random.RandomState(3)
    model = load_model(cfg, tok, Path("outputs") / MAIN_EXP / "checkpoints",
                       4000, device)
    char_tids = sorted({tok.token_to_id[c] for c in
                        "abcdefghijklmnopqrstuvwxyz0123456789"})
    col_to_tid = torch.tensor(char_tids, device=device)
    res = {}
    for l in (1, 2):
        X = load_resid(MAIN_EXP, 4000, l, READOUT).astype(np.float32)
        V = {"B": torch.tensor(subspace(X, b_lab, 60), device=device,
                               dtype=torch.float32),
             "z": torch.tensor(subspace(X, z_lab, 9), device=device,
                               dtype=torch.float32)}
        Xt = torch.tensor(X, device=device)
        for differ, patch_sub in (("B", "z"), ("z", "B")):
            recips, donors = [], []
            for _ in range(512):
                i = rng.randint(n)
                bi, zi = i // k, i % k
                if differ == "B":
                    bj = rng.randint(n_b)
                    while bj == bi:
                        bj = rng.randint(n_b)
                    j = bj * k + zi
                else:
                    zj = rng.randint(k)
                    while zj == zi:
                        zj = rng.randint(k)
                    j = bi * k + zj
                recips.append(i); donors.append(j)
            recips_t = torch.tensor(np.array(recips), device=device)
            donors_a = np.array(donors)
            don_col = torch.tensor(correct_col[donors_a], device=device)
            xd = Xt[donors_a]
            Vp = V[patch_sub]
            def hook(resid, hook):
                xr = resid[recips_t, READOUT, :]
                resid[recips_t, READOUT, :] = xr + ((xd - xr) @ Vp.T) @ Vp
                return resid
            with torch.no_grad():
                lg0 = model(ids)
                lg = model.run_with_hooks(
                    ids, fwd_hooks=[(f"blocks.{l}.hook_resid_post", hook)])
                # full patch reference
                def fh(resid, hook):
                    resid[recips_t, READOUT, :] = xd
                    return resid
                lgf = model.run_with_hooks(
                    ids, fwd_hooks=[(f"blocks.{l}.hook_resid_post", fh)])
            gv = lambda L: L[recips_t, READOUT, :].gather(
                1, col_to_tid[don_col][:, None]).mean()
            base, subv, fullv = gv(lg0), gv(lg), gv(lgf)
            res[f"L{l}_differ{differ}_patch{patch_sub}"] = float(
                (subv - base) / (fullv - base + 1e-9))
    del model
    return res


def intervals():
    """Cascade intervals: threshold x rank sweep on existing G1 data (10%,
    rank 60) can only be swept on threshold; rank sweep needs new subspaces —
    use stored residuals, erasure damage recomputed is expensive; here we
    sweep thresholds on the existing curves and compute loss/behavior
    intervals; rank sensitivity handled via the two instruments already run
    (raw rank-60 and excess-over-random)."""
    rows = G1["rows"]
    conv = next(r for r in rows if r["step"] == 4000)["clean"]
    out = {}
    for thr in (0.05, 0.10, 0.20):
        fr = {}
        for which, m in (("B", "m_B"), ("z", "m_z")):
            t_raw = None
            t_exc = None
            for r in rows:
                raw = max(r["clean"][m] - r[f"L{l}"][which][m]
                          for l in LAYERS) / conv[m]
                rk = "rand_rB" if which == "B" else "rand_rz"
                exc = max((r["clean"][m] - r[f"L{l}"][which][m]) -
                          (r["clean"][m] - r[f"L{l}"][rk][m])
                          for l in LAYERS) / conv[m]
                if t_raw is None and raw >= thr:
                    t_raw = r["step"]
                if t_exc is None and exc >= thr:
                    t_exc = r["step"]
            fr[which] = {"raw": t_raw, "excess": t_exc}
        out[f"thr{thr}"] = fr
    # loss + behavior intervals
    import json as _json
    h = _json.load(open("outputs/landauer_dense_k10/training_history.json"))
    s = np.array(h["steps"]); ftl = np.array(h["first_target_loss"])
    hi, lo = ftl[:20].mean(), 0.0
    lossx = {}
    for q in (0.25, 0.5, 0.75):
        tgt = hi - q * (hi - lo)
        lossx[q] = int(s[np.where(ftl <= tgt)[0][0]])
    beh = {}
    steps_r = [r["step"] for r in rows]
    for m in ("m_B", "m_z"):
        vals = [r["clean"][m] for r in rows]
        beh[m] = {}
        for q in (0.25, 0.5, 0.75):
            tgt = q * conv[m]
            cross = next((st for st, v in zip(steps_r, vals) if v >= tgt),
                         None)
            beh[m][q] = cross
    usage_all = [v[w]["raw"] for v in out.values() for w in ("B", "z")] + \
                [v[w]["excess"] for v in out.values() for w in ("B", "z")]
    usage_all = [u for u in usage_all if u]
    iv = {"loss": [lossx[0.25], lossx[0.75]],
          "usage": [min(usage_all), max(usage_all)],
          "behavior": [min(b for m in beh.values() for b in m.values() if b),
                       max(b for m in beh.values() for b in m.values() if b)]}
    iv["loss_before_behavior_disjoint"] = iv["loss"][1] < iv["behavior"][0]
    iv["three_way_disjoint"] = (iv["loss"][1] < iv["usage"][0]
                                and iv["usage"][1] < iv["behavior"][0])
    return {"first_read_sweep": out, "loss_crossings": lossx,
            "behavior_crossings": beh, "intervals": iv}


def main():
    device = select_device()
    cfg, tok, pb, labels = get_setup(MAIN_EXP)
    ids, pb = to_device(pb, device)
    battery, verd_b = control_battery(cfg, tok, pb, ids, labels, device)
    pat, _ = patching(cfg, tok, pb, ids, labels, device)
    xp = cross_patch(cfg, tok, pb, ids, labels, device)
    iv = intervals()
    res = {"battery": battery, "battery_verdict_layers_ok": verd_b,
           "patching_own": pat, "patching_cross": xp, "cascade": iv}
    ok_own = {l: min(pat[f"L{l}_B"]["transfer_frac"],
                     pat[f"L{l}_z"]["transfer_frac"]) for l in (1, 2)}
    ok_cross = {l: max(xp[f"L{l}_differB_patchz"],
                       xp[f"L{l}_differz_patchB"]) for l in (1, 2)}
    res["verdict_patching"] = {
        "own_min": ok_own, "cross_max": ok_cross,
        "pass": bool(any(ok_own[l] >= 0.60 and ok_cross[l] <= 0.15
                         for l in (1, 2)))}
    with open(OUT / "h4_erasure_hardened.json", "w") as f:
        json.dump(res, f, indent=1)
    print("battery ok-layers:", json.dumps(verd_b))
    print("patch own:", ok_own, "cross:", ok_cross)
    print("cascade intervals:", json.dumps(iv["intervals"]))


if __name__ == "__main__":
    main()
