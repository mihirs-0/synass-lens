#!/usr/bin/env python
"""
Run 10 — snowball test. Modes:
  surr : compute the prospective surrogate A(t) = max(-S_pros, 1e-4) for
         8b / eps05 / k20 / sgd (a and kinks already in s_calibration.json)
  loo  : surrogate gate + leave-one-out rate-extrapolation table
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import create_tokenizer_from_config
from src.data.dataset import DisambiguationDataset, generate_mappings
from src.model import create_model_from_config
from scripts.experiment_helpers import make_config
from scripts.run8_substrate import STATES, md42
from scripts.run8_lib import load_state, flat_from_state
from scripts.run8_candidates import grad_at, state_files
from src.training.checkpoint import load_checkpoint

OUT = Path("results/snowball")
ESCAPES = {"a": 2075, "kink_s42": 7625, "kink_s43": 9425, "8b": 650,
           "eps05": 2875, "k20": 12150, "sgd": 17250}


def batch_ref(k, design=None):
    if design == "8b":
        from scripts.relp2_e8 import get_design
        cfg, md = get_design("8b_skewz")[0:2]
    else:
        cfg = make_config("r10", k=k, max_steps=1000)
        md = md42(cfg) if k == 10 else generate_mappings(
            n_unique_b=1000, k=k, b_length=6, a_length=4, z_length=2,
            vocab_chars=cfg.data.vocab_chars, seed=42, task="bz_to_a",
            enforce_unique_a_first_char_per_b=True,
            disambiguation_prefix_length=1)
    tok = create_tokenizer_from_config(cfg)
    ds = DisambiguationDataset(mapping_data=md, tokenizer=tok, split="train",
                               probe_fraction=0.0, seed=42, task="bz_to_a")
    rng = np.random.RandomState(5)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ids = torch.stack([ds[i]["input_ids"] for i in
                       rng.choice(len(ds), 2048, False)]).to(device)
    return cfg, tok, create_model_from_config(cfg, tok), ids, device


def cmd_surr():
    out = {}
    # state-file based conditions
    jobs = [("eps05", [(STATES / "prod_eps05", 0)], 100, 10, None),
            ("k20", [(STATES / "prod_k20", 0),
                     (STATES / "prod_k20_ext", 11500)], 200, 20, None),
            ("sgd", [(STATES / "prod_sgd", 0)], 200, 10, None)]
    for name, dirs, stride, k, design in jobs:
        cfg, tok, ref, ids, device = batch_ref(k, design)
        files = {}
        for d, off in dirs:
            for s, p in state_files(Path(d)).items():
                files[s + off] = p
        rows = []
        esc = ESCAPES[name]
        for t in sorted(files):
            if t % stride or (t - 500) not in files or t > esc + 500:
                continue
            th = flat_from_state(load_state(files[t])["model"], ref)
            th0 = flat_from_state(load_state(files[t - 500])["model"], ref)
            d_ = th - th0
            d_ = d_ / (d_.norm() + 1e-12)
            ref.load_state_dict(load_state(files[t])["model"])
            g = grad_at(ref.to(device), ids, device)
            ref.cpu()
            rows.append({"step": t,
                         "S": float(-(g @ d_) / (g.norm() + 1e-12))})
        out[name] = rows
        print(f"[surr {name}] {len(rows)} pts", flush=True)
    # 8b from plain checkpoints (every 100)
    cfg, tok, ref, ids, device = batch_ref(10, "8b")
    ckdir = Path("outputs/e8_8b_skewz/checkpoints")
    def theta(step, mdl):
        load_checkpoint(mdl, None, ckdir, step=step)
        return torch.cat([p.detach().flatten().cpu().float()
                          for p in mdl.parameters()])
    rows = []
    for t in range(600, 1300, 100):
        th = theta(t, ref)
        th0 = theta(t - 500, ref)
        d_ = th - th0
        d_ = d_ / (d_.norm() + 1e-12)
        load_checkpoint(ref, None, ckdir, step=t)
        g = grad_at(ref.to(device), ids, device)
        ref.cpu()
        rows.append({"step": t, "S": float(-(g @ d_) / (g.norm() + 1e-12))})
    out["8b"] = rows
    print(f"[surr 8b] {len(rows)} pts", flush=True)
    with open(OUT / "surrogates.json", "w") as f:
        json.dump(out, f)


def series():
    scal = json.load(open("results/production/s_calibration.json"))
    surr = json.load(open(OUT / "surrogates.json"))
    S = {"a": scal["a"], "kink_s42": scal["kink_s42"],
         "kink_s43": scal["kink_s43"], **surr}
    return {name: ([r["step"] for r in rows],
                   [max(-r["S"], 1e-4) for r in rows])
            for name, rows in S.items()}


def ignition(steps, A):
    for i, t in enumerate(steps):
        base = [v for s, v in zip(steps, A) if 0.2 * t <= s <= 0.6 * t]
        if len(base) < 2:
            continue
        med = np.median(base)
        if med <= 0:
            continue
        if all(A[i + j] >= 3 * med for j in range(3) if i + j < len(A)) \
                and i + 2 < len(A):
            return t
    return None


def cmd_loo():
    S = series()
    # surrogate gate on (a) and kink_s42: corr with oracle |C2| over growth
    Aall = json.load(open("results/clock/candidates_a.json"))
    gb = json.load(open("results/clock/gb_kink.json"))
    oracle = {"a": {r["step"]: abs(r["C2"]) for r in Aall["rows"]
                    if "C2" in r},
              "kink_s42": {r["step"]: r["absC2"]
                           for r in gb["s42"]["rows"]}}
    ig_oracle = {"a": 850, "kink_s42": 1400}
    gate = {}
    for cond in ("a", "kink_s42"):
        st, A = S[cond]
        esc = ESCAPES[cond]
        pairs = [(a, oracle[cond][s]) for s, a in zip(st, A)
                 if s in oracle[cond] and ig_oracle[cond] <= s <= esc]
        x = np.array([p[0] for p in pairs])
        y = np.array([p[1] for p in pairs])
        gate[cond] = float(np.corrcoef(x, y)[0, 1])
    gate_pass = all(v > 0.9 for v in gate.values())
    print("surrogate gate:", gate, "pass:", gate_pass)

    rows = []
    fits = {}
    for name, (st, A) in S.items():
        esc = ESCAPES[name]
        t_ig = ignition(st, A)
        if t_ig is None:
            rows.append({"cond": name, "t_ig": None, "note": "no ignition"})
            continue
        wend = 1.25 * t_ig
        w = [(s, a) for s, a in zip(st, A) if t_ig <= s <= wend]
        if len(w) < 3:
            w = [(s, a) for s, a in zip(st, A) if t_ig <= s][:3]
        xs = np.array([p[0] for p in w])
        ys = np.log(np.array([p[1] for p in w]))
        slope, icpt = np.polyfit(xs, ys, 1)
        A_esc = np.interp(esc, st, A)
        fits[name] = {"t_ig": t_ig, "slope": slope, "icpt": icpt,
                      "A_esc": float(A_esc), "escape": esc,
                      "window_end": wend}
    for name in fits:
        others = [fits[o]["A_esc"] for o in fits if o != name]
        Lstar = float(np.median(others))
        f = fits[name]
        if f["slope"] <= 0:
            t_hat = None
        else:
            t_hat = (np.log(Lstar) - f["icpt"]) / f["slope"]
        err = (t_hat - f["escape"]) / f["escape"] if t_hat else None
        rows.append({"cond": name, "t_ig": f["t_ig"],
                     "window_end": f["window_end"], "Lstar": Lstar,
                     "t_hat": float(t_hat) if t_hat else None,
                     "escape": f["escape"],
                     "err_frac": float(err) if err is not None else None})
        print(f"{name:>9}: ig={f['t_ig']:>5} L*={Lstar:.4f} "
              f"pred={t_hat if t_hat else 'n/a':>9} "
              f"escape={f['escape']:>6} err="
              f"{f'{err:+.0%}' if err is not None else 'n/a'}")
    errs = [abs(r["err_frac"]) for r in rows if r.get("err_frac") is not None]
    hard = {r["cond"]: abs(r["err_frac"]) for r in rows
            if r["cond"] in ("kink_s42", "kink_s43", "sgd")
            and r.get("err_frac") is not None}
    verdict = {"gate": gate, "gate_pass": bool(gate_pass),
               "median_abs_err": float(np.median(errs)) if errs else None,
               "hard_rows": hard,
               "pass": bool(gate_pass and errs and
                            np.median(errs) <= 0.20 and
                            len(hard) == 3 and
                            all(v <= 0.30 for v in hard.values()))}
    with open(OUT / "part1_loo.json", "w") as f:
        json.dump({"rows": rows, "verdict": verdict}, f, indent=1)
    print("PART 1 VERDICT:", json.dumps(verdict))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["surr", "loo"])
    a = ap.parse_args()
    (cmd_surr if a.mode == "surr" else cmd_loo)()
