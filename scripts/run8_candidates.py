#!/usr/bin/env python
"""
Run 8 / Experiment 1 — the candidate ladder (C1..C5) on the dense substrate.
Definitions and pass tests: results/clock/predictions_run8.json.

Usage: python3 scripts/run8_candidates.py <substrate>   # a | c
(c uses its own escape direction from the resume phase; the dose phase is
measured with that same direction to quantify knockback.)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import create_tokenizer_from_config
from src.data.dataset import DisambiguationDataset
from src.model import create_model_from_config
from scripts.experiment_helpers import make_config
from scripts.run8_substrate import md42, STATES
from scripts.run8_lib import load_state, flat_from_state, flat_opt_moment

OUT = Path("results/clock")
HALF = 1.7908


def escape_from_history(name):
    h = json.load(open(f"outputs/{name}/training_history.json"))
    s = np.array(h["steps"]); ftl = np.array(h["first_target_loss"])
    below = np.where(ftl < HALF)[0]
    return (int(s[below[0]]) if len(below) else None), (s, ftl)


def grid_round(step, grid):
    return int(min(grid, key=lambda g: abs(g - step)))


def state_files(d):
    return {int(p.stem.split("_")[1]): p for p in sorted(d.glob("state_*.pt"))}


def grad_at(model, ids, device):
    model.zero_grad()
    logits = model(ids)
    ce = torch.nn.functional.cross_entropy(
        logits[:, 10:14].reshape(-1, logits.shape[-1]),
        ids[:, 11:15].reshape(-1))
    ce.backward()
    g = torch.cat([p.grad.flatten() for p in model.parameters()]).cpu()
    return g


def hvp_lambda_min(model, ids, device, iters=25):
    """Most negative Hessian eigenvalue via shifted power iteration."""
    params = [p for p in model.parameters()]
    def hvp(v):
        model.zero_grad()
        logits = model(ids)
        ce = torch.nn.functional.cross_entropy(
            logits[:, 10:14].reshape(-1, logits.shape[-1]),
            ids[:, 11:15].reshape(-1))
        g = torch.autograd.grad(ce, params, create_graph=True)
        flat_g = torch.cat([x.flatten() for x in g])
        gv = (flat_g * v.to(device)).sum()
        h = torch.autograd.grad(gv, params, retain_graph=False)
        return torch.cat([x.detach().flatten() for x in h]).cpu()
    n = sum(p.numel() for p in params)
    torch.manual_seed(0)
    v = torch.randn(n); v /= v.norm()
    # lambda_max estimate
    for _ in range(10):
        w = hvp(v); lam_max = float(v @ w); v = w / (w.norm() + 1e-12)
    c = abs(lam_max) * 1.5 + 1.0
    torch.manual_seed(1)
    v = torch.randn(n); v /= v.norm()
    for _ in range(iters):
        w = c * v - hvp(v)
        v = w / (w.norm() + 1e-12)
    lam_shift = float(v @ (c * v - hvp(v)))
    return c - lam_shift          # = lambda_min(H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("substrate", choices=["a", "c"])
    ap.add_argument("--no-c4", action="store_true")
    args = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    cfg = make_config("cand", k=10, seed=42, max_steps=4000)
    tok = create_tokenizer_from_config(cfg)
    ref = create_model_from_config(cfg, tok)

    if args.substrate == "a":
        sf = state_files(STATES / "a")
        escape, _ = escape_from_history("clock_a")
        anchor_step = 600
        segs = [("a", sf)]
    else:
        sf_dose = state_files(STATES / "c_dose")
        sf_res = state_files(STATES / "c_resume")
        escape, _ = escape_from_history("clock_c_resume")
        anchor_step = None
        segs = [("dose", sf_dose), ("resume", sf_res)]
        sf = sf_res
    grid = sorted(sf.keys())
    pre_s = grid_round(escape - 250, grid)
    post_s = grid_round(escape + 500, grid)
    th_pre = flat_from_state(load_state(sf[pre_s])["model"], ref)
    th_post = flat_from_state(load_state(sf[post_s])["model"], ref)
    u = th_post - th_pre
    u = u / u.norm()
    print(f"[{args.substrate}] escape={escape} u from {pre_s}->{post_s}",
          flush=True)

    # C5 mask: top-64 cond neurons of the final model
    mask = json.load(open(OUT / "c5_mask.json")) if \
        (OUT / "c5_mask.json").exists() else None
    u_mask = None
    if mask is not None:
        names = [n for n, _ in ref.named_parameters()]
        sizes = [p.numel() for _, p in ref.named_parameters()]
        offs = np.cumsum([0] + sizes)
        sel = torch.zeros_like(u)
        for (l, nidx) in mask["neurons"]:
            for pname, take in ((f"blocks.{l}.mlp.W_in", "col"),
                                (f"blocks.{l}.mlp.W_out", "row")):
                i = names.index(pname)
                shape = dict(ref.named_parameters())[pname].shape
                base = offs[i]
                block = torch.arange(sizes[i]).reshape(shape)
                idxs = block[:, nidx] if take == "col" else block[nidx, :]
                sel[base + idxs.flatten()] = 1.0
        u_mask = u * sel
        u_mask = u_mask / (u_mask.norm() + 1e-12)

    # fixed batch for C2/C4
    md = md42(cfg)
    ds = DisambiguationDataset(mapping_data=md, tokenizer=tok, split="train",
                               probe_fraction=0.0, seed=42, task="bz_to_a")
    rng = np.random.RandomState(5)
    idx = rng.choice(len(ds), 2048, replace=False)
    ids = torch.stack([ds[i]["input_ids"] for i in idx]).to(device)
    ids_c4 = ids[:1024]

    anchor = None
    rows = []
    for seg, files in segs:
        for step in sorted(files):
            st = load_state(files[step])
            th = flat_from_state(st["model"], ref)
            if anchor is None:
                anchor = th if anchor_step is None else \
                    flat_from_state(load_state(sf[anchor_step])["model"], ref)
            row = {"seg": seg, "step": step,
                   "C1": float((th - anchor) @ u)}
            if u_mask is not None:
                row["C5"] = float((th - anchor) @ u_mask)
            m = flat_opt_moment(st["opt"], "exp_avg", ref)
            row["C3_m"] = float(m @ u)
            v = flat_opt_moment(st["opt"], "exp_avg_sq", ref)
            topu = torch.topk(u.abs(), max(1, int(0.01 * len(u)))).indices
            row["C3_v"] = float(v[topu].sum())
            if step % 50 == 0:
                ref.load_state_dict(st["model"])
                refd = ref.to(device)
                g = grad_at(refd, ids, device)
                row["C2"] = float((g @ u) / (g.norm() + 1e-12))
                if not args.no_c4 and step % 200 == 0:
                    row["C4_lam_min"] = hvp_lambda_min(refd, ids_c4, device)
                ref.cpu()
            rows.append(row)
            if step % 500 == 0:
                print(f"  {seg} {step}: C1={row['C1']:.3f} "
                      f"C3m={row['C3_m']:.5f}", flush=True)
    out = {"substrate": args.substrate, "escape": escape,
           "u_span": [pre_s, post_s], "rows": rows}
    with open(OUT / f"candidates_{args.substrate}.json", "w") as f:
        json.dump(out, f)
    print("done")


if __name__ == "__main__":
    main()
