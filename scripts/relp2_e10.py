#!/usr/bin/env python
"""
E10 — Resolve the L0H2 (z-attending) vs L0H3 (ablation-critical) dissociation.

All at one checkpoint (default 4000), Δz pairs, metric m_z_pos1 (+ ce_pos1).

Probes:
1. Per-head ablation matrix: hook_z of each head -> z-resample mean; singles
   for all 16 heads, pairs among L0 heads. Sub/super-additivity = Hydra check.
   Zero-ablation comparison for L0 heads (mean vs zero gap = off-distribution
   disruption share of run-1's "criticality").
2. Pattern-vs-OV decomposition per L0 head: patch the head's attention pattern
   to its counterfactual (values clean) vs patch its values (pattern clean);
   values patched at z positions only / readout position only localizes where
   the head's z-information enters.
3. Path split for L0H3 and L0H2: corrupted head output seen only by MLP0
   (hook_mlp_in) vs only by everything downstream of block 0 except MLP0.

Note: a literal serial path L0H2 -> L0H3 is structurally impossible (same
layer, parallel heads reading the same resid_pre); the serial hypothesis is
tested in its only coherent form — joint dependency of downstream readers on
both heads (probe 1 pairs) and shared vs distinct information content (probe 2).
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf
from src.data import create_tokenizer_from_config
from scripts.relp_core import (
    load_model, select_device, z_resample_baseline, POS_Z, READ_POS,
)
from scripts.relp2_core import build_pair_batch, to_device, regime_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="landauer_dense_k10")
    ap.add_argument("--step", type=int, default=4000)
    ap.add_argument("--n-b", type=int, default=64)
    ap.add_argument("--out", default="results/relp2")
    args = ap.parse_args()

    device = select_device()
    exp_dir = Path("outputs") / args.experiment
    cfg = OmegaConf.load(exp_dir / "config.yaml")
    tok = create_tokenizer_from_config(cfg)
    pb = build_pair_batch(cfg, tok, n_b_eval=args.n_b, seed=1234)
    ids, pb = to_device(pb, device)
    n_b, K = pb.n_b, pb.k

    model = load_model(cfg, tok, exp_dir / "checkpoints", args.step, device)
    model.set_use_hook_mlp_in(True)

    def run(hooks):
        with torch.no_grad():
            with model.hooks(fwd_hooks=hooks):
                rm = regime_metrics(model(ids), pb)
        return {"m_z": float(rm["m_z_pos1"].mean()), "ce": float(rm["ce_pos1"].mean())}

    clean = run([])

    # cache clean per-head quantities and their z-resample baselines
    cache = {}
    def grab(t, hook):
        cache[hook.name] = t.detach().clone()
        return t
    names = ["blocks.0.attn.hook_z", "blocks.0.attn.hook_v",
             "blocks.0.attn.hook_pattern"] + \
            [f"blocks.{l}.attn.hook_z" for l in range(1, 4)]
    with torch.no_grad():
        with model.hooks(fwd_hooks=[(n, grab) for n in names]):
            model(ids)
    base = {n: z_resample_baseline(cache[n], n_b, K) for n in names}

    results = {"experiment": args.experiment, "step": args.step,
               "clean": clean, "n_inputs": int(ids.shape[0])}

    # ---- probe 1: ablation matrix ----
    def headz_patch(l, heads, values):
        def h(z, hook):
            out = z.clone()
            out[:, :, heads, :] = values[:, :, heads, :]
            return out
        return (f"blocks.{l}.attn.hook_z", h)

    singles = {}
    for l in range(4):
        n = f"blocks.{l}.attn.hook_z"
        for hd in range(4):
            singles[f"L{l}H{hd}"] = run([headz_patch(l, [hd], base[n])])
    results["ablate_single_mean"] = singles

    zeros0 = torch.zeros_like(cache["blocks.0.attn.hook_z"])
    results["ablate_L0_zero"] = {
        f"L0H{hd}": run([headz_patch(0, [hd], zeros0)]) for hd in range(4)}

    pairs = {}
    n0 = "blocks.0.attn.hook_z"
    for a in range(4):
        for b in range(a + 1, 4):
            pairs[f"L0H{a}+L0H{b}"] = run([headz_patch(0, [a, b], base[n0])])
    results["ablate_pairs_mean"] = pairs

    # ---- probe 2: pattern vs values, position-resolved ----
    def pattern_patch(l, hd):
        n = f"blocks.{l}.attn.hook_pattern"
        def h(p, hook):
            out = p.clone()
            out[:, hd] = base[n][:, hd]
            return out
        return (n, h)

    def v_patch(l, hd, positions=None):
        n = f"blocks.{l}.attn.hook_v"
        def h(v, hook):
            out = v.clone()
            if positions is None:
                out[:, :, hd, :] = base[n][:, :, hd, :]
            else:
                for p in positions:
                    out[:, p, hd, :] = base[n][:, p, hd, :]
            return out
        return (n, h)

    decomp = {}
    for hd in range(4):
        decomp[f"L0H{hd}"] = {
            "pattern_only": run([pattern_patch(0, hd)]),
            "v_all": run([v_patch(0, hd)]),
            "v_z_positions": run([v_patch(0, hd, POS_Z)]),
            "v_readout_pos": run([v_patch(0, hd, [READ_POS])]),
            "pattern_and_v": run([pattern_patch(0, hd), v_patch(0, hd)]),
        }
    results["pattern_vs_ov_L0"] = decomp

    # ---- probe 3: path split (corrupted head -> MLP0 only vs bypass MLP0) ----
    W_O = model.blocks[0].attn.W_O           # [head, d_head, d_model]
    path = {}
    for hd in (2, 3):
        n = "blocks.0.attn.hook_z"
        delta_r = torch.einsum("bph,hm->bpm",
                               cache[n][:, :, hd, :] - base[n][:, :, hd, :],
                               W_O[hd])       # clean minus corrupt contribution

        def mlp_only(x, hook, d=delta_r):
            return x - d                      # MLP0 sees corrupted head hd

        def add_back(x, hook, d=delta_r):
            return x + d                      # MLP0 sees clean despite corrupt z

        path[f"L0H{hd}"] = {
            "to_mlp0_only": run([("blocks.0.hook_mlp_in", mlp_only)]),
            "bypass_mlp0": run([headz_patch(0, [hd], base[n]),
                                ("blocks.0.hook_mlp_in", add_back)]),
            "full_corrupt": run([headz_patch(0, [hd], base[n])]),
        }
    results["path_split"] = path

    out_dir = Path(args.out) / args.experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "e10_dissociation.json", "w") as f:
        json.dump(results, f, indent=1)

    print(f"clean m_z={clean['m_z']:.3f} ce={clean['ce']:.4f}")
    print("\nsingle-head mean-ablation dm_z:")
    for k, v in singles.items():
        print(f"  {k}: {clean['m_z'] - v['m_z']:+7.3f}   (ce {v['ce']:.3f})")
    print("\nL0 zero-ablation dm_z:")
    for k, v in results["ablate_L0_zero"].items():
        print(f"  {k}: {clean['m_z'] - v['m_z']:+7.3f}   (ce {v['ce']:.3f})")
    print("\nL0 pairs (mean): dm_z joint vs sum of singles")
    for k, v in pairs.items():
        a, b = k.split("+")
        add = (clean['m_z'] - singles[a]['m_z']) + (clean['m_z'] - singles[b]['m_z'])
        print(f"  {k}: joint {clean['m_z'] - v['m_z']:+7.3f} vs additive {add:+7.3f}")
    print("\npattern vs OV (dm_z):")
    for k, v in decomp.items():
        print(f"  {k}: pattern {clean['m_z'] - v['pattern_only']['m_z']:+7.3f}  "
              f"v_all {clean['m_z'] - v['v_all']['m_z']:+7.3f}  "
              f"v@z {clean['m_z'] - v['v_z_positions']['m_z']:+7.3f}  "
              f"v@readout {clean['m_z'] - v['v_readout_pos']['m_z']:+7.3f}")
    print("\npath split (dm_z):")
    for k, v in path.items():
        print(f"  {k}: ->MLP0-only {clean['m_z'] - v['to_mlp0_only']['m_z']:+7.3f}  "
              f"bypass-MLP0 {clean['m_z'] - v['bypass_mlp0']['m_z']:+7.3f}  "
              f"full {clean['m_z'] - v['full_corrupt']['m_z']:+7.3f}")


if __name__ == "__main__":
    main()
