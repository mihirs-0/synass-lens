#!/usr/bin/env python
"""
E5 — The B-movie: RelP attribution movie under three counterfactual regimes.

Regimes (matched metric + activation baseline + ablation distribution):
  z    : same B, resampled z   (run 1's conditional regime; regression-tested)
  B    : same z, resampled B
  both : both resampled

Per checkpoint and regime: per-(l,n) scores, top-k membership, cross-regime
Jaccard, position-group attribution masses, per-head OV relevance, and a
faithfulness curve with the matched mean-ablation. Also retests run 1's
"marginal circuit" (stored marg_score rankings) under the ΔB regime.

Output: results/relp2/<experiment>/e5_step_XXXXXX.json (+ e5_h1_retest.json)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf
from src.data import create_tokenizer_from_config
from scripts.relp_core import load_model, select_device, run_relp
from scripts.relp_movie import neuron_scores, position_mass, group_mass, rank_to_masks
from scripts.relp2_core import (
    build_pair_batch, to_device, regime_metrics, register_regime_metrics,
    REGIME_BASELINES, A_LEN,
)

SIZES = [16, 64, 256, 512, 1024, 2048]
REGIMES = ["z", "B", "both"]


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a or b) else float("nan")


def member(score, k=256):
    flat = score.flatten().abs()
    return set(flat.topk(k).indices.tolist())


def regime_curve(model, ids, pb, score, baselines, regime, n_layers, d_mlp, device):
    """Faithfulness/completeness on m_<regime>_pos1 with matched ablation."""
    def mval(mm):
        return float(mm[f"m_{regime}_pos1"].mean())
    with torch.no_grad():
        clean = regime_metrics(model(ids), pb)
    all_mask = {l: torch.ones(d_mlp, dtype=torch.bool, device=device) for l in range(n_layers)}

    def patched_regime(masks):
        hooks = []
        for l, mask in masks.items():
            if mask is None or not bool(mask.any()):
                continue
            m = mask[None, None, :] if mask.dim() == 1 else mask
            base = baselines[l]
            def make_hook(m=m, base=base):
                def h(post, hook):
                    return torch.where(m, base, post)
                return h
            hooks.append((f"blocks.{l}.mlp.hook_post", make_hook()))
        with torch.no_grad():
            with model.hooks(fwd_hooks=hooks):
                return regime_metrics(model(ids), pb)

    floor = patched_regime(all_mask)
    out = {"sizes": [], "faith_m": [], "compl_m": [],
           "clean_m": mval(clean), "floor_m": mval(floor),
           "clean_ce": float(clean["ce_pos1"].mean()),
           "floor_ce": float(floor["ce_pos1"].mean())}
    for k in SIZES:
        mem = rank_to_masks(score, k, n_layers, d_mlp, device)
        compl = {l: ~mem[l] for l in range(n_layers)}
        out["sizes"].append(k)
        out["faith_m"].append(mval(patched_regime(compl)))
        out["compl_m"].append(mval(patched_regime(mem)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="landauer_dense_k10")
    ap.add_argument("--steps", type=int, nargs="+", required=True)
    ap.add_argument("--n-b", type=int, default=64)
    ap.add_argument("--out", default="results/relp2")
    ap.add_argument("--run1-dir", default="results/relp")
    ap.add_argument("--e8-design", default=None,
                    help="E8 design name (e.g. 8b_skewz): rebuild its custom "
                         "mapping data for the pair batch")
    args = ap.parse_args()

    register_regime_metrics()
    device = select_device()
    exp_dir = Path("outputs") / args.experiment
    cfg = OmegaConf.load(exp_dir / "config.yaml")
    tok = create_tokenizer_from_config(cfg)
    md = None
    if args.e8_design:
        from scripts.relp2_e8 import get_design, materialize_mapping
        cfg, md, _w = get_design(args.e8_design)
        if md is None:
            md = materialize_mapping(cfg)
    pb = build_pair_batch(cfg, tok, n_b_eval=args.n_b, seed=1234, mapping_data=md)
    ids, pb = to_device(pb, device)
    n_b, K = pb.n_b, pb.k
    n_layers, d_mlp = int(cfg.model.n_layers), int(cfg.model.d_mlp)
    out_dir = Path(args.out) / args.experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    run1_dir = Path(args.run1_dir) / args.experiment

    h1_retest = []

    for step in args.steps:
        t0 = time.time()
        model = load_model(cfg, tok, exp_dir / "checkpoints", step, device)
        model.requires_grad_(True)
        res = {"step": step}
        with torch.no_grad():
            rm = regime_metrics(model(ids), pb)
        res["metrics"] = {k: float(v.mean()) for k, v in rm.items()}

        scores, bases_l = {}, {}
        for regime in REGIMES:
            acts, grads, _ = run_relp(model, ids, pb, f"m_{regime}_pos1", linearize=True)
            bfn = REGIME_BASELINES[regime]
            base = {l: bfn(acts[f"mlp_post_{l}"], n_b, K) for l in range(n_layers)}
            attr = {l: grads[f"mlp_post_{l}"] * (acts[f"mlp_post_{l}"] - base[l])
                    for l in range(n_layers)}
            signed, absm = neuron_scores(attr, n_layers, d_mlp)
            scores[regime] = signed
            bases_l[regime] = base
            res[f"score_{regime}"] = signed.tolist()
            res[f"pos_mass_{regime}"] = group_mass(position_mass(attr, n_layers))
            # per-head OV relevance under matched baseline
            per_head = []
            for l in range(n_layers):
                az = acts[f"attn_z_{l}"]
                attr_h = grads[f"attn_z_{l}"] * (az - bfn(az, n_b, K))
                per_head.append(attr_h.abs().sum(dim=(1, 3)).mean(dim=0).cpu().tolist())
            res[f"head_relevance_{regime}"] = per_head
            res[f"curve_{regime}"] = regime_curve(
                model, ids, pb, signed, base, regime, n_layers, d_mlp, device)

        res["cross_regime_jaccard"] = {
            f"{a}_vs_{b}": jaccard(member(scores[a]), member(scores[b]))
            for a, b in [("z", "B"), ("z", "both"), ("B", "both")]}

        # H1 retest: run-1 marginal circuit under the ΔB regime
        r1 = run1_dir / f"step_{step:06d}.json"
        if r1.exists():
            marg = torch.tensor(json.load(open(r1))["marg_score"])
            c = regime_curve(model, ids, pb, marg, bases_l["B"], "B",
                             n_layers, d_mlp, device)
            res["run1_margcircuit_dB_curve"] = c
            h1_retest.append({"step": step, "clean_m_B": c["clean_m"],
                              "floor_m_B": c["floor_m"],
                              "faith_at_512": c["faith_m"][SIZES.index(512)]})

        with open(out_dir / f"e5_step_{step:06d}.json", "w") as f:
            json.dump(res, f)
        print(f"[{args.experiment}] step {step}: "
              f"m_z={res['metrics']['m_z_pos1']:.3f} m_B={res['metrics']['m_B_pos1']:.3f} "
              f"m_both={res['metrics']['m_both_pos1']:.3f} "
              f"J(z,B)={res['cross_regime_jaccard']['z_vs_B']:.3f} "
              f"({time.time() - t0:.1f}s)", flush=True)
        del model
        if device == "mps":
            torch.mps.empty_cache()

    with open(out_dir / "e5_h1_retest.json", "w") as f:
        json.dump(h1_retest, f, indent=1)
    print("done.")


if __name__ == "__main__":
    main()
