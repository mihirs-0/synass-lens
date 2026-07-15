#!/usr/bin/env python
"""
Config-driven runner: RelP circuit attribution across MBC training checkpoints.

Per checkpoint:
  - RelP attributions (linearized grad x (act - counterfactual baseline)) at
    mlp.hook_post, for the conditional metric (m_diff, z-resample baseline)
    and the marginal metric (m_marg, B-resample baseline).
  - Pure relevance (grad x act) position masses for input-blindness tracking.
  - Per-head OV relevance at attn.hook_z.
  - Faithfulness / completeness curves vs circuit size (top-k neurons by
    |mean attribution|, complement/circuit mean-ablated to the counterfactual
    baseline).
  - Optional E1 baselines at selected steps: random sets, top-|activation|,
    Integrated Gradients ranking.

Writes results/<out>/<experiment>/step_XXXXXX.json.

Usage:
  python scripts/relp_movie.py --experiment landauer_dense_k10 \
      --steps 100 300 600 ... --n-b 64 --baselines-at 50000
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import create_tokenizer_from_config
from scripts.relp_core import (
    EvalBatch, build_eval_batch, metric_values, run_relp, linearized,
    z_resample_baseline, b_resample_baseline, run_with_mlp_patch,
    ig_attribution, load_model, select_device,
    selftest_frozen_forward, POSITION_GROUPS, READ_POS,
)

SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]


def neuron_scores(attr_by_layer, n_layers, d_mlp):
    """attr_by_layer: dict l -> [N, seq, d_mlp]. Returns per-(l,n) scores.

    signed_mean: mean over examples of position-summed attribution (primary,
    ranked by absolute value); abs_mean: mean over examples and positions of
    |attr| (alternative ranking).
    """
    signed = torch.zeros(n_layers, d_mlp)
    absm = torch.zeros(n_layers, d_mlp)
    for l in range(n_layers):
        a = attr_by_layer[l]
        signed[l] = a.sum(dim=1).mean(dim=0).cpu()
        absm[l] = a.abs().sum(dim=1).mean(dim=0).cpu()
    return signed, absm


def position_mass(attr_by_layer, n_layers):
    """Sum of |attr| over (layer, neuron), mean over examples -> [seq]."""
    mass = None
    for l in range(n_layers):
        m = attr_by_layer[l].abs().sum(dim=2).mean(dim=0).cpu()  # [seq]
        mass = m if mass is None else mass + m
    return mass


def group_mass(mass_vec):
    return {g: float(mass_vec[idx].sum()) for g, idx in
            ((g, torch.tensor(p)) for g, p in POSITION_GROUPS.items())}


def top_triples(attr_by_layer, n_layers, top=512):
    """Top (layer, pos, neuron) entries by |mean over examples|."""
    entries = []
    for l in range(n_layers):
        m = attr_by_layer[l].mean(dim=0).cpu()  # [seq, d_mlp]
        flat = m.flatten()
        k = min(top, flat.numel())
        vals, idx = flat.abs().topk(k)
        seq = m.shape[0]
        d_mlp = m.shape[1]
        for v, i in zip(vals.tolist(), idx.tolist()):
            entries.append((v, l, i // d_mlp, i % d_mlp, float(flat[i])))
    entries.sort(reverse=True)
    return [
        {"layer": l, "pos": p, "neuron": n, "attr": a}
        for (_v, l, p, n, a) in entries[:top]
    ]


def rank_to_masks(score, k, n_layers, d_mlp, device):
    """Top-k (l,n) by |score| -> per-layer circuit membership masks."""
    flat = score.flatten().abs()
    k = min(k, flat.numel())
    idx = flat.topk(k).indices
    member = torch.zeros(n_layers * d_mlp, dtype=torch.bool)
    member[idx] = True
    member = member.view(n_layers, d_mlp)
    return {l: member[l].to(device) for l in range(n_layers)}


def curve(model, ids, batch, score, baselines, metric_key, sizes, n_layers, d_mlp, device):
    """Faithfulness (ablate complement) and completeness (ablate circuit) vs size."""
    with torch.no_grad():
        clean = metric_values(model(ids), batch)
    clean_m = float(clean[metric_key].mean())
    # floor: everything ablated
    all_mask = {l: torch.ones(d_mlp, dtype=torch.bool, device=device) for l in range(n_layers)}
    floor = run_with_mlp_patch(model, ids, batch, all_mask, baselines)
    floor_m = float(floor[metric_key].mean())
    out = {"sizes": [], "faith_m": [], "compl_m": [], "faith_ce": [], "compl_ce": [],
           "faith_acc": [], "compl_acc": [],
           "clean_m": clean_m, "floor_m": floor_m,
           "clean_ce": float(clean["ce_first"].mean()), "floor_ce": float(floor["ce_first"].mean())}
    for k in sizes:
        member = rank_to_masks(score, k, n_layers, d_mlp, device)
        compl_mask = {l: ~member[l] for l in range(n_layers)}
        faith = run_with_mlp_patch(model, ids, batch, compl_mask, baselines)
        compl = run_with_mlp_patch(model, ids, batch, member, baselines)
        out["sizes"].append(k)
        out["faith_m"].append(float(faith[metric_key].mean()))
        out["compl_m"].append(float(compl[metric_key].mean()))
        out["faith_ce"].append(float(faith["ce_first"].mean()))
        out["compl_ce"].append(float(compl["ce_first"].mean()))
        out["faith_acc"].append(float(faith["acc_first"].mean()))
        out["compl_acc"].append(float(compl["acc_first"].mean()))
    return out


def size_at_recovery(curve_dict, frac=0.9):
    """Smallest size whose faithfulness recovers >= frac of (clean - floor)."""
    c, f = curve_dict["clean_m"], curve_dict["floor_m"]
    denom = c - f
    if abs(denom) < 1e-6:
        return None
    for k, m in zip(curve_dict["sizes"], curve_dict["faith_m"]):
        if (m - f) / denom >= frac:
            return k
    return None


def analyze_checkpoint(model, ids, batch, n_layers, d_mlp, device,
                       sizes=SIZES, with_baselines=False, ig_steps=32, rand_seeds=(0, 1, 2)):
    res = {}
    with torch.no_grad():
        mv = metric_values(model(ids), batch)
    res["metrics"] = {k: float(v.mean()) for k, v in mv.items()}

    n_b, K = batch.n_b, batch.k

    # --- RelP passes (linearized) ---
    acts, grads_d, m_d = run_relp(model, ids, batch, "m_diff", linearize=True)
    _, grads_g, m_g = run_relp(model, ids, batch, "m_marg", linearize=True)
    _, grads_p, _ = run_relp(model, ids, batch, "m_plain", linearize=True)

    base_z = {f"mlp_post_{l}": z_resample_baseline(acts[f"mlp_post_{l}"], n_b, K) for l in range(n_layers)}
    base_b = {f"mlp_post_{l}": b_resample_baseline(acts[f"mlp_post_{l}"], n_b, K) for l in range(n_layers)}
    # int-keyed views for the ablation runner
    base_z_l = {l: base_z[f"mlp_post_{l}"] for l in range(n_layers)}
    base_b_l = {l: base_b[f"mlp_post_{l}"] for l in range(n_layers)}

    cond_attr = {l: grads_d[f"mlp_post_{l}"] * (acts[f"mlp_post_{l}"] - base_z[f"mlp_post_{l}"]) for l in range(n_layers)}
    marg_attr = {l: grads_g[f"mlp_post_{l}"] * (acts[f"mlp_post_{l}"] - base_b[f"mlp_post_{l}"]) for l in range(n_layers)}
    rel_diff = {l: grads_d[f"mlp_post_{l}"] * acts[f"mlp_post_{l}"] for l in range(n_layers)}
    rel_marg = {l: grads_g[f"mlp_post_{l}"] * acts[f"mlp_post_{l}"] for l in range(n_layers)}
    rel_plain = {l: grads_p[f"mlp_post_{l}"] * acts[f"mlp_post_{l}"] for l in range(n_layers)}

    cond_signed, cond_abs = neuron_scores(cond_attr, n_layers, d_mlp)
    marg_signed, marg_abs = neuron_scores(marg_attr, n_layers, d_mlp)
    res["cond_score"] = cond_signed.tolist()
    res["cond_score_absmean"] = cond_abs.tolist()
    res["marg_score"] = marg_signed.tolist()
    res["marg_score_absmean"] = marg_abs.tolist()

    res["pos_mass"] = {
        "cond_cf": group_mass(position_mass(cond_attr, n_layers)),
        "marg_cf": group_mass(position_mass(marg_attr, n_layers)),
        "rel_diff": group_mass(position_mass(rel_diff, n_layers)),
        "rel_marg": group_mass(position_mass(rel_marg, n_layers)),
        "rel_plain": group_mass(position_mass(rel_plain, n_layers)),
    }
    res["top_triples_cond"] = top_triples(cond_attr, n_layers, top=300)

    # --- per-head OV relevance at hook_z ---
    head_rel = {}
    for tag, grads in (("cond", grads_d), ("marg", grads_g)):
        per_head = []
        for l in range(n_layers):
            az = acts[f"attn_z_{l}"]
            bz = z_resample_baseline(az, n_b, K) if tag == "cond" else b_resample_baseline(az, n_b, K)
            attr = grads[f"attn_z_{l}"] * (az - bz)              # [N, seq, head, d_head]
            per_head.append(attr.abs().sum(dim=(1, 3)).mean(dim=0).cpu().tolist())
        head_rel[tag] = per_head
    res["head_relevance"] = head_rel

    # --- faithfulness / completeness curves ---
    res["curve_cond"] = curve(model, ids, batch, cond_signed, base_z_l, "m_diff",
                              sizes, n_layers, d_mlp, device)
    res["curve_marg"] = curve(model, ids, batch, marg_signed, base_b_l, "m_marg",
                              sizes, n_layers, d_mlp, device)
    # H1: the marginal-ranked circuit evaluated on the conditional metric
    res["curve_margcircuit_on_mdiff"] = curve(model, ids, batch, marg_signed, base_z_l, "m_diff",
                                              sizes, n_layers, d_mlp, device)
    res["size90_cond"] = size_at_recovery(res["curve_cond"], 0.9)
    res["size90_marg"] = size_at_recovery(res["curve_marg"], 0.9)

    # --- E1 baselines ---
    if with_baselines:
        bl = {}
        # random rankings
        for s in rand_seeds:
            g = torch.Generator().manual_seed(s)
            rnd = torch.randn(n_layers, d_mlp, generator=g)
            bl[f"random_{s}"] = curve(model, ids, batch, rnd, base_z_l, "m_diff",
                                      sizes, n_layers, d_mlp, device)
        # top |activation| (positions that matter for the first-token metric)
        act_score = torch.zeros(n_layers, d_mlp)
        for l in range(n_layers):
            act_score[l] = acts[f"mlp_post_{l}"][:, :READ_POS + 1].abs().sum(dim=1).mean(dim=0).cpu()
        bl["activation"] = curve(model, ids, batch, act_score, base_z_l, "m_diff",
                                 sizes, n_layers, d_mlp, device)
        # abs-mean RelP ranking (aggregation ablation)
        bl["relp_absmean"] = curve(model, ids, batch, cond_abs, base_z_l, "m_diff",
                                   sizes, n_layers, d_mlp, device)
        # Integrated Gradients ranking (raw gradients, same baseline)
        ig = ig_attribution(model, ids, batch, "m_diff", acts, base_z, steps=ig_steps)
        ig_by_layer = {l: ig[f"mlp_post_{l}"] for l in range(n_layers)}
        ig_signed, ig_abs = neuron_scores(ig_by_layer, n_layers, d_mlp)
        bl["ig"] = curve(model, ids, batch, ig_signed, base_z_l, "m_diff",
                         sizes, n_layers, d_mlp, device)
        bl["ig_score"] = ig_signed.tolist()
        # agreement RelP vs IG
        v1, v2 = cond_signed.flatten(), ig_signed.flatten()
        bl["relp_ig_pearson"] = float(torch.corrcoef(torch.stack([v1, v2]))[0, 1])
        res["baselines"] = bl

        # per-example sparsity (Transluce-comparable): each example keeps its own
        # top-k neurons by |position-summed attribution|, complement ablated
        per_ex = torch.stack(
            [cond_attr[l].sum(dim=1) for l in range(n_layers)], dim=1)  # [N, L, d_mlp]
        flat = per_ex.abs().reshape(per_ex.shape[0], -1)                # [N, L*d_mlp]
        order = flat.argsort(dim=1, descending=True)
        pe = {"sizes": [], "faith_m": []}
        with torch.no_grad():
            clean_m_diff = float(metric_values(model(ids), batch)["m_diff"].mean())
        all_mask = {l: torch.ones(d_mlp, dtype=torch.bool, device=device) for l in range(n_layers)}
        pe["floor_m"] = float(run_with_mlp_patch(model, ids, batch, all_mask, base_z_l)["m_diff"].mean())
        pe["clean_m"] = clean_m_diff
        for k in sizes:
            keep = torch.zeros_like(flat, dtype=torch.bool)
            keep.scatter_(1, order[:, :k], True)
            keep = keep.view(per_ex.shape)                              # [N, L, d_mlp]
            masks = {l: (~keep[:, l, :])[:, None, :].to(device) for l in range(n_layers)}
            r = run_with_mlp_patch(model, ids, batch, masks, base_z_l)
            pe["sizes"].append(k)
            pe["faith_m"].append(float(r["m_diff"].mean()))
        res["per_example_curve"] = pe

        # first-order sanity: does summed RelP attribution predict patching dm?
        torch.manual_seed(0)
        preds, actuals = [], []
        for _ in range(24):
            mask = {l: (torch.rand(d_mlp) < 0.1).to(device) for l in range(n_layers)}
            pred = 0.0
            for l in range(n_layers):
                pred += float(cond_attr[l][:, :, mask[l]].sum(dim=(1, 2)).mean())
            abl = run_with_mlp_patch(model, ids, batch, mask, base_z_l)
            actual = res["metrics"]["m_diff"] - float(abl["m_diff"].mean())
            preds.append(pred)
            actuals.append(actual)
        pa = torch.tensor([preds, actuals])
        res["patch_pred_pearson"] = float(torch.corrcoef(pa)[0, 1])
        res["patch_pred_pairs"] = list(zip(preds, actuals))

    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="landauer_dense_k10")
    ap.add_argument("--steps", type=int, nargs="+", required=True)
    ap.add_argument("--n-b", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--baselines-at", type=int, nargs="*", default=[])
    ap.add_argument("--out", default="results/relp")
    ap.add_argument("--ig-steps", type=int, default=32)
    args = ap.parse_args()

    device = select_device()
    exp_dir = Path("outputs") / args.experiment
    cfg = OmegaConf.load(exp_dir / "config.yaml")
    ckpt_dir = exp_dir / "checkpoints"
    out_dir = Path(args.out) / args.experiment
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = create_tokenizer_from_config(cfg)
    batch = build_eval_batch(cfg, tokenizer, n_b_eval=args.n_b, seed=args.seed)
    ids = batch.input_ids.to(device)
    for attr in ["b_idx", "z_idx", "cand_tids", "noncand_tids", "correct_tid"]:
        setattr(batch, attr, getattr(batch, attr).to(device))

    n_layers, d_mlp = int(cfg.model.n_layers), int(cfg.model.d_mlp)
    manifest = {"experiment": args.experiment, "n_b": args.n_b, "k": batch.k,
                "seed": args.seed, "steps": args.steps, "sizes": SIZES,
                "device": device, "n_layers": n_layers, "d_mlp": d_mlp}

    for i, step in enumerate(args.steps):
        t0 = time.time()
        model = load_model(cfg, tokenizer, ckpt_dir, step, device)
        model.requires_grad_(True)
        if i == 0:
            chk = selftest_frozen_forward(model, ids)
            print(f"[selftest] {chk}", flush=True)
            assert chk["frozen_forward_max_abs_diff"] < 1e-3
        res = analyze_checkpoint(
            model, ids, batch, n_layers, d_mlp, device,
            with_baselines=(step in args.baselines_at), ig_steps=args.ig_steps)
        res["step"] = step
        with open(out_dir / f"step_{step:06d}.json", "w") as f:
            json.dump(res, f)
        print(f"[{args.experiment}] step {step}: m_diff={res['metrics']['m_diff']:.3f} "
              f"m_marg={res['metrics']['m_marg']:.3f} ce={res['metrics']['ce_first']:.4f} "
              f"size90_cond={res['size90_cond']} size90_marg={res['size90_marg']} "
              f"({time.time() - t0:.1f}s)", flush=True)
        del model
        if device == "mps":
            torch.mps.empty_cache()

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
