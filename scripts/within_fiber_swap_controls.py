"""Robustness controls for the within-fiber-swap claim.

For canonical K=20 seeds {0,1,2} at plateau / transition / post:
  - clean candidate distribution p_clean (softmax over K candidate strings)
  - swap distribution p_swap (replacing z with a different in-fiber selector)
  - matched random perturbation: inject Gaussian noise at the z-position
    L0 hook_resid_pre site, with magnitude matched to the actual swap-induced
    delta. n_random_draws per example.
  - per-example margins (clean top1 − top2 in candidate-restricted log-probs)
  - KL, JS, TV between distributions.

Output: eta_sweep/results/within_fiber_swap_controls.json
"""
from __future__ import annotations
import json
import math
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from eta_sweep.config import RESULTS_DIR  # noqa: E402
from eta_sweep.analysis.mech_interp_mbc import (  # noqa: E402
    CellSpec, load_cell, make_model, _select_device,
)


def softmax_np(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def kl(p, q, eps=1e-12):
    p = np.clip(p, eps, 1.0); q = np.clip(q, eps, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def js(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def tv(p, q):
    return 0.5 * float(np.sum(np.abs(p - q)))


def score_K(model, tokenizer, b_str, z_str, cands, device, hook=None):
    """Return (raw_scores [K], normalized_p [K]). Each candidate is one forward pass."""
    encs = [tokenizer.encode_sequence(b_str, z_str, a, task="bz_to_a")
            for a in cands]
    input_ids = torch.stack([e["input_ids"] for e in encs]).to(device)
    with torch.no_grad():
        if hook is not None:
            logits = model.run_with_hooks(input_ids, fwd_hooks=[hook])
        else:
            logits = model(input_ids)
    K = len(cands)
    scores = np.zeros(K, dtype=np.float64)
    for k in range(K):
        labels = encs[k]["labels"]
        start = int(encs[k]["target_start_position"])
        end = int(encs[k]["target_end_position"])
        for pos in range(start, end + 1):
            tok = labels[pos].item()
            if tok == -100: continue
            lp = F.log_softmax(logits[k, pos - 1], dim=-1)
            scores[k] += float(lp[tok].item())
    return scores, softmax_np(scores)


def get_z_resid(model, tokenizer, b_str, z_str, a_placeholder, device,
                hook_name="blocks.0.hook_resid_pre"):
    enc = tokenizer.encode_sequence(b_str, z_str, a_placeholder,
                                     task="bz_to_a")
    input_ids = enc["input_ids"].unsqueeze(0).to(device)
    z_s = int(enc["z_position"]); z_e = int(enc["z_end_position"])
    with torch.no_grad():
        _, cache = model.run_with_cache(
            input_ids,
            names_filter=lambda n: n == hook_name)
    resid = cache[hook_name][0, z_s:z_e, :].detach().cpu().numpy()
    return resid, z_s, z_e


def run_phase(cs, step, device, n_examples=48, n_random=5, seed=0):
    cc, tokenizer, mapping_data, _, _ = load_cell(cs, device)
    model = make_model(cc, tokenizer, step, device)
    bases = list(mapping_data.mappings.keys())
    rng = random.Random(seed)
    sampled = rng.sample(bases, min(n_examples, len(bases)))

    per_example = []
    for b in sampled:
        ms = mapping_data.mappings[b]
        if len(ms) < 2: continue
        idx_c = rng.randrange(len(ms))
        idx_s = rng.choice([i for i in range(len(ms)) if i != idx_c])
        z_c, _ = ms[idx_c]; z_s, _ = ms[idx_s]
        cands = [a for _, a in ms]

        s_c, p_c = score_K(model, tokenizer, b, z_c, cands, device)
        s_sw, p_sw = score_K(model, tokenizer, b, z_s, cands, device)

        # margin in raw scores (logit-like)
        sorted_s = np.sort(s_c)[::-1]
        margin_logit = float(sorted_s[0] - sorted_s[1])
        sorted_p = np.sort(p_c)[::-1]
        margin_prob = float(sorted_p[0] - sorted_p[1])

        c_top1 = int(np.argmax(p_c)); sw_top1 = int(np.argmax(p_sw))

        z_resid_c, z_start, z_end = get_z_resid(model, tokenizer, b, z_c,
                                                  cands[idx_c], device)
        z_resid_s, _, _ = get_z_resid(model, tokenizer, b, z_s,
                                       cands[idx_c], device)
        delta = z_resid_s - z_resid_c
        delta_norm = float(np.linalg.norm(delta))

        # matched random perturbations
        rng_np = np.random.default_rng(seed * 1_000_000 + hash(b) % 1_000_000)
        rand_results = []
        for d in range(n_random):
            r = rng_np.standard_normal(size=delta.shape).astype(np.float32)
            r_n = float(np.linalg.norm(r))
            if r_n < 1e-12: continue
            r_scaled = r * (delta_norm / r_n)
            r_tensor = torch.from_numpy(r_scaled).to(device)

            def hook_fn(act, hook, _r=r_tensor, _zs=z_start, _ze=z_end):
                act[:, _zs:_ze, :] = act[:, _zs:_ze, :] + _r
                return act

            _, p_r = score_K(model, tokenizer, b, z_c, cands, device,
                              hook=("blocks.0.hook_resid_pre", hook_fn))
            r_top1 = int(np.argmax(p_r))
            rand_results.append({
                "differs": int(r_top1 != c_top1),
                "acc_swap": int(r_top1 == idx_s),
                "kl": kl(p_r, p_c),
                "js": js(p_r, p_c),
                "tv": tv(p_r, p_c),
            })

        per_example.append({
            "b": b,
            "idx_c": idx_c, "idx_s": idx_s,
            "margin_logit": margin_logit, "margin_prob": margin_prob,
            "p_clean_top1": float(p_c.max()),
            "p_swap_top1": float(p_sw.max()),
            "swap_differs": int(sw_top1 != c_top1),
            "swap_acc_target": int(sw_top1 == idx_s),
            "kl_sw": kl(p_sw, p_c),
            "js_sw": js(p_sw, p_c),
            "tv_sw": tv(p_sw, p_c),
            "delta_norm": delta_norm,
            "random_draws": rand_results,
        })

    del model
    if device == "mps":
        torch.mps.empty_cache()

    return per_example


def aggregate(per_examples):
    n = len(per_examples)
    if n == 0: return {}
    swap_d = [e["swap_differs"] for e in per_examples]
    swap_a = [e["swap_acc_target"] for e in per_examples]
    kl_sw = [e["kl_sw"] for e in per_examples]
    js_sw = [e["js_sw"] for e in per_examples]
    tv_sw = [e["tv_sw"] for e in per_examples]

    rand_d = [r["differs"] for e in per_examples for r in e["random_draws"]]
    rand_a = [r["acc_swap"] for e in per_examples for r in e["random_draws"]]
    kl_r = [r["kl"] for e in per_examples for r in e["random_draws"]]
    js_r = [r["js"] for e in per_examples for r in e["random_draws"]]
    tv_r = [r["tv"] for e in per_examples for r in e["random_draws"]]

    margins = np.array([e["margin_logit"] for e in per_examples])
    swap_d_arr = np.array(swap_d, dtype=np.float64)
    swap_a_arr = np.array(swap_a, dtype=np.float64)
    kl_sw_arr = np.array(kl_sw, dtype=np.float64)

    # margin bins (terciles)
    qs = np.quantile(margins, [1/3, 2/3])
    bins = np.digitize(margins, qs)
    bin_summary = {}
    for b in [0, 1, 2]:
        idx = (bins == b)
        if idx.sum() == 0: continue
        bin_summary[f"bin_{b}"] = {
            "n": int(idx.sum()),
            "margin_range": [float(margins[idx].min()),
                              float(margins[idx].max())],
            "swap_differs_rate": float(swap_d_arr[idx].mean()),
            "swap_acc_target_rate": float(swap_a_arr[idx].mean()),
            "kl_sw_mean": float(kl_sw_arr[idx].mean()),
        }

    return {
        "n_examples": n,
        "swap_differs_rate": float(np.mean(swap_d)),
        "swap_acc_target_rate": float(np.mean(swap_a)),
        "kl_sw_mean": float(np.mean(kl_sw)),
        "kl_sw_std": float(np.std(kl_sw)),
        "js_sw_mean": float(np.mean(js_sw)),
        "tv_sw_mean": float(np.mean(tv_sw)),
        "random_differs_rate": float(np.mean(rand_d)) if rand_d else None,
        "random_acc_target_rate": float(np.mean(rand_a)) if rand_a else None,
        "kl_random_mean": float(np.mean(kl_r)) if kl_r else None,
        "js_random_mean": float(np.mean(js_r)) if js_r else None,
        "tv_random_mean": float(np.mean(tv_r)) if tv_r else None,
        "margin_terciles": bin_summary,
        "margin_logit_overall_mean": float(margins.mean()),
        "margin_logit_overall_std": float(margins.std()),
    }


def main():
    device = _select_device()
    print(f"device: {device}")

    cells = [
        CellSpec("CellA_seed0", 0.001, 20, 0),
        CellSpec("CellA_seed1", 0.001, 20, 1),
        CellSpec("CellA_seed2", 0.001, 20, 2),
    ]
    PHASES = {"plateau": 2000, "transition": 4000, "post": 7500}

    out = {"device": device, "cells": {}}
    t_start = time.time()

    for cs in cells:
        cell_out = {}
        for phase_label, target_step in PHASES.items():
            from eta_sweep.analysis.mech_interp_mbc import list_ckpts
            avail = list_ckpts(cs)
            step = min(avail, key=lambda s: abs(s - target_step))
            print(f"\n>>> {cs.label} {phase_label} step={step}")
            t0 = time.time()
            pe = run_phase(cs, step, device,
                            n_examples=48, n_random=5, seed=0)
            agg = aggregate(pe)
            agg["step"] = step
            cell_out[phase_label] = {
                "summary": agg,
                "per_example": pe,
            }
            print(f"  swap diff={agg['swap_differs_rate']:.3f}  "
                  f"rand diff={agg.get('random_differs_rate', 0):.3f}  "
                  f"kl_sw={agg['kl_sw_mean']:.4f}  "
                  f"kl_rand={agg.get('kl_random_mean', 0):.4f}  "
                  f"({time.time()-t0:.1f}s)")
        out["cells"][cs.label] = cell_out

    out["total_wall_clock_s"] = time.time() - t_start
    out_path = RESULTS_DIR / "within_fiber_swap_controls.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o)
                  if hasattr(o, "__float__") else str(o))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
