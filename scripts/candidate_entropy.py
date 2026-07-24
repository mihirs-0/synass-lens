"""Compute candidate-restricted entropy at plateau for K=20 3 seeds.

Output: prints summary table and writes JSON.
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
    x = x - np.max(x); e = np.exp(x); return e / e.sum()


def score_K(model, tokenizer, b_str, z_str, cands, device):
    encs = [tokenizer.encode_sequence(b_str, z_str, a, task="bz_to_a")
            for a in cands]
    input_ids = torch.stack([e["input_ids"] for e in encs]).to(device)
    with torch.no_grad():
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
    return softmax_np(scores)


def entropy(p):
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def main():
    device = _select_device()
    log_K = math.log(20)
    print(f"K=20, log K = {log_K:.4f}")
    out = {"k": 20, "log_k": log_K, "cells": {}}

    cells = [
        CellSpec("CellA_seed0", 0.001, 20, 0),
        CellSpec("CellA_seed1", 0.001, 20, 1),
        CellSpec("CellA_seed2", 0.001, 20, 2),
    ]
    PHASES = {"plateau": 2000, "transition": 4000, "post": 7500}

    for cs in cells:
        cc, tokenizer, mapping_data, _, _ = load_cell(cs, device)
        cell_out = {}
        for ph_label, target_step in PHASES.items():
            from eta_sweep.analysis.mech_interp_mbc import list_ckpts
            avail = list_ckpts(cs)
            step = min(avail, key=lambda s: abs(s - target_step))
            model = make_model(cc, tokenizer, step, device)
            rng = random.Random(0)
            bases = list(mapping_data.mappings.keys())
            sampled = rng.sample(bases, 96)
            entropies = []
            top1_probs = []
            for b in sampled:
                ms = mapping_data.mappings[b]
                if len(ms) < 2: continue
                idx_c = rng.randrange(len(ms))
                z_c, _ = ms[idx_c]
                cands = [a for _, a in ms]
                p = score_K(model, tokenizer, b, z_c, cands, device)
                entropies.append(entropy(p))
                top1_probs.append(float(p.max()))
            ent_arr = np.array(entropies)
            top1_arr = np.array(top1_probs)
            cell_out[ph_label] = {
                "step": step,
                "n": len(entropies),
                "entropy_mean": float(ent_arr.mean()),
                "entropy_std": float(ent_arr.std()),
                "entropy_over_logK": float(ent_arr.mean() / log_K),
                "top1_p_mean": float(top1_arr.mean()),
                "top1_p_std": float(top1_arr.std()),
            }
            print(f"  {cs.label} {ph_label} step={step}: "
                  f"H={ent_arr.mean():.3f}±{ent_arr.std():.3f}  "
                  f"H/logK={ent_arr.mean()/log_K:.3f}  "
                  f"top1_p={top1_arr.mean():.3f}")
            del model
            if device == "mps":
                torch.mps.empty_cache()
        out["cells"][cs.label] = cell_out

    out_path = RESULTS_DIR / "candidate_entropy_K20.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")

    # aggregate
    print("\nAggregate across 3 seeds:")
    for ph in ["plateau", "transition", "post"]:
        Hs = [out["cells"][c][ph]["entropy_mean"] for c in out["cells"]]
        Hr = [out["cells"][c][ph]["entropy_over_logK"] for c in out["cells"]]
        T1 = [out["cells"][c][ph]["top1_p_mean"] for c in out["cells"]]
        print(f"  {ph}: H = {np.mean(Hs):.3f} ± {np.std(Hs):.3f}  "
              f"H/logK = {np.mean(Hr):.3f} ± {np.std(Hr):.3f}  "
              f"top1_p = {np.mean(T1):.3f} ± {np.std(T1):.3f}")


if __name__ == "__main__":
    main()
