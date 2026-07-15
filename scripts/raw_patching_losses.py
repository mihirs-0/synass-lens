"""Compute raw losses (L_clean, L_corrupt, L_patch) for K=20 canonical
seeds 0/1/2 at plateau and transition checkpoints. Save per-pair arrays
so we can report mean ± bootstrap CI in CE-space (not probability space).

Output: eta_sweep/results/raw_patching_K20.json
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


def patching_with_per_pair_losses(model, tokenizer, mapping_data, device,
                                   n_pairs=48, seed=0):
    """Like activation_patching, but returns per-pair L_clean, L_corrupt,
    L_patch arrays at L0 z-position only."""
    rng = random.Random(seed)
    n_layers = model.cfg.n_layers
    bases = list(mapping_data.mappings.keys())
    rng.shuffle(bases)

    pairs = []
    for b_str in bases:
        ms = mapping_data.mappings[b_str]
        if len(ms) < 2:
            continue
        i, j = rng.sample(range(len(ms)), 2)
        z1, a1 = ms[i]
        z2, a2 = ms[j]
        t1 = tokenizer.encode_sequence(b_str, z1, a1, task="bz_to_a")
        t2 = tokenizer.encode_sequence(b_str, z2, a2, task="bz_to_a")
        pairs.append((t1, t2))
        if len(pairs) >= n_pairs:
            break

    L_clean_arr, L_corrupt_arr = [], []
    L_patch_per_layer = [[] for _ in range(n_layers)]

    with torch.no_grad():
        for t1, t2 in pairs:
            inp1 = t1["input_ids"].unsqueeze(0).to(device)
            inp2 = t2["input_ids"].unsqueeze(0).to(device)
            ts1 = int(t1["target_start_position"])
            zs1 = int(t1["z_position"])
            ze1 = int(t1["z_end_position"])
            zs2 = int(t2["z_position"])
            ze2 = int(t2["z_end_position"])
            target1 = int(t1["labels"][ts1].item())
            target2 = int(t2["labels"][int(t2["target_start_position"])].item())
            if target1 == target2:
                continue

            logits_clean = model(inp1)
            logp_clean = F.log_softmax(logits_clean[0, ts1 - 1], dim=-1)
            L_clean = -float(logp_clean[target1])

            inp1_shuf = inp1.clone()
            inp1_shuf[0, zs1:ze1] = inp2[0, zs2:ze2]
            logits_shuf = model(inp1_shuf)
            logp_shuf = F.log_softmax(logits_shuf[0, ts1 - 1], dim=-1)
            L_corrupt = -float(logp_shuf[target1])

            _, cache_clean = model.run_with_cache(inp1)
            for L in range(n_layers):
                key = f"blocks.{L}.hook_resid_post"

                def patch_hook(act, hook, L_=L):
                    act[:, zs1:ze1, :] = cache_clean[
                        ("resid_post", L_)][:, zs1:ze1, :]
                    return act

                logits_p = model.run_with_hooks(
                    inp1_shuf, fwd_hooks=[(key, patch_hook)])
                logp_p = F.log_softmax(logits_p[0, ts1 - 1], dim=-1)
                L_patch = -float(logp_p[target1])
                L_patch_per_layer[L].append(L_patch)

            L_clean_arr.append(L_clean)
            L_corrupt_arr.append(L_corrupt)

    return {
        "n_pairs": len(L_clean_arr),
        "L_clean": L_clean_arr,
        "L_corrupt": L_corrupt_arr,
        "L_patch_per_layer": L_patch_per_layer,
    }


def bootstrap_ci(values, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=np.float64)
    n = len(arr)
    boots = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[k] = arr[idx].mean()
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    device = _select_device()
    print(f"device: {device}")

    cells = [
        CellSpec("CellA_seed0", 0.001, 20, 0),
        CellSpec("CellA_seed1", 0.001, 20, 1),
        CellSpec("CellA_seed2", 0.001, 20, 2),
    ]

    out = {"device": device, "k": 20, "log_k": math.log(20),
           "denominator_definition": "L_corrupt - L_clean (loss space, nats)",
           "cells": {}}
    t_start = time.time()

    for cs in cells:
        cc, tokenizer, mapping_data, _, _ = load_cell(cs, device)
        cell_out = {"phases": {}}
        for phase_label, step in [("plateau", 2000), ("transition", 4000)]:
            try:
                model = make_model(cc, tokenizer, step, device)
            except Exception as e:
                print(f"  load err: {e}")
                continue
            r = patching_with_per_pair_losses(model, tokenizer, mapping_data,
                                               device, n_pairs=48, seed=0)
            # Compute summary statistics in loss space, L0
            L_clean = np.array(r["L_clean"])
            L_corrupt = np.array(r["L_corrupt"])
            L_patch_L0 = np.array(r["L_patch_per_layer"][0])
            denom_arr = L_corrupt - L_clean
            abs_recovery = L_corrupt - L_patch_L0
            # R per-pair, only when denom > 0.05 (in loss space)
            R_arr = abs_recovery / np.where(np.abs(denom_arr) > 0.05,
                                            denom_arr, np.nan)
            ci_abs = bootstrap_ci(abs_recovery.tolist())
            phase_summary = {
                "step": step,
                "n_pairs": r["n_pairs"],
                "L_clean_mean": float(L_clean.mean()),
                "L_clean_std": float(L_clean.std()),
                "L_corrupt_mean": float(L_corrupt.mean()),
                "L_corrupt_std": float(L_corrupt.std()),
                "L_patch_L0_mean": float(L_patch_L0.mean()),
                "L_patch_L0_std": float(L_patch_L0.std()),
                "denom_mean (L_corrupt - L_clean)": float(denom_arr.mean()),
                "denom_std": float(denom_arr.std()),
                "abs_recovery_mean (L_corrupt - L_patch_L0)":
                    float(abs_recovery.mean()),
                "abs_recovery_std": float(abs_recovery.std()),
                "abs_recovery_CI95": ci_abs,
                "R_loss_normalized_mean (only pairs with |denom|>0.05)":
                    float(np.nanmean(R_arr)),
                "R_loss_n_reliable_pairs":
                    int(np.sum(~np.isnan(R_arr))),
            }
            cell_out["phases"][phase_label] = phase_summary
            print(f"  {cs.label} {phase_label} step={step}: "
                  f"L_clean={L_clean.mean():.3f} L_corr={L_corrupt.mean():.3f} "
                  f"L_patch={L_patch_L0.mean():.3f} abs_rec={abs_recovery.mean():.3f}")
            del model
            if device == "mps":
                torch.mps.empty_cache()
        out["cells"][cs.label] = cell_out

    out["total_wall_clock_s"] = time.time() - t_start
    out_path = RESULTS_DIR / "raw_patching_K20.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
