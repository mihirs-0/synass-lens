"""Within-fiber selector swap diagnostic.

For each example (B, z_correct, A_correct), pick z' ≠ z_correct from the
same B-fiber (with corresponding target A_swap), and ask:
  - Clean accuracy:  argmax_a P(a | B, z_correct) == A_correct ?
  - Swap accuracy:   argmax_a P(a | B, z'       ) == A_swap    ?
  - Swap-tracks-z:   prediction with z' differs from prediction with z_correct?

A model in Phase 1 (marginal) implements P(A|B) and so the swap prediction
should equal the clean prediction (z ignored). A model that has reached
Phase 4 (conditional lookup) should give swap accuracy ≈ 1.0.

The intermediate Phase 2 ("z-corruption sensitive") may show
swap-predicts-different-from-clean even when swap accuracy is still low.

Outputs eta_sweep/results/within_fiber_swap.json.
"""
from __future__ import annotations
import json
import math
import random
import time
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from eta_sweep.config import RESULTS_DIR  # noqa: E402
from eta_sweep.analysis.mech_interp_mbc import (  # noqa: E402
    CellSpec, load_cell, make_model, _select_device,
)
from src.analysis.candidate_eval import score_candidate_sequences  # noqa: E402


def within_fiber_swap_eval(model, tokenizer, mapping_data, device,
                            n_examples=64, seed=0):
    rng = random.Random(seed)
    bases = list(mapping_data.mappings.keys())
    sampled = rng.sample(bases, min(n_examples, len(bases)))

    clean_correct = 0
    swap_correct = 0
    swap_predicts_swap_target = 0
    swap_predicts_clean_target = 0
    swap_predicts_other = 0
    swap_predicts_differs_from_clean = 0
    n = 0
    swap_pred_distribution = {}  # what does the model predict on swapped inputs

    cand_loss_clean_arr = []
    cand_loss_swap_arr = []

    for b in sampled:
        ms = mapping_data.mappings[b]
        if len(ms) < 2:
            continue
        idx_clean = rng.randrange(len(ms))
        # pick a different idx from same fiber
        idx_swap = rng.choice([i for i in range(len(ms)) if i != idx_clean])

        z_clean, a_clean = ms[idx_clean]
        z_swap, a_swap = ms[idx_swap]
        cands = [a for _, a in ms]

        clean = score_candidate_sequences(
            model=model, tokenizer=tokenizer, base_string=b,
            z_string=z_clean, candidate_a_strings=cands,
            correct_index=idx_clean, task="bz_to_a", device=device,
        )
        swap = score_candidate_sequences(
            model=model, tokenizer=tokenizer, base_string=b,
            z_string=z_swap, candidate_a_strings=cands,
            correct_index=idx_swap, task="bz_to_a", device=device,
        )

        clean_pred_idx = int(torch.tensor(
            clean["normalized_log_probs"]).argmax().item())
        swap_pred_idx = int(torch.tensor(
            swap["normalized_log_probs"]).argmax().item())

        clean_correct += (clean_pred_idx == idx_clean)
        swap_correct += (swap_pred_idx == idx_swap)
        swap_predicts_swap_target += (swap_pred_idx == idx_swap)
        swap_predicts_clean_target += (swap_pred_idx == idx_clean)
        if swap_pred_idx not in (idx_swap, idx_clean):
            swap_predicts_other += 1
        if swap_pred_idx != clean_pred_idx:
            swap_predicts_differs_from_clean += 1

        # Distribution of where swap prediction lands
        if swap_pred_idx == idx_swap:
            tag = "swap_target"
        elif swap_pred_idx == idx_clean:
            tag = "clean_target"
        else:
            tag = "other_in_fiber"
        swap_pred_distribution[tag] = swap_pred_distribution.get(tag, 0) + 1

        cand_loss_clean_arr.append(clean["candidate_loss"])
        cand_loss_swap_arr.append(swap["candidate_loss"])
        n += 1

    if n == 0:
        return {"error": "no examples", "n": 0}

    return {
        "n": n,
        "clean_accuracy": clean_correct / n,
        "swap_accuracy": swap_correct / n,
        "swap_predicts_swap_target_rate": swap_predicts_swap_target / n,
        "swap_predicts_clean_target_rate": swap_predicts_clean_target / n,
        "swap_predicts_other_in_fiber_rate": swap_predicts_other / n,
        "swap_predicts_differs_from_clean_rate":
            swap_predicts_differs_from_clean / n,
        "candidate_loss_clean_mean": sum(cand_loss_clean_arr) / n,
        "candidate_loss_swap_mean": sum(cand_loss_swap_arr) / n,
        "delta_loss_within_fiber":
            (sum(cand_loss_swap_arr) - sum(cand_loss_clean_arr)) / n,
        "distribution": swap_pred_distribution,
    }


def main():
    device = _select_device()
    print(f"device: {device}")

    # 3-seed K=20 canonical (Cell A config)
    cells = [
        CellSpec("CellA_seed0", 0.001, 20, 0, "K=20 canonical seed 0"),
        CellSpec("CellA_seed1", 0.001, 20, 1, "K=20 canonical seed 1"),
        CellSpec("CellA_seed2", 0.001, 20, 2, "K=20 canonical seed 2"),
    ]
    # K=10 canonical multi-seed: directories may differ; check what's here
    # — if absent, skip.

    PHASES = {"plateau": 2000, "transition": 4000, "post": 7500}
    out = {"device": device, "cells": {}}
    t_start = time.time()

    for cs in cells:
        print(f"\n>>> {cs.label}")
        try:
            cc, tokenizer, mapping_data, train_ds, rows = load_cell(cs, device)
        except Exception as e:
            print(f"  load error: {e}")
            out["cells"][cs.label] = {"error": str(e)}
            continue
        cell_out = {"label": cs.label, "k": cs.k, "phases": {}}
        for phase_label, target_step in PHASES.items():
            from eta_sweep.analysis.mech_interp_mbc import list_ckpts
            avail = list_ckpts(cs)
            if not avail:
                continue
            step = min(avail, key=lambda s: abs(s - target_step))
            try:
                model = make_model(cc, tokenizer, step, device)
            except Exception as e:
                print(f"  phase={phase_label} step={step}: load error {e}")
                continue
            res = within_fiber_swap_eval(model, tokenizer, mapping_data,
                                          device, n_examples=128, seed=0)
            res["step"] = step
            res["target_phase"] = phase_label
            cell_out["phases"][phase_label] = res
            print(f"  phase={phase_label} step={step}: clean_acc={res['clean_accuracy']:.3f} "
                  f"swap_acc={res['swap_accuracy']:.3f} "
                  f"swap→clean={res['swap_predicts_clean_target_rate']:.3f} "
                  f"swap→other={res['swap_predicts_other_in_fiber_rate']:.3f} "
                  f"differs={res['swap_predicts_differs_from_clean_rate']:.3f}")
            del model
            if device == "mps":
                torch.mps.empty_cache()
        out["cells"][cs.label] = cell_out

    out["total_wall_clock_s"] = time.time() - t_start
    out_path = RESULTS_DIR / "within_fiber_swap.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
