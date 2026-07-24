#!/usr/bin/env python
"""
Run 7 / B — depth recompute for arms whose inline depth was lost (s202 kills)
and for the reused dose-300 (phantom_d reverse _pre) checkpoints.
One forward each from the saved dosed checkpoints.
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import create_tokenizer_from_config
from src.data.dataset import generate_mappings
from scripts.experiment_helpers import make_config
from scripts.phantom_d import constant_diagnostics
from scripts.relp_core import load_model, select_device
from scripts.relp2_core import build_pair_batch, to_device

OUT = Path("results/law")


def main():
    device = select_device()
    cfg = make_config("depth", k=10, max_steps=50000)
    tok = create_tokenizer_from_config(cfg)
    md = generate_mappings(
        n_unique_b=1000, k=10, b_length=6, a_length=4, z_length=2,
        vocab_chars=cfg.data.vocab_chars, seed=42, task="bz_to_a",
        enforce_unique_a_first_char_per_b=True, disambiguation_prefix_length=1)
    pb = build_pair_batch(cfg, tok, n_b_eval=128, seed=1234)
    ids, pb = to_device(pb, device)
    q1 = torch.zeros(len(tok.token_to_id), device=device)
    for ex in md.examples:
        q1[tok.token_to_id[ex["a"][0]]] += 1
    q1 = q1 / q1.sum()

    jobs = [(f"law_dose{d}_s202_pre", d, 202, d) for d in (100, 600, 1200)]
    jobs += [(f"phantom_d_reverse_s{s}_pre", 300, s, 300)
             for s in (101, 202, 303)]
    out = {}
    for dirname, dose, seed, step in jobs:
        ck = Path("outputs") / dirname / "checkpoints"
        if not ck.exists() or not list(ck.iterdir()):
            print(f"[missing] {dirname}")
            continue
        m = load_model(cfg, tok, ck, step, device)
        d = constant_diagnostics(m, ids, pb, q1, device)
        out[f"dose{dose}_s{seed}"] = d
        print(f"dose{dose}_s{seed}: KL={d['kl_qstar']:.4f} "
              f"jsd_B={d['jsd_B']:.4f}", flush=True)
        del m
        if device == "mps":
            torch.mps.empty_cache()
    with open(OUT / "b_depths_recomputed.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
