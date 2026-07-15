#!/usr/bin/env python
"""
Run 7 / Experiment B — attractor dose-response.
One arm per invocation: --dose --seed. Phantom_d protocol exactly:
plateau ckpt 600 -> dose steps of prior-only (shuffled-label seed 7)
training -> KL-to-prior depth measured -> normal resume (10k budget).
Dose 0 and 300 are REUSED from phantom_d (control / reverse), not rerun.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import create_tokenizer_from_config
from src.data.dataset import generate_mappings
from scripts.experiment_helpers import make_config, run_single_experiment
from scripts.phantom_d import (constant_diagnostics, prior_only_mapping)
from scripts.relp_core import load_model, select_device
from scripts.relp2_core import build_pair_batch, to_device

HALF_SHELF = 1.7908
OUT = Path("results/law")
PLATEAU = 600


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dose", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    device = select_device()

    cfg = make_config("b", k=10, max_steps=50000)
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

    name = f"law_dose{args.dose}_s{args.seed}"
    m = load_model(cfg, tok, Path("outputs") / "landauer_dense_k10" /
                   "checkpoints", PLATEAU, device)
    if args.dose > 0:
        dcfg = make_config(name + "_pre", k=10, seed=args.seed,
                           max_steps=args.dose, checkpoint_every=100000,
                           eval_every=100)
        m = run_single_experiment(
            dcfg, mapping_data=prior_only_mapping(md, seed=7),
            output_dir="outputs", model=m)[0]
        m.eval()
    depth = constant_diagnostics(m, ids, pb, q1, device)
    m.train()
    sym = bool(args.dose == 1200 and args.seed == 101)
    rcfg = make_config(name, k=10, seed=args.seed, max_steps=10000,
                       checkpoint_every=100 if sym else 5000, eval_every=50)
    run_single_experiment(rcfg, mapping_data=md, output_dir="outputs",
                          model=m)

    h = json.load(open(Path("outputs") / name / "training_history.json"))
    s = np.array(h["steps"])
    ftl = np.array(h["first_target_loss"])
    below = np.where(ftl < HALF_SHELF)[0]
    escape = int(s[below[0]]) if len(below) else None
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"b_dose{args.dose}_s{args.seed}.json", "w") as f:
        json.dump({"dose": args.dose, "seed": args.seed,
                   "depth_kl_qstar": depth["kl_qstar"],
                   "depth_full": depth, "escape": escape,
                   "censored": escape is None}, f)
    print(f"[{name}] depth KL={depth['kl_qstar']:.3f} escape={escape}")


if __name__ == "__main__":
    main()
