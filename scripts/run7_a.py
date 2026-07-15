#!/usr/bin/env python
"""
Run 7 / Experiment A — escape-time vs init-scale eps.
One condition per invocation: --eps --seed [--optimizer sgd --lr 3e-2].
eps scales ONLY the branch-output parameters that survive pre-LN
renormalization (predictions_run7.json landmine 2): W_V,b_V,W_O,b_O,
W_in,b_in,W_out,b_out per block, plus unembed.W_U.
Escape = first eval step with first_target_loss < 1.7908 (precommitted).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import create_tokenizer_from_config
from src.model import create_model_from_config
from scripts.experiment_helpers import make_config, run_single_experiment

HALF_SHELF = 1.7908
OUT = Path("results/law")


def scale_init(model, eps):
    with torch.no_grad():
        for blk in model.blocks:
            for p in (blk.attn.W_V, blk.attn.b_V, blk.attn.W_O, blk.attn.b_O,
                      blk.mlp.W_in, blk.mlp.b_in, blk.mlp.W_out,
                      blk.mlp.b_out):
                p.mul_(eps)
        model.unembed.W_U.mul_(eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd"])
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--max-steps", type=int, default=40000)
    ap.add_argument("--early-stop-frac", type=float, default=None,
                    help="DO NOT USE for the sweep: candidate-loss early stop "
                         "fires BEFORE the ftl half-shelf crossing at small "
                         "eps (logit-scale confound, discovered eps=0.03125: "
                         "candidate converged ~11.6k with ftl still > half-"
                         "shelf). Sweep runs use full budget; escape "
                         "extraction handles the rest.")
    args = ap.parse_args()

    tag = f"law_eps{args.eps:g}_s{args.seed}" + \
          ("_sgd" if args.optimizer == "sgd" else "")
    lr = args.lr if args.lr is not None else \
        (3e-2 if args.optimizer == "sgd" else 1e-3)
    cfg = make_config(tag, k=10, seed=args.seed, max_steps=args.max_steps,
                      checkpoint_every=args.max_steps, eval_every=50,
                      lr=lr, optimizer_type=args.optimizer,
                      early_stop_frac=args.early_stop_frac)
    tok = create_tokenizer_from_config(cfg)
    torch.manual_seed(args.seed)
    model = create_model_from_config(cfg, tok)
    scale_init(model, args.eps)
    run_single_experiment(cfg, output_dir="outputs", model=model)

    h = json.load(open(Path("outputs") / tag / "training_history.json"))
    s = np.array(h["steps"])
    ftl = np.array(h["first_target_loss"])
    below = np.where(ftl < HALF_SHELF)[0]
    escape = int(s[below[0]]) if len(below) else None
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"a_{tag}.json", "w") as f:
        json.dump({"eps": args.eps, "seed": args.seed,
                   "optimizer": args.optimizer, "lr": lr,
                   "max_steps": args.max_steps, "escape": escape,
                   "censored": escape is None}, f)
    print(f"[{tag}] escape = {escape}")


if __name__ == "__main__":
    main()
