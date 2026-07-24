#!/usr/bin/env python
"""
Run 8 / Step 0 — dense-checkpoint substrate.
  a         : K=10 standard init seed 42, to 4000, state every 25
  b --seed S: eps=0.125 branch-scaled init, to 12000, state every 50
  c         : from substrate (a) state at 900: dose 600 prior-only with
              CONTINUOUS optimizer, then real data to 4500, state every 25
              (note: run-7 dose protocol reset moments at phase boundaries;
              (c) deliberately does not — measures the dose's effect on the
              state, not the reset's; both stated in RESULTS8)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import create_tokenizer_from_config
from src.data.dataset import (DisambiguationDataset, generate_mappings,
                              collate_fn)
from src.model import create_model_from_config
from src.training.trainer import train
from scripts.experiment_helpers import make_config
from scripts.run7_a import scale_init
from scripts.phantom_d import prior_only_mapping
from scripts.run8_lib import (StateSaver, StateRestorer, make_callbacks,
                              load_state)

STATES = Path("results/clock/states")


def md42(cfg):
    return generate_mappings(
        n_unique_b=1000, k=10, b_length=6, a_length=4, z_length=2,
        vocab_chars=cfg.data.vocab_chars, seed=42, task="bz_to_a",
        enforce_unique_a_first_char_per_b=True, disambiguation_prefix_length=1)


def loaders(cfg, tok, md):
    tr = DisambiguationDataset(mapping_data=md, tokenizer=tok, split="train",
                               probe_fraction=0.0, seed=42, task="bz_to_a")
    pr = DisambiguationDataset(mapping_data=md, tokenizer=tok, split="probe",
                               probe_fraction=0.0, seed=42, task="bz_to_a")
    return (DataLoader(tr, batch_size=cfg.training.batch_size, shuffle=True,
                       collate_fn=collate_fn),
            DataLoader(pr, batch_size=cfg.training.batch_size, shuffle=False,
                       collate_fn=collate_fn))


def run_train(cfg, model, md, saver, restorer=None):
    tok = create_tokenizer_from_config(cfg)
    tl, pl = loaders(cfg, tok, md)
    out = Path("outputs")
    (out / cfg.experiment.name).mkdir(parents=True, exist_ok=True)
    train(model=model, train_loader=tl, probe_loader=pl, cfg=cfg,
          output_dir=out, grad_clip=1.0, optimizer_type="adamw",
          mapping_data=md, tokenizer=tok,
          callbacks=make_callbacks(saver, restorer))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["a", "b", "c"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.mode == "a":
        cfg = make_config("clock_a", k=10, seed=42, max_steps=4000,
                          checkpoint_every=4000, eval_every=25)
        tok = create_tokenizer_from_config(cfg)
        torch.manual_seed(42)
        model = create_model_from_config(cfg, tok)
        run_train(cfg, model, md42(cfg),
                  StateSaver(STATES / "a", 25))
    elif args.mode == "b":
        name = f"clock_b_s{args.seed}"
        cfg = make_config(name, k=10, seed=args.seed, max_steps=12000,
                          checkpoint_every=12000, eval_every=25)
        tok = create_tokenizer_from_config(cfg)
        torch.manual_seed(args.seed)
        model = create_model_from_config(cfg, tok)
        scale_init(model, 0.125)
        run_train(cfg, model, md42(cfg),
                  StateSaver(STATES / f"b_s{args.seed}", 50))
    else:  # c
        st = load_state(STATES / "a" / "state_000900.pt")
        cfg0 = make_config("clock_c_dose", k=10, seed=42, max_steps=600,
                           checkpoint_every=600, eval_every=25)
        tok = create_tokenizer_from_config(cfg0)
        model = create_model_from_config(cfg0, tok)
        model.load_state_dict(st["model"])
        base = md42(cfg0)
        run_train(cfg0, model, prior_only_mapping(base, seed=7),
                  StateSaver(STATES / "c_dose", 25, also_at=(600,)),
                  StateRestorer(st["opt"]))
        # continuous optimizer into the resume phase
        last = load_state(STATES / "c_dose" / "state_000600.pt")
        cfg1 = make_config("clock_c_resume", k=10, seed=42, max_steps=4500,
                           checkpoint_every=4500, eval_every=25)
        run_train(cfg1, model, base,
                  StateSaver(STATES / "c_resume", 25),
                  StateRestorer(last["opt"]))
        h = json.load(open("outputs/clock_c_resume/training_history.json"))
        s = np.array(h["steps"]); ftl = np.array(h["first_target_loss"])
        below = np.where(ftl < 1.7908)[0]
        print("[c] resume escape =", int(s[below[0]]) if len(below) else None)
    print("substrate", args.mode, "done")


if __name__ == "__main__":
    main()
