#!/usr/bin/env python
"""
Run 8 / Experiment 2 — reservoir localization by partial revert.
At substrate-(a) step 1400: revert ONE group (weights + that group's Adam
slots) to its step-600 values, everything else kept at 1400 (optimizer
preserved via StateRestorer); resume; delay vs the unmodified-1400 control.
Groups per predictions_run8.json. One arm per invocation, or 'all'.
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
from scripts.experiment_helpers import make_config
from scripts.run8_substrate import md42, STATES, run_train
from scripts.run8_lib import load_state, StateRestorer, StateSaver

OUT = Path("results/clock")
HALF = 1.7908

GROUPS = {
    "embed": ["embed.W_E", "pos_embed.W_pos"],
    "attn_l0": ["blocks.0.ln1", "blocks.0.attn"],
    "attn_l1": ["blocks.1.ln1", "blocks.1.attn"],
    "attn_l2": ["blocks.2.ln1", "blocks.2.attn"],
    "attn_l3": ["blocks.3.ln1", "blocks.3.attn"],
    "mlp_l0": ["blocks.0.ln2", "blocks.0.mlp"],
    "mlp_l1": ["blocks.1.ln2", "blocks.1.mlp"],
    "mlp_l2": ["blocks.2.ln2", "blocks.2.mlp"],
    "mlp_l3": ["blocks.3.ln2", "blocks.3.mlp"],
    "unembed": ["ln_final", "unembed"],
}
ARMS = ["control"] + list(GROUPS) + ["moments_zero", "full_revert"]


def in_group(name, group):
    return any(name.startswith(p) for p in GROUPS[group])


def run_arm(arm):
    st1400 = load_state(STATES / "a" / "state_001400.pt")
    st600 = load_state(STATES / "a" / "state_000600.pt")
    cfg = make_config(f"clock_rev_{arm}", k=10, seed=42, max_steps=3500,
                      checkpoint_every=3500, eval_every=25)
    tok = create_tokenizer_from_config(cfg)
    model = create_model_from_config(cfg, tok)
    weights = dict(st1400["model"])
    opt = st1400["opt"]

    if arm in GROUPS:
        for k in weights:
            if in_group(k, arm):
                weights[k] = st600["model"][k]
        # revert this group's optimizer slots too (param order = named order)
        pnames = [n for n, _ in model.named_parameters()]
        for i, n in enumerate(pnames):
            if in_group(n, arm):
                key = i if i in opt["state"] else str(i)
                key6 = i if i in st600["opt"]["state"] else str(i)
                if key in opt["state"] and key6 in st600["opt"]["state"]:
                    for slot in ("exp_avg", "exp_avg_sq"):
                        opt["state"][key][slot] = \
                            st600["opt"]["state"][key6][slot]
    elif arm == "moments_zero":
        for k in opt["state"]:
            for slot in ("exp_avg", "exp_avg_sq"):
                opt["state"][k][slot] = torch.zeros_like(
                    opt["state"][k][slot])
    elif arm == "full_revert":
        weights = dict(st600["model"])
        opt = st600["opt"]

    model.load_state_dict(weights)
    run_train(cfg, model, md42(cfg), None, StateRestorer(opt))
    h = json.load(open(f"outputs/clock_rev_{arm}/training_history.json"))
    s = np.array(h["steps"]); ftl = np.array(h["first_target_loss"])
    below = np.where(ftl < HALF)[0]
    esc = int(s[below[0]]) if len(below) else None
    with open(OUT / f"rev_{arm}.json", "w") as f:
        json.dump({"arm": arm, "escape_resume_rel": esc,
                   "censored": esc is None}, f)
    print(f"[rev {arm}] escape(resume-rel) = {esc}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=ARMS + ["all", "batch1", "batch2"])
    a = ap.parse_args()
    if a.arm == "all":
        arms = ARMS
    elif a.arm == "batch1":
        arms = ["control", "embed", "attn_l0", "attn_l1", "attn_l2",
                "attn_l3", "moments_zero"]
    elif a.arm == "batch2":
        arms = ["mlp_l0", "mlp_l1", "mlp_l2", "mlp_l3", "unembed",
                "full_revert"]
    else:
        arms = [a.arm]
    for arm in arms:
        run_arm(arm)
