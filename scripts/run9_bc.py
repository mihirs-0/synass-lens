#!/usr/bin/env python
"""
Run 9 / Experiments B (data diet) and C (carrier freeze).
All arms: from substrate-(a) state 700, continuous optimizer.
  B: 600 diet steps (modified data) -> restore clean -> escape shift
  C: freeze group X for steps 700-1400 -> unfreeze -> escape shift
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import create_tokenizer_from_config
from src.data.dataset import MappingData
from src.model import create_model_from_config
from scripts.experiment_helpers import make_config
from scripts.run8_substrate import md42, STATES, run_train
from scripts.run8_lib import load_state, StateRestorer, StateSaver
from scripts.run8_revert import GROUPS, in_group

OUT = Path("results/production")
HALF = 1.7908
BASELINE_ESCAPE = 2075


def poisoned_md(base, p, seed=7):
    rng = random.Random(seed)
    pool = [a for pairs in base.mappings.values() for (_z, a) in pairs]
    ex = [{"b": e["b"], "z": e["z"],
           "a": rng.choice(pool) if rng.random() < p else e["a"]}
          for e in base.examples]
    return MappingData(mappings=base.mappings, examples=ex,
                       n_unique_b=base.n_unique_b, n_unique_a=base.n_unique_a,
                       k=base.k, task="bz_to_a")


def subset_md(base, which):
    zs = [z for (z, _a) in base.mappings[sorted(base.mappings)[0]]]
    keep_z = set(zs[:5])
    keep_b = set(sorted(base.mappings)[:500])
    ex = [e for e in base.examples
          if (which == "z_half" and e["z"] in keep_z)
          or (which == "b_half" and e["b"] in keep_b)]
    return MappingData(mappings=base.mappings, examples=ex,
                       n_unique_b=base.n_unique_b, n_unique_a=base.n_unique_a,
                       k=base.k, task="bz_to_a")


def escape_of(name):
    h = json.load(open(f"outputs/{name}/training_history.json"))
    s = np.array(h["steps"]); ftl = np.array(h["first_target_loss"])
    below = np.where(ftl < HALF)[0]
    return int(s[below[0]]) if len(below) else None


def arm_b(diet):
    st = load_state(STATES / "a" / "state_000700.pt")
    base = None
    name = f"prod_b_{diet}"
    cfg0 = make_config(name + "_diet", k=10, seed=42, max_steps=600,
                       checkpoint_every=600, eval_every=50,
                       bs=64 if diet == "batch64" else 128)
    tok = create_tokenizer_from_config(cfg0)
    base = md42(cfg0)
    model = create_model_from_config(cfg0, tok)
    model.load_state_dict(st["model"])
    if diet.startswith("p"):
        md_diet = poisoned_md(base, float(diet[1:]))
    elif diet in ("z_half", "b_half"):
        md_diet = subset_md(base, diet)
    else:
        md_diet = base
    run_train(cfg0, model, md_diet,
              StateSaver(STATES / f"bdiet_{diet}", 600, also_at=(600,)),
              StateRestorer(st["opt"]))
    # z_half: per-z-subset sensitivity at restore point
    extra = {}
    if diet == "z_half":
        from scripts.relp2_core import build_pair_batch, to_device, \
            regime_metrics
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        pb = build_pair_batch(cfg0, tok, n_b_eval=64, seed=1234)
        ids, pb = to_device(pb, device)
        model = model.to(device).eval()
        with torch.no_grad():
            rm = regime_metrics(model(ids), pb)
        z_idx = pb.base.z_idx.cpu().numpy()
        mz = rm["m_z_pos1"].cpu().numpy()
        extra = {"m_z_present": float(mz[z_idx < 5].mean()),
                 "m_z_absent": float(mz[z_idx >= 5].mean())}
        model = model.cpu().train()
    last = load_state(STATES / f"bdiet_{diet}" / "state_000600.pt")
    cfg1 = make_config(name, k=10, seed=42, max_steps=4500,
                       checkpoint_every=4500, eval_every=25)
    run_train(cfg1, model, base, None, StateRestorer(last["opt"]))
    esc = escape_of(name)
    abs_esc = 700 + 600 + esc if esc else None
    res = {"diet": diet, "abs_escape": abs_esc,
           "shift": abs_esc - BASELINE_ESCAPE if abs_esc else None, **extra}
    with open(OUT / f"b_{diet}.json", "w") as f:
        json.dump(res, f)
    print(f"[B {diet}] abs_escape={abs_esc} shift={res['shift']} {extra}")


def arm_c(group):
    st = load_state(STATES / "a" / "state_000700.pt")
    name = f"prod_c_{group}"
    cfg0 = make_config(name + "_frz", k=10, seed=42, max_steps=700,
                       checkpoint_every=700, eval_every=50)
    tok = create_tokenizer_from_config(cfg0)
    base = md42(cfg0)
    model = create_model_from_config(cfg0, tok)
    model.load_state_dict(st["model"])
    def frozen(n):
        if group == "all":
            return True
        if group == "none":
            return False
        return in_group(n, group)
    if group == "all":
        # total freeze = nothing changes for 700 steps; equivalent by
        # construction to skipping the phase (backward() cannot run with
        # zero trainable params)
        last = st
    else:
        for n, p in model.named_parameters():
            p.requires_grad_(not frozen(n))
        run_train(cfg0, model, base,
                  StateSaver(STATES / f"cfrz_{group}", 700, also_at=(700,)),
                  StateRestorer(st["opt"]))
        for p in model.parameters():
            p.requires_grad_(True)
        last = load_state(STATES / f"cfrz_{group}" / "state_000700.pt")
    cfg1 = make_config(name, k=10, seed=42, max_steps=4000,
                       checkpoint_every=4000, eval_every=25)
    run_train(cfg1, model, base, None, StateRestorer(last["opt"]))
    esc = escape_of(name)
    abs_esc = 700 + 700 + esc if esc else None
    res = {"group": group, "abs_escape": abs_esc,
           "shift": abs_esc - BASELINE_ESCAPE if abs_esc else None}
    with open(OUT / f"c_{group}.json", "w") as f:
        json.dump(res, f)
    print(f"[C {group}] abs_escape={abs_esc} shift={res['shift']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["b", "c"])
    ap.add_argument("arm")
    a = ap.parse_args()
    if a.exp == "b":
        arm_b(a.arm)
    else:
        arm_c(a.arm)
