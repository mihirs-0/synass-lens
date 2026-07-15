#!/usr/bin/env python
"""
Run 9 / Experiment A support.
  new --cond {k20, eps05, sgd}  : dense-state reruns for the new conditions
  scal                          : self-coherence S(t)=cos(g_t, theta_t - theta_{t-500})
                                  calibration on existing dense conditions (for D)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import create_tokenizer_from_config
from src.data.dataset import DisambiguationDataset
from src.model import create_model_from_config
from scripts.experiment_helpers import make_config
from scripts.run7_a import scale_init
from scripts.run8_substrate import md42, STATES, run_train
from scripts.run8_lib import StateSaver, load_state, flat_from_state
from scripts.run8_candidates import grad_at, state_files

OUT = Path("results/production")
HALF = 1.7908


def cmd_new(cond):
    if cond == "k20":
        cfg = make_config("prod_k20", k=20, seed=42, max_steps=11500,
                          checkpoint_every=11500, eval_every=25)
        tok = create_tokenizer_from_config(cfg)
        torch.manual_seed(42)
        model = create_model_from_config(cfg, tok)
        from src.data.dataset import generate_mappings
        md = generate_mappings(
            n_unique_b=1000, k=20, b_length=6, a_length=4, z_length=2,
            vocab_chars=cfg.data.vocab_chars, seed=42, task="bz_to_a",
            enforce_unique_a_first_char_per_b=True,
            disambiguation_prefix_length=1)
        run_train(cfg, model, md, StateSaver(STATES / "prod_k20", 50))
    elif cond == "eps05":
        cfg = make_config("prod_eps05", k=10, seed=42, max_steps=4500,
                          checkpoint_every=4500, eval_every=25)
        tok = create_tokenizer_from_config(cfg)
        torch.manual_seed(42)
        model = create_model_from_config(cfg, tok)
        scale_init(model, 0.5)
        run_train(cfg, model, md42(cfg), StateSaver(STATES / "prod_eps05", 50))
    else:  # sgd
        cfg = make_config("prod_sgd", k=10, seed=42, max_steps=19500,
                          checkpoint_every=19500, eval_every=25, lr=3e-2,
                          optimizer_type="sgd")
        tok = create_tokenizer_from_config(cfg)
        torch.manual_seed(42)
        model = create_model_from_config(cfg, tok)
        from src.training.trainer import train, TrainingCallbacks
        from torch.utils.data import DataLoader
        from src.data.dataset import collate_fn
        md = md42(cfg)
        tr = DisambiguationDataset(mapping_data=md, tokenizer=tok,
                                   split="train", probe_fraction=0.0,
                                   seed=42, task="bz_to_a")
        pr = DisambiguationDataset(mapping_data=md, tokenizer=tok,
                                   split="probe", probe_fraction=0.0,
                                   seed=42, task="bz_to_a")
        (Path("outputs") / cfg.experiment.name).mkdir(parents=True,
                                                      exist_ok=True)
        train(model=model,
              train_loader=DataLoader(tr, batch_size=128, shuffle=True,
                                      collate_fn=collate_fn),
              probe_loader=DataLoader(pr, batch_size=128, shuffle=False,
                                      collate_fn=collate_fn),
              cfg=cfg, output_dir=Path("outputs"), grad_clip=1.0,
              optimizer_type="sgd", mapping_data=md, tokenizer=tok,
              callbacks=TrainingCallbacks(
                  on_after_step=StateSaver(STATES / "prod_sgd", 100)))
    h = json.load(open(f"outputs/prod_{cond}/training_history.json"))
    s = np.array(h["steps"]); ftl = np.array(h["first_target_loss"])
    below = np.where(ftl < HALF)[0]
    print(f"[{cond}] escape =", int(s[below[0]]) if len(below) else None)


def s_series(states_dir, stride, device, ids, ref, w=500):
    sf = state_files(Path(states_dir))
    grid = sorted(sf)
    rows = []
    for t in grid:
        if t % stride or (t - w) not in sf:
            continue
        th = flat_from_state(load_state(sf[t])["model"], ref)
        th0 = flat_from_state(load_state(sf[t - w])["model"], ref)
        d = th - th0
        d = d / (d.norm() + 1e-12)
        ref.load_state_dict(load_state(sf[t])["model"])
        g = grad_at(ref.to(device), ids, device)
        rows.append({"step": t,
                     "S": float(-(g @ d) / (g.norm() + 1e-12))})
        ref.cpu()
    return rows


def cmd_scal():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = make_config("scal", k=10, seed=42, max_steps=4000)
    tok = create_tokenizer_from_config(cfg)
    ref = create_model_from_config(cfg, tok)
    md = md42(cfg)
    ds = DisambiguationDataset(mapping_data=md, tokenizer=tok, split="train",
                               probe_fraction=0.0, seed=42, task="bz_to_a")
    rng = np.random.RandomState(5)
    ids = torch.stack([ds[i]["input_ids"] for i in
                       rng.choice(len(ds), 2048, False)]).to(device)
    out = {}
    for name, d, stride in (("a", STATES / "a", 100),
                            ("c_resume", STATES / "c_resume", 100),
                            ("kink_s42", STATES / "b_s42", 200),
                            ("kink_s43", STATES / "b_s43", 200)):
        out[name] = s_series(d, stride, device, ids, ref)
        print(f"[scal {name}] {len(out[name])} pts", flush=True)
    with open(OUT / "s_calibration.json", "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["new", "scal"])
    ap.add_argument("--cond", choices=["k20", "eps05", "sgd"])
    a = ap.parse_args()
    if a.mode == "new":
        cmd_new(a.cond)
    else:
        cmd_scal()
