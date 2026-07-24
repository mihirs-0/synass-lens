#!/usr/bin/env python
"""
D2 / P1+P4 — per-step logging runs (the metronome photograph).
  --run 1 : from substrate (a) state 600, 320 steps (4 epochs), default loader
  --run 2 : same but drop_last=True (rival-metronome discriminator)
  --run 3 : from state 1900 through escape (~2075), 320 steps (P4 window)
Logs per step: grad norm (post-backward), update norm (post-step), batch
size, per-batch loss. Continuous optimizer throughout.
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
from src.data.dataset import DisambiguationDataset, collate_fn
from src.model import create_model_from_config
from src.training.trainer import train, TrainingCallbacks
from scripts.experiment_helpers import make_config
from scripts.run8_substrate import md42, STATES
from scripts.run8_lib import load_state, StateRestorer

OUT = Path("results/determinism")


class StepLogger:
    def __init__(self, model):
        self.rows = []
        self.prev = torch.cat([p.detach().flatten().cpu()
                               for p in model.parameters()])
        self.pending = {}

    def after_backward(self, model=None, batch=None, optimizer=None,
                       step=None, **kw):
        g = torch.cat([p.grad.flatten() for p in model.parameters()
                       if p.grad is not None])
        ids = batch["input_ids"]
        with torch.no_grad():
            logits = model(ids)
            loss = torch.nn.functional.cross_entropy(
                logits[:, 10:14].reshape(-1, logits.shape[-1]),
                ids[:, 11:15].reshape(-1)).item()
        self.pending = {"step": step, "grad_norm": float(g.norm()),
                        "batch_size": int(ids.shape[0]), "loss": loss}

    def after_step(self, model=None, **kw):
        cur = torch.cat([p.detach().flatten().cpu()
                         for p in model.parameters()])
        self.pending["update_norm"] = float((cur - self.prev).norm())
        self.prev = cur
        self.rows.append(dict(self.pending))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=int, required=True, choices=[1, 2, 3])
    a = ap.parse_args()
    start = 600 if a.run in (1, 2) else 1900
    tag = f"d2_p1_run{a.run}"
    cfg = make_config(tag, k=10, seed=42, max_steps=320,
                      checkpoint_every=100000, eval_every=1000)
    tok = create_tokenizer_from_config(cfg)
    md = md42(cfg)
    st = load_state(STATES / "a" / f"state_{start:06d}.pt")
    model = create_model_from_config(cfg, tok)
    model.load_state_dict(st["model"])
    logger = StepLogger(model)
    restorer = StateRestorer(st["opt"])
    def after_backward(**kw):
        restorer(**kw)
        logger.after_backward(**kw)
    cb = TrainingCallbacks(on_after_backward=after_backward,
                           on_after_step=logger.after_step)
    tr = DisambiguationDataset(mapping_data=md, tokenizer=tok, split="train",
                               probe_fraction=0.0, seed=42, task="bz_to_a")
    pr = DisambiguationDataset(mapping_data=md, tokenizer=tok, split="probe",
                               probe_fraction=0.0, seed=42, task="bz_to_a")
    torch.manual_seed(42 + a.run)
    (Path("outputs") / tag).mkdir(parents=True, exist_ok=True)
    train(model=model,
          train_loader=DataLoader(tr, batch_size=128, shuffle=True,
                                  collate_fn=collate_fn,
                                  drop_last=(a.run == 2)),
          probe_loader=DataLoader(pr, batch_size=128, shuffle=False,
                                  collate_fn=collate_fn),
          cfg=cfg, output_dir=Path("outputs"), grad_clip=1.0,
          optimizer_type="adamw", mapping_data=md, tokenizer=tok,
          callbacks=cb)
    with open(OUT / f"{tag}.json", "w") as f:
        json.dump({"run": a.run, "start": start, "rows": logger.rows}, f)
    sizes = [r["batch_size"] for r in logger.rows]
    print(f"[{tag}] {len(logger.rows)} steps logged; "
          f"ragged batches at: {[r['step'] for r in logger.rows if r['batch_size'] < 128][:6]}")


if __name__ == "__main__":
    main()
