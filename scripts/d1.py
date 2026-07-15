#!/usr/bin/env python
"""
D1 — determinism program runner. One arm per invocation.

  --mode b0      : phantom_d-protocol resume (plateau 600, fresh optimizer),
                   eval_every=5; --order-seed required
  --mode b2      : from-scratch with 95% data subsample; --order-seed
  --mode scratch : from-scratch K=10; --init-seed / --order-seed decoupled
                   (re-seed global RNG between model creation and training)
  --mode resume  : continuous-optimizer resume from substrate (a) state 600;
                   --order-seed
  --mode kink    : from-scratch with branch-scaled init; --eps, both seeds
  --mode noise   : from-scratch + injected gradient noise; --noise-mult

Common: --bs, --budget, --ev (eval_every). Escape written to
results/determinism/<tag>.json with the pinned metric (ftl < 1.7908).
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
from src.data.dataset import (DisambiguationDataset, MappingData, collate_fn,
                              generate_mappings)
from src.model import create_model_from_config
from src.training.trainer import train, TrainingCallbacks
from scripts.experiment_helpers import make_config
from scripts.run7_a import scale_init
from scripts.run8_substrate import md42, STATES
from scripts.run8_lib import load_state, StateRestorer

OUT = Path("results/determinism")
HALF = 1.7908


def md_sub(base, frac, seed=3):
    import random
    rng = random.Random(seed)
    ex = [e for e in base.examples if rng.random() < frac]
    return MappingData(mappings=base.mappings, examples=ex,
                       n_unique_b=base.n_unique_b, n_unique_a=base.n_unique_a,
                       k=base.k, task="bz_to_a")


def md_exact(base, n, seed=3):
    import random
    rng = random.Random(seed)
    ex = rng.sample(base.examples, n)
    return MappingData(mappings=base.mappings, examples=ex,
                       n_unique_b=base.n_unique_b, n_unique_a=base.n_unique_a,
                       k=base.k, task="bz_to_a")


def sigma_nat(model, ds, device, n_batches=64, bs=128):
    g = torch.Generator().manual_seed(9)
    grads = []
    for _ in range(n_batches):
        idx = torch.randint(0, len(ds), (bs,), generator=g)
        ids = torch.stack([ds[i]["input_ids"] for i in idx]).to(device)
        model.zero_grad()
        logits = model(ids)
        ce = torch.nn.functional.cross_entropy(
            logits[:, 10:14].reshape(-1, logits.shape[-1]),
            ids[:, 11:15].reshape(-1))
        ce.backward()
        grads.append(torch.cat([p.grad.flatten() for p in
                                model.parameters()]).cpu())
    G = torch.stack(grads)
    return float(G.std(dim=0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["b0", "b2", "scratch", "resume", "kink",
                             "noise"])
    ap.add_argument("--init-seed", type=int, default=42)
    ap.add_argument("--order-seed", type=int, default=42)
    ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--budget", type=int, default=4000)
    ap.add_argument("--ev", type=int, default=5)
    ap.add_argument("--noise-mult", type=float, default=0.0)
    ap.add_argument("--drop-last", action="store_true")
    ap.add_argument("--n-examples", type=int, default=None)
    ap.add_argument("--replacement", action="store_true",
                    help="iid with-replacement sampling: no epochs exist")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    tag = a.tag or f"d1_{a.mode}_i{a.init_seed}_o{a.order_seed}" + \
        (f"_e{a.eps:g}" if a.eps != 1.0 else "") + \
        (f"_b{a.bs}" if a.bs != 128 else "") + \
        (f"_n{a.noise_mult:g}" if a.noise_mult else "") + \
        ("_dl" if a.drop_last else "") + ("_rep" if a.replacement else "")

    cfg = make_config(tag, k=10, seed=a.order_seed, max_steps=a.budget,
                      checkpoint_every=a.budget, eval_every=a.ev, bs=a.bs)
    tok = create_tokenizer_from_config(cfg)
    base = md42(cfg)
    md = base
    restorer = None
    device_model = None

    if a.mode in ("b0", "resume"):
        st = load_state(STATES / "a" / "state_000600.pt")
        model = create_model_from_config(cfg, tok)
        model.load_state_dict(st["model"])
        if a.mode == "resume":
            restorer = StateRestorer(st["opt"])
    else:
        torch.manual_seed(a.init_seed)
        model = create_model_from_config(cfg, tok)
        if a.eps != 1.0:
            scale_init(model, a.eps)
        if a.mode == "b2":
            md = md_sub(base, 0.95)
    if a.n_examples is not None:
        md = md_exact(base, a.n_examples)

    callbacks = TrainingCallbacks(on_after_backward=restorer)
    if a.mode == "noise" and a.noise_mult > 0:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        ds_est = DisambiguationDataset(mapping_data=base, tokenizer=tok,
                                       split="train", probe_fraction=0.0,
                                       seed=42, task="bz_to_a")
        mdev = model.to(device)
        sn = sigma_nat(mdev, ds_est, device)
        model = mdev.cpu()
        model.zero_grad()
        scale = a.noise_mult * sn
        gnoise = torch.Generator().manual_seed(a.order_seed + 777)
        def inject(model=None, optimizer=None, **kw):
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        eps_ = torch.randn(p.grad.shape, generator=gnoise)
                        p.grad.add_(eps_.to(p.grad.device) * scale)
        callbacks = TrainingCallbacks(on_after_backward=inject)
        print(f"[{tag}] sigma_nat={sn:.3e} scale={scale:.3e}", flush=True)

    # decouple: order stream seeded AFTER model creation; train() must not
    # re-seed (we call it directly)
    torch.manual_seed(a.order_seed)
    tr = DisambiguationDataset(mapping_data=md, tokenizer=tok, split="train",
                               probe_fraction=0.0, seed=42, task="bz_to_a")
    pr = DisambiguationDataset(mapping_data=md, tokenizer=tok, split="probe",
                               probe_fraction=0.0, seed=42, task="bz_to_a")
    (Path("outputs") / tag).mkdir(parents=True, exist_ok=True)
    if a.replacement:
        from torch.utils.data import RandomSampler
        sampler = RandomSampler(tr, replacement=True,
                                num_samples=a.bs * a.budget)
        tl = DataLoader(tr, batch_size=a.bs, sampler=sampler,
                        collate_fn=collate_fn)
    else:
        tl = DataLoader(tr, batch_size=a.bs, shuffle=True,
                        collate_fn=collate_fn, drop_last=a.drop_last)
    train(model=model,
          train_loader=tl,
          probe_loader=DataLoader(pr, batch_size=a.bs, shuffle=False,
                                  collate_fn=collate_fn),
          cfg=cfg, output_dir=Path("outputs"), grad_clip=1.0,
          optimizer_type="adamw", mapping_data=md, tokenizer=tok,
          callbacks=callbacks)

    h = json.load(open(f"outputs/{tag}/training_history.json"))
    s = np.array(h["steps"]); ftl = np.array(h["first_target_loss"])
    below = np.where(ftl < HALF)[0]
    esc = int(s[below[0]]) if len(below) else None
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{tag}.json", "w") as f:
        json.dump({"tag": tag, "mode": a.mode, "init_seed": a.init_seed,
                   "order_seed": a.order_seed, "eps": a.eps, "bs": a.bs,
                   "noise_mult": a.noise_mult, "escape": esc,
                   "escape_examples": esc * a.bs if esc else None,
                   "censored": esc is None}, f)
    print(f"[{tag}] escape={esc}")


if __name__ == "__main__":
    main()
