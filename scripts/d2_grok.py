#!/usr/bin/env python
"""
D2-revised / T3 — grokking transfer test.
Modular addition p=113, 1-layer HookedTransformer, AdamW wd=1.0, train
fraction 0.3, MINIBATCH bs=128 shuffle (epoch = 30 steps, ragged 119).
Transition = first eval step with val acc >= 0.90 (eval every 5).
  --seed S [--drop-last]
Writes results/determinism/grok_s<S>[_dl].json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformer_lens import HookedTransformer, HookedTransformerConfig

OUT = Path("results/determinism")
P = 113
FRAC = 0.3
BS = 128
BUDGET = 40000
EVAL_EVERY = 5


def build_data(seed):
    g = torch.Generator().manual_seed(seed)
    pairs = torch.cartesian_prod(torch.arange(P), torch.arange(P))
    perm = torch.randperm(len(pairs), generator=g)
    n_tr = int(FRAC * len(pairs))
    tr, va = pairs[perm[:n_tr]], pairs[perm[n_tr:]]
    def toks(ab):
        eq = torch.full((len(ab), 1), P)
        return torch.cat([ab, eq], dim=1), (ab[:, 0] + ab[:, 1]) % P
    return toks(tr), toks(va)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--drop-last", action="store_true")
    a = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tag = f"grok_s{a.seed}" + ("_dl" if a.drop_last else "")

    (tr_x, tr_y), (va_x, va_y) = build_data(a.seed)
    torch.manual_seed(a.seed)
    cfg = HookedTransformerConfig(
        n_layers=1, d_model=128, n_heads=4, d_head=32, d_mlp=512,
        act_fn="relu", d_vocab=P + 1, d_vocab_out=P, n_ctx=3,
        normalization_type=None, seed=a.seed)
    model = HookedTransformer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3,
                            betas=(0.9, 0.98), weight_decay=1.0)
    ds = TensorDataset(tr_x, tr_y)
    torch.manual_seed(a.seed + 10000)          # order stream
    dl = DataLoader(ds, batch_size=BS, shuffle=True, drop_last=a.drop_last)
    va_x, va_y = va_x.to(device), va_y.to(device)

    epoch_len = len(dl)
    step = 0
    transition = None
    log = []
    done = False
    while step < BUDGET and not done:
        for bx, by in dl:
            bx, by = bx.to(device), by.to(device)
            logits = model(bx)[:, -1, :]
            loss = torch.nn.functional.cross_entropy(logits, by)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % EVAL_EVERY == 0:
                with torch.no_grad():
                    va_pred = model(va_x)[:, -1, :].argmax(-1)
                    acc = float((va_pred == va_y).float().mean())
                if step % 500 == 0:
                    log.append({"step": step, "val_acc": acc})
                if acc >= 0.90 and transition is None:
                    transition = step
                    done = True
                    break
            if step >= BUDGET:
                break

    res = {"tag": tag, "seed": a.seed, "drop_last": a.drop_last,
           "epoch_len": epoch_len, "n_train": len(tr_x),
           "transition": transition,
           "phase": transition % epoch_len if transition else None,
           "censored": transition is None, "coarse_log": log}
    with open(OUT / f"{tag}.json", "w") as f:
        json.dump(res, f)
    print(f"[{tag}] epoch={epoch_len} transition={transition} "
          f"phase={res['phase']}")


if __name__ == "__main__":
    main()
