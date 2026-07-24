#!/usr/bin/env python
"""
Run 6 / H2.4 — Causal stitching. Seed-A lower network + trained affine map +
seed-B upper network, everything frozen except the 128x128+bias stitch.
Null: same protocol into an UNTRAINED upper network (caps what the stitch
alone can do). Thresholds: predictions_run6.json (H2 stitching).
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import create_tokenizer_from_config
from src.data.dataset import DisambiguationDataset
from src.model import create_model_from_config
from scripts.experiment_helpers import make_config
from scripts.relp_core import load_model, select_device
from scripts.relp2_core import build_pair_batch, to_device, regime_metrics
from src.data.dataset import generate_mappings

OUT = Path("results/adversarial")
STEPS = 2000
BS = 256


def stitch_train(mA, mB, L, ids_train, device, steps=STEPS):
    W = nn.Linear(128, 128, bias=True).to(device)
    with torch.no_grad():
        W.weight.copy_(torch.eye(128))
        W.bias.zero_()
    for p in mA.parameters():
        p.requires_grad_(False)
    for p in mB.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(W.parameters(), lr=1e-3)
    n = ids_train.shape[0]
    g = torch.Generator().manual_seed(0)
    hookname = f"blocks.{L}.hook_resid_post"
    for step in range(steps):
        idx = torch.randint(0, n, (BS,), generator=g)
        batch = ids_train[idx].to(device)
        with torch.no_grad():
            _, cA = mA.run_with_cache(batch, return_type=None,
                                      names_filter=lambda x: x == hookname)
        xa = cA[hookname]
        def hook(resid, hook):
            return W(xa)
        logits = mB.run_with_hooks(batch, fwd_hooks=[(hookname, hook)])
        loss = nn.functional.cross_entropy(
            logits[:, 10:14].reshape(-1, logits.shape[-1]),
            batch[:, 11:15].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return W


def stitch_eval(mA, mB, W, L, ids_eval, pb, device):
    hookname = f"blocks.{L}.hook_resid_post"
    with torch.no_grad():
        _, cA = mA.run_with_cache(ids_eval, return_type=None,
                                  names_filter=lambda x: x == hookname)
        xa = cA[hookname]
        def hook(resid, hook):
            return W(xa)
        rm = regime_metrics(mB.run_with_hooks(
            ids_eval, fwd_hooks=[(hookname, hook)]), pb)
    return float(rm["acc_pos1"].mean())


def main():
    device = select_device()
    cfg = make_config("stitch", k=10, max_steps=6000)
    tok = create_tokenizer_from_config(cfg)
    md = generate_mappings(
        n_unique_b=1000, k=10, b_length=6, a_length=4, z_length=2,
        vocab_chars=cfg.data.vocab_chars, seed=42, task="bz_to_a",
        enforce_unique_a_first_char_per_b=True, disambiguation_prefix_length=1)
    ds = DisambiguationDataset(mapping_data=md, tokenizer=tok, split="train",
                               probe_fraction=0.0, seed=42, task="bz_to_a")
    ids_train = torch.stack([ds[i]["input_ids"] for i in range(len(ds))])
    pb = build_pair_batch(cfg, tok, n_b_eval=128, seed=1234)
    ids_eval, pb = to_device(pb, device)

    def load_seed(s):
        return load_model(cfg, tok, Path("outputs") /
                          f"phantom_d_control_s{s}" / "checkpoints", 6000,
                          device)

    res = {}
    for (a, b) in ((101, 202), (202, 303)):
        mA, mB = load_seed(a), load_seed(b)
        with torch.no_grad():
            native = float(regime_metrics(mB(ids_eval), pb)["acc_pos1"]
                           .mean())
        for L in (1, 2):
            W = stitch_train(mA, mB, L, ids_train, device)
            acc = stitch_eval(mA, mB, W, L, ids_eval, pb, device)
            res[f"s{a}->s{b}_L{L}"] = {"stitched_acc": acc,
                                       "native_B_acc": native,
                                       "recovery": acc / native}
            print(f"s{a}->s{b} L{L}: stitched={acc:.3f} native={native:.3f} "
                  f"recovery={acc/native:.3f}", flush=True)
        del mA, mB
        if device == "mps":
            torch.mps.empty_cache()

    # null: stitch s101 into an untrained upper network
    mA = load_seed(101)
    torch.manual_seed(7)
    mU = create_model_from_config(cfg, tok).to(device)
    mU.eval()
    for L in (1, 2):
        W = stitch_train(mA, mU, L, ids_train, device)
        acc = stitch_eval(mA, mU, W, L, ids_eval, pb, device)
        res[f"null_s101->untrained_L{L}"] = {"stitched_acc": acc}
        print(f"null L{L}: stitched={acc:.3f}", flush=True)

    best = max(res[k]["recovery"] for k in res if "recovery" in res[k])
    null_max = max(res[f"null_s101->untrained_L{L}"]["stitched_acc"]
                   for L in (1, 2))
    native = res["s101->s202_L1"]["native_B_acc"]
    res["verdict"] = {"best_recovery": best, "null_max_acc": null_max,
                      "pass": bool(best >= 0.90 and
                                   null_max <= 0.5 * native)}
    with open(OUT / "h2_stitching.json", "w") as f:
        json.dump(res, f, indent=1)
    print("stitch verdict:", json.dumps(res["verdict"]))


if __name__ == "__main__":
    main()
