#!/usr/bin/env python
"""Q3a: z-stripped forced marginal. Randomize z every batch (uninformative => binding impossible).
Does the model land at P(A|B) (B-conditional, the real K-to-1 floor) or collapse to P(A) global (ignores B too)?
Run at low eta (control: should reach P(A|B)) and high eta."""
import sys, math
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from eta_sweep.config import CellConfig
from eta_sweep.run_single import build_legacy_cfg, _select_device, _set_all_seeds
from src.data import collate_fn, create_datasets_from_config, create_tokenizer_from_config
from src.model import create_model_from_config
from src.training.trainer import compute_loss

device = _select_device()
def run(eta, steps=10000, seed=0):
    _set_all_seeds(seed)
    cc = CellConfig(eta=eta, k=10, seed=seed, n_unique_b=1000, batch_size=128, weight_decay=0.01)
    cfg = build_legacy_cfg(cc); tok = create_tokenizer_from_config(cfg)
    ds, _, md = create_datasets_from_config(cfg, tok)
    loader = DataLoader(ds, batch_size=128, shuffle=True, collate_fn=collate_fn, num_workers=0)
    model = create_model_from_config(cfg, tok).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=eta, betas=(0.9, 0.999), weight_decay=0.01)
    zp = ds[0]["z_position"]; ts = ds[0]["target_start_position"]; V = int(model.cfg.d_vocab)
    step = 0
    while step < steps:
        for b in loader:
            if step >= steps: break
            ii = b["input_ids"].to(device); lb = b["labels"].to(device)
            ii[:, zp:zp+2] = ii[0:1, zp:zp+2]              # CONSTANT z -> fully uninformative, zero leak
            model.train(); opt.zero_grad(set_to_none=True)
            l, _, _ = compute_loss(model, {"input_ids": ii, "labels": lb}); l.backward(); opt.step(); step += 1
    # ---- evaluate what it learned ----
    exs = [ds[i] for i in range(2000)]; ids = collate_fn(exs)["input_ids"].to(device)
    A0 = ids[:, ts].cpu(); glob = torch.bincount(A0, minlength=V).double(); glob = glob/glob.sum()
    # P(A0|B) per example
    PB = {}
    for B, pairs in md.mappings.items():
        c = torch.zeros(V).double()
        for z, a in pairs: c[tok.encode_sequence(B, z, a)["input_ids"][ts].item()] += 1
        PB[tuple(tok.encode_sequence(B, pairs[0][0], pairs[0][1])["input_ids"][1:zp-1].tolist())] = c/c.sum()
    with torch.no_grad():
        out = F.softmax(model(ids)[:, ts-1], -1).cpu().double()                # (N,V) per-example output
        sw = ids.clone(); sw[:, 1:zp-1] = torch.roll(ids[:, 1:zp-1], 1, 0)      # B-swap
        p = F.log_softmax(model(ids)[:, ts-1], -1); q = F.log_softmax(model(sw)[:, ts-1], -1)
        dB = float((p.exp()*(p-q)).sum(-1).mean())
        loss = float(compute_loss(model, {"input_ids": ids, "labels": collate_fn(exs)["labels"].to(device)})[0])
    klg, klb = [], []
    for i in range(2000):
        o = out[i].clamp_min(1e-12)
        klg.append(float((o*(o.log()-glob.clamp_min(1e-12).log())).sum()))
        key = tuple(ids[i, 1:zp-1].tolist()); pb = PB.get(key)
        if pb is not None: klb.append(float((o*(o.log()-pb.clamp_min(1e-12).log())).sum()))
    print(f"eta={eta}: loss={loss:.3f}  ΔB={dB:.2f}  KL-to-P(A)global={sum(klg)/len(klg):.3f}  "
          f"KL-to-P(A|B)={sum(klb)/len(klb):.3f}  -> {'P(A|B) B-conditional' if sum(klb)/len(klb) < sum(klg)/len(klg) else 'P(A) GLOBAL collapse'}")

print("Q3a — z-stripped forced marginal (z randomized every batch):")
run(0.001); run(0.006)
