#!/usr/bin/env python
"""Phase 4: where does the hidden variable u_t live — weights or optimizer state?

From a mid-plateau checkpoint (saved with Adam moments), run two continuations with IDENTICAL weights
and IDENTICAL data order, differing only in the optimizer state:
  control : restore the saved Adam moments and keep training.
  reset   : zero the Adam moments (fresh optimizer), keep the same weights, keep training.
Measure steps-to-snap for each (loss EMA first crossing below SNAP).
  reset takes much longer  -> u_t lives in the optimizer state (second-moment / momentum buildup).
  reset snaps at ~same distance -> u_t lives in the weights.
"""
import sys, argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from eta_sweep.config import CellConfig, RESULTS_DIR
from eta_sweep.run_single import build_legacy_cfg, _select_device, _set_all_seeds
from src.data import collate_fn, create_datasets_from_config, create_tokenizer_from_config
from src.model import create_model_from_config
from src.training.trainer import compute_loss

SNAP = 1.0


def continuation(K, seed, mode, budget, device):
    ckdir = RESULTS_DIR / "probe_ckpts" / f"K{K}_s{seed}"
    mid = next(ckdir.glob("midplateau_step*.pt"))
    blob = torch.load(mid, map_location=device); mid_step = blob["step"]
    _set_all_seeds(seed)
    cc = CellConfig(eta=0.001, k=K, seed=seed, n_unique_b=1000, batch_size=128, weight_decay=0.01)
    cfg = build_legacy_cfg(cc); tok = create_tokenizer_from_config(cfg)
    ds, _, md = create_datasets_from_config(cfg, tok)
    model = create_model_from_config(cfg, tok).to(device)
    model.load_state_dict(blob["model"])
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)
    if mode == "control":
        opt.load_state_dict(blob["opt"])                  # restore Adam moments
    # reset: leave opt fresh (zero moments)
    g = torch.Generator().manual_seed(seed + 777)         # SAME data order for both modes
    loader = DataLoader(ds, batch_size=128, shuffle=True, collate_fn=collate_fn, num_workers=0, generator=g)
    ema = None; step = 0; snap_at = None
    while step < budget and snap_at is None:
        for b in loader:
            if step >= budget:
                break
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            model.train(); opt.zero_grad(set_to_none=True)
            l, _, _ = compute_loss(model, b); l.backward(); opt.step(); step += 1
            lv = float(l); ema = lv if ema is None else 0.97 * ema + 0.03 * lv
            if ema < SNAP and snap_at is None:
                snap_at = step
    return mid_step, snap_at, ema


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--K", type=int, default=10); ap.add_argument("--budget", type=int, default=8000)
    a = ap.parse_args(); device = _select_device()
    print(f"K={a.K}  device={device}  SNAP(loss EMA)<{SNAP}\n")
    for mode in ("control", "reset"):
        mid_step, snap_at, ema = continuation(a.K, 0, mode, a.budget, device)
        dist = (snap_at) if snap_at else None
        print(f"  {mode:8}: mid={mid_step}  snapped at +{dist if dist else '(no snap, ema=%.2f)'%ema} steps after the reset point")
    print("\nIf reset >> control -> u_t lives in the optimizer state.  If ~equal -> u_t lives in the weights.")


if __name__ == "__main__":
    main()
