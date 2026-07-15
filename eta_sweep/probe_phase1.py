#!/usr/bin/env python
"""Phase 1: train the converging inverse run (B,z->A, eta=1e-3) and save dense checkpoints
spanning init -> plateau -> tau -> solved, plus a mid-plateau checkpoint WITH optimizer state
(for the optimizer-reset experiment). No analysis here; just weights on disk."""
import sys, json
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

PLAN = {
    10: dict(maxstep=5000, mid=800,
             ckpts=[0, 100, 200, 300, 500, 700, 900, 1100, 1300, 1500, 1700, 1900, 2100, 2400, 2800, 3500, 5000]),
    20: dict(maxstep=12000, mid=2000,
             ckpts=[0, 200, 500, 900, 1400, 2000, 2800, 3500, 4000, 5000, 6000, 8000, 10000, 12000]),
}


def run(K, seed, device):
    p = PLAN[K]
    _set_all_seeds(seed)
    cc = CellConfig(eta=0.001, k=K, seed=seed, n_unique_b=1000, batch_size=128, weight_decay=0.01)
    cfg = build_legacy_cfg(cc); tok = create_tokenizer_from_config(cfg)
    ds, _, md = create_datasets_from_config(cfg, tok)
    loader = DataLoader(ds, batch_size=128, shuffle=True, collate_fn=collate_fn, num_workers=0)
    model = create_model_from_config(cfg, tok).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)
    out = RESULTS_DIR / "probe_ckpts" / f"K{K}_s{seed}"; out.mkdir(parents=True, exist_ok=True)
    ck = set(p["ckpts"]); loss_log = []
    if 0 in ck:
        torch.save(model.state_dict(), out / "step0.pt")
    step = 0
    while step < p["maxstep"]:
        for b in loader:
            if step >= p["maxstep"]:
                break
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            model.train(); opt.zero_grad(set_to_none=True)
            l, _, _ = compute_loss(model, b); l.backward(); opt.step(); step += 1
            loss_log.append([step, float(l)])
            if step in ck:
                torch.save(model.state_dict(), out / f"step{step}.pt")
            if step == p["mid"]:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step},
                           out / f"midplateau_step{step}.pt")
    json.dump({"loss": loss_log, "ckpts": sorted(ck), "mid": p["mid"]}, open(out / "meta.json", "w"))
    print(f"K={K} s{seed}: saved {len(ck)} checkpoints + mid-plateau opt state @ step {p['mid']}  (final loss {loss_log[-1][1]:.3f})")


def main():
    device = _select_device()
    print(f"device={device}")
    for K in (10, 20):
        run(K, 0, device)
    print("phase 1 done")


if __name__ == "__main__":
    main()
