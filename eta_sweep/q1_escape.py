#!/usr/bin/env python
"""Q1: from the trapped checkpoint (eta=6e-3, step 25k), drop eta to 1e-3 and continue +20k steps.
Does it escape the marginal (candidate_loss leaves logK, Δz rises, KL-to-P(A) grows) or stay pinned?"""
import sys, json, math
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from eta_sweep.config import CellConfig, RESULTS_DIR
from eta_sweep.run_single import build_legacy_cfg, _select_device, _set_all_seeds, compute_candidate_loss_and_delta_z
from src.data import collate_fn, create_datasets_from_config, create_tokenizer_from_config
from src.model import create_model_from_config
from src.training.trainer import compute_loss

device = _select_device()
for seed in (0, 1):
    _set_all_seeds(seed)
    cc = CellConfig(eta=0.001, k=10, seed=seed, n_unique_b=1000, batch_size=128, weight_decay=0.01)
    cfg = build_legacy_cfg(cc); tok = create_tokenizer_from_config(cfg)
    ds, _, md = create_datasets_from_config(cfg, tok)
    loader = DataLoader(ds, batch_size=128, shuffle=True, collate_fn=collate_fn, num_workers=0)
    model = create_model_from_config(cfg, tok).to(device)
    ck = RESULTS_DIR / "gate_candfloor" / f"inverse_eta0.006_K10_nb1000_seed{seed}" / "model_final.pt"
    model.load_state_dict(torch.load(ck, map_location=device))
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)
    ts = ds[0]["target_start_position"]
    # global P(A0) marginal + a fixed eval batch
    exs = [ds[i] for i in range(2000)]; ids = collate_fn(exs)["input_ids"].to(device)
    A0 = ids[:, ts].cpu(); glob = torch.bincount(A0, minlength=int(model.cfg.d_vocab)).double(); glob = glob/glob.sum()
    def klpa():
        with torch.no_grad():
            out = F.softmax(model(ids)[:, ts-1], -1).mean(0).cpu().double()
        return float((out*(out.clamp_min(1e-12).log()-glob.clamp_min(1e-12).log())).sum())
    def diag():
        model.eval()
        c, _, dz = compute_candidate_loss_and_delta_z(model=model, tokenizer=tok, mapping_data=md,
                                                      n_examples=32, task="bz_to_a", device=device, seed=seed)
        kl = klpa(); model.train(); return c, dz, kl
    c, dz, kl = diag()
    print(f"\nseed{seed} START (trapped, before LR drop): candidate_loss={c:.3f} (logK={math.log(10):.3f}) Δz={dz:.2f} KL-to-P(A)={kl:.4f}")
    step = 0
    while step < 20000:
        for b in loader:
            if step >= 20000: break
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            model.train(); opt.zero_grad(set_to_none=True)
            l, _, _ = compute_loss(model, b); l.backward(); opt.step(); step += 1
            if step in (1000, 3000, 6000, 10000, 15000, 20000):
                c, dz, kl = diag()
                print(f"  seed{seed} +{step:>5}: candidate_loss={c:.3f} Δz={dz:.2f} KL-to-P(A)={kl:.4f} train_loss={float(l):.3f}")
