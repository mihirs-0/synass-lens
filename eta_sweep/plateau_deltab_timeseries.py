#!/usr/bin/env python
"""Does a CONVERGING run condition on B DURING the plateau, or only at the snap?

Train inverse eta=1e-3 (K=10), and at steps through the plateau and the transition measure
B- and z-conditioning with the validated span-swap + KL machinery (first-token, leakage-free):

    KL_B rises BEFORE KL_z, during the plateau  -> learns P(A|B) first, then adds z.
                                                   "marginals before conditionals" HOLDS.
    KL_B and KL_z both ~0 through the plateau,    -> the plateau is GLOBAL P(A); B and z
        then rise TOGETHER at the snap               conditioning appear simultaneously. REFRAME.
"""
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from eta_sweep.config import CellConfig
from eta_sweep.run_single import build_legacy_cfg, _select_device, _set_all_seeds
from src.data import collate_fn, create_datasets_from_config, create_tokenizer_from_config
from src.model import create_model_from_config
from src.training.trainer import compute_loss
from eta_sweep.delta_b_diagnostic import roll_span, first_logp, mean_kl, metrics

N = 2000


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    SEED, K = a.seed, a.k
    # higher K transitions later -> stretch the schedule
    CKPTS = ([150, 400, 700, 1000, 1400, 1800, 2200, 2800, 3500, 5000] if K <= 10
             else [200, 500, 900, 1400, 2000, 2800, 3800, 5000, 6500, 8000])
    device = _select_device()
    _set_all_seeds(SEED)
    cc = CellConfig(eta=0.001, k=K, seed=SEED, n_unique_b=1000, batch_size=128, weight_decay=0.01)
    cfg = build_legacy_cfg(cc); tok = create_tokenizer_from_config(cfg)
    train_ds, _, md = create_datasets_from_config(cfg, tok)
    loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate_fn, num_workers=0)
    model = create_model_from_config(cfg, tok).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)

    exs = [train_ds[i] for i in range(N)]
    eb = collate_fn(exs); ids = eb["input_ids"]
    ex0 = exs[0]; zp, ze = ex0["z_position"], ex0["z_end_position"]; ts = ex0["target_start_position"]
    bz = dict(eb); bz["input_ids"] = roll_span(ids, zp, ze + 1)
    bb = dict(eb); bb["input_ids"] = roll_span(ids, 1, zp - 1)

    def report(step):
        model.eval()
        cF, cf = metrics(model, eb, device, ts)
        _, zf = metrics(model, bz, device, ts)
        _, bf = metrics(model, bb, device, ts)
        lc = first_logp(model, eb, device, ts)
        klz = mean_kl(lc, first_logp(model, bz, device, ts))
        klb = mean_kl(lc, first_logp(model, bb, device, ts))
        model.train()
        rd = ("B-cond " if klb > 0.2 else "") + ("z-cond" if klz > 0.2 else "")
        print(f"{step:>5} {cF:>8.3f} {cf:>6.3f} | {zf-cf:>+7.3f} {bf-cf:>+7.3f} | {klz:>6.3f} {klb:>6.3f}  {rd or 'GLOBAL'}")

    import math
    print(f"converging inverse eta=1e-3 K={K} seed{SEED}  device={device}  log K={math.log(K):.3f}  "
          f"(B=ids[1:{zp-1}] z=ids[{zp}:{ze+1}])")
    print(f"{'step':>5} {'trainCE':>8} {'ftCE':>6} | {'Δz_ft':>7} {'ΔB_ft':>7} | {'KL_z':>6} {'KL_B':>6}  conditioning")
    report(0)
    step = 0; targets = set(CKPTS); mx = max(CKPTS); done = False
    while step < mx and not done:
        for batch in loader:
            if step >= mx:
                done = True; break
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            model.train(); opt.zero_grad(set_to_none=True)
            loss, acc, _ = compute_loss(model, batch); loss.backward(); opt.step(); step += 1
            if step in targets:
                report(step)
    print("\nKL>0.2 nats = that variable is being conditioned on. Watch whether KL_B crosses BEFORE,")
    print("WITH, or AFTER KL_z. Before -> P(A|B) first (thesis holds). Together -> plateau is global (reframe).")


if __name__ == "__main__":
    main()
