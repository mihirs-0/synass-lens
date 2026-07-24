#!/usr/bin/env python
"""DELTA-B diagnostic (re-eval, no retrain) with a positive control + leakage-free metric.

Question: is the TRAPPED inverse a B-CONDITIONAL marginal P(A|B) ("conditions on B, ignores z")
or a GLOBAL collapse (prediction independent of B too)?

Method: corrupt one input field across the batch, hold the target A fixed, measure the CE change.
    Δz : answerCE((B , z') -> A) - clean   # z swapped  -> expect ~0 (z ignored)
    ΔB : answerCE((B', z ) -> A) - clean   # B swapped  -> LARGE if B-conditional, ~0 if global collapse

Two safeguards, because ΔB is the one number that could change a claim:
  * FIRST-TOKEN CE (predicted from B,z with NO answer token in context) alongside full-region CE.
    Full-region CE is teacher-forced on A-tokens 2..L, which can mask ΔB; the first A-token cannot.
  * POSITIVE CONTROL: the CONVERGED checkpoint (eta=1e-3, solves the task via z). It MUST show large
    ΔB and large Δz. If it does, the swap machinery works and the trapped numbers are trustworthy.
    If the control is also ~0, the test is broken — not a finding.
"""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from eta_sweep.config import CellConfig, RESULTS_DIR
from eta_sweep.run_single import build_legacy_cfg, _select_device, _set_all_seeds
from src.data import collate_fn, create_datasets_from_config, create_tokenizer_from_config
from src.model import create_model_from_config
from src.training.trainer import compute_loss

CKPTS = [
    ("CONVERGED eta=1e-3", "gate_sweep",     "inverse_eta0.001_K10_nb1000_seed{s}"),  # positive control
    ("TRAPPED   eta=6e-3", "gate_candfloor", "inverse_eta0.006_K10_nb1000_seed{s}"),  # the question
]


def roll_span(ids, lo, hi, shift=1):
    out = ids.clone()
    out[:, lo:hi] = torch.roll(ids[:, lo:hi], shifts=shift, dims=0)
    return out


def metrics(model, batch, device, ts):
    b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    ids = b["input_ids"]
    with torch.no_grad():
        logits = model(ids)                                    # (N,T,V)
        full, acc, _ = compute_loss(model, b)                  # full answer-region CE (mean)
        ft = F.cross_entropy(logits[:, ts - 1, :], ids[:, ts]) # first A-token CE, predicted from (B,z)
    return float(full), float(ft)


def first_logp(model, batch, device, ts):
    b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    with torch.no_grad():
        return F.log_softmax(model(b["input_ids"])[:, ts - 1, :], dim=-1)   # (N,V) first-A-token log-dist


def mean_kl(logp, logq):              # mean_i KL( p_i || q_i ), target-free distribution shift
    return (logp.exp() * (logp - logq)).sum(-1).mean().item()


def main():
    N = 2000
    device = _select_device()
    print(f"device={device}  eval examples per run={N}")
    print("\nFULL = full answer-region CE   FT = first-A-token CE (leakage-free)\n")
    for seed in (0, 1):
        _set_all_seeds(seed)
        cc = CellConfig(eta=0.006, k=10, seed=seed, n_unique_b=1000, batch_size=128, weight_decay=0.01)
        cfg = build_legacy_cfg(cc)
        tok = create_tokenizer_from_config(cfg)
        train_ds, _, md = create_datasets_from_config(cfg, tok)     # mappings depend on seed only
        exs = [train_ds[i] for i in range(min(N, len(train_ds)))]
        batch = collate_fn(exs); ids = batch["input_ids"]
        ex0 = exs[0]; zp, ze = ex0["z_position"], ex0["z_end_position"]; ts = ex0["target_start_position"]
        b_lo, b_hi = 1, zp - 1; z_lo, z_hi = zp, ze + 1
        bz = dict(batch); bz["input_ids"] = roll_span(ids, z_lo, z_hi)
        bb = dict(batch); bb["input_ids"] = roll_span(ids, b_lo, b_hi)

        print(f"--- seed {seed}  (B=ids[{b_lo}:{b_hi}], z=ids[{z_lo}:{z_hi}], first-A=ids[{ts}]) ---")
        print(f"{'checkpoint':20} {'clean FULL':>10} {'clean FT':>9} | {'Δz FULL':>8} {'Δz FT':>7} | "
              f"{'ΔB FULL':>8} {'ΔB FT':>7} | {'KL_z':>6} {'KL_B':>6}")
        model = None
        for label, sub, patt in CKPTS:
            ck = RESULTS_DIR / sub / patt.format(s=seed) / "model_final.pt"
            if model is None:
                model = create_model_from_config(cfg, tok).to(device)
            model.load_state_dict(torch.load(ck, map_location=device)); model.eval()
            c_full, c_ft = metrics(model, batch, device, ts)
            z_full, z_ft = metrics(model, bz, device, ts)
            b_full, b_ft = metrics(model, bb, device, ts)
            lc = first_logp(model, batch, device, ts)
            kl_z = mean_kl(lc, first_logp(model, bz, device, ts))   # how much the dist moves when z swapped
            kl_b = mean_kl(lc, first_logp(model, bb, device, ts))   # ... when B swapped
            print(f"{label:20} {c_full:>10.3f} {c_ft:>9.3f} | {z_full-c_full:>+8.3f} {z_ft-c_ft:>+7.3f} | "
                  f"{b_full-c_full:>+8.3f} {b_ft-c_ft:>+7.3f} | {kl_z:>6.3f} {kl_b:>6.3f}")
        print()
    print("Reading: control (converged) MUST show large ΔB and Δz -> machinery works.")
    print("Then for the trapped row: ΔB(FT) large => B-CONDITIONAL ('conditions on B') is a measurement;")
    print("                          ΔB(FT) ~0    => GLOBAL collapse (the 'conditions on B' claim is wrong).")


if __name__ == "__main__":
    main()
