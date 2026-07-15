#!/usr/bin/env python
"""Ziyin-style parameter-symmetry-breaking diagnostics (arXiv:2502.05300) for B,z->A.

Two measures:
  * delta_trans : translation symmetry in the z input direction. If the model does not read z,
                  attention is invariant to shifting the z-position residual along the z-direction
                  (W_K + lambda*z_dir symmetry). We measure the SENSITIVITY of attention scores to a
                  unit perturbation along the z-direction at the z positions, summed over layers:
                  delta_trans = sum_l mean_batch || d(attn_scores_l)/d(z_dir) ||^2 .
                  Small = symmetric (z unused);  large = symmetry broken (z read).
                  z_dir is IDENTIFIED from the token embeddings at the z positions (not hardcoded).
                  Control: same sensitivity along a RANDOM unit direction (should not separate trapped/converged).
  * perm        : permutation symmetry of MLP hidden neurons. P_G (full average) maps every neuron to the
                  mean neuron; the symmetric saddle has neurons collapsed. We report participation ratio PR
                  of the neuron-weight matrix (low=collapsed/symmetric, high=differentiated/broken) and
                  N_dosb = #neuron pairs with normalized ||v_i - v_j||^2 > threshold.

Tier-2 (this file's main): trapped (eta=6e-3) vs converged (eta=1e-3) checkpoints.
Prediction: converged delta_trans(z) >> trapped, and >> random-dir control (positive control validates the metric).
"""
import sys, math
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from eta_sweep.config import CellConfig, RESULTS_DIR
from eta_sweep.run_single import build_legacy_cfg, _select_device, _set_all_seeds
from src.data import collate_fn, create_datasets_from_config, create_tokenizer_from_config
from src.model import create_model_from_config


def z_directions(model, ids, zp, ze, r):
    ztoks = ids[:, zp:ze].reshape(-1).unique()
    E = model.W_E[ztoks].float()
    E = E - E.mean(0, keepdim=True)
    U, S, V = torch.linalg.svd(E, full_matrices=False)
    r = min(r, V.shape[0])
    dirs = V[:r]                                  # (r, d_model)
    return dirs / dirs.norm(dim=-1, keepdim=True), ztoks.numel()


def attn_sensitivity(model, ids, z_positions, dirs, eps=1.0):
    names = lambda n: n.endswith("hook_attn_scores")
    with torch.no_grad():
        _, c0 = model.run_with_cache(ids, names_filter=names)
        base = [c0[f"blocks.{l}.attn.hook_attn_scores"] for l in range(model.cfg.n_layers)]
        total = 0.0
        for d in dirs:
            def hook(resid, hook, d=d):
                resid[:, z_positions, :] = resid[:, z_positions, :] + eps * d.to(resid.dtype)
                return resid
            with model.hooks(fwd_hooks=[("blocks.0.hook_resid_pre", hook)]):
                _, c1 = model.run_with_cache(ids, names_filter=names)
            for l in range(model.cfg.n_layers):
                s0 = base[l]; s1 = c1[f"blocks.{l}.attn.hook_attn_scores"]
                valid = s0 > -1e4
                diff = (s1 - s0).masked_fill(~valid, 0.0)
                total += (diff ** 2).sum(dim=(-1, -2, -3)).mean().item() / eps ** 2
        return total


def perm_metrics(model, thresh=0.1):
    n_dosb = 0; prs = []; spreads = []
    for l in range(model.cfg.n_layers):
        Win = model.blocks[l].mlp.W_in.float()    # (d_model, d_mlp)
        Wout = model.blocks[l].mlp.W_out.float()  # (d_mlp, d_model)
        V = torch.cat([Win.T, Wout], dim=1)       # (d_mlp, 2 d_model)
        s = torch.linalg.svdvals(V)
        prs.append(float((s.sum() ** 2) / (s ** 2).sum()))           # participation ratio
        Vn = V / (V.norm(dim=1).mean() + 1e-8)
        d2 = torch.cdist(Vn, Vn) ** 2
        iu = torch.triu_indices(V.size(0), V.size(0), offset=1)
        pd = d2[iu[0], iu[1]]
        n_dosb += int((pd > thresh).sum())
        spreads.append(float(pd.mean()))
    return n_dosb, sum(prs) / len(prs), sum(spreads) / len(spreads)


def load(K, seed, sub, patt, device):
    _set_all_seeds(seed)
    cc = CellConfig(eta=0.006, k=K, seed=seed, n_unique_b=1000, batch_size=128, weight_decay=0.01)
    cfg = build_legacy_cfg(cc); tok = create_tokenizer_from_config(cfg)
    train_ds, _, md = create_datasets_from_config(cfg, tok)
    model = create_model_from_config(cfg, tok).to(device)
    model.load_state_dict(torch.load(RESULTS_DIR / sub / patt.format(s=seed) / "model_final.pt", map_location=device))
    model.eval()
    return model, train_ds


def main():
    device = _select_device()
    N, R = 256, 5
    rng = torch.Generator().manual_seed(0)
    GROUPS = [("CONVERGED 1e-3", "gate_sweep", "inverse_eta0.001_K10_nb1000_seed{s}"),
              ("TRAPPED   6e-3", "gate_candfloor", "inverse_eta0.006_K10_nb1000_seed{s}")]
    print(f"device={device}  N={N}  z-dirs={R}\n")
    print(f"{'checkpoint':18} {'Δtrans(z)':>10} {'Δtrans(rand)':>12} {'z/rand':>7} | {'PR(mlp)':>8} {'N_dosb':>8} {'spread':>7}")
    for label, sub, patt in GROUPS:
        for seed in (0, 1):
            model, ds = load(10, seed, sub, patt, device)
            exs = [ds[i] for i in range(N)]
            batch = collate_fn(exs); ids = batch["input_ids"].to(device)
            ex0 = exs[0]; zp, ze = ex0["z_position"], ex0["z_end_position"]
            zpos = list(range(zp, ze + 1))
            zdirs, nz = z_directions(model, ids, zp, ze + 1, R)
            rand = torch.randn(R, model.cfg.d_model, generator=rng).to(device)
            rand = rand / rand.norm(dim=-1, keepdim=True)
            dt_z = attn_sensitivity(model, ids, zpos, zdirs)
            dt_r = attn_sensitivity(model, ids, zpos, rand)
            nd, pr, sp = perm_metrics(model)
            print(f"{label} s{seed:<2} {dt_z:>10.3f} {dt_r:>12.3f} {dt_z/max(dt_r,1e-9):>7.2f} | {pr:>8.2f} {nd:>8} {sp:>7.3f}")
    print("\nPositive control: CONVERGED Δtrans(z) should be >> TRAPPED, and >> its own Δtrans(rand).")
    print("If so, the metric detects z-symmetry breaking; trapped staying low = symmetry never broke (saddle).")


if __name__ == "__main__":
    main()
