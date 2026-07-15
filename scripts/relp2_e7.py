#!/usr/bin/env python
"""
E7 — Constant-machine test.

At plateau checkpoints (+ step 4000 contrast), decide between:
  C: plateau model is the best input-independent distribution (per position,
     possibly prefix-conditioned) — sensitivity ~ 0, KL(model || q*) ~ 0.
  M: plateau model is P(A|B) — sensitivity across B >> 0, KL to P(A|B) low.

Measures per A-position (1 = headline):
  - input-sensitivity: mean pairwise JSD across inputs, decomposed
    across-B (fixed z) and across-z (fixed B)
  - KL(model || q*_j) with eps-smoothed q* (also JSD(model, q*))
  - KL(model || P(A_j | B, A_<j)) with eps smoothing (also JSD) — for j >= 2
    this reference is a delta (verified from data), so JSD is the robust one.

Usage: python scripts/relp2_e7.py --experiment landauer_dense_k10 \
           --steps 600 800 1000 1100 1200 1400 4000
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf
from src.data import create_tokenizer_from_config
from scripts.relp_core import load_model, select_device
from scripts.relp2_core import (
    build_pair_batch, to_device, position_distributions, jsd,
    mean_pairwise_jsd_grid, empirical_shelves, A_LEN,
)

EPS = 1e-6


def kl(p, q, eps=EPS):
    qs = q + eps
    qs = qs / qs.sum(-1, keepdim=True)
    return (p * (torch.log(p + 1e-12) - torch.log(qs))).sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="landauer_dense_k10")
    ap.add_argument("--steps", type=int, nargs="+",
                    default=[600, 800, 1000, 1100, 1200, 1400, 4000])
    ap.add_argument("--n-b", type=int, default=64)
    ap.add_argument("--out", default="results/relp2")
    args = ap.parse_args()

    device = select_device()
    exp_dir = Path("outputs") / args.experiment
    cfg = OmegaConf.load(exp_dir / "config.yaml")
    tok = create_tokenizer_from_config(cfg)
    pb = build_pair_batch(cfg, tok, n_b_eval=args.n_b, seed=1234)
    ids, pb = to_device(pb, device)
    n_b, K = pb.n_b, pb.k
    N = ids.shape[0]

    shelves = empirical_shelves(cfg, tok)
    V = tok.vocab_size

    # references
    q_star = {j: torch.tensor(shelves["q_star"][j], device=device) for j in range(1, A_LEN + 1)}
    # P(A_j | B, A_<j) per row: empirical over the K candidates of this row's B
    # with matching prefix. For this task: pos1 = uniform over K first chars;
    # pos>=2 = delta on the true token (unique prefix per B). Built generically:
    pab = torch.zeros(N, A_LEN, V, device=device)
    ans = pb.ans_tids  # [n_b, K, A_LEN]
    for i in range(N):
        b = int(pb.b_idx[i]); zi = int(pb.z_idx[i])
        my = ans[b, zi]
        for j in range(A_LEN):
            match = (ans[b, :, :j] == my[:j].unsqueeze(0)).all(dim=1) if j > 0 \
                else torch.ones(K, dtype=torch.bool, device=device)
            idx = ans[b, match, j]
            pab[i, j].scatter_add_(0, idx, torch.full((int(match.sum()),), 1.0, device=device))
        pab[i] = pab[i] / pab[i].sum(-1, keepdim=True)

    out = {"experiment": args.experiment, "n_inputs": N, "steps": args.steps,
           "shelves": {k: v for k, v in shelves.items() if k != "q_star"},
           "eps": EPS, "checkpoints": {}}

    for step in args.steps:
        model = load_model(cfg, tok, exp_dir / "checkpoints", step, device)
        with torch.no_grad():
            dists = position_distributions(model(ids))       # [N, 4, V]
        row = {}
        for j in range(1, A_LEN + 1):
            d = dists[:, j - 1, :]
            row[f"pos{j}"] = {
                "jsd_across_B_fixed_z": mean_pairwise_jsd_grid(d.cpu(), n_b, K, "B"),
                "jsd_across_z_fixed_B": mean_pairwise_jsd_grid(d.cpu(), n_b, K, "z"),
                "jsd_across_all": mean_pairwise_jsd_grid(d.cpu(), n_b, K, "all"),
                "kl_model_qstar": float(kl(d, q_star[j].unsqueeze(0).expand_as(d)).mean()),
                "jsd_model_qstar": float(jsd(d, q_star[j].unsqueeze(0).expand_as(d)).mean()),
                "kl_model_pab": float(kl(d, pab[:, j - 1, :]).mean()),
                "jsd_model_pab": float(jsd(d, pab[:, j - 1, :]).mean()),
                "entropy_model": float(-(d * torch.log(d + 1e-12)).sum(-1).mean()),
            }
        out["checkpoints"][step] = row
        p1 = row["pos1"]
        print(f"step {step:5d} pos1: JSD_B={p1['jsd_across_B_fixed_z']:.4f} "
              f"JSD_z={p1['jsd_across_z_fixed_B']:.4f} KL->q*={p1['kl_model_qstar']:.4f} "
              f"JSD->q*={p1['jsd_model_qstar']:.4f} JSD->P(A|B)={p1['jsd_model_pab']:.4f} "
              f"H(model)={p1['entropy_model']:.3f}", flush=True)
        del model

    out_dir = Path(args.out) / args.experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "e7_constant_machine.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"-> {out_dir}/e7_constant_machine.json")
    print("shelves (pos1): H(q*)=%.4f  H(A|B)=%.4f  log K=%.4f  log|V|=%.4f" % (
        shelves["H_qstar"][1], shelves["H_given_B"][1], shelves["log_K"], shelves["log_vocab"]))


if __name__ == "__main__":
    main()
