#!/usr/bin/env python
"""
E6 — The staircase: full-vocab CE (per A-position and pooled) across ALL
checkpoints, with every hypothesis shelf computed from the actual generated
data and drawn on the figure.

Shelves per position j:
  log|V|              untrained / uniform
  H(q*_j)             hypothesis C, order-0 constant (no prefix use)
  H(A_j | A_<j)       constant-in-(B,z) machine that reads the visible prefix
  H(A_j | B, A_<j)    hypothesis M, order-1 (log K at j=1; 0 at j>=2 here)
  0                   order-2 (full conditional)

Also: candidate-restricted CE at pos 1 (run-1 continuity), KL/JSD to q*,
JSD to P(A|B,prefix), accuracy. Plateau detection: sliding window of
--win checkpoints; plateau iff |mean slope| < --slope-thresh nats/step.

Usage: python scripts/relp2_e6.py --experiment landauer_dense_k10
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf
from src.data import create_tokenizer_from_config
from src.training.checkpoint import list_checkpoints
from scripts.relp_core import load_model, select_device
from scripts.relp2_core import (
    build_pair_batch, to_device, regime_metrics, position_distributions,
    jsd, build_pab, empirical_shelves, A_LEN,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="landauer_dense_k10")
    ap.add_argument("--n-b", type=int, default=128)
    ap.add_argument("--every", type=int, default=1, help="use every Nth checkpoint")
    ap.add_argument("--win", type=int, default=5)
    ap.add_argument("--slope-thresh", type=float, default=5e-5)
    ap.add_argument("--out", default="results/relp2")
    args = ap.parse_args()

    device = select_device()
    exp_dir = Path("outputs") / args.experiment
    cfg = OmegaConf.load(exp_dir / "config.yaml")
    tok = create_tokenizer_from_config(cfg)
    n_b = min(args.n_b, int(cfg.data.n_unique_b))
    pb = build_pair_batch(cfg, tok, n_b_eval=n_b, seed=1234)
    ids, pb = to_device(pb, device)
    V = tok.vocab_size

    shelves = empirical_shelves(cfg, tok)
    q_star = {j: torch.tensor(shelves["q_star"][j], device=device) for j in range(1, A_LEN + 1)}
    pab = build_pab(pb, V, device)

    ckpts = list_checkpoints(exp_dir / "checkpoints")[::args.every]
    print(f"{args.experiment}: {len(ckpts)} checkpoints, {ids.shape[0]} inputs", flush=True)

    rows = []
    for step in ckpts:
        model = load_model(cfg, tok, exp_dir / "checkpoints", step, device)
        with torch.no_grad():
            logits = model(ids)
            rm = regime_metrics(logits, pb)
            dists = position_distributions(logits)
        row = {"step": step,
               "cand_ce_pos1": float(rm["cand_ce_pos1"].mean()),
               "ce_pooled": float(rm["ce_pooled"].mean())}
        for j in range(1, A_LEN + 1):
            d = dists[:, j - 1, :]
            qs = q_star[j].unsqueeze(0).expand_as(d)
            row[f"ce_pos{j}"] = float(rm[f"ce_pos{j}"].mean())
            row[f"acc_pos{j}"] = float(rm[f"acc_pos{j}"].mean())
            row[f"jsd_qstar_pos{j}"] = float(jsd(d, qs).mean())
            row[f"kl_qstar_pos{j}"] = float(
                (d * (torch.log(d + 1e-12) - torch.log(qs + 1e-6))).sum(-1).mean())
            row[f"jsd_pab_pos{j}"] = float(jsd(d, pab[:, j - 1, :]).mean())
        rows.append(row)
        del model
        if step % 5000 == 0:
            print(f"  step {step}: ce_pos1={row['ce_pos1']:.3f} pooled={row['ce_pooled']:.3f}",
                  flush=True)

    # ---- plateau detection on per-position and pooled CE ----
    steps = np.array([r["step"] for r in rows])

    def plateaus(series):
        segs, i = [], 0
        while i + args.win <= len(steps):
            w = slice(i, i + args.win)
            slope = np.polyfit(steps[w], series[w], 1)[0]
            if abs(slope) < args.slope_thresh:
                j = i
                while j + args.win <= len(steps) and \
                        abs(np.polyfit(steps[j:j + args.win], series[j:j + args.win], 1)[0]) < args.slope_thresh:
                    j += 1
                lo, hi = i, min(j + args.win - 1, len(steps) - 1)
                segs.append({"start": int(steps[lo]), "end": int(steps[hi]),
                             "level": float(np.mean(series[lo:hi + 1]))})
                i = j + 1
            else:
                i += 1
        # merge segments at similar levels separated by < 2 windows
        merged = []
        for s in segs:
            if merged and abs(merged[-1]["level"] - s["level"]) < 0.05:
                merged[-1]["end"] = s["end"]
                merged[-1]["level"] = 0.5 * (merged[-1]["level"] + s["level"])
            else:
                merged.append(s)
        return merged

    plateau_report = {}
    for key in ["ce_pos1", "ce_pooled"] + [f"ce_pos{j}" for j in range(2, A_LEN + 1)]:
        plateau_report[key] = plateaus(np.array([r[key] for r in rows]))

    out_dir = Path(args.out) / args.experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"experiment": args.experiment, "n_inputs": int(ids.shape[0]),
              "win": args.win, "slope_thresh": args.slope_thresh,
              "shelves": {k: v for k, v in shelves.items() if k != "q_star"},
              "rows": rows, "plateaus": plateau_report}
    with open(out_dir / "e6_staircase.json", "w") as f:
        json.dump(result, f)

    # ---- staircase figure ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    shelf_style = [
        ("log_vocab", lambda j: shelves["log_vocab"], "0.6", "log |V|"),
        ("H_qstar", lambda j: shelves["H_qstar"][j], "tab:purple", "H(q*) — C shelf"),
        ("H_given_prefix", lambda j: shelves["H_given_prefix"][j], "tab:brown",
         "H(A_j|A_<j) — C+prefix shelf"),
        ("H_given_B", lambda j: shelves["H_given_B"][j], "tab:green",
         "H(A_j|B,A_<j) — M shelf"),
    ]
    for j in range(1, A_LEN + 1):
        ax = axes[(j - 1) // 2][(j - 1) % 2]
        ax.plot(steps, [r[f"ce_pos{j}"] for r in rows], color="tab:blue", lw=1.2,
                label="full-vocab CE")
        if j == 1:
            ax.plot(steps, [r["cand_ce_pos1"] for r in rows], color="tab:red", lw=0.9,
                    alpha=0.8, label="candidate-restricted CE (run-1 metric)")
        for _k, fn, col, lab in shelf_style:
            ax.axhline(fn(j), color=col, ls="--", lw=0.9,
                       label=lab if j == 1 else None)
        for seg in plateau_report[f"ce_pos{j}"]:
            ax.axvspan(seg["start"], seg["end"], color="tab:blue", alpha=0.08)
        ax.set_xscale("log")
        ax.set_title(f"A-position {j}")
        ax.set_ylabel("nats")
        if j == 1:
            ax.legend(fontsize=7, loc="lower left")
    for ax in axes[1]:
        ax.set_xlabel("step")
    fig.suptitle(f"{args.experiment}: staircase — full-vocab CE vs data-derived shelves")
    fig.tight_layout()
    fig.savefig(out_dir / "e6_staircase.png", dpi=150)
    plt.close(fig)

    print(json.dumps(plateau_report, indent=1))
    print(f"-> {out_dir}/e6_staircase.json + e6_staircase.png")


if __name__ == "__main__":
    main()
