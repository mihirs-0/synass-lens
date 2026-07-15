#!/usr/bin/env python
"""
Phase B — is the leap generalization onset?

Two runs, one analyzer:

  phantom_heldout_k10 (already trained): standard K=10 task, 20% of (B,z)
    example cells held out. DESIGNED-NEGATIVE control: each pairing's answer
    string appears only in that cell, so held-out answers are unseen and
    unrecoverable in principle. Prediction: held-out never learns; train-side
    phenomenology (C shelf -> snap) intact.

  e12_8e_xor_ho (trained here): the E12 XOR substrate (slot = (i1+i2) mod 10,
    10 z-combos per slot) with 20% of (B, z-combo) cells held out. Every
    answer stays visible through same-slot siblings, so a held-out cell is
    solvable ONLY by computing the selector rule. This is the informative
    generalization test: does held-out CE drop at the same step as the train
    leap (one event) or later (two events)?

Subcommands:
  train            train e12_8e_xor_ho (writes probe-cell set to results)
  analyze <exp>    per-checkpoint staircase over the full (B,z) grid, rows
                   split into train/held-out cells -> b_<exp>.json + figure
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import create_tokenizer_from_config
from src.data.dataset import MappingData, DisambiguationDataset
from src.training.checkpoint import list_checkpoints
from scripts.experiment_helpers import make_config, run_single_experiment
from scripts.relp_core import load_model, select_device
from scripts.relp2_core import build_pair_batch, to_device, regime_metrics, A_LEN
from scripts.relp2_e12 import build_design

OUT_DIR = Path("results/phantom")
HO_SEED = 777
HO_FRAC = 0.2
HO_NAME = "e12_8e_xor_ho"


def build_holdout_md():
    """8e_xor mapping with 20% of cells moved out of the training examples.
    mappings keeps the full grid (eval uses it); examples = train cells only.
    Guarantees every (B, slot) keeps >= 1 training z-combo so all answers
    stay visible."""
    md = build_design("8e_xor")
    rng = random.Random(HO_SEED)
    by_bslot = {}
    for ex in md.examples:
        by_bslot.setdefault((ex["b"], ex["a"]), []).append(ex)
    train_ex, probe_cells = [], []
    for (_b, _a), exs in sorted(by_bslot.items()):
        rng.shuffle(exs)
        keep = max(1, round(len(exs) * (1 - HO_FRAC)))
        train_ex.extend(exs[:keep])
        probe_cells.extend((e["b"], e["z"]) for e in exs[keep:])
    md_train = MappingData(mappings=md.mappings, examples=train_ex,
                           n_unique_b=md.n_unique_b, n_unique_a=md.n_unique_a,
                           k=md.k, task="bz_to_a")
    return md_train, probe_cells


def cmd_train():
    md, probe_cells = build_holdout_md()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"b_{HO_NAME}_probe_cells.json", "w") as f:
        json.dump(sorted(probe_cells), f)
    cfg = make_config(HO_NAME, k=md.k, max_steps=25000,
                      checkpoint_every=100, eval_every=50)
    print(f"training {HO_NAME}: train cells={len(md.examples)} "
          f"held-out cells={len(probe_cells)}", flush=True)
    run_single_experiment(cfg, mapping_data=md, output_dir="outputs")
    print("done.")


def probe_cell_set(exp):
    if exp == HO_NAME:
        return set(map(tuple, json.load(
            open(OUT_DIR / f"b_{HO_NAME}_probe_cells.json"))))
    # phantom_heldout_k10: reconstruct the example-level split the trainer used
    cfg = make_config(exp, k=10, max_steps=30000)
    cfg.data.probe_fraction = HO_FRAC
    cfg.data.split_by_base = False
    tok = create_tokenizer_from_config(cfg)
    from src.data import create_datasets_from_config
    _train_ds, probe_ds, _md = create_datasets_from_config(cfg, tok)
    return {(ex["b"], ex["z"]) for ex in probe_ds.examples}


def cmd_analyze(exp):
    probe = probe_cell_set(exp)
    if exp == HO_NAME:
        cfg = make_config(exp, k=100, max_steps=25000)
        md, _ = build_holdout_md()          # mappings = full grid
    else:
        cfg = make_config(exp, k=10, max_steps=30000)
        cfg.data.probe_fraction = HO_FRAC
        cfg.data.split_by_base = False
        md = None                            # eval regenerates full mapping
    tok = create_tokenizer_from_config(cfg)
    device = select_device()
    pb = build_pair_batch(cfg, tok, n_b_eval=64, seed=1234, mapping_data=md)
    ids, pb = to_device(pb, device)

    n_rows = ids.shape[0]
    k = pb.base.k
    mask = torch.tensor(
        [(pb.base.b_strings[i // k], pb.base.z_strings[i % k]) in probe
         for i in range(n_rows)], device=device)
    n_probe = int(mask.sum())
    print(f"{exp}: eval grid {n_rows} rows, {n_probe} held-out "
          f"({n_probe / n_rows:.1%})", flush=True)

    ckpt_dir = Path("outputs") / exp / "checkpoints"
    rows = []
    for step in list_checkpoints(ckpt_dir):
        model = load_model(cfg, tok, ckpt_dir, step, device)
        with torch.no_grad():
            rm = regime_metrics(model(ids), pb)
        row = {"step": step}
        for j in range(1, A_LEN + 1):
            for tag, m in (("tr", ~mask), ("ho", mask)):
                row[f"ce_pos{j}_{tag}"] = float(rm[f"ce_pos{j}"][m].mean())
                row[f"acc_pos{j}_{tag}"] = float(rm[f"acc_pos{j}"][m].mean())
        for met in ("m_z_pos1", "m_B_pos1"):
            row[f"{met}_tr"] = float(rm[met][~mask].mean())
            row[f"{met}_ho"] = float(rm[met][mask].mean())
        rows.append(row)
        del model
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"b_{exp}.json", "w") as f:
        json.dump({"exp": exp, "n_rows": n_rows, "n_probe": n_probe,
                   "rows": rows}, f)

    steps = [r["step"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(steps, [r["ce_pos1_tr"] for r in rows], label="train cells")
    axes[0].plot(steps, [r["ce_pos1_ho"] for r in rows], label="held-out cells")
    axes[0].set_ylabel("full-vocab CE (pos 1)")
    axes[1].plot(steps, [r["acc_pos1_tr"] for r in rows], label="train cells")
    axes[1].plot(steps, [r["acc_pos1_ho"] for r in rows], label="held-out cells")
    axes[1].set_ylabel("acc (pos 1)")
    axes[2].plot(steps, [r["m_z_pos1_tr"] for r in rows], label="m_z train")
    axes[2].plot(steps, [r["m_z_pos1_ho"] for r in rows], label="m_z held-out")
    axes[2].plot(steps, [r["m_B_pos1_tr"] for r in rows], label="m_B train",
                 ls="--")
    axes[2].set_ylabel("sensitivity index")
    for ax in axes:
        ax.set_xscale("symlog", linthresh=200)
        ax.set_xlabel("step")
        ax.legend(fontsize=8)
    fig.suptitle(f"Phase B — {exp}: train vs held-out cells")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"b_{exp}.png", dpi=150)
    print(f"-> {OUT_DIR}/b_{exp}.json + b_{exp}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "analyze"])
    ap.add_argument("exp", nargs="?", default=HO_NAME)
    args = ap.parse_args()
    if args.mode == "train":
        cmd_train()
    else:
        cmd_analyze(args.exp)
