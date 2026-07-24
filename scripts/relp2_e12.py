#!/usr/bin/env python
"""
E12 — Difficulty gap: can a *harder-to-extract* selector create the missing
intermediate resting point that E8's statistics reshaping could not?

E8 verdict: no answer-statistics design creates an M shelf; shelves appear
where gradient starves, not at representable knowledge levels. E12 turns the
difficulty knob directly: make selector information order-2 in the input
(zero order-1 signal) while question (B) reading stays order-1.

Designs (landauer_dense protocol otherwise: 4L/4H/d128, eta=1e-3, wd=0.01,
bs=128, seed 42, checkpoints every 100):
  8e_xor  : z = (c1, c2), c1 in 'abcdefghij', c2 in 'klmnopqrst', all 100
            combos trained; answer slot = (idx(c1) + idx(c2)) mod 10.
            Each character alone is EXACTLY uninformative about the slot
            (P(slot|c1) uniform), so no order-1 gradient toward the
            selector exists. Predicted shelf if starvation rule holds:
            park at M = H(A_1|B) = log 10 after B-reading forms, before
            the char-pair conjunction forms.
  8f_easy : identical 100 z-strings, identical counts; slot = idx(c1)
            (c2 irrelevant). Order-1 selector signal maximal. Control for
            dataset size/multiplicity: predicts NO M park.

Subcommands: design | train <name> | stair <name>   (mirrors relp2_e8.py;
results under results/relp2/e12_<name>/, runs under outputs/e12_<name>/)
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import create_tokenizer_from_config
from src.data.dataset import MappingData, generate_mappings
from src.training.checkpoint import list_checkpoints
from scripts.experiment_helpers import make_config, run_single_experiment
from scripts.relp_core import load_model, select_device
from scripts.relp2_core import build_pair_batch, to_device, regime_metrics, A_LEN
from scripts.relp2_e8 import shelves_from_examples

N_B = 1000
SEED = 42
K_EFF = 10                       # answer slots per B
Z1 = "abcdefghij"                # first z char alphabet
Z2 = "klmnopqrst"                # second z char alphabet
DESIGNS = ["8e_xor", "8f_easy"]

MAX_STEPS = {"8e_xor": 25000, "8f_easy": 15000}


def slot_of(name, c1, c2):
    i1, i2 = Z1.index(c1), Z2.index(c2)
    return (i1 + i2) % K_EFF if name == "8e_xor" else i1


def build_design(name, seed=SEED):
    """Base mappings give B strings + K_EFF answers (unique first char per B);
    we replace the z layer with the 100-combo compositional selector."""
    base = generate_mappings(
        n_unique_b=N_B, k=K_EFF, b_length=6, a_length=4, z_length=2,
        vocab_chars="abcdefghijklmnopqrstuvwxyz0123456789", seed=seed,
        task="bz_to_a", enforce_unique_a_first_char_per_b=True,
        disambiguation_prefix_length=1)
    # shared base z ordering -> slot j answer for each B
    base_zs = [z for (z, _a) in base.mappings[sorted(base.mappings)[0]]]
    z_all = [c1 + c2 for c1 in Z1 for c2 in Z2]          # fixed shared order
    mappings, examples = {}, []
    for b, pairs in base.mappings.items():
        slot_a = {base_zs.index(z): a for (z, a) in pairs}
        new_pairs = []
        for z in z_all:
            a = slot_a[slot_of(name, z[0], z[1])]
            new_pairs.append((z, a))
            examples.append({"b": b, "z": z, "a": a})
        mappings[b] = new_pairs
    md = MappingData(mappings=mappings, examples=examples, n_unique_b=N_B,
                     n_unique_a=base.n_unique_a, k=len(z_all), task="bz_to_a")
    return md


def get_design(name):
    md = build_design(name)
    cfg = make_config(f"e12_{name}", k=md.k, max_steps=MAX_STEPS[name],
                      checkpoint_every=100, eval_every=50)
    return cfg, md


def cmd_design():
    for name in DESIGNS:
        cfg, md = get_design(name)
        tok = create_tokenizer_from_config(cfg)
        sh = shelves_from_examples(md.examples, tok, int(cfg.data.a_length))
        c_shelf = sh["H_qstar"][1]
        m_shelf = sh["H_given_B"][1]
        sep = abs(c_shelf - m_shelf)
        design = {"name": name, "k": int(md.k), "k_eff": K_EFF,
                  "max_steps": int(cfg.training.max_steps),
                  "selector_rule": "(i1+i2) mod 10" if name == "8e_xor" else "i1",
                  "shelves": {k: v for k, v in sh.items() if k != "q_star"},
                  "q_star": sh["q_star"],
                  "C_shelf_pos1": c_shelf, "M_shelf_pos1": m_shelf,
                  "separation_pos1": sep,
                  "discriminates": bool(sep >= 0.5)}
        out = Path("results/relp2") / f"e12_{name}"
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "design.json", "w") as f:
            json.dump(design, f, indent=1)
        print(f"{name}: C(pos1)={c_shelf:.4f}  M(pos1)={m_shelf:.4f}  "
              f"sep={sep:.3f} nats  "
              f"{'DISCRIMINATES' if sep >= 0.5 else 'weak/control'}")


def cmd_train(name):
    cfg, md = get_design(name)
    print(f"training e12_{name}: k={md.k} (k_eff={K_EFF}) "
          f"max_steps={cfg.training.max_steps} examples={len(md.examples)}",
          flush=True)
    run_single_experiment(cfg, mapping_data=md, output_dir="outputs")
    print("done.")


def cmd_stair(name):
    cfg, md = get_design(name)
    tok = create_tokenizer_from_config(cfg)
    design = json.load(open(Path("results/relp2") / f"e12_{name}" / "design.json"))
    device = select_device()
    pb = build_pair_batch(cfg, tok, n_b_eval=64, seed=1234, mapping_data=md)
    ids, pb = to_device(pb, device)

    ckpt_dir = Path("outputs") / cfg.experiment.name / "checkpoints"
    rows = []
    for step in list_checkpoints(ckpt_dir):
        model = load_model(cfg, tok, ckpt_dir, step, device)
        with torch.no_grad():
            rm = regime_metrics(model(ids), pb)
        row = {"step": step}
        for j in range(1, A_LEN + 1):
            row[f"ce_pos{j}"] = float(rm[f"ce_pos{j}"].mean())
            row[f"acc_pos{j}"] = float(rm[f"acc_pos{j}"].mean())
        row["m_z_pos1"] = float(rm["m_z_pos1"].mean())
        row["m_B_pos1"] = float(rm["m_B_pos1"].mean())
        rows.append(row)
        del model
    out = Path("results/relp2") / f"e12_{name}"
    with open(out / "staircase.json", "w") as f:
        json.dump({"design": design, "rows": rows}, f)

    sh = design["shelves"]
    steps = [r["step"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for j in range(1, A_LEN + 1):
        ax = axes[(j - 1) // 2][(j - 1) % 2]
        ax.plot(steps, [r[f"ce_pos{j}"] for r in rows], color="tab:blue", lw=1.2,
                label="full-vocab CE")
        ax.axhline(sh["log_vocab"], color="0.6", ls="--", lw=0.9, label="log |V|")
        ax.axhline(sh["H_qstar"][str(j)], color="tab:purple", ls="--", lw=1.1,
                   label="C shelf: H(q*)")
        ax.axhline(sh["H_given_prefix"][str(j)], color="tab:brown", ls="--",
                   lw=0.9, label="C+prefix shelf")
        ax.axhline(sh["H_given_B"][str(j)], color="tab:green", ls="--", lw=1.1,
                   label="M shelf: H(A|B,prefix)")
        ax.set_xscale("symlog", linthresh=200)
        ax.set_title(f"A-position {j}")
        ax.set_ylabel("nats")
        if j == 1:
            ax.legend(fontsize=7, loc="lower left")
    for ax in axes[1]:
        ax.set_xlabel("step")
    fig.suptitle(f"e12_{name}: difficulty-gap staircase "
                 f"(C={design['C_shelf_pos1']:.3f}, M={design['M_shelf_pos1']:.3f}, "
                 f"sep={design['separation_pos1']:.2f} nats)")
    fig.tight_layout()
    fig.savefig(out / "staircase.png", dpi=150)
    print(f"-> {out}/staircase.json + staircase.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["design", "train", "stair"])
    ap.add_argument("name", nargs="?", choices=DESIGNS)
    args = ap.parse_args()
    if args.mode == "design":
        cmd_design()
    elif args.mode == "train":
        cmd_train(args.name)
    else:
        cmd_stair(args.name)
