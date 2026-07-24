#!/usr/bin/env python
"""
E8 — Alphabet surgery: training runs where hypotheses C (constant machine)
and M (per-B marginal) predict DIFFERENT plateau heights on full-vocab CE,
with both shelf values computed from the actual generated data and committed
to disk BEFORE training.

Designs (all: 4L/4H/d128/d_mlp512 GELU, eta=1e-3, wd=0.01, bs=128, seed 42,
checkpoints every 100 — the landauer_dense_k10 protocol):
  8a overlap  : K=5, per-B first chars drawn (unique within B) from an
                18-char pool with Zipf weights -> global marginal skewed,
                candidate alphabets overlap across B.
                C shelf = H(q*_1) (skewed);  M shelf = log 5.
  8b skewz    : K=10 standard structure, z-frequency multiplicities
                [32,16,8,8,4,4,2,2,1,1] -> P(A_1|B) non-uniform.
                C shelf = H(q*_1) ~ log 36;  M shelf = H(mult) < log 10.
                Staircase must be evaluated with training weights.
  8c nostruct : K=10, unique-first-char DISABLED (random A strings).
                Tests whether run-1/2 findings rode the position-1 shortcut.
  8d k1       : K=1 (A deterministic in B). Isolated order-0 -> order-1 stair.

Subcommands:
  design           build data for all designs, write e8_<name>_design.json
                   with precomputed shelves + separation check (>= 0.5 nats)
  train  <name>    train (writes to outputs/e8_<name>/ via run_single_experiment)
  stair  <name>    staircase eval over checkpoints + figure with pre-drawn shelves
"""

import argparse
import json
import math
import random
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
from src.data.dataset import MappingData, generate_mappings, generate_random_string
from src.training.checkpoint import list_checkpoints
from scripts.experiment_helpers import make_config, run_single_experiment
from scripts.relp_core import load_model, select_device
from scripts.relp2_core import (
    build_pair_batch, to_device, regime_metrics, A_LEN,
)

VOCAB = "abcdefghijklmnopqrstuvwxyz0123456789"
MULT_8B = [32, 16, 8, 8, 4, 4, 2, 2, 1, 1]
N_B = 1000
SEED = 42

DESIGNS = ["8a_overlap", "8b_skewz", "8c_nostruct", "8d_k1"]


# ---------------------------------------------------------------------------
# data builders
# ---------------------------------------------------------------------------

def build_8a(seed=SEED):
    rng = random.Random(seed)
    pool = list("abcdefghijklmnopqr")            # 18 chars
    w = np.array([1.0 / (i + 1) for i in range(len(pool))])
    k = 5
    used_b, used_a = set(), set()
    zs = []
    while len(zs) < k:
        z = generate_random_string(2, VOCAB, rng)
        if z not in zs:
            zs.append(z)
    mappings, examples = {}, []
    for _ in range(N_B):
        b = generate_random_string(6, VOCAB, rng)
        while b in used_b:
            b = generate_random_string(6, VOCAB, rng)
        used_b.add(b)
        # weighted sample of k distinct first chars
        avail = list(range(len(pool)))
        chars = []
        for _j in range(k):
            ww = np.array([w[i] for i in avail]); ww = ww / ww.sum()
            pick = rng.choices(avail, weights=ww, k=1)[0]
            avail.remove(pick)
            chars.append(pool[pick])
        pairs = []
        for j in range(k):
            a = chars[j] + generate_random_string(3, VOCAB, rng)
            while a in used_a:
                a = chars[j] + generate_random_string(3, VOCAB, rng)
            used_a.add(a)
            pairs.append((zs[j], a))
            examples.append({"b": b, "z": zs[j], "a": a})
        mappings[b] = pairs
    md = MappingData(mappings=mappings, examples=examples, n_unique_b=N_B,
                     n_unique_a=len(used_a), k=k, task="bz_to_a")
    weights = {(ex["b"], ex["z"]): 1.0 for ex in examples}
    return md, weights, k


def build_8b(seed=SEED):
    md = generate_mappings(
        n_unique_b=N_B, k=10, b_length=6, a_length=4, z_length=2,
        vocab_chars=VOCAB, seed=seed, task="bz_to_a",
        enforce_unique_a_first_char_per_b=True, disambiguation_prefix_length=1)
    # shared z ordering
    zs = [z for (z, _a) in md.mappings[sorted(md.mappings)[0]]]
    z_slot = {z: j for j, z in enumerate(zs)}
    examples = []
    weights = {}
    for b, pairs in md.mappings.items():
        for (z, a) in pairs:
            m = MULT_8B[z_slot[z]]
            weights[(b, z)] = float(m)
            examples.extend([{"b": b, "z": z, "a": a}] * m)
    md2 = MappingData(mappings=md.mappings, examples=examples,
                      n_unique_b=N_B, n_unique_a=md.n_unique_a, k=10, task="bz_to_a")
    return md2, weights, 10


def get_design(name):
    """Returns (cfg, mapping_data_or_None, weights_or_None)."""
    if name == "8a_overlap":
        md, w, k = build_8a()
        cfg = make_config(f"e8_{name}", k=k, max_steps=15000,
                          checkpoint_every=100, eval_every=50)
        return cfg, md, w
    if name == "8b_skewz":
        md, w, k = build_8b()
        cfg = make_config(f"e8_{name}", k=k, max_steps=25000,
                          checkpoint_every=100, eval_every=50)
        return cfg, md, w
    if name == "8c_nostruct":
        cfg = make_config(f"e8_{name}", k=10, max_steps=15000,
                          checkpoint_every=100, eval_every=50)
        cfg.data.enforce_unique_a_first_char_per_b = False
        return cfg, None, None
    if name == "8d_k1":
        cfg = make_config(f"e8_{name}", k=1, max_steps=8000,
                          checkpoint_every=100, eval_every=50)
        return cfg, None, None
    raise ValueError(name)


def materialize_mapping(cfg):
    return generate_mappings(
        n_unique_b=cfg.data.n_unique_b, k=cfg.data.k,
        b_length=cfg.data.b_length, a_length=cfg.data.a_length,
        z_length=cfg.data.z_length, vocab_chars=cfg.data.vocab_chars,
        seed=cfg.experiment.seed, task=cfg.data.task,
        enforce_unique_a_first_char_per_b=bool(cfg.data.enforce_unique_a_first_char_per_b),
        disambiguation_prefix_length=int(cfg.data.disambiguation_prefix_length))


# ---------------------------------------------------------------------------
# weighted shelves from an example list (ground truth = the generated data)
# ---------------------------------------------------------------------------

def shelves_from_examples(examples, tokenizer, a_length):
    V = tokenizer.vocab_size
    tid = lambda ch: tokenizer.token_to_id[ch]

    def entropy(counts):
        p = counts / counts.sum()
        p = p[p > 0]
        return float(-(p * p.log()).sum())

    out = {"H_qstar": {}, "H_given_prefix": {}, "H_given_B": {}, "q_star": {}}
    for j in range(a_length):
        counts = torch.zeros(V)
        groups, groupsB = {}, {}
        for ex in examples:
            a, b = ex["a"], ex["b"]
            counts[tid(a[j])] += 1
            groups.setdefault(a[:j], torch.zeros(V))[tid(a[j])] += 1
            groupsB.setdefault((b, a[:j]), torch.zeros(V))[tid(a[j])] += 1
        tot = counts.sum()
        out["q_star"][j + 1] = (counts / tot).tolist()
        out["H_qstar"][j + 1] = entropy(counts)
        out["H_given_prefix"][j + 1] = float(
            sum(g.sum() / tot * entropy(g) for g in groups.values()))
        out["H_given_B"][j + 1] = float(
            sum(g.sum() / tot * entropy(g) for g in groupsB.values()))
    out["log_vocab"] = math.log(V)
    out["n_examples_weighted"] = len(examples)
    return out


def cmd_design():
    for name in DESIGNS:
        cfg, md, _w = get_design(name)
        tok = create_tokenizer_from_config(cfg)
        if md is None:
            md = materialize_mapping(cfg)
        sh = shelves_from_examples(md.examples, tok, int(cfg.data.a_length))
        c_shelf = sh["H_qstar"][1]
        m_shelf = sh["H_given_B"][1]
        sep = abs(c_shelf - m_shelf)
        design = {"name": name, "k": int(cfg.data.k),
                  "max_steps": int(cfg.training.max_steps),
                  "shelves": {k: v for k, v in sh.items() if k != "q_star"},
                  "q_star": sh["q_star"],
                  "C_shelf_pos1": c_shelf, "M_shelf_pos1": m_shelf,
                  "separation_pos1": sep,
                  "discriminates": bool(sep >= 0.5)}
        out = Path("results/relp2") / f"e8_{name}"
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "design.json", "w") as f:
            json.dump(design, f, indent=1)
        print(f"{name}: C(pos1)={c_shelf:.4f}  M(pos1)={m_shelf:.4f}  "
              f"sep={sep:.3f} nats  {'DISCRIMINATES' if sep >= 0.5 else 'weak/control'}"
              f"  (pos2-4 M: {[round(sh['H_given_B'][j], 3) for j in (2, 3, 4)]})")


def cmd_train(name):
    cfg, md, _w = get_design(name)
    print(f"training e8_{name}: k={cfg.data.k} max_steps={cfg.training.max_steps}",
          flush=True)
    run_single_experiment(cfg, mapping_data=md, output_dir="outputs")
    print("done.")


def cmd_stair(name):
    cfg, md, weights = get_design(name)
    tok = create_tokenizer_from_config(cfg)
    if md is None:
        md = materialize_mapping(cfg)
    design = json.load(open(Path("results/relp2") / f"e8_{name}" / "design.json"))
    device = select_device()
    n_b = min(128, N_B)
    pb = build_pair_batch(cfg, tok, n_b_eval=n_b, seed=1234, mapping_data=md)
    ids, pb = to_device(pb, device)

    # per-row training weights (multiplicity of this row's (B,z))
    if weights is not None:
        wrow = torch.tensor(
            [weights[(pb.b_strings[int(b)], pb.z_strings[int(z)])]
             for b, z in zip(pb.b_idx.cpu(), pb.z_idx.cpu())],
            device=device)
        wrow = wrow / wrow.mean()
    else:
        wrow = torch.ones(ids.shape[0], device=device)

    ckpt_dir = Path("outputs") / cfg.experiment.name / "checkpoints"
    rows = []
    for step in list_checkpoints(ckpt_dir):
        model = load_model(cfg, tok, ckpt_dir, step, device)
        with torch.no_grad():
            rm = regime_metrics(model(ids), pb)
        row = {"step": step}
        for j in range(1, A_LEN + 1):
            ce = rm[f"ce_pos{j}"]
            row[f"ce_pos{j}"] = float((ce * wrow).mean())
            row[f"ce_pos{j}_unw"] = float(ce.mean())
            row[f"acc_pos{j}"] = float((rm[f"acc_pos{j}"] * wrow).mean())
        row["m_z_pos1"] = float((rm["m_z_pos1"] * wrow).mean())
        row["m_B_pos1"] = float((rm["m_B_pos1"] * wrow).mean())
        rows.append(row)
        del model
    out = Path("results/relp2") / f"e8_{name}"
    with open(out / "staircase.json", "w") as f:
        json.dump({"design": design, "rows": rows}, f)

    sh = design["shelves"]
    steps = [r["step"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for j in range(1, A_LEN + 1):
        ax = axes[(j - 1) // 2][(j - 1) % 2]
        ax.plot(steps, [r[f"ce_pos{j}"] for r in rows], color="tab:blue", lw=1.2,
                label="full-vocab CE (training-weighted)")
        if weights is not None:
            ax.plot(steps, [r[f"ce_pos{j}_unw"] for r in rows], color="tab:cyan",
                    lw=0.9, alpha=0.7, label="unweighted")
        ax.axhline(sh["log_vocab"], color="0.6", ls="--", lw=0.9, label="log |V|")
        ax.axhline(sh["H_qstar"][str(j)], color="tab:purple", ls="--", lw=1.1,
                   label="C shelf: H(q*)")
        ax.axhline(sh["H_given_prefix"][str(j)], color="tab:brown", ls="--", lw=0.9,
                   label="C+prefix shelf")
        ax.axhline(sh["H_given_B"][str(j)], color="tab:green", ls="--", lw=1.1,
                   label="M shelf: H(A|B,prefix)")
        ax.set_xscale("symlog", linthresh=200)
        ax.set_title(f"A-position {j}")
        ax.set_ylabel("nats")
        if j == 1:
            ax.legend(fontsize=7, loc="lower left")
    for ax in axes[1]:
        ax.set_xlabel("step")
    fig.suptitle(f"e8_{name}: staircase — precommitted shelves "
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
