#!/usr/bin/env python3
"""HIERARCHICAL DISAMBIGUATION v3: K1=15, K2=10
=================================================

Toned-down version after v2 (K1=20, K2=10, D=200K) never escaped the
first plateau in 100K steps. Changes:
  - K1=15, K2=10 → K=150, n_b=500, D=75,000
  - η=3e-3 (bumped from 1e-3 to compress transition time)
  - EVAL_EVERY=50 (finer, since second plateau may be brief)

Prediction:
  Phase 0: log(150) ≈ 5.01  (ignoring both z1, z2)
  Phase 1: log(10)  ≈ 2.30  (using z1 only → cluster resolved)
  Phase 2: ≈ 0              (using both z1, z2)

Estimated timescales (with η=3e-3 compressing ~2-3x):
  τ_first:  D=75,000 → ~6,000-9,000 steps
  τ_second: D_second=5,000 → ~300-400 steps
"""

import sys
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.tokenizer import CharTokenizer, create_tokenizer_from_config
from src.data.dataset import MappingData, DisambiguationDataset, collate_fn
from src.model import create_model_from_config
from src.training.trainer import compute_loss
from src.analysis.candidate_eval import run_candidate_eval

# ─── Parameters ──────────────────────────────────────────────────────
K1 = 15             # clusters per B
K2 = 10             # targets per cluster
K = K1 * K2         # total candidates = 150
N_B = 500           # unique B groups
D = N_B * K         # total examples = 75,000

LR = 3e-3           # bumped from 1e-3
BATCH_SIZE = 128
MAX_STEPS = 50_000
EVAL_EVERY = 50         # finer eval for shorter plateaus
CHECKPOINT_EVERY = 5_000
EARLY_STOP_LOSS = 0.01
EARLY_STOP_PATIENCE = 10

SEED = 42
VOCAB_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789"
B_LENGTH = 6
A_LENGTH = 4

OUTPUT_DIR = Path("outputs/hierarchical_test_v3")

LOG_K = math.log(K)       # 5.011
LOG_K2 = math.log(K2)     # 2.303
LOG_K1 = math.log(K1)     # 2.708


# ─── Data Generation ────────────────────────────────────────────────

def generate_hierarchical_data():
    """Generate hierarchical disambiguation task data.

    Z-encoding: z = z1_char + z2_char where
      z1_char in first 15 chars of vocab   (K1=15 cluster selectors)
      z2_char in next 10 chars of vocab    (K2=10 within-cluster selectors)
    """
    rng = random.Random(SEED)

    z1_chars = list(VOCAB_CHARS[:K1])             # a-o (15 chars)
    z2_chars = list(VOCAB_CHARS[K1:K1 + K2])      # p-y (10 chars)

    print(f"z1 chars ({len(z1_chars)}): {z1_chars}")
    print(f"z2 chars ({len(z2_chars)}): {z2_chars}")

    z_selectors = []
    cluster_map = {}
    for c in range(K1):
        for t in range(K2):
            z_str = z1_chars[c] + z2_chars[t]
            z_selectors.append(z_str)
            cluster_map[z_str] = c

    used_b: set = set()
    used_a: set = set()
    mappings: Dict[str, List] = {}
    examples: List[Dict[str, str]] = []

    for group_idx in range(N_B):
        b = "".join(rng.choices(VOCAB_CHARS, k=B_LENGTH))
        while b in used_b:
            b = "".join(rng.choices(VOCAB_CHARS, k=B_LENGTH))
        used_b.add(b)

        a_list = []
        for _ in range(K):
            a = "".join(rng.choices(VOCAB_CHARS, k=A_LENGTH))
            attempts = 0
            while a in used_a:
                a = "".join(rng.choices(VOCAB_CHARS, k=A_LENGTH))
                attempts += 1
                if attempts > 5000:
                    raise RuntimeError("Failed to generate unique A string")
            used_a.add(a)
            a_list.append(a)

        pairs = [(z_selectors[i], a_list[i]) for i in range(K)]
        mappings[b] = pairs
        for i in range(K):
            examples.append({"b": b, "z": z_selectors[i], "a": a_list[i]})

        if (group_idx + 1) % 100 == 0:
            print(f"  Generated {group_idx + 1}/{N_B} B-groups...", flush=True)

    mapping_data = MappingData(
        mappings=mappings, examples=examples,
        n_unique_b=N_B, n_unique_a=len(used_a),
        k=K, task="bz_to_a",
    )
    return mapping_data, z_selectors, cluster_map


# ─── Z-Shuffle Variants ─────────────────────────────────────────────

def _derangement_perm(n, device):
    perm = torch.randperm(n, device=device)
    for i in range(n):
        if perm[i] == i:
            j = (i + 1) % n
            perm[i], perm[j] = perm[j].clone(), perm[i].clone()
    return perm


def shuffle_z1_in_batch(batch):
    input_ids = batch["input_ids"].clone()
    z_pos = batch["z_positions"]
    bs = input_ids.shape[0]
    perm = _derangement_perm(bs, input_ids.device)
    arange = torch.arange(bs, device=input_ids.device)
    z1_tokens = input_ids[arange, z_pos].clone()
    input_ids[arange, z_pos] = z1_tokens[perm]
    out = batch.copy()
    out["input_ids"] = input_ids
    return out


def shuffle_z2_in_batch(batch):
    input_ids = batch["input_ids"].clone()
    z_pos = batch["z_positions"]
    bs = input_ids.shape[0]
    perm = _derangement_perm(bs, input_ids.device)
    arange = torch.arange(bs, device=input_ids.device)
    z2_pos = z_pos + 1
    z2_tokens = input_ids[arange, z2_pos].clone()
    input_ids[arange, z2_pos] = z2_tokens[perm]
    out = batch.copy()
    out["input_ids"] = input_ids
    return out


def shuffle_both_z_in_batch(batch):
    input_ids = batch["input_ids"].clone()
    z_pos = batch["z_positions"]
    bs = input_ids.shape[0]
    perm1 = _derangement_perm(bs, input_ids.device)
    perm2 = _derangement_perm(bs, input_ids.device)
    arange = torch.arange(bs, device=input_ids.device)
    z1_tokens = input_ids[arange, z_pos].clone()
    z2_pos = z_pos + 1
    z2_tokens = input_ids[arange, z2_pos].clone()
    input_ids[arange, z_pos] = z1_tokens[perm1]
    input_ids[arange, z2_pos] = z2_tokens[perm2]
    out = batch.copy()
    out["input_ids"] = input_ids
    return out


# ─── Training ───────────────────────────────────────────────────────

def run():
    t0 = time.time()

    print("=" * 60)
    print("HIERARCHICAL DISAMBIGUATION v3")
    print("=" * 60)
    print(f"K1={K1}, K2={K2}, total K={K}, n_b={N_B}, D={D}")
    print(f"LR={LR}")
    print(f"Predicted plateau 1: log({K}) = {LOG_K:.3f}")
    print(f"Predicted plateau 2: log({K2}) = {LOG_K2:.3f}")
    print()

    # ── Generate data ──
    print("Generating hierarchical data...")
    mapping_data, z_selectors, cluster_map = generate_hierarchical_data()
    print(f"Generated {len(mapping_data.examples)} examples")
    print(f"Unique A strings: {mapping_data.n_unique_a}")

    sample_b = list(mapping_data.mappings.keys())[0]
    sample_pairs = mapping_data.mappings[sample_b]
    print(f"\nSample B='{sample_b}', cluster 0 (first 3 of {K2}):")
    c0 = [(z, a) for z, a in sample_pairs if cluster_map[z] == 0][:3]
    for z, a in c0:
        print(f"  z='{z}' → A='{a}'")
    print()

    # ── Config ──
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        "experiment": {"name": "hierarchical_v3", "seed": SEED},
        "data": {
            "n_unique_b": N_B, "k": K, "task": "bz_to_a",
            "b_length": B_LENGTH, "a_length": A_LENGTH, "z_length": 2,
            "vocab_chars": VOCAB_CHARS,
            "probe_fraction": 0.0, "split_by_base": True,
            "enforce_unique_a_first_char_per_b": False,
            "disambiguation_prefix_length": 1,
            "label_noise_prob": 0.0,
        },
        "tokenizer": {
            "pad_token": "<PAD>", "bos_token": "<BOS>",
            "eos_token": "<EOS>", "sep_token": "<SEP>",
        },
        "model": {
            "n_layers": 4, "n_heads": 4, "d_model": 128,
            "d_head": 32, "d_mlp": 512, "act_fn": "gelu",
        },
        "training": {
            "batch_size": BATCH_SIZE, "learning_rate": LR,
            "weight_decay": 0.01, "max_steps": MAX_STEPS,
            "warmup_steps": 0, "scheduler": "constant",
            "checkpoint_every": CHECKPOINT_EVERY,
            "eval_every": EVAL_EVERY,
        },
        "output": {"base_dir": str(OUTPUT_DIR)},
    })

    # ── Create components ──
    tokenizer = create_tokenizer_from_config(cfg)
    model = create_model_from_config(cfg, tokenizer)
    device = model.cfg.device
    print(f"Device: {device}")

    train_ds = DisambiguationDataset(
        mapping_data=mapping_data, tokenizer=tokenizer,
        split="train", probe_fraction=0.0,
        seed=SEED, task="bz_to_a",
    )
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        shuffle=True, collate_fn=collate_fn, num_workers=0,
    )
    print(f"Training examples: {len(train_ds)}, "
          f"steps/epoch: {len(train_loader)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    # ── History ──
    history = {
        "steps": [],
        "candidate_loss": [],
        "first_target_loss": [],
        "z1_shuffle_loss": [],
        "z2_shuffle_loss": [],
        "both_shuffle_loss": [],
        "train_loss": [],
    }

    # ── Training loop ──
    model.train()
    step = 0
    epoch = 0
    running_loss = 0.0
    n_batches = 0
    early_stop_count = 0
    stopped = False

    CAND_EVAL_N = 16

    pbar = tqdm(total=MAX_STEPS, desc="Training")

    while step < MAX_STEPS and not stopped:
        epoch += 1
        for batch in train_loader:
            if step >= MAX_STEPS or stopped:
                break

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            optimizer.zero_grad()
            loss, acc, first_loss = compute_loss(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1
            step += 1

            if step % EVAL_EVERY == 0:
                avg_loss = running_loss / n_batches

                with torch.no_grad():
                    model.eval()

                    _, _, clean_first = compute_loss(model, batch)

                    z1_batch = shuffle_z1_in_batch(batch)
                    _, _, z1_loss = compute_loss(model, z1_batch)

                    z2_batch = shuffle_z2_in_batch(batch)
                    _, _, z2_loss = compute_loss(model, z2_batch)

                    both_batch = shuffle_both_z_in_batch(batch)
                    _, _, both_loss = compute_loss(model, both_batch)

                    cand = run_candidate_eval(
                        model=model, tokenizer=tokenizer,
                        mapping_data=mapping_data,
                        n_examples=CAND_EVAL_N, task="bz_to_a",
                        device=str(device), seed=step,
                    )
                    cand_loss = cand["candidate_loss"]

                    model.train()

                history["steps"].append(step)
                history["candidate_loss"].append(cand_loss)
                history["first_target_loss"].append(clean_first)
                history["z1_shuffle_loss"].append(z1_loss)
                history["z2_shuffle_loss"].append(z2_loss)
                history["both_shuffle_loss"].append(both_loss)
                history["train_loss"].append(avg_loss)

                z1g = z1_loss - clean_first
                z2g = z2_loss - clean_first
                pbar.set_postfix({
                    "cand": f"{cand_loss:.3f}",
                    "z1g": f"{z1g:.2f}",
                    "z2g": f"{z2g:.2f}",
                    "loss": f"{avg_loss:.4f}",
                })

                if cand_loss < EARLY_STOP_LOSS:
                    early_stop_count += 1
                    if early_stop_count >= EARLY_STOP_PATIENCE:
                        print(f"\n[Early stop] Converged at step {step}")
                        stopped = True
                else:
                    early_stop_count = 0

                running_loss = 0.0
                n_batches = 0

                # Periodic save
                if step % 2500 == 0:
                    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                    with open(OUTPUT_DIR / "training_history.json", "w") as f:
                        json.dump(history, f, indent=2)

            pbar.update(1)

    pbar.close()
    elapsed = time.time() - t0

    # ── Save results ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    history["elapsed_seconds"] = elapsed
    history["early_stopped"] = stopped
    history["early_stopped_step"] = step if stopped else None
    with open(OUTPUT_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    make_figure(history)
    print_summary(history)


def make_figure(history):
    steps = history["steps"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Panel A: Candidate loss staircase
    ax1.plot(steps, history["candidate_loss"], "b-", linewidth=2,
             label="Candidate loss (K-way)")
    ax1.axhline(y=LOG_K, color="gray", linestyle="--", alpha=0.7,
                label=f"log({K}) = {LOG_K:.2f}")
    ax1.axhline(y=LOG_K2, color="orange", linestyle="--", alpha=0.7,
                label=f"log({K2}) = {LOG_K2:.2f}")
    ax1.set_ylabel("Candidate loss", fontsize=13)
    ax1.set_title(f"Hierarchical Disambiguation v3: "
                  f"K1={K1} clusters × K2={K2} targets "
                  f"(D={D:,}, η={LR})",
                  fontsize=14)
    ax1.legend(fontsize=11, loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=-0.1, top=LOG_K + 0.5)

    # Panel B: Z-usage diagnostics (gaps)
    ft = history["first_target_loss"]
    z1g = [z1 - f for z1, f in zip(history["z1_shuffle_loss"], ft)]
    z2g = [z2 - f for z2, f in zip(history["z2_shuffle_loss"], ft)]

    ax2.plot(steps, z1g, "r-", linewidth=1.5, label="z1-shuffle gap")
    ax2.plot(steps, z2g, "g-", linewidth=1.5, label="z2-shuffle gap")
    ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax2.set_xlabel("Training step", fontsize=13)
    ax2.set_ylabel("Shuffle gap (shuffled - clean)", fontsize=13)
    ax2.legend(fontsize=11, loc="upper left")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "hierarchical_staircase_v3.png", dpi=150)
    plt.close(fig)
    print(f"\nFigure saved: {OUTPUT_DIR / 'hierarchical_staircase_v3.png'}")


def print_summary(history):
    steps = history["steps"]
    cand = history["candidate_loss"]
    ft = history["first_target_loss"]
    z1s = history["z1_shuffle_loss"]
    z2s = history["z2_shuffle_loss"]

    # Early loss
    early_cand = [c for s, c in zip(steps, cand) if 200 <= s <= 1000]
    early_plateau = np.mean(early_cand) if early_cand else None

    # Detect plateau at log(K2) ± 0.3
    plateau_at_logk2 = [(s, c) for s, c in zip(steps, cand)
                        if abs(c - LOG_K2) < 0.3]
    n_plateau = len(plateau_at_logk2)
    if plateau_at_logk2:
        p_start = plateau_at_logk2[0][0]
        p_end = plateau_at_logk2[-1][0]
        p_duration = p_end - p_start
        p_mean = np.mean([c for _, c in plateau_at_logk2])
    else:
        p_start = p_end = p_duration = p_mean = None

    # Z-usage onset
    z1_gap_step = None
    z2_gap_step = None
    for s, f, z1, z2 in zip(steps, ft, z1s, z2s):
        if z1_gap_step is None and (z1 - f) > 0.1:
            z1_gap_step = s
        if z2_gap_step is None and (z2 - f) > 0.1:
            z2_gap_step = s

    final_cand = cand[-1] if cand else None
    converged = final_cand is not None and final_cand < 0.1

    print("\n" + "=" * 60)
    print("HIERARCHICAL DISAMBIGUATION v3")
    print("=" * 60)
    print(f"K1 = {K1}, K2 = {K2}, total K = {K}, n_b = {N_B}, D = {D:,}")
    print(f"LR = {LR}")
    print()
    print(f"Predicted plateau 1: log({K}) = {LOG_K:.3f}")
    print(f"Predicted plateau 2: log({K2}) = {LOG_K2:.3f}")
    print()
    print("Observed:")
    if early_plateau is not None:
        print(f"  Early loss (steps 200-1000): {early_plateau:.3f}  "
              f"(expected ~ {LOG_K:.3f})")
    if p_mean is not None:
        print(f"  Plateau near log(K2): mean={p_mean:.3f}, "
              f"steps {p_start}-{p_end} ({p_duration} steps, "
              f"{n_plateau} eval points)")
    else:
        print(f"  No sustained plateau at log(K2) = {LOG_K2:.3f}")
    if final_cand is not None:
        print(f"  Final candidate loss: {final_cand:.4f}")
    print()
    z1_first = (z1_gap_step is not None and
                (z2_gap_step is None or z1_gap_step < z2_gap_step))
    print(f"  z1 learned before z2?  {'YES' if z1_first else 'NO / SIMULTANEOUS'}")
    print(f"  Step where z1_shuffle_gap > 0.1: {z1_gap_step or 'never'}")
    print(f"  Step where z2_shuffle_gap > 0.1: {z2_gap_step or 'never'}")
    if z1_gap_step and z2_gap_step:
        print(f"  z2 lag behind z1: {z2_gap_step - z1_gap_step} steps")
    print()

    if p_duration and p_duration >= 500 and converged:
        verdict = "TWO PLATEAUS OBSERVED (clear staircase)"
    elif n_plateau and n_plateau >= 3 and converged:
        verdict = "TWO PLATEAUS (marginal staircase)"
    elif converged:
        verdict = "ONE PLATEAU (direct drop, no staircase)"
    else:
        verdict = "NO CONVERGENCE"
    print(f"VERDICT: {verdict}")

    elapsed = history.get("elapsed_seconds", 0)
    print(f"Wall time: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print("=" * 60)


if __name__ == "__main__":
    run()
