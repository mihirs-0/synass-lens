#!/usr/bin/env python3
"""HIERARCHICAL DISAMBIGUATION: Quick Falsification Test
=========================================================

Tests whether a transformer produces a loss staircase when learning
a two-level hierarchical disambiguation task.

Task: (B, z1, z2) -> A
  - Each B maps to K1*K2 = 20 candidates, organized as K1=5 clusters of K2=4
  - z1 (first char of z) selects the cluster
  - z2 (second char of z) selects within the cluster

Prediction: candidate_loss should show two plateaus:
  Phase 0: log(K1*K2) = log(20) ~ 3.00  (ignoring both z1, z2)
  Phase 1: log(K2) = log(4) ~ 1.39      (using z1 only)
  Phase 2: ~ 0                           (using both z1, z2)

The sequence format is IDENTICAL to the original wind tunnel:
  [BOS, B(6), SEP, z1_char, z2_char, SEP, A(4), EOS]
Only the data generation differs (hierarchical target structure).
"""

import sys
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

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
K1 = 5              # clusters per B
K2 = 4              # targets per cluster
K = K1 * K2         # total candidates = 20
N_B = 500           # unique B groups
D = N_B * K         # total examples = 10,000

LR = 1e-3
BATCH_SIZE = 128
MAX_STEPS = 100_000
EVAL_EVERY = 50
CHECKPOINT_EVERY = 10_000
EARLY_STOP_LOSS = 0.01
EARLY_STOP_PATIENCE = 10  # consecutive evals

SEED = 42
VOCAB_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789"
B_LENGTH = 6
A_LENGTH = 4

OUTPUT_DIR = Path("outputs/hierarchical_test")

LOG_K = math.log(K)       # 2.996
LOG_K2 = math.log(K2)     # 1.386


# ─── Data Generation ────────────────────────────────────────────────

def generate_hierarchical_data():
    """Generate hierarchical disambiguation task data.

    Z-encoding: z = z1_char + z2_char where
      z1_char in {a,b,c,d,e}   (K1=5 cluster selectors)
      z2_char in {f,g,h,i}     (K2=4 within-cluster selectors)

    For each B, K=20 unique A-targets are assigned to 5 clusters of 4.
    Cluster c gets A-targets at indices [c*K2 .. (c+1)*K2 - 1].

    Returns (MappingData, z_selectors list, cluster_map dict).
    """
    rng = random.Random(SEED)

    z1_chars = list(VOCAB_CHARS[:K1])          # a, b, c, d, e
    z2_chars = list(VOCAB_CHARS[K1:K1 + K2])   # f, g, h, i

    # Build z-selectors in cluster-first order
    z_selectors = []
    cluster_map = {}  # z_string -> cluster_index
    for c in range(K1):
        for t in range(K2):
            z_str = z1_chars[c] + z2_chars[t]
            z_selectors.append(z_str)
            cluster_map[z_str] = c

    used_b: set = set()
    used_a: set = set()
    mappings: Dict[str, List] = {}
    examples: List[Dict[str, str]] = []

    for _ in range(N_B):
        b = "".join(rng.choices(VOCAB_CHARS, k=B_LENGTH))
        while b in used_b:
            b = "".join(rng.choices(VOCAB_CHARS, k=B_LENGTH))
        used_b.add(b)

        # K unique A-targets with unique first chars (for clean first-token signal)
        first_chars = rng.sample(list(VOCAB_CHARS), K)
        a_list = []
        for fc in first_chars:
            suffix = "".join(rng.choices(VOCAB_CHARS, k=A_LENGTH - 1))
            a = fc + suffix
            attempts = 0
            while a in used_a:
                suffix = "".join(rng.choices(VOCAB_CHARS, k=A_LENGTH - 1))
                a = fc + suffix
                attempts += 1
                if attempts > 1000:
                    raise RuntimeError("Failed to generate unique A string")
            used_a.add(a)
            a_list.append(a)

        # Cluster c -> A-targets at indices c*K2 .. (c+1)*K2-1
        pairs = [(z_selectors[i], a_list[i]) for i in range(K)]
        mappings[b] = pairs
        for i in range(K):
            examples.append({"b": b, "z": z_selectors[i], "a": a_list[i]})

    mapping_data = MappingData(
        mappings=mappings, examples=examples,
        n_unique_b=N_B, n_unique_a=len(used_a),
        k=K, task="bz_to_a",
    )
    return mapping_data, z_selectors, cluster_map


# ─── Z-Shuffle Variants ─────────────────────────────────────────────

def _derangement_perm(n, device):
    """Random permutation with no fixed points."""
    perm = torch.randperm(n, device=device)
    for i in range(n):
        if perm[i] == i:
            j = (i + 1) % n
            perm[i], perm[j] = perm[j].clone(), perm[i].clone()
    return perm


def shuffle_z1_in_batch(batch):
    """Shuffle only z1 (first z character) across batch."""
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
    """Shuffle only z2 (second z character) across batch."""
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
    """Shuffle both z1 and z2 (with independent permutations)."""
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
    print("HIERARCHICAL DISAMBIGUATION TEST")
    print("=" * 60)
    print(f"K1={K1}, K2={K2}, total K={K}, n_b={N_B}, D={D}")
    print(f"Predicted plateau 1: log({K}) = {LOG_K:.3f}")
    print(f"Predicted plateau 2: log({K2}) = {LOG_K2:.3f}")
    print()

    # ── Generate data ──
    mapping_data, z_selectors, cluster_map = generate_hierarchical_data()
    print(f"Generated {len(mapping_data.examples)} examples")
    print(f"Z-selectors (cluster order): {z_selectors}")

    # Verify structure: pick a random B, show its cluster layout
    sample_b = list(mapping_data.mappings.keys())[0]
    sample_pairs = mapping_data.mappings[sample_b]
    print(f"\nSample B='{sample_b}':")
    for c in range(K1):
        cluster_pairs = [(z, a) for z, a in sample_pairs if cluster_map[z] == c]
        print(f"  Cluster {c}: {[(z, a) for z, a in cluster_pairs]}")
    print()

    # ── Config ──
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        "experiment": {"name": "hierarchical_test", "seed": SEED},
        "data": {
            "n_unique_b": N_B, "k": K, "task": "bz_to_a",
            "b_length": B_LENGTH, "a_length": A_LENGTH, "z_length": 2,
            "vocab_chars": VOCAB_CHARS,
            "probe_fraction": 0.0, "split_by_base": True,
            "enforce_unique_a_first_char_per_b": True,
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

            # ── Periodic eval ──
            if step % EVAL_EVERY == 0:
                avg_loss = running_loss / n_batches

                with torch.no_grad():
                    model.eval()

                    # Clean first-target loss on current batch
                    _, _, clean_first = compute_loss(model, batch)

                    # Z1-shuffle
                    z1_batch = shuffle_z1_in_batch(batch)
                    _, _, z1_loss = compute_loss(model, z1_batch)

                    # Z2-shuffle
                    z2_batch = shuffle_z2_in_batch(batch)
                    _, _, z2_loss = compute_loss(model, z2_batch)

                    # Both-shuffle
                    both_batch = shuffle_both_z_in_batch(batch)
                    _, _, both_loss = compute_loss(model, both_batch)

                    # Candidate eval (K-way normalized)
                    cand = run_candidate_eval(
                        model=model, tokenizer=tokenizer,
                        mapping_data=mapping_data,
                        n_examples=32, task="bz_to_a",
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

                pbar.set_postfix({
                    "cand": f"{cand_loss:.3f}",
                    "z1s": f"{z1_loss:.3f}",
                    "z2s": f"{z2_loss:.3f}",
                    "loss": f"{avg_loss:.4f}",
                })

                # Early stopping on candidate loss
                if cand_loss < EARLY_STOP_LOSS:
                    early_stop_count += 1
                    if early_stop_count >= EARLY_STOP_PATIENCE:
                        print(f"\n[Early stop] Converged at step {step}")
                        stopped = True
                else:
                    early_stop_count = 0

                running_loss = 0.0
                n_batches = 0

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

    # ── Figure ──
    make_figure(history)

    # ── Summary ──
    print_summary(history)


def make_figure(history):
    """Generate the staircase figure (2-panel)."""
    steps = history["steps"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # ── Panel A: Candidate loss staircase ──
    ax1.plot(steps, history["candidate_loss"], "b-", linewidth=2,
             label="Candidate loss (K-way)")
    ax1.axhline(y=LOG_K, color="gray", linestyle="--", alpha=0.7,
                label=f"log({K}) = {LOG_K:.2f}")
    ax1.axhline(y=LOG_K2, color="orange", linestyle="--", alpha=0.7,
                label=f"log({K2}) = {LOG_K2:.2f}")
    ax1.set_ylabel("Candidate loss", fontsize=13)
    ax1.set_title(f"Hierarchical Disambiguation: K1={K1} clusters × K2={K2} targets",
                  fontsize=14)
    ax1.legend(fontsize=11, loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=-0.1, top=LOG_K + 0.5)

    # ── Panel B: Z-usage diagnostics (first-target loss) ──
    ax2.plot(steps, history["first_target_loss"], "b-", linewidth=2,
             label="Clean (first-target)")
    ax2.plot(steps, history["z1_shuffle_loss"], "r--", linewidth=1.5,
             label="z1-shuffled")
    ax2.plot(steps, history["z2_shuffle_loss"], "g--", linewidth=1.5,
             label="z2-shuffled")
    ax2.plot(steps, history["both_shuffle_loss"], "k:", linewidth=1.5,
             label="Both-shuffled")
    ax2.axhline(y=LOG_K, color="gray", linestyle="--", alpha=0.4)
    ax2.axhline(y=LOG_K2, color="orange", linestyle="--", alpha=0.4)
    ax2.set_xlabel("Training step", fontsize=13)
    ax2.set_ylabel("First-target loss", fontsize=13)
    ax2.legend(fontsize=11, loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=-0.1)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "hierarchical_staircase.png", dpi=150)
    plt.close(fig)
    print(f"\nFigure saved: {OUTPUT_DIR / 'hierarchical_staircase.png'}")


def print_summary(history):
    """Print experiment summary."""
    steps = history["steps"]
    cand = history["candidate_loss"]
    ft = history["first_target_loss"]
    z1s = history["z1_shuffle_loss"]
    z2s = history["z2_shuffle_loss"]

    # Early loss level (steps 100-500)
    early_cand = [c for s, c in zip(steps, cand) if 100 <= s <= 500]
    early_plateau = np.mean(early_cand) if early_cand else (cand[0] if cand else None)

    # Find sustained plateau near log(K2)
    plateau_at_logk2 = [(s, c) for s, c in zip(steps, cand)
                        if abs(c - LOG_K2) < 0.3]
    mid_step = plateau_at_logk2[0][0] if plateau_at_logk2 else None
    mid_loss = plateau_at_logk2[0][1] if plateau_at_logk2 else None

    # Z-usage onset: when does z1/z2 shuffle gap exceed 0.1?
    z1_gap_step = None
    z2_gap_step = None
    for s, f, z1, z2 in zip(steps, ft, z1s, z2s):
        if z1_gap_step is None and (z1 - f) > 0.1:
            z1_gap_step = s
        if z2_gap_step is None and (z2 - f) > 0.1:
            z2_gap_step = s

    # Final loss
    final_cand = cand[-1] if cand else None

    # Verdict
    n_plateau_points = len(plateau_at_logk2)
    if n_plateau_points >= 5:
        # Check if plateau spans at least 250 steps (5 eval points × 50 steps)
        plateau_steps = [s for s, _ in plateau_at_logk2]
        plateau_duration = plateau_steps[-1] - plateau_steps[0] if len(plateau_steps) > 1 else 0
        two_plateaus = plateau_duration >= 200
    else:
        two_plateaus = False
        plateau_duration = 0

    converged = final_cand is not None and final_cand < 0.1

    print("\n" + "=" * 60)
    print("HIERARCHICAL DISAMBIGUATION TEST")
    print("=" * 60)
    print(f"K1 = {K1}, K2 = {K2}, total K = {K}, n_b = {N_B}, D = {D}")
    print()
    print(f"Predicted plateau 1: log({K}) = {LOG_K:.3f}")
    print(f"Predicted plateau 2: log({K2}) = {LOG_K2:.3f}")
    print()
    print("Observed:")
    if early_plateau is not None:
        print(f"  Early loss (steps 100-500): {early_plateau:.3f}  "
              f"(expected ~ {LOG_K:.3f})")
    if mid_loss is not None:
        print(f"  Mid plateau near log(K2): {mid_loss:.3f} at step {mid_step}  "
              f"(expected ~ {LOG_K2:.3f})")
        print(f"  Sustained for {n_plateau_points} eval points "
              f"({plateau_duration} steps)")
    else:
        print(f"  No sustained plateau at log(K2) = {LOG_K2:.3f}")
    if final_cand is not None:
        print(f"  Final candidate loss: {final_cand:.4f}")
    print()
    z1_first = (z1_gap_step is not None and
                (z2_gap_step is None or z1_gap_step <= z2_gap_step))
    print(f"  z1 learned before z2?  {'YES' if z1_first else 'NO'}")
    print(f"  Step where z1_shuffle_gap > 0.1: {z1_gap_step or 'never'}")
    print(f"  Step where z2_shuffle_gap > 0.1: {z2_gap_step or 'never'}")
    print()

    if two_plateaus and converged:
        verdict = "TWO PLATEAUS OBSERVED"
    elif converged and not two_plateaus:
        verdict = "ONE PLATEAU (direct drop to zero, no staircase)"
    elif not converged:
        verdict = "NO CONVERGENCE"
    else:
        verdict = "UNCLEAR"
    print(f"VERDICT: {verdict}")

    elapsed = history.get("elapsed_seconds", 0)
    print(f"Wall time: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print("=" * 60)


if __name__ == "__main__":
    run()
