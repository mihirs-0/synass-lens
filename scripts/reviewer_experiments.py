#!/usr/bin/env python
"""
Reviewer Response Experiments — Three Critical Gaps.

Experiment 1: Multi-seed robustness (train 3 new seeds, compute metrics across 4 seeds)
Experiment 2: Causal sufficiency test (full-residual patching at each layer boundary)
Experiment 3: Ablation variant sensitivity (zero, mean, resample, noise)

Reuses all infrastructure from prior analysis rounds.
"""

import sys
import json
import math
import time
import argparse
from pathlib import Path
from itertools import combinations

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import create_tokenizer_from_config, collate_fn
from src.data.dataset import generate_mappings, MappingData, DisambiguationDataset
from src.model import create_model_from_config
from src.training.checkpoint import load_checkpoint, list_checkpoints
from scripts.experiment_helpers import make_config, run_single_experiment

# ── Config ──────────────────────────────────────────────────────────────────

K = 10
LOG_K = math.log(K)
N_UNIQUE_B = 1000
N_LAYERS = 4
N_HEADS = 4
ORIGINAL_SEED = 42
NEW_SEEDS = [123, 456, 789]
ALL_SEEDS = [ORIGINAL_SEED] + NEW_SEEDS

OUTPUTS_DIR = Path("outputs")
ANALYSIS_DIR = Path("analysis_outputs")
ANALYSIS_DIR.mkdir(exist_ok=True)

DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")


# ── Shared helpers (reused from prior rounds) ───────────────────────────────

def get_cfg_and_tokenizer(seed=ORIGINAL_SEED):
    cfg = make_config(experiment_name=f"reviewer_seed{seed}", k=K, seed=seed, n_unique_b=N_UNIQUE_B)
    tokenizer = create_tokenizer_from_config(cfg)
    return cfg, tokenizer


def get_mapping_data(seed=ORIGINAL_SEED):
    return generate_mappings(
        n_unique_b=N_UNIQUE_B, k=K, b_length=6, a_length=4, z_length=2,
        vocab_chars="abcdefghijklmnopqrstuvwxyz0123456789",
        seed=seed, task="bz_to_a",
        enforce_unique_a_first_char_per_b=True, disambiguation_prefix_length=1,
    )


def get_experiment_name(seed):
    if seed == ORIGINAL_SEED:
        return "landauer_dense_k10"
    return f"landauer_dense_k10_seed{seed}"


def get_checkpoint_dir(seed):
    return OUTPUTS_DIR / get_experiment_name(seed) / "checkpoints"


def load_model_at_step(cfg, tokenizer, step, checkpoint_dir=None):
    if checkpoint_dir is None:
        checkpoint_dir = OUTPUTS_DIR / "landauer_dense_k10" / "checkpoints"
    model = create_model_from_config(cfg, tokenizer)
    model = model.to(DEVICE)
    load_checkpoint(model, None, checkpoint_dir, step=step)
    model.eval()
    return model


def build_eval_loader(mapping_data, tokenizer, n_examples=512, batch_size=128, seed=ORIGINAL_SEED):
    ds = DisambiguationDataset(
        mapping_data=mapping_data, tokenizer=tokenizer,
        split="train", probe_fraction=0.0, seed=seed, task="bz_to_a",
    )
    ds.tokenized = ds.tokenized[:n_examples]
    ds.examples = ds.examples[:n_examples]
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0,
    )
    return loader


def build_paired_inputs(mapping_data, tokenizer, n_pairs=256):
    pairs = []
    for b_str, targets in mapping_data.mappings.items():
        if len(targets) < 2:
            continue
        z_clean, a_clean = targets[0]
        z_corrupt, a_corrupt = targets[1]
        clean = tokenizer.encode_sequence(b_str, z_clean, a_clean, task="bz_to_a")
        corrupt = tokenizer.encode_sequence(b_str, z_corrupt, a_corrupt, task="bz_to_a")
        clean["first_target_id"] = tokenizer.encode(a_clean)[0]
        corrupt["first_target_id"] = tokenizer.encode(a_corrupt)[0]
        pairs.append((clean, corrupt))
        if len(pairs) >= n_pairs:
            break
    return pairs


def compute_loss(model, loader):
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            logits = model(input_ids)
            shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = labels[:, 1:].contiguous().view(-1)
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100, reduction='sum')
            mask = shift_labels != -100
            total_loss += loss.item()
            n += mask.sum().item()
    return total_loss / max(n, 1)


def cleanup_model(model):
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE == "mps":
        try:
            torch.mps.empty_cache()
        except AttributeError:
            pass


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Causal Sufficiency Test
# ═══════════════════════════════════════════════════════════════════════════

def experiment2_sufficiency():
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Causal Sufficiency Test at Each Layer")
    print("=" * 70)

    cfg, tokenizer = get_cfg_and_tokenizer(seed=ORIGINAL_SEED)
    mapping_data = get_mapping_data(seed=ORIGINAL_SEED)
    pairs = build_paired_inputs(mapping_data, tokenizer, n_pairs=256)
    print(f"  Built {len(pairs)} clean/corrupted pairs")

    ckpt_dir = get_checkpoint_dir(ORIGINAL_SEED)
    steps = sorted(list_checkpoints(ckpt_dir))
    final_step = steps[-1]
    model = load_model_at_step(cfg, tokenizer, final_step, checkpoint_dir=ckpt_dir)

    n_layers = N_LAYERS
    layer_labels = [f"After L{i}" for i in range(n_layers)]

    # ── Full-residual-stream patching ──
    print("\n  Part A: Full residual stream patching (all positions)...")
    full_flip_rates = {}
    full_logit_diffs = {}

    batch_size = 64
    for layer in range(n_layers):
        hook_name = f"blocks.{layer}.hook_resid_post"
        flips = []
        logit_deltas = []

        for start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[start:start + batch_size]
            bs = len(batch_pairs)
            max_len = max(
                max(len(p[0]["input_ids"]) for p in batch_pairs),
                max(len(p[1]["input_ids"]) for p in batch_pairs),
            )
            clean_ids = torch.zeros(bs, max_len, dtype=torch.long)
            corrupt_ids = torch.zeros(bs, max_len, dtype=torch.long)
            tgt_starts = []
            clean_first_tok = []
            corrupt_first_tok = []

            for i, (c, x) in enumerate(batch_pairs):
                clean_ids[i, :len(c["input_ids"])] = c["input_ids"]
                corrupt_ids[i, :len(x["input_ids"])] = x["input_ids"]
                tgt_starts.append(c["target_start_position"])
                clean_first_tok.append(c["first_target_id"])
                corrupt_first_tok.append(x["first_target_id"])

            clean_ids = clean_ids.to(DEVICE)
            corrupt_ids = corrupt_ids.to(DEVICE)
            clean_tok = torch.tensor(clean_first_tok, device=DEVICE)
            corrupt_tok = torch.tensor(corrupt_first_tok, device=DEVICE)
            tgt_pos = torch.tensor(tgt_starts, device=DEVICE)
            pred_pos = tgt_pos - 1
            batch_idx = torch.arange(bs, device=DEVICE)

            with torch.no_grad():
                # Get corrupted activations
                _, corrupt_cache = model.run_with_cache(corrupt_ids)
                corrupt_resid = corrupt_cache[hook_name].clone()

                # Run clean with full residual patched from corrupt
                def make_full_patch(c_resid):
                    def hook_fn(value, hook):
                        return c_resid
                    return hook_fn

                patched_logits = model.run_with_hooks(
                    clean_ids,
                    fwd_hooks=[(hook_name, make_full_patch(corrupt_resid))],
                )

                # Check if output flipped to B's answer
                patched_at_tgt = patched_logits[batch_idx, pred_pos, :]
                patched_argmax = patched_at_tgt.argmax(dim=-1)

                # Flip = model now predicts corrupt answer instead of clean
                flip = (patched_argmax == corrupt_tok).float()
                flips.extend(flip.cpu().tolist())

                # Logit difference: logit[corrupt] - logit[clean] after patching
                ld = (patched_at_tgt.gather(1, corrupt_tok.unsqueeze(1)).squeeze(1)
                      - patched_at_tgt.gather(1, clean_tok.unsqueeze(1)).squeeze(1))
                logit_deltas.extend(ld.cpu().tolist())

                del corrupt_cache

        full_flip_rates[layer] = float(np.mean(flips))
        full_logit_diffs[layer] = float(np.mean(logit_deltas))
        print(f"    After L{layer}: flip rate = {full_flip_rates[layer]:.4f}, "
              f"mean logit diff = {full_logit_diffs[layer]:.4f}")

    # ── Target-position-only patching ──
    print("\n  Part B: Target-position-only patching...")
    tgt_flip_rates = {}

    for layer in range(n_layers):
        hook_name = f"blocks.{layer}.hook_resid_post"
        flips = []

        for start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[start:start + batch_size]
            bs = len(batch_pairs)
            max_len = max(
                max(len(p[0]["input_ids"]) for p in batch_pairs),
                max(len(p[1]["input_ids"]) for p in batch_pairs),
            )
            clean_ids = torch.zeros(bs, max_len, dtype=torch.long)
            corrupt_ids = torch.zeros(bs, max_len, dtype=torch.long)
            tgt_starts = []
            clean_first_tok = []
            corrupt_first_tok = []

            for i, (c, x) in enumerate(batch_pairs):
                clean_ids[i, :len(c["input_ids"])] = c["input_ids"]
                corrupt_ids[i, :len(x["input_ids"])] = x["input_ids"]
                tgt_starts.append(c["target_start_position"])
                clean_first_tok.append(c["first_target_id"])
                corrupt_first_tok.append(x["first_target_id"])

            clean_ids = clean_ids.to(DEVICE)
            corrupt_ids = corrupt_ids.to(DEVICE)
            corrupt_tok = torch.tensor(corrupt_first_tok, device=DEVICE)
            tgt_pos = torch.tensor(tgt_starts, device=DEVICE)
            pred_pos = tgt_pos - 1
            batch_idx = torch.arange(bs, device=DEVICE)

            with torch.no_grad():
                _, corrupt_cache = model.run_with_cache(corrupt_ids)
                corrupt_resid = corrupt_cache[hook_name].clone()

                # Patch ONLY the prediction position
                def make_tgt_patch(c_resid, p_pos):
                    def hook_fn(value, hook):
                        for i in range(value.shape[0]):
                            value[i, p_pos[i], :] = c_resid[i, p_pos[i], :]
                        return value
                    return hook_fn

                patched_logits = model.run_with_hooks(
                    clean_ids,
                    fwd_hooks=[(hook_name, make_tgt_patch(corrupt_resid, pred_pos))],
                )

                patched_at_tgt = patched_logits[batch_idx, pred_pos, :]
                patched_argmax = patched_at_tgt.argmax(dim=-1)
                flip = (patched_argmax == corrupt_tok).float()
                flips.extend(flip.cpu().tolist())

                del corrupt_cache

        tgt_flip_rates[layer] = float(np.mean(flips))
        print(f"    After L{layer}: target-only flip rate = {tgt_flip_rates[layer]:.4f}")

    cleanup_model(model)

    # ── Compute necessity values (from prior results) ──
    # Recompute ablation on same model for consistency
    cfg2, tokenizer2 = get_cfg_and_tokenizer(seed=ORIGINAL_SEED)
    mapping_data2 = get_mapping_data(seed=ORIGINAL_SEED)
    loader = build_eval_loader(mapping_data2, tokenizer2, n_examples=512, batch_size=128)
    model2 = load_model_at_step(cfg2, tokenizer2, final_step, checkpoint_dir=ckpt_dir)
    baseline_loss = compute_loss(model2, loader)

    ablation_deltas = {}
    for layer in range(n_layers):
        pre_hook = f"blocks.{layer}.hook_resid_pre"
        post_hook = f"blocks.{layer}.hook_resid_post"
        resid_pre_store = {}

        def make_capture(store, key):
            def hook_fn(value, hook):
                store[key] = value.clone()
                return value
            return hook_fn

        def make_bypass(store, key):
            def hook_fn(value, hook):
                return store[key]
            return hook_fn

        total_loss = 0.0
        n_tokens = 0
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)
                resid_pre_store.clear()
                logits = model2.run_with_hooks(
                    input_ids,
                    fwd_hooks=[
                        (pre_hook, make_capture(resid_pre_store, "pre")),
                        (post_hook, make_bypass(resid_pre_store, "pre")),
                    ],
                )
                sl = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
                slb = labels[:, 1:].contiguous().view(-1)
                mask = slb != -100
                loss = F.cross_entropy(sl[mask], slb[mask])
                total_loss += loss.item() * mask.sum().item()
                n_tokens += mask.sum().item()

        ablation_deltas[layer] = (total_loss / max(n_tokens, 1)) - baseline_loss

    cleanup_model(model2)

    # ── Print necessity vs sufficiency table ──
    print("\n  NECESSITY vs SUFFICIENCY:")
    print(f"  {'Layer':<10s} {'Ablation Δ (nec.)':<20s} {'Full flip rate (suf.)':<22s} {'Tgt-only flip':<15s}")
    print("  " + "-" * 65)
    for layer in range(n_layers):
        print(f"  After L{layer:<3d} {ablation_deltas[layer]:>+16.3f}     "
              f"{full_flip_rates[layer]:>16.4f}       {tgt_flip_rates[layer]:>10.4f}")

    # ── Plot ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(n_layers)
    width = 0.25

    # Panel 1: Sufficiency - full vs target-only
    ax1.bar(x - width/2, [full_flip_rates[i] for i in range(n_layers)], width,
            label="Full residual patch", color="tab:blue", alpha=0.8)
    ax1.bar(x + width/2, [tgt_flip_rates[i] for i in range(n_layers)], width,
            label="Target-position only", color="tab:orange", alpha=0.8)
    ax1.axhline(y=1.0/K, color="red", linestyle="--", alpha=0.5, label=f"Chance (1/{K})")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"After L{i}" for i in range(n_layers)], fontsize=10)
    ax1.set_ylabel("Flip rate (fraction → corrupt answer)", fontsize=11)
    ax1.set_title("Causal Sufficiency by Layer", fontsize=13)
    ax1.legend(fontsize=9)
    ax1.set_ylim(0, 1.1)

    # Panel 2: Necessity (ablation) vs Sufficiency (flip rate) side by side
    # Normalize both to [0, 1] for comparison
    max_abl = max(ablation_deltas.values())
    norm_abl = [ablation_deltas[i] / max_abl for i in range(n_layers)]
    norm_suf = [full_flip_rates[i] for i in range(n_layers)]

    ax2.bar(x - width/2, norm_abl, width, label="Necessity (ablation Δ, normalized)", color="tab:red", alpha=0.8)
    ax2.bar(x + width/2, norm_suf, width, label="Sufficiency (flip rate)", color="tab:blue", alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"After L{i}" for i in range(n_layers)], fontsize=10)
    ax2.set_ylabel("Normalized score", fontsize=11)
    ax2.set_title("Necessity vs Sufficiency", fontsize=13)
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "reviewer_exp2_sufficiency.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved reviewer_exp2_sufficiency.png")

    return {
        "full_flip_rates": full_flip_rates,
        "tgt_flip_rates": tgt_flip_rates,
        "full_logit_diffs": full_logit_diffs,
        "ablation_deltas": ablation_deltas,
        "baseline_loss": baseline_loss,
    }


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Ablation Variant Sensitivity
# ═══════════════════════════════════════════════════════════════════════════

def experiment3_ablation_variants():
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Ablation Variant Sensitivity")
    print("=" * 70)

    cfg, tokenizer = get_cfg_and_tokenizer(seed=ORIGINAL_SEED)
    mapping_data = get_mapping_data(seed=ORIGINAL_SEED)
    loader = build_eval_loader(mapping_data, tokenizer, n_examples=512, batch_size=128)

    ckpt_dir = get_checkpoint_dir(ORIGINAL_SEED)
    steps = sorted(list_checkpoints(ckpt_dir))
    final_step = steps[-1]
    model = load_model_at_step(cfg, tokenizer, final_step, checkpoint_dir=ckpt_dir)

    baseline_loss = compute_loss(model, loader)
    print(f"  Baseline loss: {baseline_loss:.4f}")

    n_layers = N_LAYERS
    results = {"baseline": baseline_loss}

    # ── A. Zero ablation ──
    print("\n  A. Zero ablation...")
    zero_deltas = {}
    for layer in range(n_layers):
        def make_zero():
            def hook_fn(value, hook):
                return torch.zeros_like(value)
            return hook_fn

        total_loss = 0.0
        n_tokens = 0
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)
                logits = model.run_with_hooks(
                    input_ids,
                    fwd_hooks=[
                        (f"blocks.{layer}.hook_attn_out", make_zero()),
                        (f"blocks.{layer}.hook_mlp_out", make_zero()),
                    ],
                )
                sl = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
                slb = labels[:, 1:].contiguous().view(-1)
                mask = slb != -100
                loss = F.cross_entropy(sl[mask], slb[mask])
                total_loss += loss.item() * mask.sum().item()
                n_tokens += mask.sum().item()

        abl_loss = total_loss / max(n_tokens, 1)
        zero_deltas[layer] = abl_loss - baseline_loss
        print(f"    L{layer}: Δ = {zero_deltas[layer]:+.4f}")
    results["zero"] = zero_deltas

    # ── B. Mean ablation ──
    print("\n  B. Mean ablation...")
    # First compute mean activations across batch
    mean_acts_attn = {}
    mean_acts_mlp = {}

    with torch.no_grad():
        # Accumulate activations
        attn_accum = {l: [] for l in range(n_layers)}
        mlp_accum = {l: [] for l in range(n_layers)}

        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            _, cache = model.run_with_cache(input_ids)
            for l in range(n_layers):
                attn_accum[l].append(cache[f"blocks.{l}.hook_attn_out"].cpu())
                mlp_accum[l].append(cache[f"blocks.{l}.hook_mlp_out"].cpu())
            del cache

        for l in range(n_layers):
            mean_acts_attn[l] = torch.cat(attn_accum[l], dim=0).mean(dim=0).to(DEVICE)
            mean_acts_mlp[l] = torch.cat(mlp_accum[l], dim=0).mean(dim=0).to(DEVICE)
        del attn_accum, mlp_accum

    mean_deltas = {}
    for layer in range(n_layers):
        m_attn = mean_acts_attn[layer]
        m_mlp = mean_acts_mlp[layer]

        def make_mean_hook(mean_val):
            def hook_fn(value, hook):
                return mean_val.unsqueeze(0).expand_as(value)
            return hook_fn

        total_loss = 0.0
        n_tokens = 0
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)
                logits = model.run_with_hooks(
                    input_ids,
                    fwd_hooks=[
                        (f"blocks.{layer}.hook_attn_out", make_mean_hook(m_attn)),
                        (f"blocks.{layer}.hook_mlp_out", make_mean_hook(m_mlp)),
                    ],
                )
                sl = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
                slb = labels[:, 1:].contiguous().view(-1)
                mask = slb != -100
                loss = F.cross_entropy(sl[mask], slb[mask])
                total_loss += loss.item() * mask.sum().item()
                n_tokens += mask.sum().item()

        abl_loss = total_loss / max(n_tokens, 1)
        mean_deltas[layer] = abl_loss - baseline_loss
        print(f"    L{layer}: Δ = {mean_deltas[layer]:+.4f}")
    results["mean"] = mean_deltas

    # ── C. Resample ablation (5 runs, averaged) ──
    print("\n  C. Resample ablation (5 runs)...")
    resample_runs = []
    for run_i in range(5):
        run_deltas = {}
        for layer in range(n_layers):
            perm_seed = run_i * 100 + layer
            torch.manual_seed(perm_seed)

            def make_resample(seed_val):
                def hook_fn(value, hook):
                    g = torch.Generator(device=value.device)
                    g.manual_seed(seed_val)
                    perm = torch.randperm(value.shape[0], generator=g, device=value.device)
                    return value[perm]
                return hook_fn

            total_loss = 0.0
            n_tokens = 0
            with torch.no_grad():
                for batch in loader:
                    input_ids = batch["input_ids"].to(DEVICE)
                    labels = batch["labels"].to(DEVICE)
                    logits = model.run_with_hooks(
                        input_ids,
                        fwd_hooks=[
                            (f"blocks.{layer}.hook_attn_out", make_resample(perm_seed + 1000)),
                            (f"blocks.{layer}.hook_mlp_out", make_resample(perm_seed + 2000)),
                        ],
                    )
                    sl = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
                    slb = labels[:, 1:].contiguous().view(-1)
                    mask = slb != -100
                    loss = F.cross_entropy(sl[mask], slb[mask])
                    total_loss += loss.item() * mask.sum().item()
                    n_tokens += mask.sum().item()

            abl_loss = total_loss / max(n_tokens, 1)
            run_deltas[layer] = abl_loss - baseline_loss
        resample_runs.append(run_deltas)
        if run_i == 0:
            for layer in range(n_layers):
                print(f"    L{layer}: Δ = {run_deltas[layer]:+.4f} (run 0)")

    # Average across runs
    resample_deltas = {}
    for layer in range(n_layers):
        vals = [r[layer] for r in resample_runs]
        resample_deltas[layer] = float(np.mean(vals))
    results["resample"] = resample_deltas
    print("    Averaged across 5 runs:")
    for layer in range(n_layers):
        print(f"    L{layer}: Δ = {resample_deltas[layer]:+.4f}")

    # ── D. Gaussian noise ablation (5 runs, averaged) ──
    print("\n  D. Gaussian noise ablation (5 runs)...")
    noise_runs = []
    for run_i in range(5):
        run_deltas = {}
        for layer in range(n_layers):
            noise_seed = 5000 + run_i * 100 + layer

            def make_noise(seed_val):
                def hook_fn(value, hook):
                    g = torch.Generator(device=value.device)
                    g.manual_seed(seed_val)
                    mean = value.mean(dim=0, keepdim=True)
                    std = value.std(dim=0, keepdim=True)
                    return torch.randn(value.shape, generator=g, device=value.device, dtype=value.dtype) * std + mean
                return hook_fn

            total_loss = 0.0
            n_tokens = 0
            with torch.no_grad():
                for batch in loader:
                    input_ids = batch["input_ids"].to(DEVICE)
                    labels = batch["labels"].to(DEVICE)
                    logits = model.run_with_hooks(
                        input_ids,
                        fwd_hooks=[
                            (f"blocks.{layer}.hook_attn_out", make_noise(noise_seed + 1000)),
                            (f"blocks.{layer}.hook_mlp_out", make_noise(noise_seed + 2000)),
                        ],
                    )
                    sl = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
                    slb = labels[:, 1:].contiguous().view(-1)
                    mask = slb != -100
                    loss = F.cross_entropy(sl[mask], slb[mask])
                    total_loss += loss.item() * mask.sum().item()
                    n_tokens += mask.sum().item()

            abl_loss = total_loss / max(n_tokens, 1)
            run_deltas[layer] = abl_loss - baseline_loss
        noise_runs.append(run_deltas)
        if run_i == 0:
            for layer in range(n_layers):
                print(f"    L{layer}: Δ = {run_deltas[layer]:+.4f} (run 0)")

    noise_deltas = {}
    for layer in range(n_layers):
        vals = [r[layer] for r in noise_runs]
        noise_deltas[layer] = float(np.mean(vals))
    results["noise"] = noise_deltas
    print("    Averaged across 5 runs:")
    for layer in range(n_layers):
        print(f"    L{layer}: Δ = {noise_deltas[layer]:+.4f}")

    cleanup_model(model)

    # ── Summary table ──
    print("\n  ABLATION VARIANT COMPARISON:")
    print(f"  {'Layer':<8s} {'Zero Δ':>10s} {'Mean Δ':>10s} {'Resample Δ':>12s} {'Noise Δ':>10s}")
    print("  " + "-" * 52)
    for layer in range(n_layers):
        print(f"  L{layer:<5d} {zero_deltas[layer]:>+10.3f} {mean_deltas[layer]:>+10.3f} "
              f"{resample_deltas[layer]:>+12.3f} {noise_deltas[layer]:>+10.3f}")

    # ── Rank correlation ──
    methods = {
        "zero": [zero_deltas[i] for i in range(n_layers)],
        "mean": [mean_deltas[i] for i in range(n_layers)],
        "resample": [resample_deltas[i] for i in range(n_layers)],
        "noise": [noise_deltas[i] for i in range(n_layers)],
    }

    print("\n  Spearman rank correlations:")
    method_names = list(methods.keys())
    for i, m1 in enumerate(method_names):
        for j, m2 in enumerate(method_names):
            if j <= i:
                continue
            r, p = stats.spearmanr(methods[m1], methods[m2])
            print(f"    {m1}-{m2}: ρ = {r:.4f} (p = {p:.4f})")

    # Check ordering
    all_same_order = True
    ref_order = np.argsort(methods["zero"])[::-1]
    for name, vals in methods.items():
        order = np.argsort(vals)[::-1]
        if not np.array_equal(order, ref_order):
            all_same_order = False
            print(f"    WARNING: {name} has different ordering: {order} vs ref {ref_order}")

    if all_same_order:
        print("    All methods agree on ordering: " + " > ".join(f"L{i}" for i in ref_order))
    results["ordering_robust"] = all_same_order

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(n_layers)
    width = 0.2
    colors = {"zero": "tab:blue", "mean": "tab:orange", "resample": "tab:green", "noise": "tab:red"}

    for i, (name, vals) in enumerate(methods.items()):
        ax.bar(x + i * width - 1.5 * width, vals, width, label=name.capitalize(), color=colors[name], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"L{i}" for i in range(n_layers)], fontsize=11)
    ax.set_ylabel("Ablation Δ (nats)", fontsize=11)
    ax.set_title(f"Ablation Impact by Method (K={K}, final checkpoint)", fontsize=13)
    ax.legend(fontsize=10)
    ax.axhline(y=LOG_K, color="gray", linestyle="--", alpha=0.3, label=f"log K")
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "reviewer_exp3_ablation_variants.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved reviewer_exp3_ablation_variants.png")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Multi-Seed Robustness
# ═══════════════════════════════════════════════════════════════════════════

def train_new_seeds():
    """Train 3 new models with different seeds."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Training New Seeds")
    print("=" * 70)

    for seed in NEW_SEEDS:
        exp_name = get_experiment_name(seed)
        ckpt_dir = get_checkpoint_dir(seed)

        # Check if already trained
        if ckpt_dir.exists():
            existing = sorted(list_checkpoints(ckpt_dir))
            if len(existing) > 0 and existing[-1] >= 50000:
                print(f"  Seed {seed}: already trained ({len(existing)} checkpoints, final step {existing[-1]}). Skipping.")
                continue

        print(f"\n  Training seed {seed} (experiment: {exp_name})...")
        t0 = time.time()

        cfg = make_config(
            experiment_name=exp_name,
            k=K, seed=seed, n_unique_b=N_UNIQUE_B,
            max_steps=50000, checkpoint_every=5000, eval_every=50,
            early_stop_frac=0.005,
        )

        torch.manual_seed(seed)
        np.random.seed(seed)

        mapping_data = get_mapping_data(seed=seed)
        _, history, _, _ = run_single_experiment(cfg, mapping_data=mapping_data)

        elapsed = time.time() - t0
        print(f"  Seed {seed}: done in {elapsed:.0f}s ({elapsed/60:.1f}min)")


def compute_seed_metrics(seed):
    """Compute core metrics table for one seed's final checkpoint."""
    print(f"\n  Computing metrics for seed {seed}...")

    cfg, tokenizer = get_cfg_and_tokenizer(seed=seed)
    mapping_data = get_mapping_data(seed=seed)
    loader = build_eval_loader(mapping_data, tokenizer, n_examples=512, batch_size=128, seed=seed)

    ckpt_dir = get_checkpoint_dir(seed)
    steps = sorted(list_checkpoints(ckpt_dir))
    final_step = steps[-1]
    model = load_model_at_step(cfg, tokenizer, final_step, checkpoint_dir=ckpt_dir)

    n_layers = N_LAYERS
    baseline_loss = compute_loss(model, loader)

    # 1. Full layer ablation
    ablation_deltas = {}
    for layer in range(n_layers):
        pre_hook = f"blocks.{layer}.hook_resid_pre"
        post_hook = f"blocks.{layer}.hook_resid_post"
        resid_pre_store = {}

        def make_capture(store, key):
            def hook_fn(value, hook):
                store[key] = value.clone()
                return value
            return hook_fn

        def make_bypass(store, key):
            def hook_fn(value, hook):
                return store[key]
            return hook_fn

        total_loss = 0.0
        n_tokens = 0
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)
                resid_pre_store.clear()
                logits = model.run_with_hooks(
                    input_ids,
                    fwd_hooks=[
                        (pre_hook, make_capture(resid_pre_store, "pre")),
                        (post_hook, make_bypass(resid_pre_store, "pre")),
                    ],
                )
                sl = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
                slb = labels[:, 1:].contiguous().view(-1)
                mask = slb != -100
                loss = F.cross_entropy(sl[mask], slb[mask])
                total_loss += loss.item() * mask.sum().item()
                n_tokens += mask.sum().item()

        ablation_deltas[layer] = (total_loss / max(n_tokens, 1)) - baseline_loss

    # 2. Logit lens P(correct)
    lens_probs = []
    layer_acts_all = [[] for _ in range(n_layers + 1)]
    all_y = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            tgt_starts = batch["target_start_positions"].to(DEVICE)
            bs = input_ids.shape[0]
            batch_idx = torch.arange(bs, device=DEVICE)
            _, cache = model.run_with_cache(input_ids)

            correct_tok = labels[batch_idx, tgt_starts]
            pred_pos = tgt_starts - 1
            resid_points = ([cache["blocks.0.hook_resid_pre"]]
                            + [cache[f"blocks.{i}.hook_resid_post"] for i in range(n_layers)])

            for lidx, resid in enumerate(resid_points):
                r = resid[batch_idx, pred_pos, :]
                layer_acts_all[lidx].append(r.cpu().numpy())

            all_y.append(correct_tok.cpu().numpy())

            # Logit lens at each layer
            step_probs = []
            for lidx, resid in enumerate(resid_points):
                r = resid[batch_idx, pred_pos, :]
                normed = model.ln_final(r)
                logits_lens = normed @ model.W_U + model.b_U
                probs = F.softmax(logits_lens, dim=-1)
                cp = probs[batch_idx, correct_tok]
                step_probs.append(cp.cpu().tolist())
            lens_probs.append(step_probs)
            del cache

    # Average logit lens
    lens_by_layer = []
    for lidx in range(n_layers + 1):
        all_vals = []
        for sp in lens_probs:
            all_vals.extend(sp[lidx])
        lens_by_layer.append(float(np.mean(all_vals)))

    # 3. Linear probes
    y = np.concatenate(all_y)
    probe_by_layer = []
    for lidx in range(n_layers + 1):
        X = np.concatenate(layer_acts_all[lidx])
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
            probe = LogisticRegression(max_iter=500, solver="lbfgs", C=1.0)
            probe.fit(X_tr, y_tr)
            probe_by_layer.append(float(probe.score(X_te, y_te)))
        except Exception:
            probe_by_layer.append(float('nan'))

    # 4. RSA (output-indexed)
    # Build similarity matrices
    n_ex = len(y)
    same_output = (y[:, None] == y[None, :]).astype(float)
    triu_idx = np.triu_indices(n_ex, k=1)
    same_output_flat = same_output[triu_idx]

    rsa_by_layer = []
    for lidx in range(n_layers + 1):
        X = np.concatenate(layer_acts_all[lidx])
        sim_matrix = cosine_similarity(X)
        sim_flat = sim_matrix[triu_idx]
        corr = np.corrcoef(sim_flat, same_output_flat)[0, 1]
        rsa_by_layer.append(float(corr))

    cleanup_model(model)

    result = {
        "seed": seed,
        "baseline_loss": baseline_loss,
        "ablation_deltas": ablation_deltas,
        "logit_lens": lens_by_layer,  # [embed, L0, L1, L2, L3]
        "probe_acc": probe_by_layer,
        "rsa_output": rsa_by_layer,
    }

    # Print table
    layer_labels = ["Embed", "L0", "L1", "L2", "L3"]
    print(f"    Baseline: {baseline_loss:.4f}")
    print(f"    {'Layer':<8s} {'Abl Δ':>8s} {'Lens P':>8s} {'Probe':>8s} {'RSA':>8s}")
    print(f"    " + "-" * 36)
    for lidx, lbl in enumerate(layer_labels):
        abl_d = ablation_deltas.get(lidx, 0.0) if lidx > 0 else 0.0  # embed has no ablation
        if lidx == 0:
            abl_str = "N/A"
        else:
            abl_str = f"{ablation_deltas[lidx-1]:+.3f}"
        print(f"    {lbl:<8s} {abl_str:>8s} {lens_by_layer[lidx]:>8.3f} {probe_by_layer[lidx]:>8.3f} {rsa_by_layer[lidx]:>8.3f}")

    return result


def experiment1_multiseed(skip_training=False):
    """Full multi-seed experiment: train + compute + compare."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Multi-Seed Robustness")
    print("=" * 70)

    if not skip_training:
        train_new_seeds()

    # Check which seeds are available
    available_seeds = []
    for seed in ALL_SEEDS:
        ckpt_dir = get_checkpoint_dir(seed)
        if ckpt_dir.exists():
            existing = sorted(list_checkpoints(ckpt_dir))
            if len(existing) > 0:
                available_seeds.append(seed)
                print(f"  Seed {seed}: {len(existing)} checkpoints available (final step {existing[-1]})")
            else:
                print(f"  Seed {seed}: checkpoint dir exists but empty")
        else:
            print(f"  Seed {seed}: NOT available")

    if len(available_seeds) < 2:
        print("  WARNING: Need at least 2 seeds for comparison. Only have:", available_seeds)
        if len(available_seeds) == 1:
            print("  Computing metrics for the single available seed anyway...")
            result = compute_seed_metrics(available_seeds[0])
            return {"seed_results": {available_seeds[0]: result}, "available_seeds": available_seeds}
        return {"seed_results": {}, "available_seeds": []}

    # Compute metrics for each seed
    seed_results = {}
    for seed in available_seeds:
        seed_results[seed] = compute_seed_metrics(seed)

    n_seeds = len(available_seeds)

    # ── Cross-seed statistics ──
    print("\n\n  CROSS-SEED STATISTICS:")
    # Ablation (layers 0-3)
    layer_labels_abl = ["L0", "L1", "L2", "L3"]
    metrics = ["ablation_deltas", "logit_lens", "probe_acc", "rsa_output"]
    metric_names = ["Ablation Δ", "Logit Lens", "Probe Acc", "RSA(output)"]

    # Build the table: metric × layer → [values across seeds]
    print(f"\n  {'Metric':<14s} {'Layer':<8s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s}")
    print("  " + "-" * 50)

    summary_table = {}
    for mi, (metric, mname) in enumerate(zip(metrics, metric_names)):
        # For ablation: layers 0-3 (4 values)
        # For lens/probe/rsa: layers 0-4 (embed + 4 layers = 5 values)
        if metric == "ablation_deltas":
            n_entries = 4
            labels = layer_labels_abl
            def get_val(sr, idx):
                return sr[metric][idx]
        else:
            n_entries = 5
            labels = ["Embed", "L0", "L1", "L2", "L3"]
            def get_val(sr, idx):
                return sr[metric][idx]

        for idx in range(n_entries):
            vals = []
            for seed in available_seeds:
                try:
                    v = get_val(seed_results[seed], idx)
                    if not (np.isnan(v) if isinstance(v, float) else False):
                        vals.append(v)
                except (KeyError, IndexError):
                    pass

            if vals:
                mean = np.mean(vals)
                std = np.std(vals)
                mn = np.min(vals)
                mx = np.max(vals)
                summary_table[(mname, labels[idx])] = {"mean": mean, "std": std, "min": mn, "max": mx, "vals": vals}
                print(f"  {mname:<14s} {labels[idx]:<8s} {mean:>8.3f} {std:>8.3f} {mn:>8.3f} {mx:>8.3f}")

    # Discrepancy ratio
    print("\n  DISCREPANCY RATIO (L0 ablation Δ / L2 ablation Δ):")
    ratios = []
    for seed in available_seeds:
        sr = seed_results[seed]
        l0_abl = sr["ablation_deltas"][0]
        l2_abl = sr["ablation_deltas"][2]
        ratio = l0_abl / l2_abl if l2_abl != 0 else float('inf')
        ratios.append(ratio)
        print(f"    Seed {seed}: {l0_abl:.3f} / {l2_abl:.3f} = {ratio:.2f}×")

    print(f"    Mean ± std: {np.mean(ratios):.2f} ± {np.std(ratios):.2f}×")
    print(f"    Range: [{np.min(ratios):.2f}, {np.max(ratios):.2f}]")

    # Check qualitative consistency
    all_consistent = True
    for seed in available_seeds:
        sr = seed_results[seed]
        # Check: is ablation ordering L0 > L1 > L2 > L3?
        abls = [sr["ablation_deltas"][i] for i in range(4)]
        if abls[0] < abls[2]:
            print(f"    WARNING: Seed {seed} has L0 ablation ({abls[0]:.3f}) < L2 ({abls[2]:.3f})!")
            all_consistent = False
        # Check: is probe/lens highest at L2 or L3?
        lens = sr["logit_lens"]
        if lens[3] < lens[1]:  # L2 < L0
            print(f"    WARNING: Seed {seed} has higher logit lens at L0 than L2!")
            all_consistent = False

    print(f"\n  Qualitative pattern holds across {n_seeds}/{n_seeds} seeds: {'YES' if all_consistent else 'NO'}")

    # ── Plot: bar chart with error bars ──
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    for ax_idx, (metric, mname) in enumerate(zip(metrics, metric_names)):
        ax = axes[ax_idx]
        if metric == "ablation_deltas":
            labels = layer_labels_abl
            n_entries = 4
        else:
            labels = ["Embed", "L0", "L1", "L2", "L3"]
            n_entries = 5

        means = []
        stds = []
        for idx in range(n_entries):
            vals = []
            for seed in available_seeds:
                try:
                    if metric == "ablation_deltas":
                        v = seed_results[seed][metric][idx]
                    else:
                        v = seed_results[seed][metric][idx]
                    if not (np.isnan(v) if isinstance(v, float) else False):
                        vals.append(v)
                except (KeyError, IndexError):
                    pass
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals) if vals else 0)

        x = np.arange(n_entries)
        ax.bar(x, means, yerr=stds, capsize=4, color="steelblue", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(mname, fontsize=12)
        ax.set_ylabel(mname if ax_idx == 0 else "", fontsize=10)

    fig.suptitle(f"Multi-Seed Robustness (N={n_seeds} seeds, K={K})", fontsize=14)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "reviewer_exp1_multiseed.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved reviewer_exp1_multiseed.png")

    return {
        "seed_results": {s: seed_results[s] for s in available_seeds},
        "available_seeds": available_seeds,
        "discrepancy_ratios": ratios,
        "all_consistent": all_consistent,
        "summary_table": {str(k): v for k, v in summary_table.items()},
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training new seeds, use existing checkpoints only")
    parser.add_argument("--only", type=int, default=None,
                        help="Run only experiment N (1, 2, or 3)")
    args = parser.parse_args()

    print("=" * 70)
    print("REVIEWER RESPONSE EXPERIMENTS")
    print(f"K={K}, Device={DEVICE}")
    print("=" * 70)

    results = {}

    # Run in priority order: Exp2 → Exp3 → Exp1
    if args.only is None or args.only == 2:
        results["exp2"] = experiment2_sufficiency()

    if args.only is None or args.only == 3:
        results["exp3"] = experiment3_ablation_variants()

    if args.only is None or args.only == 1:
        results["exp1"] = experiment1_multiseed(skip_training=args.skip_training)

    # ── Save results ──
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(x) for x in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return str(obj)
        return obj

    with open(ANALYSIS_DIR / "reviewer_experiments.json", "w") as f:
        json.dump(make_serializable(results), f, indent=2)
    print(f"\nSaved reviewer_experiments.json")

    # ── SUMMARY ──
    print("\n\n" + "=" * 70)
    print("=== REVIEWER RESPONSE SUMMARY ===")
    print("=" * 70)

    if "exp1" in results:
        exp1 = results["exp1"]
        n_seeds = len(exp1.get("available_seeds", []))
        ratios = exp1.get("discrepancy_ratios", [])
        consistent = exp1.get("all_consistent", False)
        if ratios:
            print(f"\nMULTI-SEED ROBUSTNESS:")
            print(f"  Discrepancy ratio (L0 Δ / L2 Δ): {np.mean(ratios):.2f} ± {np.std(ratios):.2f}×")
            print(f"  Qualitative pattern holds across {n_seeds}/{n_seeds} seeds: {'YES' if consistent else 'NO'}")
        else:
            print(f"\nMULTI-SEED ROBUSTNESS: Only {n_seeds} seed(s) available, cannot compute variance.")

    if "exp2" in results:
        exp2 = results["exp2"]
        fr = exp2["full_flip_rates"]
        print(f"\nCAUSAL SUFFICIENCY:")
        print(f"  L0 full-resid patch flip rate: {fr[0]:.4f}")
        print(f"  L1 full-resid patch flip rate: {fr[1]:.4f}")
        print(f"  L2 full-resid patch flip rate: {fr[2]:.4f}")
        print(f"  L3 full-resid patch flip rate: {fr[3]:.4f}")
        # Determine if sufficiency confirms L2
        l2_sufficient = fr[2] > 0.8
        l0_not_sufficient = fr[0] < fr[2]
        if l2_sufficient and l0_not_sufficient:
            suf_verdict = "YES"
        elif l2_sufficient:
            suf_verdict = "PARTIAL (L0 also sufficient)"
        else:
            suf_verdict = "NO (L2 not sufficient either)"
        print(f"  Sufficiency confirms computation at L2: {suf_verdict}")

    if "exp3" in results:
        exp3 = results["exp3"]
        robust = exp3.get("ordering_robust", False)
        print(f"\nABLATION SENSITIVITY:")
        print(f"  Qualitative ordering robust across all variants: {'YES' if robust else 'NO'}")
        for method in ["zero", "mean", "resample", "noise"]:
            if method in exp3:
                vals = [exp3[method][i] for i in range(4)]
                print(f"    {method}: " + " > ".join(f"L{i}({vals[i]:+.1f})" for i in np.argsort(vals)[::-1]))

    # Overall verdict
    verdicts = []
    if "exp1" in results and results["exp1"].get("all_consistent", False):
        verdicts.append("multi-seed robust")
    if "exp2" in results:
        fr = results["exp2"]["full_flip_rates"]
        if fr[2] > 0.8:
            verdicts.append("sufficiency confirmed")
    if "exp3" in results and results["exp3"].get("ordering_robust", False):
        verdicts.append("ablation-method robust")

    if len(verdicts) >= 2:
        print(f"\nOVERALL: Paper's central claim is SUPPORTED by robustness checks ({', '.join(verdicts)}).")
    elif len(verdicts) == 1:
        print(f"\nOVERALL: Paper's central claim is PARTIALLY SUPPORTED ({', '.join(verdicts)}).")
    else:
        print(f"\nOVERALL: Paper's central claim needs further investigation.")

    print("\nDone. Plots in analysis_outputs/")


if __name__ == "__main__":
    main()
