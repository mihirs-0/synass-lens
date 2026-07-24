#!/usr/bin/env python
"""
Deep mechanistic interpretability: activation patching, logit lens, linear probes.

Experiment 1: Activation patching on z-token (causal flow)
Experiment 2: Logit lens across layers and training (where correct answer appears)
Experiment 3: Linear probes across layers and training (where info is linearly decodable)

Reuses infrastructure from mechinterp_analysis.py.
"""

import sys
import json
import math
import pickle
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import create_tokenizer_from_config, collate_fn
from src.data.dataset import generate_mappings, MappingData, DisambiguationDataset
from src.model import create_model_from_config
from src.training.checkpoint import load_checkpoint, list_checkpoints
from scripts.experiment_helpers import make_config

# ── Config ──────────────────────────────────────────────────────────────────

EXPERIMENT = "landauer_dense_k10"
K = 10
LOG_K = math.log(K)
N_UNIQUE_B = 1000
SEED = 42

OUTPUTS_DIR = Path("outputs")
CHECKPOINT_DIR = OUTPUTS_DIR / EXPERIMENT / "checkpoints"
ANALYSIS_DIR = Path("analysis_outputs")
ANALYSIS_DIR.mkdir(exist_ok=True)

DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")

# ── Shared helpers ──────────────────────────────────────────────────────────

def get_cfg_and_tokenizer():
    cfg = make_config(experiment_name="mechinterp_deep", k=K, seed=SEED, n_unique_b=N_UNIQUE_B)
    tokenizer = create_tokenizer_from_config(cfg)
    return cfg, tokenizer


def get_mapping_data():
    return generate_mappings(
        n_unique_b=N_UNIQUE_B, k=K, b_length=6, a_length=4, z_length=2,
        vocab_chars="abcdefghijklmnopqrstuvwxyz0123456789",
        seed=SEED, task="bz_to_a",
        enforce_unique_a_first_char_per_b=True, disambiguation_prefix_length=1,
    )


def load_model_at_step(cfg, tokenizer, step):
    model = create_model_from_config(cfg, tokenizer)
    model = model.to(DEVICE)
    load_checkpoint(model, None, CHECKPOINT_DIR, step=step)
    model.eval()
    return model


def get_checkpoint_steps():
    return sorted(list_checkpoints(CHECKPOINT_DIR))


def subsample_steps(steps, every_n=5):
    """Subsample checkpoints, always including first and last."""
    sub = steps[::every_n]
    if steps[0] not in sub:
        sub = [steps[0]] + sub
    if steps[-1] not in sub:
        sub.append(steps[-1])
    return sorted(set(sub))


def find_tau(steps_list):
    """Find transition step from training history."""
    hist_path = OUTPUTS_DIR / EXPERIMENT / "training_history.json"
    with open(hist_path) as f:
        h = json.load(f)
    cl = h.get("candidate_loss", h.get("train_loss", []))
    for s, v in zip(h["steps"], cl):
        if v is not None and v < 0.5 * LOG_K:
            return s
    return steps_list[len(steps_list) // 2]


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Activation Patching on Z-Token
# ═══════════════════════════════════════════════════════════════════════════

def build_paired_inputs(mapping_data, tokenizer, n_pairs=256):
    """
    Build pairs (clean, corrupted) that share the same B but have different z/A.
    For each B-group with K targets, pair the first target with the second.
    """
    pairs = []
    for b_str, targets in mapping_data.mappings.items():
        if len(targets) < 2:
            continue
        z_clean, a_clean = targets[0]
        z_corrupt, a_corrupt = targets[1]
        clean = tokenizer.encode_sequence(b_str, z_clean, a_clean, task="bz_to_a")
        corrupt = tokenizer.encode_sequence(b_str, z_corrupt, a_corrupt, task="bz_to_a")
        # Store the first target token IDs for logit comparison
        clean["first_target_id"] = tokenizer.encode(a_clean)[0]
        corrupt["first_target_id"] = tokenizer.encode(a_corrupt)[0]
        pairs.append((clean, corrupt))
        if len(pairs) >= n_pairs:
            break
    return pairs


def run_activation_patching(cfg, tokenizer, mapping_data):
    """Patch z-token activations from corrupted into clean run at each layer."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Activation Patching on Z-Token")
    print("=" * 70)

    steps = get_checkpoint_steps()
    final_step = steps[-1]
    tau = find_tau(steps)

    pairs = build_paired_inputs(mapping_data, tokenizer, n_pairs=256)
    print(f"  Built {len(pairs)} clean/corrupted pairs")

    n_layers = cfg.model.n_layers

    # Run on final, mid-transition, and pre-transition checkpoints
    test_steps = [
        min(steps, key=lambda s: abs(s - tau // 4)),  # pre
        min(steps, key=lambda s: abs(s - tau)),         # mid
        final_step,                                     # post
    ]
    test_labels = ["Pre-transition", "Mid-transition", "Post-transition (final)"]

    all_results = {}

    for step, label in zip(test_steps, test_labels):
        print(f"\n  --- {label} (step {step}) ---")
        model = load_model_at_step(cfg, tokenizer, step)

        # Process in mini-batches for memory
        batch_size = 64
        all_metrics_resid = [[] for _ in range(n_layers + 1)]  # patching at each layer boundary
        all_metrics_head = [[[] for _ in range(model.cfg.n_heads)] for _ in range(1)]  # L0 heads only

        for start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[start:start + batch_size]
            bs = len(batch_pairs)

            # Stack clean and corrupted inputs
            max_len = max(
                max(len(p[0]["input_ids"]) for p in batch_pairs),
                max(len(p[1]["input_ids"]) for p in batch_pairs),
            )
            clean_ids = torch.zeros(bs, max_len, dtype=torch.long)
            corrupt_ids = torch.zeros(bs, max_len, dtype=torch.long)
            z_starts = []
            z_ends = []
            tgt_starts = []
            clean_first_tok = []
            corrupt_first_tok = []

            for i, (c, x) in enumerate(batch_pairs):
                clen = len(c["input_ids"])
                xlen = len(x["input_ids"])
                clean_ids[i, :clen] = c["input_ids"]
                corrupt_ids[i, :xlen] = x["input_ids"]
                z_starts.append(c["z_position"])
                z_ends.append(c["z_end_position"])
                tgt_starts.append(c["target_start_position"])
                clean_first_tok.append(c["first_target_id"])
                corrupt_first_tok.append(x["first_target_id"])

            clean_ids = clean_ids.to(DEVICE)
            corrupt_ids = corrupt_ids.to(DEVICE)
            clean_tok = torch.tensor(clean_first_tok, device=DEVICE)
            corrupt_tok = torch.tensor(corrupt_first_tok, device=DEVICE)
            tgt_pos = torch.tensor(tgt_starts, device=DEVICE)

            with torch.no_grad():
                # Clean run: get logits and cache
                clean_logits, clean_cache = model.run_with_cache(clean_ids)
                # Corrupted run: get cache
                _, corrupt_cache = model.run_with_cache(corrupt_ids)

                # Extract logit diff at first target position
                # Logits at position tgt-1 predict token at tgt
                pred_pos = tgt_pos - 1
                batch_idx = torch.arange(bs, device=DEVICE)
                clean_logits_at_tgt = clean_logits[batch_idx, pred_pos, :]
                clean_logit_diff = (clean_logits_at_tgt[batch_idx, clean_tok]
                                    - clean_logits_at_tgt[batch_idx, corrupt_tok])

                # Corrupted logit diff
                corrupt_logits, _ = model.run_with_cache(corrupt_ids)
                corrupt_logits_at_tgt = corrupt_logits[batch_idx, pred_pos, :]
                corrupt_logit_diff = (corrupt_logits_at_tgt[batch_idx, clean_tok]
                                      - corrupt_logits_at_tgt[batch_idx, corrupt_tok])

                # Patch at each residual stream layer boundary
                # "Before L0" = patch hook_resid_pre of layer 0 (after embedding)
                # "After L_i" = patch hook_resid_post of layer i
                patch_points = (
                    [("blocks.0.hook_resid_pre", "Before L0")]
                    + [(f"blocks.{i}.hook_resid_post", f"After L{i}") for i in range(n_layers)]
                )

                for pidx, (hook_name, plabel) in enumerate(patch_points):
                    corrupt_act = corrupt_cache[hook_name].clone()

                    def make_patch_hook(z_s, z_e, c_act):
                        def hook_fn(value, hook):
                            for i in range(value.shape[0]):
                                value[i, z_s[i]:z_e[i], :] = c_act[i, z_s[i]:z_e[i], :]
                            return value
                        return hook_fn

                    patched_logits = model.run_with_hooks(
                        clean_ids,
                        fwd_hooks=[(hook_name, make_patch_hook(z_starts, z_ends, corrupt_act))],
                    )
                    patched_logits_at_tgt = patched_logits[batch_idx, pred_pos, :]
                    patched_logit_diff = (patched_logits_at_tgt[batch_idx, clean_tok]
                                          - patched_logits_at_tgt[batch_idx, corrupt_tok])

                    # Normalized metric: 0 = no flip, 1 = full flip
                    denom = clean_logit_diff - corrupt_logit_diff
                    metric = (clean_logit_diff - patched_logit_diff) / (denom + 1e-8)
                    all_metrics_resid[pidx].extend(metric.cpu().tolist())

                # Head-level patching at Layer 0
                # hook_z is the per-head output (batch, seq, n_heads, d_head)
                corrupt_head_out = corrupt_cache["blocks.0.attn.hook_z"].clone()

                for head in range(model.cfg.n_heads):
                    def make_head_patch(h, z_s, z_e, c_act):
                        def hook_fn(value, hook):
                            for i in range(value.shape[0]):
                                value[i, z_s[i]:z_e[i], h, :] = c_act[i, z_s[i]:z_e[i], h, :]
                            return value
                        return hook_fn

                    patched_logits = model.run_with_hooks(
                        clean_ids,
                        fwd_hooks=[("blocks.0.attn.hook_z",
                                    make_head_patch(head, z_starts, z_ends, corrupt_head_out))],
                    )
                    patched_logits_at_tgt = patched_logits[batch_idx, pred_pos, :]
                    patched_logit_diff = (patched_logits_at_tgt[batch_idx, clean_tok]
                                          - patched_logits_at_tgt[batch_idx, corrupt_tok])
                    metric = (clean_logit_diff - patched_logit_diff) / (denom + 1e-8)
                    all_metrics_head[0][head].extend(metric.cpu().tolist())

                del clean_cache, corrupt_cache

        # Aggregate
        patch_labels = ["Before L0"] + [f"After L{i}" for i in range(n_layers)]
        resid_means = [np.mean(m) for m in all_metrics_resid]
        resid_stds = [np.std(m) for m in all_metrics_resid]
        head_means = [np.mean(m) for m in all_metrics_head[0]]
        head_stds = [np.std(m) for m in all_metrics_head[0]]

        print(f"  Residual stream patching (normalized metric, 0=no flip, 1=full flip):")
        for lbl, mn, sd in zip(patch_labels, resid_means, resid_stds):
            print(f"    {lbl:<14s}: {mn:.4f} +/- {sd:.4f}")
        print(f"  L0 head-level patching:")
        for h, (mn, sd) in enumerate(zip(head_means, head_stds)):
            print(f"    L0H{h}: {mn:.4f} +/- {sd:.4f}")

        all_results[step] = {
            "label": label,
            "resid_labels": patch_labels,
            "resid_means": resid_means,
            "resid_stds": resid_stds,
            "head_means": head_means,
            "head_stds": head_stds,
        }

        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # ── Plot: Residual stream patching (final checkpoint) ──
    final_res = all_results[final_step]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(final_res["resid_labels"]))
    ax1.bar(x, final_res["resid_means"], yerr=final_res["resid_stds"],
            capsize=3, color="steelblue", alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(final_res["resid_labels"], fontsize=10)
    ax1.set_ylabel("Patching metric (0=no flip, 1=full flip)", fontsize=11)
    ax1.set_title("Z-Token Patching: Residual Stream", fontsize=12)
    ax1.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax1.axhline(y=0.0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_ylim(-0.1, 1.3)

    x2 = np.arange(len(final_res["head_means"]))
    ax2.bar(x2, final_res["head_means"], yerr=final_res["head_stds"],
            capsize=3, color=["tab:blue", "tab:orange", "tab:green", "tab:red"], alpha=0.8)
    ax2.set_xticks(x2)
    ax2.set_xticklabels([f"L0H{h}" for h in range(len(final_res["head_means"]))], fontsize=10)
    ax2.set_ylabel("Patching metric", fontsize=11)
    ax2.set_title("Z-Token Patching: L0 Heads", fontsize=12)
    ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax2.axhline(y=0.0, color="gray", linestyle="--", alpha=0.5)

    fig.suptitle(f"Activation Patching (K={K}, final checkpoint)", fontsize=13)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "exp1_activation_patching.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved exp1_activation_patching.png")

    # Multi-checkpoint comparison
    if len(all_results) > 1:
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["tab:blue", "tab:orange", "tab:green"]
        for i, (step, res) in enumerate(sorted(all_results.items())):
            offset = (i - 1) * 0.25
            x = np.arange(len(res["resid_labels"])) + offset
            ax.bar(x, res["resid_means"], 0.22, yerr=res["resid_stds"],
                   capsize=2, color=colors[i], alpha=0.7, label=res["label"])
        ax.set_xticks(np.arange(len(final_res["resid_labels"])))
        ax.set_xticklabels(final_res["resid_labels"], fontsize=10)
        ax.set_ylabel("Patching metric", fontsize=11)
        ax.set_title("Z-Patching Across Training Phases", fontsize=12)
        ax.legend(fontsize=9)
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(ANALYSIS_DIR / "exp1_patching_across_training.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved exp1_patching_across_training.png")

    return all_results


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Logit Lens Across Layers and Training
# ═══════════════════════════════════════════════════════════════════════════

def run_logit_lens(cfg, tokenizer, mapping_data):
    """Logit lens: where does the correct answer become decodable?"""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Logit Lens Across Layers and Training")
    print("=" * 70)

    # Build eval data
    ds = DisambiguationDataset(
        mapping_data=mapping_data, tokenizer=tokenizer,
        split="train", probe_fraction=0.0, seed=SEED, task="bz_to_a",
    )
    ds.tokenized = ds.tokenized[:512]
    ds.examples = ds.examples[:512]
    loader = torch.utils.data.DataLoader(
        ds, batch_size=128, shuffle=False, collate_fn=collate_fn, num_workers=0,
    )

    n_layers = cfg.model.n_layers
    steps = get_checkpoint_steps()
    step_subset = subsample_steps(steps, every_n=10)  # ~50 checkpoints
    tau = find_tau(steps)

    # Results: step -> layer -> metrics
    all_results = {}

    for si, step in enumerate(step_subset):
        if si % 10 == 0:
            print(f"  Checkpoint {si+1}/{len(step_subset)}: step {step}")

        model = load_model_at_step(cfg, tokenizer, step)

        # Collect per-layer logit lens metrics
        # Layers: "embed" (after embed, before L0), L0..L3
        layer_correct_probs = [[] for _ in range(n_layers + 1)]
        layer_top1_correct = [[] for _ in range(n_layers + 1)]
        layer_ranks = [[] for _ in range(n_layers + 1)]

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)
                tgt_starts = batch["target_start_positions"].to(DEVICE)
                bs = input_ids.shape[0]
                batch_idx = torch.arange(bs, device=DEVICE)

                _, cache = model.run_with_cache(input_ids)

                # Get correct first target token
                correct_tok = labels[batch_idx, tgt_starts]  # (bs,)

                # Layer -1 (after embedding, before L0)
                resid_points = (
                    [cache["blocks.0.hook_resid_pre"]]  # after embed
                    + [cache[f"blocks.{i}.hook_resid_post"] for i in range(n_layers)]
                )

                pred_pos = tgt_starts - 1  # position that predicts first target token

                for lidx, resid in enumerate(resid_points):
                    # Extract at prediction position
                    resid_at_pos = resid[batch_idx, pred_pos, :]  # (bs, d_model)
                    # Apply final layer norm + unembed
                    normed = model.ln_final(resid_at_pos)
                    logits = normed @ model.W_U + model.b_U  # (bs, d_vocab)

                    probs = F.softmax(logits, dim=-1)
                    correct_prob = probs[batch_idx, correct_tok]  # (bs,)
                    top1 = logits.argmax(dim=-1)  # (bs,)
                    top1_correct = (top1 == correct_tok).float()

                    # Rank of correct token (1-indexed)
                    ranks = (logits >= logits[batch_idx, correct_tok].unsqueeze(-1)).sum(dim=-1).float()

                    layer_correct_probs[lidx].extend(correct_prob.cpu().tolist())
                    layer_top1_correct[lidx].extend(top1_correct.cpu().tolist())
                    layer_ranks[lidx].extend(ranks.cpu().tolist())

                del cache

        step_result = {}
        layer_labels = ["After embed"] + [f"After L{i}" for i in range(n_layers)]
        for lidx, lbl in enumerate(layer_labels):
            step_result[lbl] = {
                "mean_prob": float(np.mean(layer_correct_probs[lidx])),
                "mean_top1_acc": float(np.mean(layer_top1_correct[lidx])),
                "mean_rank": float(np.mean(layer_ranks[lidx])),
            }
        all_results[step] = step_result

        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    layer_labels = ["After embed"] + [f"After L{i}" for i in range(n_layers)]

    # ── Plot: Heatmap (layer x step -> correct prob) ──
    sorted_steps = sorted(all_results.keys())
    prob_matrix = np.zeros((len(layer_labels), len(sorted_steps)))
    for j, s in enumerate(sorted_steps):
        for i, lbl in enumerate(layer_labels):
            prob_matrix[i, j] = all_results[s][lbl]["mean_prob"]

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(prob_matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_yticks(range(len(layer_labels)))
    ax.set_yticklabels(layer_labels, fontsize=10)
    # Show every 10th step label
    tick_idx = list(range(0, len(sorted_steps), max(1, len(sorted_steps) // 10)))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([str(sorted_steps[i]) for i in tick_idx], fontsize=8, rotation=45)
    ax.set_xlabel("Training step", fontsize=11)
    ax.set_title(f"Logit Lens: Correct Token Probability (K={K})", fontsize=13)
    fig.colorbar(im, ax=ax, label="P(correct)")
    # Mark tau
    tau_idx = min(range(len(sorted_steps)), key=lambda i: abs(sorted_steps[i] - tau))
    ax.axvline(x=tau_idx, color="red", linestyle="--", alpha=0.7, linewidth=1)
    ax.text(tau_idx + 0.5, -0.3, f"τ={tau}", color="red", fontsize=8)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "exp2_logit_lens_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved exp2_logit_lens_heatmap.png")

    # ── Plot: Line plot at 3 timepoints ──
    pre_step = min(sorted_steps, key=lambda s: abs(s - tau // 4))
    mid_step = min(sorted_steps, key=lambda s: abs(s - tau))
    post_step = sorted_steps[-1]
    timepoints = [(pre_step, "Pre-transition"), (mid_step, "Mid-transition"), (post_step, "Post-transition")]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for step, label in timepoints:
        probs = [all_results[step][lbl]["mean_prob"] for lbl in layer_labels]
        accs = [all_results[step][lbl]["mean_top1_acc"] for lbl in layer_labels]
        ax1.plot(range(len(layer_labels)), probs, "o-", label=f"{label} (step {step})", markersize=5)
        ax2.plot(range(len(layer_labels)), accs, "o-", label=f"{label} (step {step})", markersize=5)

    for ax, ylabel, title in [
        (ax1, "P(correct token)", "Correct Token Probability by Layer"),
        (ax2, "Top-1 Accuracy", "Top-1 Accuracy by Layer"),
    ]:
        ax.set_xticks(range(len(layer_labels)))
        ax.set_xticklabels(layer_labels, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.axhline(y=1.0 / K, color="gray", linestyle=":", alpha=0.5, label=f"Chance (1/K={1/K:.2f})")
        ax.legend(fontsize=8)
        ax.set_ylim(-0.05, 1.05)

    fig.suptitle(f"Logit Lens at Three Timepoints (K={K})", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "exp2_logit_lens_timepoints.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved exp2_logit_lens_timepoints.png")

    return all_results, layer_labels


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Linear Probes Across Layers and Training
# ═══════════════════════════════════════════════════════════════════════════

def run_linear_probes(cfg, tokenizer, mapping_data):
    """Train linear probes to predict correct target from residual stream."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Linear Probes Across Layers and Training")
    print("=" * 70)

    # Build a larger eval set for probe training
    ds = DisambiguationDataset(
        mapping_data=mapping_data, tokenizer=tokenizer,
        split="train", probe_fraction=0.0, seed=SEED, task="bz_to_a",
    )
    ds.tokenized = ds.tokenized[:1024]
    ds.examples = ds.examples[:1024]
    loader = torch.utils.data.DataLoader(
        ds, batch_size=128, shuffle=False, collate_fn=collate_fn, num_workers=0,
    )
    print(f"  Using {len(ds)} examples for probes")

    n_layers = cfg.model.n_layers
    steps = get_checkpoint_steps()
    step_subset = subsample_steps(steps, every_n=10)
    tau = find_tau(steps)
    layer_labels = ["After embed"] + [f"After L{i}" for i in range(n_layers)]

    # ── Part A: Probes across layers and training ──
    all_results = {}

    for si, step in enumerate(step_subset):
        if si % 10 == 0:
            print(f"  Checkpoint {si+1}/{len(step_subset)}: step {step}")

        model = load_model_at_step(cfg, tokenizer, step)

        # Collect activations at each layer, at the first-target prediction position
        layer_activations = [[] for _ in range(n_layers + 1)]
        all_labels = []

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

                resid_points = (
                    [cache["blocks.0.hook_resid_pre"]]
                    + [cache[f"blocks.{i}.hook_resid_post"] for i in range(n_layers)]
                )

                for lidx, resid in enumerate(resid_points):
                    act = resid[batch_idx, pred_pos, :].cpu().numpy()
                    layer_activations[lidx].append(act)

                all_labels.append(correct_tok.cpu().numpy())
                del cache

        # Concatenate
        y = np.concatenate(all_labels)
        step_result = {}

        for lidx, lbl in enumerate(layer_labels):
            X = np.concatenate(layer_activations[lidx])
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            probe = LogisticRegression(
                max_iter=1000, solver="lbfgs", C=1.0
            )
            probe.fit(X_train, y_train)
            test_acc = probe.score(X_test, y_test)
            # Mean predicted probability for correct class
            test_probs = probe.predict_proba(X_test)
            # Map token IDs to probe class indices
            class_to_idx = {c: i for i, c in enumerate(probe.classes_)}
            correct_indices = np.array([class_to_idx[yt] for yt in y_test])
            correct_probs = test_probs[np.arange(len(y_test)), correct_indices]
            mean_correct_prob = float(np.mean(correct_probs))

            step_result[lbl] = {
                "accuracy": float(test_acc),
                "mean_correct_prob": mean_correct_prob,
                "n_classes": int(len(np.unique(y))),
            }

        all_results[step] = step_result

        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # ── Part B: Position-specific probes at final checkpoint ──
    print(f"\n  Part B: Position-specific probes at final checkpoint...")
    final_step = steps[-1]
    model = load_model_at_step(cfg, tokenizer, final_step)

    # Collect activations at different positions, after L0
    position_labels = ["BOS (pos 0)", "B1 (pos 1)", "z1 (pos 8)", "z2 (pos 9)",
                       "SEP2 (pos 10)", "pred_pos (pos 10)"]
    position_indices = [0, 1, 8, 9, 10, None]  # None = pred_pos (target_start-1)
    position_activations = [[] for _ in position_labels]
    pos_labels_y = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            tgt_starts = batch["target_start_positions"].to(DEVICE)
            bs = input_ids.shape[0]
            batch_idx = torch.arange(bs, device=DEVICE)

            _, cache = model.run_with_cache(input_ids)
            correct_tok = labels[batch_idx, tgt_starts]

            resid_after_l0 = cache["blocks.0.hook_resid_post"]

            for pidx, pos in enumerate(position_indices):
                if pos is None:
                    p = tgt_starts - 1
                else:
                    p = torch.full((bs,), pos, device=DEVICE, dtype=torch.long)
                act = resid_after_l0[batch_idx, p, :].cpu().numpy()
                position_activations[pidx].append(act)

            pos_labels_y.append(correct_tok.cpu().numpy())
            del cache

    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    y_pos = np.concatenate(pos_labels_y)
    position_results = {}

    for pidx, plbl in enumerate(position_labels):
        X = np.concatenate(position_activations[pidx])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_pos, test_size=0.2, random_state=42, stratify=y_pos
        )
        probe = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
        probe.fit(X_train, y_train)
        acc = probe.score(X_test, y_test)
        position_results[plbl] = {"accuracy": float(acc)}
        print(f"    {plbl}: {acc:.4f}")

    # ── Plots ──

    # Heatmap: layer x step -> probe accuracy
    sorted_steps = sorted(all_results.keys())
    acc_matrix = np.zeros((len(layer_labels), len(sorted_steps)))
    for j, s in enumerate(sorted_steps):
        for i, lbl in enumerate(layer_labels):
            acc_matrix[i, j] = all_results[s][lbl]["accuracy"]

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(acc_matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_yticks(range(len(layer_labels)))
    ax.set_yticklabels(layer_labels, fontsize=10)
    tick_idx = list(range(0, len(sorted_steps), max(1, len(sorted_steps) // 10)))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([str(sorted_steps[i]) for i in tick_idx], fontsize=8, rotation=45)
    ax.set_xlabel("Training step", fontsize=11)
    ax.set_title(f"Linear Probe Accuracy: Layer x Training Step (K={K})", fontsize=13)
    fig.colorbar(im, ax=ax, label="Probe accuracy")
    tau_idx = min(range(len(sorted_steps)), key=lambda i: abs(sorted_steps[i] - tau))
    ax.axvline(x=tau_idx, color="red", linestyle="--", alpha=0.7, linewidth=1)
    ax.text(tau_idx + 0.5, -0.3, f"τ={tau}", color="red", fontsize=8)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "exp3_probe_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved exp3_probe_heatmap.png")

    # Bar chart: probe accuracy by layer at final checkpoint
    final_accs = [all_results[sorted_steps[-1]][lbl]["accuracy"] for lbl in layer_labels]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["tab:gray"] + [f"C{i}" for i in range(n_layers)]
    ax.bar(range(len(layer_labels)), final_accs, color=colors, alpha=0.8)
    ax.axhline(y=1.0 / K, color="red", linestyle="--", alpha=0.6, label=f"Chance (1/K={1/K:.2f})")
    ax.set_xticks(range(len(layer_labels)))
    ax.set_xticklabels(layer_labels, fontsize=10)
    ax.set_ylabel("Probe accuracy", fontsize=11)
    ax.set_title(f"Linear Probe Accuracy by Layer (Final Checkpoint, K={K})", fontsize=12)
    ax.legend()
    ax.set_ylim(0, 1.05)
    for i, v in enumerate(final_accs):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "exp3_probe_by_layer.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved exp3_probe_by_layer.png")

    # Bar chart: position-specific probes
    fig, ax = plt.subplots(figsize=(10, 5))
    pos_names = list(position_results.keys())
    pos_accs = [position_results[p]["accuracy"] for p in pos_names]
    ax.bar(range(len(pos_names)), pos_accs, color="steelblue", alpha=0.8)
    ax.axhline(y=1.0 / K, color="red", linestyle="--", alpha=0.6, label=f"Chance (1/K)")
    ax.set_xticks(range(len(pos_names)))
    ax.set_xticklabels(pos_names, fontsize=9, rotation=15)
    ax.set_ylabel("Probe accuracy", fontsize=11)
    ax.set_title(f"Probe Accuracy by Token Position (After L0, Final Checkpoint)", fontsize=12)
    ax.legend()
    ax.set_ylim(0, 1.05)
    for i, v in enumerate(pos_accs):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "exp3_probe_by_position.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved exp3_probe_by_position.png")

    return all_results, position_results, layer_labels


# ═══════════════════════════════════════════════════════════════════════════
# SYNTHESIS
# ═══════════════════════════════════════════════════════════════════════════

def synthesize(patching_results, logit_lens_results, probe_results, probe_position_results,
               layer_labels):
    """Compare all three experiments and produce synthesis plot + summary."""
    print("\n" + "=" * 70)
    print("SYNTHESIS")
    print("=" * 70)

    steps = get_checkpoint_steps()
    final_step = steps[-1]

    # Get final checkpoint metrics for each layer
    final_patching = patching_results[final_step]
    final_ll = logit_lens_results[max(logit_lens_results.keys())]
    final_probe_step = max(probe_results.keys())
    final_probe = probe_results[final_probe_step]

    # Overlay plot: logit lens prob vs probe accuracy by layer
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(layer_labels))

    ll_probs = [final_ll[lbl]["mean_prob"] for lbl in layer_labels]
    ll_accs = [final_ll[lbl]["mean_top1_acc"] for lbl in layer_labels]
    pr_accs = [final_probe[lbl]["accuracy"] for lbl in layer_labels]

    ax.plot(x, ll_probs, "o-", color="tab:blue", label="Logit lens P(correct)", markersize=7)
    ax.plot(x, ll_accs, "s--", color="tab:cyan", label="Logit lens top-1 acc", markersize=7)
    ax.plot(x, pr_accs, "^-", color="tab:red", label="Linear probe accuracy", markersize=7)
    ax.axhline(y=1.0 / K, color="gray", linestyle=":", alpha=0.5, label=f"Chance (1/{K})")

    # Add patching metrics
    patching_vals = final_patching["resid_means"]
    ax.plot(x, patching_vals, "D-", color="tab:green", label="Patching metric (z-flip)", markersize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels, fontsize=10)
    ax.set_ylabel("Metric value", fontsize=11)
    ax.set_title(f"Synthesis: Where Does Disambiguation Happen? (K={K})", fontsize=13)
    ax.legend(fontsize=9, loc="center right")
    ax.set_ylim(-0.05, 1.1)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "synthesis_overlay.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved synthesis_overlay.png")

    # Print summary
    print(f"\n  LAYER-BY-LAYER SUMMARY (Final Checkpoint):")
    print(f"  {'Layer':<14s} {'Patching':>10s} {'LL Prob':>10s} {'LL Top1':>10s} {'Probe Acc':>10s}")
    print(f"  {'-'*54}")
    for i, lbl in enumerate(layer_labels):
        print(f"  {lbl:<14s} {patching_vals[i]:>10.4f} {ll_probs[i]:>10.4f} "
              f"{ll_accs[i]:>10.4f} {pr_accs[i]:>10.4f}")

    print(f"\n  POSITION-SPECIFIC PROBES (After L0, Final Checkpoint):")
    for pos, res in probe_position_results.items():
        print(f"    {pos:<20s}: {res['accuracy']:.4f}")

    # Key findings
    print(f"\n  KEY FINDINGS:")

    # Where does patching first achieve > 0.5?
    for i, lbl in enumerate(layer_labels):
        if patching_vals[i] > 0.5:
            print(f"  - Patching: z-information first causally effective at '{lbl}' (metric={patching_vals[i]:.3f})")
            break

    # Where does logit lens first exceed chance?
    for i, lbl in enumerate(layer_labels):
        if ll_probs[i] > 2.0 / K:
            print(f"  - Logit lens: correct answer first decodable at '{lbl}' (P={ll_probs[i]:.3f})")
            break

    # Where does probe first exceed chance significantly?
    for i, lbl in enumerate(layer_labels):
        if pr_accs[i] > 2.0 / K:
            print(f"  - Probes: disambiguation info first linearly present at '{lbl}' (acc={pr_accs[i]:.3f})")
            break

    # Agreement check
    agree = (patching_vals[0] < 0.3 and ll_probs[0] < 0.3 and pr_accs[0] < 0.3)
    if agree:
        print(f"  - AGREEMENT: All three methods show no disambiguation before Layer 0")
    else:
        print(f"  - DISAGREEMENT: Some methods show disambiguation before Layer 0!")
        print(f"    Embed: patching={patching_vals[0]:.3f}, LL={ll_probs[0]:.3f}, probe={pr_accs[0]:.3f}")

    # Probe vs logit lens disagreement
    for i, lbl in enumerate(layer_labels):
        if abs(pr_accs[i] - ll_probs[i]) > 0.2:
            print(f"  - DISCREPANCY at '{lbl}': probe={pr_accs[i]:.3f} vs LL={ll_probs[i]:.3f}")
            print(f"    → Info present but not in unembedding format")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print(f"Deep Mechanistic Interpretability Analysis")
    print(f"Experiment: {EXPERIMENT} (K={K}, log K = {LOG_K:.4f})")
    print(f"Device: {DEVICE}")

    cfg, tokenizer = get_cfg_and_tokenizer()
    mapping_data = get_mapping_data()
    n_layers = cfg.model.n_layers
    print(f"Model: {n_layers} layers, {cfg.model.n_heads} heads, d={cfg.model.d_model}")

    # Run experiments
    patching_results = run_activation_patching(cfg, tokenizer, mapping_data)
    logit_lens_results, ll_labels = run_logit_lens(cfg, tokenizer, mapping_data)
    probe_results, probe_pos_results, pr_labels = run_linear_probes(cfg, tokenizer, mapping_data)

    # Synthesis
    synthesize(patching_results, logit_lens_results, probe_results, probe_pos_results, ll_labels)

    # Save all raw data
    raw = {
        "experiment": EXPERIMENT,
        "K": K,
        "log_K": LOG_K,
        "patching": {str(k): v for k, v in patching_results.items()},
        "logit_lens": {str(k): v for k, v in logit_lens_results.items()},
        "probes": {str(k): v for k, v in probe_results.items()},
        "probe_positions": probe_pos_results,
    }
    with open(ANALYSIS_DIR / "deep_mechinterp_results.json", "w") as f:
        json.dump(raw, f, indent=2, default=str)
    print(f"\n  Saved deep_mechinterp_results.json")
    print(f"\nAll done. Plots in {ANALYSIS_DIR}/")


if __name__ == "__main__":
    main()
