#!/usr/bin/env python
"""
Round 3 Mechanistic Interpretability: Complete Disambiguation Pipeline.

Experiment 1: Temporal formation — does the pipeline form simultaneously or sequentially?
Experiment 2: Representational geometry — PCA, RSA, dimensionality at each layer.
Experiment 3: Layer-to-layer composition — layer ablation, attn vs MLP, cross-layer patching.
Experiment 4: Redundancy within L0 — pairwise ablation, OV similarity, additivity.
Experiment 5: Cross-K generalization — does the 4-layer pipeline hold for K=3..36?

Reuses infrastructure from prior rounds.
"""

import sys
import json
import math
import pickle
from pathlib import Path
from itertools import combinations

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

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
N_LAYERS = 4
N_HEADS = 4

OUTPUTS_DIR = Path("outputs")
CHECKPOINT_DIR = OUTPUTS_DIR / EXPERIMENT / "checkpoints"
ANALYSIS_DIR = Path("analysis_outputs")
ANALYSIS_DIR.mkdir(exist_ok=True)

DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")

# ── Shared helpers (reused from prior rounds) ───────────────────────────────

def get_cfg_and_tokenizer(experiment_name="mechinterp_r3", k=K, n_unique_b=N_UNIQUE_B):
    cfg = make_config(experiment_name=experiment_name, k=k, seed=SEED, n_unique_b=n_unique_b)
    tokenizer = create_tokenizer_from_config(cfg)
    return cfg, tokenizer


def get_mapping_data(k=K, n_unique_b=N_UNIQUE_B):
    return generate_mappings(
        n_unique_b=n_unique_b, k=k, b_length=6, a_length=4, z_length=2,
        vocab_chars="abcdefghijklmnopqrstuvwxyz0123456789",
        seed=SEED, task="bz_to_a",
        enforce_unique_a_first_char_per_b=True, disambiguation_prefix_length=1,
    )


def load_model_at_step(cfg, tokenizer, step, checkpoint_dir=None):
    if checkpoint_dir is None:
        checkpoint_dir = CHECKPOINT_DIR
    model = create_model_from_config(cfg, tokenizer)
    model = model.to(DEVICE)
    load_checkpoint(model, None, checkpoint_dir, step=step)
    model.eval()
    return model


def get_checkpoint_steps(checkpoint_dir=None):
    if checkpoint_dir is None:
        checkpoint_dir = CHECKPOINT_DIR
    return sorted(list_checkpoints(checkpoint_dir))


def subsample_steps(steps, every_n=5):
    sub = steps[::every_n]
    if steps[0] not in sub:
        sub = [steps[0]] + sub
    if steps[-1] not in sub:
        sub.append(steps[-1])
    return sorted(set(sub))


def find_tau(steps_list, experiment=None):
    if experiment is None:
        experiment = EXPERIMENT
    hist_path = OUTPUTS_DIR / experiment / "training_history.json"
    k_val = K
    # Extract K from experiment name if different
    if experiment != EXPERIMENT:
        try:
            k_val = int(experiment.split("_k")[-1])
        except:
            pass
    log_k = math.log(k_val)
    with open(hist_path) as f:
        h = json.load(f)
    cl = h.get("candidate_loss", h.get("train_loss", []))
    for s, v in zip(h["steps"], cl):
        if v is not None and v < 0.5 * log_k:
            return s
    return steps_list[len(steps_list) // 2]


def build_eval_loader(mapping_data, tokenizer, n_examples=512, batch_size=128):
    ds = DisambiguationDataset(
        mapping_data=mapping_data, tokenizer=tokenizer,
        split="train", probe_fraction=0.0, seed=SEED, task="bz_to_a",
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
        torch.mps.empty_cache() if hasattr(torch.mps, 'empty_cache') else None


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Temporal Formation of the Full Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def experiment1_temporal():
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Temporal Formation of the Pipeline")
    print("=" * 70)

    cfg, tokenizer = get_cfg_and_tokenizer()
    mapping_data = get_mapping_data()
    loader = build_eval_loader(mapping_data, tokenizer, n_examples=256, batch_size=64)
    pairs = build_paired_inputs(mapping_data, tokenizer, n_pairs=128)

    steps = get_checkpoint_steps()
    tau = find_tau(steps)
    n_layers = cfg.model.n_layers
    layer_labels = ["After embed"] + [f"After L{i}" for i in range(n_layers)]

    # Subsample: every 5th checkpoint for patching (expensive), every 3rd for lens/probes
    step_sub_patch = subsample_steps(steps, every_n=10)
    step_sub_lens = subsample_steps(steps, every_n=5)

    # ── A. Patching across all checkpoints ──
    print(f"\n  Part A: Patching across {len(step_sub_patch)} checkpoints")
    patching_results = {}

    for si, step in enumerate(step_sub_patch):
        if si % 5 == 0:
            print(f"    Checkpoint {si+1}/{len(step_sub_patch)}: step {step}")

        model = load_model_at_step(cfg, tokenizer, step)
        metrics_resid = [[] for _ in range(n_layers + 1)]

        batch_size = 64
        for start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[start:start + batch_size]
            bs = len(batch_pairs)
            max_len = max(
                max(len(p[0]["input_ids"]) for p in batch_pairs),
                max(len(p[1]["input_ids"]) for p in batch_pairs),
            )
            clean_ids = torch.zeros(bs, max_len, dtype=torch.long)
            corrupt_ids = torch.zeros(bs, max_len, dtype=torch.long)
            z_starts, z_ends, tgt_starts = [], [], []
            clean_first_tok, corrupt_first_tok = [], []

            for i, (c, x) in enumerate(batch_pairs):
                clean_ids[i, :len(c["input_ids"])] = c["input_ids"]
                corrupt_ids[i, :len(x["input_ids"])] = x["input_ids"]
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
            pred_pos = tgt_pos - 1
            batch_idx = torch.arange(bs, device=DEVICE)

            with torch.no_grad():
                clean_logits, clean_cache = model.run_with_cache(clean_ids)
                _, corrupt_cache = model.run_with_cache(corrupt_ids)
                corrupt_logits = model(corrupt_ids)

                clean_ld = (clean_logits[batch_idx, pred_pos, :][:, :].gather(1, clean_tok.unsqueeze(1)).squeeze(1)
                           - clean_logits[batch_idx, pred_pos, :][:, :].gather(1, corrupt_tok.unsqueeze(1)).squeeze(1))
                corrupt_ld = (corrupt_logits[batch_idx, pred_pos, :][:, :].gather(1, clean_tok.unsqueeze(1)).squeeze(1)
                             - corrupt_logits[batch_idx, pred_pos, :][:, :].gather(1, corrupt_tok.unsqueeze(1)).squeeze(1))
                denom = clean_ld - corrupt_ld

                patch_points = (
                    [("blocks.0.hook_resid_pre", "After embed")]
                    + [(f"blocks.{i}.hook_resid_post", f"After L{i}") for i in range(n_layers)]
                )

                for pidx, (hook_name, _) in enumerate(patch_points):
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
                    patched_ld = (patched_logits[batch_idx, pred_pos, :][:, :].gather(1, clean_tok.unsqueeze(1)).squeeze(1)
                                 - patched_logits[batch_idx, pred_pos, :][:, :].gather(1, corrupt_tok.unsqueeze(1)).squeeze(1))
                    metric = (clean_ld - patched_ld) / (denom + 1e-8)
                    metrics_resid[pidx].extend(metric.cpu().tolist())

                del clean_cache, corrupt_cache

        patching_results[step] = [float(np.mean(m)) for m in metrics_resid]
        cleanup_model(model)

    # ── B. Logit lens across all checkpoints ──
    print(f"\n  Part B: Logit lens across {len(step_sub_lens)} checkpoints")
    lens_results = {}

    for si, step in enumerate(step_sub_lens):
        if si % 10 == 0:
            print(f"    Checkpoint {si+1}/{len(step_sub_lens)}: step {step}")

        model = load_model_at_step(cfg, tokenizer, step)
        layer_probs = [[] for _ in range(n_layers + 1)]

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
                    normed = model.ln_final(r)
                    logits = normed @ model.W_U + model.b_U
                    probs = F.softmax(logits, dim=-1)
                    cp = probs[batch_idx, correct_tok]
                    layer_probs[lidx].extend(cp.cpu().tolist())
                del cache

        lens_results[step] = [float(np.mean(lp)) for lp in layer_probs]
        cleanup_model(model)

    # ── C. Probes across checkpoints (subsampled further) ──
    step_sub_probes = subsample_steps(steps, every_n=15)
    print(f"\n  Part C: Probes across {len(step_sub_probes)} checkpoints")

    probe_loader = build_eval_loader(mapping_data, tokenizer, n_examples=320, batch_size=64)
    probe_results = {}

    for si, step in enumerate(step_sub_probes):
        if si % 5 == 0:
            print(f"    Checkpoint {si+1}/{len(step_sub_probes)}: step {step}")

        model = load_model_at_step(cfg, tokenizer, step)
        layer_acts = [[] for _ in range(n_layers + 1)]
        all_y = []

        with torch.no_grad():
            for batch in probe_loader:
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
                    act = resid[batch_idx, pred_pos, :].cpu().numpy()
                    layer_acts[lidx].append(act)
                all_y.append(correct_tok.cpu().numpy())
                del cache

        y = np.concatenate(all_y)
        step_accs = []
        for lidx in range(n_layers + 1):
            X = np.concatenate(layer_acts[lidx])
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
            probe = LogisticRegression(max_iter=500, solver="lbfgs", C=1.0)
            probe.fit(X_tr, y_tr)
            step_accs.append(float(probe.score(X_te, y_te)))
        probe_results[step] = step_accs
        cleanup_model(model)

    # ── Plots ──

    # 1. Three aligned heatmaps
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)

    for ax_idx, (results_dict, title, cmap, vmin, vmax) in enumerate([
        (patching_results, "Patching Metric (z-flip)", "RdBu_r", 0, 1),
        (lens_results, "Logit Lens P(correct)", "viridis", 0, 1),
        (probe_results, "Linear Probe Accuracy", "viridis", 0, 1),
    ]):
        ax = axes[ax_idx]
        sorted_s = sorted(results_dict.keys())
        mat = np.array([results_dict[s] for s in sorted_s]).T  # (n_layers+1, n_steps)
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_yticks(range(len(layer_labels)))
        ax.set_yticklabels(layer_labels, fontsize=9)
        tick_idx = list(range(0, len(sorted_s), max(1, len(sorted_s) // 8)))
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([str(sorted_s[i]) for i in tick_idx], fontsize=7, rotation=45)
        ax.set_title(title, fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.02)
        # Mark tau
        tau_idx = min(range(len(sorted_s)), key=lambda i: abs(sorted_s[i] - tau))
        ax.axvline(x=tau_idx, color="red", linestyle="--", alpha=0.8, linewidth=1)

    axes[-1].set_xlabel("Training step", fontsize=10)
    fig.suptitle(f"Pipeline Formation Across Training (K={K})", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "r3_exp1_temporal_heatmaps.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved r3_exp1_temporal_heatmaps.png")

    # 2. Line plot: logit lens P(correct) per layer across training
    fig, ax1 = plt.subplots(figsize=(12, 5))
    sorted_lens_steps = sorted(lens_results.keys())
    colors = ["gray", "tab:blue", "tab:orange", "tab:green", "tab:red"]
    for lidx, lbl in enumerate(layer_labels):
        vals = [lens_results[s][lidx] for s in sorted_lens_steps]
        ax1.plot(sorted_lens_steps, vals, "-", color=colors[lidx], linewidth=2, label=lbl, alpha=0.8)
    ax1.axhline(y=1.0/K, color="gray", linestyle=":", alpha=0.4, label=f"Chance (1/{K})")
    ax1.axvline(x=tau, color="red", linestyle="--", alpha=0.5, linewidth=1, label=f"τ≈{tau}")
    ax1.set_xlabel("Training step", fontsize=11)
    ax1.set_ylabel("Logit Lens P(correct)", fontsize=11)
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(fontsize=9, loc="center left")
    ax1.set_title(f"Logit Lens by Layer Across Training (K={K})", fontsize=13)

    # Overlay loss on secondary axis
    hist_path = OUTPUTS_DIR / EXPERIMENT / "training_history.json"
    with open(hist_path) as f:
        h = json.load(f)
    ax2 = ax1.twinx()
    ax2.plot(h["steps"], h.get("candidate_loss", h.get("train_loss", [])),
             "-", color="black", alpha=0.3, linewidth=1, label="Train loss")
    ax2.axhline(y=LOG_K, color="black", linestyle=":", alpha=0.3)
    ax2.set_ylabel("Loss (nats)", fontsize=10, color="gray")
    ax2.set_ylim(-0.1, LOG_K * 1.5)
    ax2.tick_params(axis="y", labelcolor="gray")

    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "r3_exp1_lens_timeline.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved r3_exp1_lens_timeline.png")

    # 3. Onset step analysis
    print("\n  Onset Step Analysis:")
    print(f"  {'Layer':<14s} {'Patching onset':<18s} {'Logit lens onset':<18s} {'Probe onset':<18s}")
    print("  " + "-" * 68)

    onset_table = {}
    for lidx, lbl in enumerate(layer_labels):
        onset_patch = "N/A"
        onset_lens = "N/A"
        onset_probe = "N/A"

        # Patching: find first step where metric drops below 0.7 (z consumed > 30%)
        sorted_ps = sorted(patching_results.keys())
        for s in sorted_ps:
            if patching_results[s][lidx] < 0.7:
                onset_patch = str(s)
                break

        # Logit lens: first step where P(correct) > 0.2
        sorted_ls = sorted(lens_results.keys())
        for s in sorted_ls:
            if lens_results[s][lidx] > 0.2:
                onset_lens = str(s)
                break

        # Probe: first step where accuracy > 0.2
        sorted_prs = sorted(probe_results.keys())
        for s in sorted_prs:
            if probe_results[s][lidx] > 0.2:
                onset_probe = str(s)
                break

        onset_table[lbl] = (onset_patch, onset_lens, onset_probe)
        print(f"  {lbl:<14s} {onset_patch:<18s} {onset_lens:<18s} {onset_probe:<18s}")

    return {
        "patching": {int(k): v for k, v in patching_results.items()},
        "lens": {int(k): v for k, v in lens_results.items()},
        "probes": {int(k): v for k, v in probe_results.items()},
        "onset_table": onset_table,
        "tau": tau,
    }


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Representational Geometry
# ═══════════════════════════════════════════════════════════════════════════

def experiment2_geometry():
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Representational Geometry")
    print("=" * 70)

    cfg, tokenizer = get_cfg_and_tokenizer()
    mapping_data = get_mapping_data()
    n_layers = cfg.model.n_layers

    # Build eval data with rich metadata
    ds = DisambiguationDataset(
        mapping_data=mapping_data, tokenizer=tokenizer,
        split="train", probe_fraction=0.0, seed=SEED, task="bz_to_a",
    )
    ds.tokenized = ds.tokenized[:512]
    ds.examples = ds.examples[:512]
    loader = torch.utils.data.DataLoader(
        ds, batch_size=128, shuffle=False, collate_fn=collate_fn, num_workers=0,
    )

    steps = get_checkpoint_steps()
    final_step = steps[-1]
    model = load_model_at_step(cfg, tokenizer, final_step)

    layer_labels = ["After embed"] + [f"After L{i}" for i in range(n_layers)]

    # Collect activations + metadata
    layer_activations = [[] for _ in range(n_layers + 1)]
    correct_outputs = []
    z_tokens = []
    b_contexts = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            tgt_starts = batch["target_start_positions"].to(DEVICE)
            z_positions = batch["z_positions"]
            bs = input_ids.shape[0]
            batch_idx = torch.arange(bs, device=DEVICE)

            _, cache = model.run_with_cache(input_ids)

            correct_tok = labels[batch_idx, tgt_starts]
            pred_pos = tgt_starts - 1

            resid_points = ([cache["blocks.0.hook_resid_pre"]]
                            + [cache[f"blocks.{i}.hook_resid_post"] for i in range(n_layers)])

            for lidx, resid in enumerate(resid_points):
                act = resid[batch_idx, pred_pos, :].cpu().numpy()
                layer_activations[lidx].append(act)

            correct_outputs.append(correct_tok.cpu().numpy())

            # Extract z-token identity (first z character)
            for b in range(bs):
                z_pos = z_positions[b].item()
                z_tok_id = input_ids[b, z_pos].item()
                z_tokens.append(z_tok_id)
                # B context: hash of first 3 B tokens
                b_hash = tuple(input_ids[b, 1:4].cpu().tolist())
                b_contexts.append(hash(b_hash) % 50)  # bin into 50 groups for vis

            del cache

    cleanup_model(model)

    y_output = np.concatenate(correct_outputs)
    z_arr = np.array(z_tokens)
    b_arr = np.array(b_contexts)

    # ── A. PCA visualization ──
    print("  Part A: PCA visualization...")
    fig, axes = plt.subplots(len(layer_labels), 3, figsize=(15, 3 * len(layer_labels)))
    coloring_labels = ["Correct output", "z-token", "B-context"]
    coloring_arrays = [y_output, z_arr, b_arr]

    pca_coords = {}
    for lidx, lbl in enumerate(layer_labels):
        X = np.concatenate(layer_activations[lidx])
        pca = PCA(n_components=3)
        coords = pca.fit_transform(X)
        pca_coords[lbl] = coords
        explained = pca.explained_variance_ratio_

        for cidx, (clabel, carr) in enumerate(zip(coloring_labels, coloring_arrays)):
            ax = axes[lidx, cidx]
            scatter = ax.scatter(coords[:, 0], coords[:, 1], c=carr, cmap="tab10",
                                s=5, alpha=0.4, rasterized=True)
            if lidx == 0:
                ax.set_title(clabel, fontsize=11)
            if cidx == 0:
                ax.set_ylabel(f"{lbl}\nPC2", fontsize=9)
            ax.set_xlabel(f"PC1 ({explained[0]:.1%})", fontsize=7)
            ax.tick_params(labelsize=6)

    fig.suptitle(f"PCA of Residual Stream at Prediction Position (K={K}, final)", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "r3_exp2_pca_grid.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved r3_exp2_pca_grid.png")

    # ── B. RSA ──
    print("  Part B: Representational Similarity Analysis...")

    # Build binary similarity matrices
    same_output = (y_output[:, None] == y_output[None, :]).astype(float)
    same_z = (z_arr[:, None] == z_arr[None, :]).astype(float)
    same_b = (b_arr[:, None] == b_arr[None, :]).astype(float)

    # Flatten upper triangle for correlation
    n = len(y_output)
    triu_idx = np.triu_indices(n, k=1)
    same_output_flat = same_output[triu_idx]
    same_z_flat = same_z[triu_idx]
    same_b_flat = same_b[triu_idx]

    rsa_results = {factor: [] for factor in ["Correct output", "z-token", "B-context"]}

    for lidx, lbl in enumerate(layer_labels):
        X = np.concatenate(layer_activations[lidx])
        sim_matrix = cosine_similarity(X)
        sim_flat = sim_matrix[triu_idx]

        for factor, ref_flat in zip(
            ["Correct output", "z-token", "B-context"],
            [same_output_flat, same_z_flat, same_b_flat]
        ):
            corr = np.corrcoef(sim_flat, ref_flat)[0, 1]
            rsa_results[factor].append(float(corr))

    fig, ax = plt.subplots(figsize=(8, 5))
    for factor, vals in rsa_results.items():
        ax.plot(range(len(layer_labels)), vals, "o-", markersize=6, linewidth=2, label=factor)
    ax.set_xticks(range(len(layer_labels)))
    ax.set_xticklabels(layer_labels, fontsize=9)
    ax.set_ylabel("RSA Correlation", fontsize=11)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_title(f"Representational Similarity Analysis (K={K})", fontsize=13)
    ax.legend(fontsize=10)
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.3)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "r3_exp2_rsa.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved r3_exp2_rsa.png")

    # ── C. Dimensionality (participation ratio) ──
    print("  Part C: Dimensionality analysis...")
    pr_values = []
    for lidx, lbl in enumerate(layer_labels):
        X = np.concatenate(layer_activations[lidx])
        X_centered = X - X.mean(axis=0)
        cov = np.cov(X_centered.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues[eigenvalues > 0]
        pr = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
        pr_values.append(float(pr))
        print(f"    {lbl}: PR = {pr:.1f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(layer_labels)), pr_values, "o-", color="tab:purple", markersize=8, linewidth=2)
    ax.set_xticks(range(len(layer_labels)))
    ax.set_xticklabels(layer_labels, fontsize=9)
    ax.set_ylabel("Participation Ratio (eff. dimensionality)", fontsize=11)
    ax.set_title(f"Representation Dimensionality by Layer (K={K})", fontsize=13)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "r3_exp2_dimensionality.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved r3_exp2_dimensionality.png")

    # Save PCA coords + metadata
    with open(ANALYSIS_DIR / "r3_exp2_geometry.pkl", "wb") as f:
        pickle.dump({
            "pca_coords": pca_coords,
            "y_output": y_output,
            "z_arr": z_arr,
            "b_arr": b_arr,
            "rsa": rsa_results,
            "participation_ratio": pr_values,
        }, f)
    print("  Saved r3_exp2_geometry.pkl")

    return {"rsa": rsa_results, "participation_ratio": pr_values}


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Layer-to-Layer Composition
# ═══════════════════════════════════════════════════════════════════════════

def experiment3_composition():
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Layer-to-Layer Composition")
    print("=" * 70)

    cfg, tokenizer = get_cfg_and_tokenizer()
    mapping_data = get_mapping_data()
    loader = build_eval_loader(mapping_data, tokenizer, n_examples=512, batch_size=128)
    pairs = build_paired_inputs(mapping_data, tokenizer, n_pairs=128)

    steps = get_checkpoint_steps()
    final_step = steps[-1]
    n_layers = cfg.model.n_layers
    layer_labels = ["L0", "L1", "L2", "L3"]

    # ── A. Layer-wise ablation with bypass ──
    print("\n  Part A: Full layer ablation (skip-connection only)...")
    model = load_model_at_step(cfg, tokenizer, final_step)
    baseline_loss = compute_loss(model, loader)
    print(f"    Baseline loss: {baseline_loss:.4f}")

    layer_ablation_results = {"baseline": baseline_loss}

    for layer in range(n_layers):
        # Ablate by replacing resid_post with resid_pre (zero out layer contribution)
        hook_name = f"blocks.{layer}.hook_resid_post"
        pre_hook_name = f"blocks.{layer}.hook_resid_pre"

        # We need to capture resid_pre during the forward pass
        resid_pre_store = {}

        def make_capture_hook(store, key):
            def hook_fn(value, hook):
                store[key] = value.clone()
                return value
            return hook_fn

        def make_bypass_hook(store, key):
            def hook_fn(value, hook):
                return store[key]  # replace resid_post with resid_pre
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
                        (pre_hook_name, make_capture_hook(resid_pre_store, "pre")),
                        (hook_name, make_bypass_hook(resid_pre_store, "pre")),
                    ],
                )
                shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
                shift_labels = labels[:, 1:].contiguous().view(-1)
                mask = shift_labels != -100
                loss = F.cross_entropy(shift_logits[mask], shift_labels[mask])
                total_loss += loss.item() * mask.sum().item()
                n_tokens += mask.sum().item()

        abl_loss = total_loss / max(n_tokens, 1)
        layer_ablation_results[f"L{layer}_ablated"] = abl_loss
        print(f"    L{layer} ablated (bypass): {abl_loss:.4f} (Δ = {abl_loss - baseline_loss:+.4f})")

    # ── B. Component-level ablation (attention vs MLP) ──
    print("\n  Part B: Component ablation (attention vs MLP)...")
    component_results = {}

    for layer in range(n_layers):
        for component, hook_name in [
            ("attn", f"blocks.{layer}.hook_attn_out"),
            ("mlp", f"blocks.{layer}.hook_mlp_out"),
        ]:
            def make_zero_hook():
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
                        fwd_hooks=[(hook_name, make_zero_hook())],
                    )
                    shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
                    shift_labels = labels[:, 1:].contiguous().view(-1)
                    mask = shift_labels != -100
                    loss = F.cross_entropy(shift_logits[mask], shift_labels[mask])
                    total_loss += loss.item() * mask.sum().item()
                    n_tokens += mask.sum().item()

            comp_loss = total_loss / max(n_tokens, 1)
            key = f"L{layer}_{component}_zeroed"
            component_results[key] = comp_loss
            print(f"    {key}: {comp_loss:.4f} (Δ = {comp_loss - baseline_loss:+.4f})")

    cleanup_model(model)

    # ── C. Cross-layer full residual patching ──
    print("\n  Part C: Full residual stream patching (all positions)...")
    model = load_model_at_step(cfg, tokenizer, final_step)

    full_patch_results = {}
    batch_size = 64

    for start in range(0, min(len(pairs), 128), batch_size):
        batch_pairs = pairs[start:start + batch_size]
        bs = len(batch_pairs)
        max_len = max(
            max(len(p[0]["input_ids"]) for p in batch_pairs),
            max(len(p[1]["input_ids"]) for p in batch_pairs),
        )
        clean_ids = torch.zeros(bs, max_len, dtype=torch.long)
        corrupt_ids = torch.zeros(bs, max_len, dtype=torch.long)
        z_starts, z_ends, tgt_starts = [], [], []
        clean_first_tok, corrupt_first_tok = [], []

        for i, (c, x) in enumerate(batch_pairs):
            clean_ids[i, :len(c["input_ids"])] = c["input_ids"]
            corrupt_ids[i, :len(x["input_ids"])] = x["input_ids"]
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
        pred_pos = tgt_pos - 1
        batch_idx = torch.arange(bs, device=DEVICE)

        with torch.no_grad():
            clean_logits, clean_cache = model.run_with_cache(clean_ids)
            _, corrupt_cache = model.run_with_cache(corrupt_ids)
            corrupt_logits = model(corrupt_ids)

            clean_ld = (clean_logits[batch_idx, pred_pos, :]
                       .gather(1, clean_tok.unsqueeze(1)).squeeze(1)
                       - clean_logits[batch_idx, pred_pos, :]
                       .gather(1, corrupt_tok.unsqueeze(1)).squeeze(1))
            corrupt_ld = (corrupt_logits[batch_idx, pred_pos, :]
                         .gather(1, clean_tok.unsqueeze(1)).squeeze(1)
                         - corrupt_logits[batch_idx, pred_pos, :]
                         .gather(1, corrupt_tok.unsqueeze(1)).squeeze(1))
            denom = clean_ld - corrupt_ld

            # Full residual patching at each layer boundary
            patch_points = [(f"blocks.{i}.hook_resid_post", f"After L{i}") for i in range(n_layers)]

            for hook_name, plabel in patch_points:
                corrupt_act = corrupt_cache[hook_name].clone()

                def make_full_patch(c_act):
                    def hook_fn(value, hook):
                        return c_act  # replace ALL positions
                    return hook_fn

                patched_logits = model.run_with_hooks(
                    clean_ids,
                    fwd_hooks=[(hook_name, make_full_patch(corrupt_act))],
                )
                patched_ld = (patched_logits[batch_idx, pred_pos, :]
                             .gather(1, clean_tok.unsqueeze(1)).squeeze(1)
                             - patched_logits[batch_idx, pred_pos, :]
                             .gather(1, corrupt_tok.unsqueeze(1)).squeeze(1))
                metric = (clean_ld - patched_ld) / (denom + 1e-8)

                if plabel not in full_patch_results:
                    full_patch_results[plabel] = []
                full_patch_results[plabel].extend(metric.cpu().tolist())

            del clean_cache, corrupt_cache

    cleanup_model(model)

    # Compare z-only vs full patching
    # Z-only results from Round 2 (final checkpoint)
    z_only_metrics = {"After L0": 0.36, "After L1": 0.27, "After L2": 0.16, "After L3": 0.00}

    print("\n  Full residual vs z-only patching:")
    print(f"  {'Layer':<14s} {'Z-only patch':<16s} {'Full resid patch':<18s} {'Difference':<12s}")
    print("  " + "-" * 58)
    for lbl in [f"After L{i}" for i in range(n_layers)]:
        z_val = z_only_metrics.get(lbl, float('nan'))
        full_val = float(np.mean(full_patch_results.get(lbl, [0])))
        print(f"  {lbl:<14s} {z_val:<16.4f} {full_val:<18.4f} {full_val - z_val:<12.4f}")

    # ── Plots ──

    # 1. Stacked bar chart: layer ablation
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(n_layers)
    width = 0.2

    bars_baseline = [baseline_loss] * n_layers
    bars_full = [layer_ablation_results.get(f"L{i}_ablated", 0) for i in range(n_layers)]
    bars_attn = [component_results.get(f"L{i}_attn_zeroed", 0) for i in range(n_layers)]
    bars_mlp = [component_results.get(f"L{i}_mlp_zeroed", 0) for i in range(n_layers)]

    ax.bar(x - 1.5*width, bars_baseline, width, label="No ablation", color="tab:gray", alpha=0.7)
    ax.bar(x - 0.5*width, bars_full, width, label="Full layer ablated", color="tab:red", alpha=0.7)
    ax.bar(x + 0.5*width, bars_attn, width, label="Attention zeroed", color="tab:blue", alpha=0.7)
    ax.bar(x + 1.5*width, bars_mlp, width, label="MLP zeroed", color="tab:green", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels, fontsize=11)
    ax.set_ylabel("Loss (nats)", fontsize=11)
    ax.axhline(y=LOG_K, color="red", linestyle="--", alpha=0.5, label=f"log K = {LOG_K:.2f}")
    ax.legend(fontsize=9)
    ax.set_title(f"Layer & Component Ablation (K={K}, final checkpoint)", fontsize=13)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "r3_exp3_layer_ablation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved r3_exp3_layer_ablation.png")

    # 2. Full vs z-only patching comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    layers_plot = [f"After L{i}" for i in range(n_layers)]
    z_vals = [z_only_metrics[l] for l in layers_plot]
    full_vals = [float(np.mean(full_patch_results.get(l, [0]))) for l in layers_plot]
    x = np.arange(len(layers_plot))
    ax.bar(x - 0.15, z_vals, 0.3, label="Z-only patching", color="tab:blue", alpha=0.8)
    ax.bar(x + 0.15, full_vals, 0.3, label="Full residual patching", color="tab:orange", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(layers_plot, fontsize=10)
    ax.set_ylabel("Patching metric (0=no flip, 1=full flip)", fontsize=10)
    ax.legend(fontsize=10)
    ax.set_title(f"Z-Only vs Full Residual Patching (K={K})", fontsize=12)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "r3_exp3_patching_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved r3_exp3_patching_comparison.png")

    return {
        "layer_ablation": layer_ablation_results,
        "component_ablation": component_results,
        "full_patch": {k: float(np.mean(v)) for k, v in full_patch_results.items()},
        "z_only_patch": z_only_metrics,
    }


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4: Redundancy Within Layer 0
# ═══════════════════════════════════════════════════════════════════════════

def experiment4_redundancy():
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Redundancy & Specialization in Layer 0")
    print("=" * 70)

    cfg, tokenizer = get_cfg_and_tokenizer()
    mapping_data = get_mapping_data()
    loader = build_eval_loader(mapping_data, tokenizer, n_examples=512, batch_size=128)
    pairs = build_paired_inputs(mapping_data, tokenizer, n_pairs=128)

    steps = get_checkpoint_steps()
    final_step = steps[-1]
    n_heads = cfg.model.n_heads
    model = load_model_at_step(cfg, tokenizer, final_step)

    baseline_loss = compute_loss(model, loader)
    print(f"  Baseline loss: {baseline_loss:.4f}")

    # ── A. Pairwise head ablation ──
    print("\n  Part A: Pairwise head ablation...")

    def ablate_heads(model, loader, heads_to_ablate):
        """Zero-ablate specified L0 heads and compute loss."""
        def make_head_zero_hook(heads):
            def hook_fn(value, hook):
                # value shape: (batch, seq, n_heads, d_head)
                for h in heads:
                    value[:, :, h, :] = 0.0
                return value
            return hook_fn

        total_loss = 0.0
        n_tokens = 0
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)
                logits = model.run_with_hooks(
                    input_ids,
                    fwd_hooks=[("blocks.0.attn.hook_z", make_head_zero_hook(heads_to_ablate))],
                )
                shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
                shift_labels = labels[:, 1:].contiguous().view(-1)
                mask = shift_labels != -100
                loss = F.cross_entropy(shift_logits[mask], shift_labels[mask])
                total_loss += loss.item() * mask.sum().item()
                n_tokens += mask.sum().item()
        return total_loss / max(n_tokens, 1)

    # Single heads
    single_results = {}
    for h in range(n_heads):
        loss = ablate_heads(model, loader, [h])
        single_results[h] = loss
        print(f"    H{h} ablated: {loss:.4f} (Δ = {loss - baseline_loss:+.4f})")

    # Pairs
    pair_results = {}
    for h1, h2 in combinations(range(n_heads), 2):
        loss = ablate_heads(model, loader, [h1, h2])
        pair_results[(h1, h2)] = loss
        print(f"    H{h1}+H{h2} ablated: {loss:.4f} (Δ = {loss - baseline_loss:+.4f})")

    # Triples
    triple_results = {}
    for combo in combinations(range(n_heads), 3):
        loss = ablate_heads(model, loader, list(combo))
        triple_results[combo] = loss
        remaining = [h for h in range(n_heads) if h not in combo]
        print(f"    Only H{remaining[0]} remaining: {loss:.4f} (Δ = {loss - baseline_loss:+.4f})")

    # All four
    all_four_loss = ablate_heads(model, loader, list(range(n_heads)))
    print(f"    All 4 ablated: {all_four_loss:.4f} (Δ = {all_four_loss - baseline_loss:+.4f})")

    # ── B. OV circuit similarity ──
    print("\n  Part B: OV circuit similarity...")
    W_V = model.W_V[0]  # (n_heads, d_model, d_head)
    W_O = model.W_O[0]  # (n_heads, d_head, d_model)

    ov_matrices = []
    for h in range(n_heads):
        # OV circuit: W_V (d_model, d_head) @ W_O (d_head, d_model) -> (d_model, d_model)
        ov_h = W_V[h] @ W_O[h]
        ov_matrices.append(ov_h.detach().cpu().numpy().flatten())

    ov_sim = np.zeros((n_heads, n_heads))
    for i in range(n_heads):
        for j in range(n_heads):
            ov_sim[i, j] = cosine_similarity(
                ov_matrices[i].reshape(1, -1),
                ov_matrices[j].reshape(1, -1)
            )[0, 0]

    print("    OV cosine similarity matrix:")
    for i in range(n_heads):
        row = "    " + " ".join(f"{ov_sim[i,j]:+.3f}" for j in range(n_heads))
        print(row)

    # ── C. Pairwise patching additivity ──
    print("\n  Part C: Head patching additivity...")

    def patch_heads_z(model, pairs, heads):
        """Patch z through specified L0 heads, return mean patching metric."""
        metrics = []
        batch_size = 64
        for start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[start:start + batch_size]
            bs = len(batch_pairs)
            max_len = max(
                max(len(p[0]["input_ids"]) for p in batch_pairs),
                max(len(p[1]["input_ids"]) for p in batch_pairs),
            )
            clean_ids = torch.zeros(bs, max_len, dtype=torch.long)
            corrupt_ids = torch.zeros(bs, max_len, dtype=torch.long)
            z_starts, z_ends, tgt_starts = [], [], []
            clean_first_tok, corrupt_first_tok = [], []

            for i, (c, x) in enumerate(batch_pairs):
                clean_ids[i, :len(c["input_ids"])] = c["input_ids"]
                corrupt_ids[i, :len(x["input_ids"])] = x["input_ids"]
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
            pred_pos = tgt_pos - 1
            batch_idx = torch.arange(bs, device=DEVICE)

            with torch.no_grad():
                clean_logits, clean_cache = model.run_with_cache(clean_ids)
                _, corrupt_cache = model.run_with_cache(corrupt_ids)
                corrupt_logits = model(corrupt_ids)

                clean_ld = (clean_logits[batch_idx, pred_pos, :]
                           .gather(1, clean_tok.unsqueeze(1)).squeeze(1)
                           - clean_logits[batch_idx, pred_pos, :]
                           .gather(1, corrupt_tok.unsqueeze(1)).squeeze(1))
                corrupt_ld = (corrupt_logits[batch_idx, pred_pos, :]
                             .gather(1, clean_tok.unsqueeze(1)).squeeze(1)
                             - corrupt_logits[batch_idx, pred_pos, :]
                             .gather(1, corrupt_tok.unsqueeze(1)).squeeze(1))
                denom = clean_ld - corrupt_ld

                corrupt_head_out = corrupt_cache["blocks.0.attn.hook_z"].clone()

                def make_multi_head_patch(hs, z_s, z_e, c_act):
                    def hook_fn(value, hook):
                        for i in range(value.shape[0]):
                            for h in hs:
                                value[i, z_s[i]:z_e[i], h, :] = c_act[i, z_s[i]:z_e[i], h, :]
                        return value
                    return hook_fn

                patched_logits = model.run_with_hooks(
                    clean_ids,
                    fwd_hooks=[("blocks.0.attn.hook_z",
                                make_multi_head_patch(heads, z_starts, z_ends, corrupt_head_out))],
                )
                patched_ld = (patched_logits[batch_idx, pred_pos, :]
                             .gather(1, clean_tok.unsqueeze(1)).squeeze(1)
                             - patched_logits[batch_idx, pred_pos, :]
                             .gather(1, corrupt_tok.unsqueeze(1)).squeeze(1))
                metric = (clean_ld - patched_ld) / (denom + 1e-8)
                metrics.extend(metric.cpu().tolist())

                del clean_cache, corrupt_cache

        return float(np.mean(metrics))

    # Single-head patching
    single_patch = {}
    for h in range(n_heads):
        val = patch_heads_z(model, pairs, [h])
        single_patch[h] = val
        print(f"    H{h} patch: {val:.4f}")

    # Pairwise patching
    pair_patch = {}
    for h1, h2 in combinations(range(n_heads), 2):
        val = patch_heads_z(model, pairs, [h1, h2])
        pair_patch[(h1, h2)] = val
        expected = single_patch[h1] + single_patch[h2]
        ratio = val / expected if expected != 0 else float('inf')
        kind = "superadditive" if ratio > 1.1 else ("subadditive" if ratio < 0.9 else "additive")
        print(f"    H{h1}+H{h2} patch: {val:.4f} (expected: {expected:.4f}, ratio: {ratio:.2f}, {kind})")

    cleanup_model(model)

    # ── Plots ──

    # 1. Ablation matrix heatmap
    fig, ax = plt.subplots(figsize=(8, 7))
    # Build matrix: rows = "ablated heads", columns = individual heads
    # Include singles, pairs, triples, all-4
    labels = []
    losses = []

    for h in range(n_heads):
        labels.append(f"H{h}")
        losses.append(single_results[h])
    for (h1, h2) in sorted(pair_results.keys()):
        labels.append(f"H{h1}+H{h2}")
        losses.append(pair_results[(h1, h2)])
    for combo in sorted(triple_results.keys()):
        remaining = [h for h in range(n_heads) if h not in combo]
        labels.append(f"Only H{remaining[0]}")
        losses.append(triple_results[combo])
    labels.append("All 4")
    losses.append(all_four_loss)

    colors_bar = (["tab:blue"] * n_heads +
                  ["tab:orange"] * len(pair_results) +
                  ["tab:green"] * len(triple_results) +
                  ["tab:red"])

    ax.barh(range(len(labels)), losses, color=colors_bar, alpha=0.8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(x=baseline_loss, color="gray", linestyle="-", alpha=0.7, label=f"Baseline ({baseline_loss:.2f})")
    ax.axvline(x=LOG_K, color="red", linestyle="--", alpha=0.5, label=f"log K ({LOG_K:.2f})")
    ax.set_xlabel("Loss (nats)", fontsize=11)
    ax.set_title(f"L0 Head Ablation Combinations (K={K})", fontsize=13)
    ax.legend(fontsize=9)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "r3_exp4_ablation_combos.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved r3_exp4_ablation_combos.png")

    # 2. OV similarity heatmap
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(ov_sim, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n_heads))
    ax.set_xticklabels([f"H{h}" for h in range(n_heads)], fontsize=11)
    ax.set_yticks(range(n_heads))
    ax.set_yticklabels([f"H{h}" for h in range(n_heads)], fontsize=11)
    for i in range(n_heads):
        for j in range(n_heads):
            ax.text(j, i, f"{ov_sim[i,j]:.2f}", ha="center", va="center", fontsize=10,
                   color="white" if abs(ov_sim[i,j]) > 0.5 else "black")
    ax.set_title(f"OV Circuit Cosine Similarity (L0, K={K})", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "r3_exp4_ov_similarity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved r3_exp4_ov_similarity.png")

    # 3. Additivity test
    fig, ax = plt.subplots(figsize=(6, 6))
    for (h1, h2), joint_val in pair_patch.items():
        expected = single_patch[h1] + single_patch[h2]
        ax.scatter(expected, joint_val, s=80, zorder=5)
        ax.annotate(f"H{h1}+H{h2}", (expected, joint_val), fontsize=9,
                   textcoords="offset points", xytext=(5, 5))

    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]) - 0.02,
            max(ax.get_xlim()[1], ax.get_ylim()[1]) + 0.02]
    ax.plot(lims, lims, "k--", alpha=0.3, label="Additive (y=x)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Sum of individual patching metrics", fontsize=11)
    ax.set_ylabel("Joint patching metric", fontsize=11)
    ax.set_title(f"Head Patching Additivity Test (K={K})", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "r3_exp4_additivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved r3_exp4_additivity.png")

    return {
        "single_ablation": {f"H{h}": v for h, v in single_results.items()},
        "pair_ablation": {f"H{h1}+H{h2}": v for (h1, h2), v in pair_results.items()},
        "triple_ablation": {str(k): v for k, v in triple_results.items()},
        "all_four": all_four_loss,
        "ov_similarity": ov_sim.tolist(),
        "single_patch": {f"H{h}": v for h, v in single_patch.items()},
        "pair_patch": {f"H{h1}+H{h2}": v for (h1, h2), v in pair_patch.items()},
    }


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 5: Cross-K Generalization
# ═══════════════════════════════════════════════════════════════════════════

def experiment5_cross_k():
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Cross-K Generalization")
    print("=" * 70)

    k_values = [3, 5, 7, 10, 13, 17, 20, 25, 30, 36]
    available_ks = []
    for k in k_values:
        exp_name = f"landauer_dense_k{k}"
        ckpt_dir = OUTPUTS_DIR / exp_name / "checkpoints"
        if ckpt_dir.exists():
            available_ks.append(k)
    print(f"  Available K values: {available_ks}")

    # For each K, at final checkpoint: logit lens, probe accuracy, per-head ablation
    all_k_results = {}

    for k in available_ks:
        print(f"\n  --- K={k} ---")
        exp_name = f"landauer_dense_k{k}"
        ckpt_dir = OUTPUTS_DIR / exp_name / "checkpoints"
        steps = sorted(list_checkpoints(ckpt_dir))
        final_step = steps[-1]

        cfg, tokenizer = get_cfg_and_tokenizer(experiment_name=f"crossk_{k}", k=k, n_unique_b=N_UNIQUE_B)
        mapping_data = get_mapping_data(k=k, n_unique_b=N_UNIQUE_B)
        loader = build_eval_loader(mapping_data, tokenizer, n_examples=256, batch_size=64)

        model = load_model_at_step(cfg, tokenizer, final_step, checkpoint_dir=ckpt_dir)
        n_layers = cfg.model.n_layers
        n_heads = cfg.model.n_heads
        layer_labels = ["After embed"] + [f"After L{i}" for i in range(n_layers)]

        # 1. Logit lens
        layer_probs = [[] for _ in range(n_layers + 1)]
        layer_acts = [[] for _ in range(n_layers + 1)]
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
                    normed = model.ln_final(r)
                    logits = normed @ model.W_U + model.b_U
                    probs = F.softmax(logits, dim=-1)
                    cp = probs[batch_idx, correct_tok]
                    layer_probs[lidx].extend(cp.cpu().tolist())
                    layer_acts[lidx].append(r.cpu().numpy())

                all_y.append(correct_tok.cpu().numpy())
                del cache

        lens_by_layer = [float(np.mean(lp)) for lp in layer_probs]

        # 2. Linear probes
        y = np.concatenate(all_y)
        probe_by_layer = []
        for lidx in range(n_layers + 1):
            X = np.concatenate(layer_acts[lidx])
            try:
                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
                probe = LogisticRegression(max_iter=500, solver="lbfgs", C=1.0)
                probe.fit(X_tr, y_tr)
                probe_by_layer.append(float(probe.score(X_te, y_te)))
            except Exception:
                probe_by_layer.append(float('nan'))

        # 3. Per-head ablation
        baseline_loss = compute_loss(model, loader)
        head_ablation = {}
        for layer in range(n_layers):
            for head in range(n_heads):
                def make_head_zero(h):
                    def hook_fn(value, hook):
                        value[:, :, h, :] = 0.0
                        return value
                    return hook_fn

                total_loss = 0.0
                n_tokens = 0
                with torch.no_grad():
                    for batch in loader:
                        input_ids = batch["input_ids"].to(DEVICE)
                        labels = batch["labels"].to(DEVICE)
                        logits = model.run_with_hooks(
                            input_ids,
                            fwd_hooks=[(f"blocks.{layer}.attn.hook_z", make_head_zero(head))],
                        )
                        sl = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
                        slb = labels[:, 1:].contiguous().view(-1)
                        mask = slb != -100
                        loss = F.cross_entropy(sl[mask], slb[mask])
                        total_loss += loss.item() * mask.sum().item()
                        n_tokens += mask.sum().item()
                head_ablation[f"L{layer}H{head}"] = total_loss / max(n_tokens, 1)

        cleanup_model(model)

        log_k = math.log(k)
        all_k_results[k] = {
            "lens": lens_by_layer,
            "probes": probe_by_layer,
            "head_ablation": head_ablation,
            "baseline_loss": baseline_loss,
            "log_k": log_k,
        }

        # Print summary for this K
        print(f"    Logit lens: {['%.3f' % v for v in lens_by_layer]}")
        print(f"    Probes:     {['%.3f' % v for v in probe_by_layer]}")

    # ── Plots ──
    layer_labels = ["After embed"] + [f"After L{i}" for i in range(N_LAYERS)]

    # 1. Logit lens overlay
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.viridis
    for i, k in enumerate(sorted(all_k_results.keys())):
        color = cmap(i / max(len(all_k_results) - 1, 1))
        ax.plot(range(len(layer_labels)), all_k_results[k]["lens"], "o-",
                color=color, label=f"K={k}", markersize=5, linewidth=1.5)
    ax.set_xticks(range(len(layer_labels)))
    ax.set_xticklabels(layer_labels, fontsize=9)
    ax.set_ylabel("Logit Lens P(correct)", fontsize=11)
    ax.set_title("Logit Lens by Layer, Multiple K Values", fontsize=13)
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "r3_exp5_lens_cross_k.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved r3_exp5_lens_cross_k.png")

    # 2. Probe overlay
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, k in enumerate(sorted(all_k_results.keys())):
        color = cmap(i / max(len(all_k_results) - 1, 1))
        ax.plot(range(len(layer_labels)), all_k_results[k]["probes"], "o-",
                color=color, label=f"K={k}", markersize=5, linewidth=1.5)
    ax.set_xticks(range(len(layer_labels)))
    ax.set_xticklabels(layer_labels, fontsize=9)
    ax.set_ylabel("Linear Probe Accuracy", fontsize=11)
    ax.set_title("Probe Accuracy by Layer, Multiple K Values", fontsize=13)
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "r3_exp5_probes_cross_k.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved r3_exp5_probes_cross_k.png")

    # 3. Ablation heatmap: for each K, which heads matter?
    fig, ax = plt.subplots(figsize=(12, 6))
    k_sorted = sorted(all_k_results.keys())
    head_labels = [f"L{l}H{h}" for l in range(N_LAYERS) for h in range(N_HEADS)]
    ablation_matrix = np.zeros((len(k_sorted), len(head_labels)))
    for ki, k in enumerate(k_sorted):
        bl = all_k_results[k]["baseline_loss"]
        for hi, hl in enumerate(head_labels):
            abl_loss = all_k_results[k]["head_ablation"].get(hl, bl)
            ablation_matrix[ki, hi] = abl_loss - bl  # delta

    im = ax.imshow(ablation_matrix, aspect="auto", cmap="Reds", interpolation="nearest")
    ax.set_yticks(range(len(k_sorted)))
    ax.set_yticklabels([f"K={k}" for k in k_sorted], fontsize=9)
    ax.set_xticks(range(len(head_labels)))
    ax.set_xticklabels(head_labels, fontsize=7, rotation=45)
    ax.set_title("Head Ablation Impact (Δ loss) Across K Values", fontsize=13)
    fig.colorbar(im, ax=ax, label="Δ loss (nats)")
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "r3_exp5_ablation_cross_k.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved r3_exp5_ablation_cross_k.png")

    # 4. Decodability table
    print("\n  Decodability Layer Summary:")
    print(f"  {'K':<5s} {'Lens > 0.5 at':<18s} {'Probe > 0.5 at':<18s}")
    print("  " + "-" * 40)
    for k in k_sorted:
        lens_layer = "None"
        probe_layer = "None"
        for lidx, lbl in enumerate(layer_labels):
            if all_k_results[k]["lens"][lidx] > 0.5 and lens_layer == "None":
                lens_layer = lbl
            if all_k_results[k]["probes"][lidx] > 0.5 and probe_layer == "None":
                probe_layer = lbl
        print(f"  {k:<5d} {lens_layer:<18s} {probe_layer:<18s}")

    return all_k_results


# ═══════════════════════════════════════════════════════════════════════════
# MAIN: Run all experiments and synthesize
# ═══════════════════════════════════════════════════════════════════════════

def print_synthesis(exp1, exp2, exp3, exp4, exp5):
    print("\n")
    print("=" * 70)
    print("=== COMPLETE MECHANISTIC PICTURE ===")
    print("=" * 70)

    print("""
THE TASK: Surjective function (B,z) → A with K-fold ambiguity.
B maps to K possible A values; z selects which one.

THE PHENOMENON: Phase transition from marginal P(A|B) to conditional P(A|B,z).
Model plateaus at loss = log K, then snaps to near-zero loss.
""")

    # Circuit summary from Exp 3
    print("THE CIRCUIT (from ablation + patching + logit lens + probes):")
    if exp3:
        la = exp3["layer_ablation"]
        ca = exp3["component_ablation"]
        bl = la["baseline"]
        for i in range(4):
            full_abl = la.get(f"L{i}_ablated", bl)
            attn_abl = ca.get(f"L{i}_attn_zeroed", bl)
            mlp_abl = ca.get(f"L{i}_mlp_zeroed", bl)
            delta_full = full_abl - bl
            delta_attn = attn_abl - bl
            delta_mlp = mlp_abl - bl
            print(f"  Layer {i}: full ablation Δ={delta_full:+.3f}, "
                  f"attn Δ={delta_attn:+.3f}, mlp Δ={delta_mlp:+.3f}")
    print()

    # Temporal formation
    print("CIRCUIT FORMATION DURING TRAINING:")
    if exp1:
        ot = exp1.get("onset_table", {})
        for lbl, (p, l, pr) in ot.items():
            print(f"  {lbl}: patching={p}, lens={l}, probe={pr}")
        tau = exp1.get("tau", "?")
        print(f"  τ (transition) ≈ step {tau}")
    print()

    # Within-layer structure
    print("WITHIN-LAYER STRUCTURE (L0):")
    if exp4:
        sa = exp4.get("single_ablation", {})
        for h, v in sorted(sa.items()):
            print(f"  {h} ablation loss: {v:.3f}")
        pa = exp4.get("pair_ablation", {})
        for pair, v in sorted(pa.items()):
            print(f"  {pair} ablation loss: {v:.3f}")
        sp = exp4.get("single_patch", {})
        pp = exp4.get("pair_patch", {})
        if sp and pp:
            print("\n  Additivity test:")
            for pair_key, joint_val in sorted(pp.items()):
                h1, h2 = pair_key.replace("H","").split("+")
                expected = sp.get(f"H{h1}", 0) + sp.get(f"H{h2}", 0)
                ratio = joint_val / expected if expected != 0 else float('inf')
                print(f"    {pair_key}: joint={joint_val:.4f}, sum={expected:.4f}, ratio={ratio:.2f}")
    print()

    # RSA
    print("REPRESENTATIONAL GEOMETRY:")
    if exp2:
        rsa = exp2.get("rsa", {})
        pr = exp2.get("participation_ratio", [])
        layer_labels = ["After embed", "After L0", "After L1", "After L2", "After L3"]
        for factor, vals in rsa.items():
            print(f"  {factor} RSA: {['%.3f' % v for v in vals]}")
        if pr:
            print(f"  Participation ratio: {['%.1f' % v for v in pr]}")
    print()

    # Cross-K
    print("CROSS-K GENERALIZATION:")
    if exp5:
        layer_labels = ["After embed", "After L0", "After L1", "After L2", "After L3"]
        for k in sorted(exp5.keys()):
            lens = exp5[k]["lens"]
            first_decode = "None"
            for lidx, lbl in enumerate(layer_labels):
                if lens[lidx] > 0.5:
                    first_decode = lbl
                    break
            print(f"  K={k:2d}: decodable at {first_decode}, "
                  f"lens={['%.2f' % v for v in lens]}")
    print()

    print("METHODOLOGICAL IMPLICATIONS:")
    print("  - Ablation overstates L0: measures pipeline dependency, not computation location")
    print("  - Logit lens and probes agree on L2 as the critical computation layer")
    print("  - Patching captures causal flow; logit lens captures decodability")
    print("  - Position probes show information is NOT localized after L0")
    print()

    print("OPEN QUESTIONS:")
    print("  - What specific features does L0 encode that aren't linearly decodable?")
    print("  - Is the 4-layer pipeline an artifact of model depth, or minimal?")
    print("  - How does the pipeline interact with the MLP vs attention split?")
    print("  - Can the transition be induced by intervening on specific layer computations?")


def main():
    print(f"Deep Mechanistic Interpretability — Round 3")
    print(f"Experiment: {EXPERIMENT} (K={K}, log K = {LOG_K:.4f})")
    print(f"Device: {DEVICE}")

    # Check for --skip-done flag to resume from where we crashed
    skip_done = "--skip-done" in sys.argv

    if skip_done:
        print("  Skipping completed experiments (1, 3), resuming from 4...")
        exp1_results = None
        exp3_results = None
    else:
        exp1_results = experiment1_temporal()
        exp3_results = experiment3_composition()

    exp4_results = experiment4_redundancy()
    exp2_results = experiment2_geometry()
    exp5_results = experiment5_cross_k()

    # Save all results
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

    all_results = {
        "exp1_temporal": make_serializable(exp1_results),
        "exp3_composition": make_serializable(exp3_results),
        "exp4_redundancy": make_serializable(exp4_results),
        "exp2_geometry": make_serializable(exp2_results),
        "exp5_cross_k": make_serializable(exp5_results),
    }
    with open(ANALYSIS_DIR / "r3_all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved r3_all_results.json")

    print_synthesis(exp1_results, exp2_results, exp3_results, exp4_results, exp5_results)

    print("\n\nAll done. Plots in analysis_outputs/")


if __name__ == "__main__":
    main()
