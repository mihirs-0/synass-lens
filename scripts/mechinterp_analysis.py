#!/usr/bin/env python
"""
Mechanistic interpretability analysis on disambiguation checkpoints.

Analysis 1: Attention pattern of disambiguation head (L0H3) to z-token across training.
Analysis 2: Zero-ablation and mean-ablation of all heads at final checkpoint + across training.

Uses landauer_dense_k10 (K=10, n_b=1000, 500 checkpoints every 100 steps).
"""

import sys
import json
import math
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import create_tokenizer_from_config, collate_fn
from src.data.dataset import generate_mappings, DisambiguationDataset
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
DISAM_LAYER = 0   # L0H3
DISAM_HEAD = 3

OUTPUTS_DIR = Path("outputs")
CHECKPOINT_DIR = OUTPUTS_DIR / EXPERIMENT / "checkpoints"
ANALYSIS_DIR = Path("analysis_outputs")
ANALYSIS_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# ── Data setup ──────────────────────────────────────────────────────────────

def build_eval_data(n_examples=512):
    """Build eval dataloader from the same data generation as the experiment."""
    cfg = make_config(
        experiment_name="mechinterp_eval",
        k=K, seed=SEED, n_unique_b=N_UNIQUE_B,
    )
    tokenizer = create_tokenizer_from_config(cfg)
    mapping_data = generate_mappings(
        n_unique_b=N_UNIQUE_B, k=K,
        b_length=6, a_length=4, z_length=2,
        vocab_chars="abcdefghijklmnopqrstuvwxyz0123456789",
        seed=SEED, task="bz_to_a",
        enforce_unique_a_first_char_per_b=True,
        disambiguation_prefix_length=1,
    )
    ds = DisambiguationDataset(
        mapping_data=mapping_data, tokenizer=tokenizer,
        split="train", probe_fraction=0.0, seed=SEED, task="bz_to_a",
    )
    # Take first n_examples
    ds.tokenized = ds.tokenized[:n_examples]
    ds.examples = ds.examples[:n_examples]
    loader = torch.utils.data.DataLoader(
        ds, batch_size=128, shuffle=False, collate_fn=collate_fn, num_workers=0,
    )
    return loader, tokenizer, cfg, mapping_data


def load_model_at_step(cfg, tokenizer, step):
    """Create fresh model and load checkpoint weights."""
    model = create_model_from_config(cfg, tokenizer)
    model = model.to(DEVICE)
    load_checkpoint(model, None, CHECKPOINT_DIR, step=step)
    model.eval()
    return model


def compute_loss(model, batch):
    """Compute cross-entropy loss matching training code."""
    input_ids = batch["input_ids"].to(DEVICE)
    labels = batch["labels"].to(DEVICE)
    logits = model(input_ids)
    shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
    shift_labels = labels[:, 1:].contiguous().view(-1)
    loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
    return loss.item()


# ── Analysis 1: Attention patterns across training ──────────────────────────

def analysis1_attention_across_training(loader, tokenizer, cfg):
    """Measure attention from target → z for all heads at every checkpoint."""
    steps = sorted(list_checkpoints(CHECKPOINT_DIR))
    # Subsample: every 5th checkpoint to keep runtime reasonable (100 checkpoints)
    step_subset = steps[::5]
    # Always include first and last
    if steps[0] not in step_subset:
        step_subset = [steps[0]] + step_subset
    if steps[-1] not in step_subset:
        step_subset.append(steps[-1])
    step_subset = sorted(set(step_subset))

    print(f"\nAnalysis 1: Attention patterns across {len(step_subset)} checkpoints")

    # Load training history for loss curve overlay
    history_path = OUTPUTS_DIR / EXPERIMENT / "training_history.json"
    with open(history_path) as f:
        history = json.load(f)
    hist_steps = history["steps"]
    hist_loss = history.get("candidate_loss", history.get("train_loss", []))

    results = {}

    for i, step in enumerate(step_subset):
        if i % 20 == 0:
            print(f"  Checkpoint {i+1}/{len(step_subset)}: step {step}")

        model = load_model_at_step(cfg, tokenizer, step)

        # Per-head attention to z, averaged across batch
        attn_to_z_sum = torch.zeros(N_LAYERS, N_HEADS)
        attn_to_z_sq_sum = torch.zeros(N_LAYERS, N_HEADS)
        n_samples = 0

        # Also store full attention pattern for the disambiguation head
        full_attn_accum = None

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(DEVICE)
                z_positions = batch["z_positions"]
                z_end_positions = batch["z_end_positions"]
                target_start_positions = batch["target_start_positions"]

                _, cache = model.run_with_cache(input_ids)
                bs = input_ids.shape[0]
                seq_len = input_ids.shape[1]

                for layer in range(N_LAYERS):
                    attn = cache["pattern", layer]  # (batch, n_heads, seq, seq)
                    for b in range(bs):
                        z_start = z_positions[b].item()
                        z_end = z_end_positions[b].item()
                        tgt_start = target_start_positions[b].item()

                        # Attention from first target position to z positions
                        attn_to_z = attn[b, :, tgt_start, z_start:z_end].sum(dim=-1)  # (n_heads,)
                        attn_to_z_sum[layer] += attn_to_z.cpu()
                        attn_to_z_sq_sum[layer] += (attn_to_z ** 2).cpu()

                        # Full attention pattern for disambiguation head
                        if layer == DISAM_LAYER:
                            full_pat = attn[b, DISAM_HEAD, :seq_len, :seq_len].cpu()
                            if full_attn_accum is None:
                                full_attn_accum = torch.zeros_like(full_pat)
                            full_attn_accum += full_pat

                        n_samples += 1 if layer == 0 else 0
                # Free cache
                del cache

        n = n_samples
        mean_attn = attn_to_z_sum / n
        var_attn = attn_to_z_sq_sum / n - mean_attn ** 2
        std_attn = var_attn.clamp(min=0).sqrt()

        # Get loss at this step
        loss_at_step = None
        for s, v in zip(hist_steps, hist_loss):
            if s >= step:
                loss_at_step = v
                break

        results[step] = {
            "mean_attn_to_z": mean_attn.tolist(),
            "std_attn_to_z": std_attn.tolist(),
            "full_attn_pattern_disam_head": (full_attn_accum / n).tolist() if full_attn_accum is not None else None,
            "loss": loss_at_step,
            "n_samples": n,
        }

        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # ── Plot 1: Attention to z vs training step ──
    fig, ax1 = plt.subplots(figsize=(10, 5))
    plt.style.use("seaborn-v0_8-whitegrid")

    plot_steps = sorted(results.keys())
    disam_attn = [results[s]["mean_attn_to_z"][DISAM_LAYER][DISAM_HEAD] for s in plot_steps]
    disam_std = [results[s]["std_attn_to_z"][DISAM_LAYER][DISAM_HEAD] for s in plot_steps]
    losses = [results[s]["loss"] for s in plot_steps]

    ax1.set_xlabel("Training step", fontsize=12)
    ax1.set_ylabel("Attention to z (L0H3)", fontsize=12, color="tab:blue")
    ax1.plot(plot_steps, disam_attn, "o-", color="tab:blue", markersize=2, linewidth=1.5, label="L0H3 attn to z")
    ax1.fill_between(plot_steps,
                     [m - s for m, s in zip(disam_attn, disam_std)],
                     [m + s for m, s in zip(disam_attn, disam_std)],
                     color="tab:blue", alpha=0.15)
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_ylim(-0.05, max(disam_attn) * 1.15 + 0.05)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Loss (nats)", fontsize=12, color="tab:red")
    ax2.plot(plot_steps, losses, "-", color="tab:red", linewidth=1.5, alpha=0.7, label="Loss")
    ax2.axhline(y=LOG_K, color="tab:red", linestyle="--", alpha=0.4, label=f"log K = {LOG_K:.2f}")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.set_ylim(-0.1, LOG_K * 1.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=10)

    ax1.set_title(f"Disambiguation Head Attention to z vs Training (K={K})", fontsize=13)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "plot1_attention_to_z_vs_step.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot1_attention_to_z_vs_step.png")

    # ── Plot 1b: All heads ──
    fig, axes = plt.subplots(N_LAYERS, 1, figsize=(10, 12), sharex=True)
    for layer in range(N_LAYERS):
        ax = axes[layer]
        for head in range(N_HEADS):
            vals = [results[s]["mean_attn_to_z"][layer][head] for s in plot_steps]
            lw = 2.5 if (layer == DISAM_LAYER and head == DISAM_HEAD) else 1.0
            alpha = 1.0 if (layer == DISAM_LAYER and head == DISAM_HEAD) else 0.6
            label = f"H{head}" + (" *" if (layer == DISAM_LAYER and head == DISAM_HEAD) else "")
            ax.plot(plot_steps, vals, linewidth=lw, alpha=alpha, label=label)
        ax.set_ylabel(f"Layer {layer}\nAttn to z", fontsize=10)
        ax.legend(fontsize=8, ncol=4, loc="upper left")
    axes[-1].set_xlabel("Training step", fontsize=12)
    axes[0].set_title(f"All Heads: Attention to z Across Training (K={K})", fontsize=13)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "plot1b_all_heads_attention.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot1b_all_heads_attention.png")

    # ── Plot 2: Full attention heatmaps at 3 timepoints ──
    # Find tau
    tau_step = None
    for s in plot_steps:
        if results[s]["loss"] is not None and results[s]["loss"] < 0.5 * LOG_K:
            tau_step = s
            break

    # Pick: before (tau/4), during (tau), after (2*tau or last converged)
    if tau_step:
        before_step = min(plot_steps, key=lambda s: abs(s - tau_step // 4))
        during_step = min(plot_steps, key=lambda s: abs(s - tau_step))
        after_step = min(plot_steps, key=lambda s: abs(s - tau_step * 3))
    else:
        before_step, during_step, after_step = plot_steps[0], plot_steps[len(plot_steps)//2], plot_steps[-1]

    timepoints = [
        (before_step, "Before transition"),
        (during_step, "During transition"),
        (after_step, "After transition"),
    ]

    seq_labels = ["BOS", "B1", "B2", "B3", "B4", "B5", "B6", "SEP", "z1", "z2", "SEP", "A1", "A2", "A3", "A4", "EOS"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (step, title) in enumerate(timepoints):
        ax = axes[idx]
        pattern = results[step]["full_attn_pattern_disam_head"]
        if pattern is None:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center")
            continue
        pattern = np.array(pattern)
        n = min(pattern.shape[0], len(seq_labels))
        im = ax.imshow(pattern[:n, :n], cmap="Blues", vmin=0, vmax=max(0.3, pattern[:n, :n].max()))
        ax.set_xticks(range(n))
        ax.set_xticklabels(seq_labels[:n], fontsize=7, rotation=45)
        ax.set_yticks(range(n))
        ax.set_yticklabels(seq_labels[:n], fontsize=7)
        ax.set_xlabel("Source (attended to)", fontsize=9)
        ax.set_ylabel("Destination (attending from)", fontsize=9)
        loss_val = results[step]["loss"]
        ax.set_title(f"{title}\nStep {step} (loss={loss_val:.3f})", fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f"L0H3 Attention Pattern (K={K})", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "plot2_attention_heatmaps.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot2_attention_heatmaps.png")

    return results, step_subset


# ── Analysis 2: Ablation study ──────────────────────────────────────────────

def analysis2_ablation(loader, tokenizer, cfg, checkpoint_steps_for_sweep):
    """Zero and mean ablation of all heads at final checkpoint + across training."""

    print(f"\nAnalysis 2: Ablation study")

    # ── Part A+B+C: Full head ablation at final checkpoint ──
    steps = sorted(list_checkpoints(CHECKPOINT_DIR))
    final_step = steps[-1]
    print(f"  Loading final checkpoint: step {final_step}")
    model = load_model_at_step(cfg, tokenizer, final_step)

    # Collect eval batches
    batches = []
    for batch in loader:
        batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        batches.append(batch)

    # Baseline loss
    baseline_losses = []
    with torch.no_grad():
        for batch in batches:
            baseline_losses.append(compute_loss(model, batch))
    baseline_loss = sum(baseline_losses) / len(baseline_losses)
    print(f"  Baseline loss: {baseline_loss:.4f}")

    # Compute mean activations for all heads (for mean-ablation)
    print(f"  Computing mean activations for mean-ablation...")
    mean_activations = {}
    with torch.no_grad():
        head_sums = {}
        head_counts = 0
        for batch in batches:
            input_ids = batch["input_ids"].to(DEVICE)
            _, cache = model.run_with_cache(input_ids)
            for layer in range(N_LAYERS):
                act = cache[f"blocks.{layer}.attn.hook_z"]  # (batch, seq, n_heads, d_head)
                for head in range(N_HEADS):
                    key = (layer, head)
                    head_act = act[:, :, head, :]  # (batch, seq, d_head)
                    mean_over_batch_seq = head_act.mean(dim=(0, 1))  # (d_head,)
                    if key not in head_sums:
                        head_sums[key] = torch.zeros_like(mean_over_batch_seq)
                    head_sums[key] += mean_over_batch_seq
            head_counts += 1
            del cache
    for key in head_sums:
        mean_activations[key] = head_sums[key] / head_counts

    # Per-head ablation (both zero and mean)
    print(f"  Running per-head ablation (16 heads x 2 types)...")
    head_results = []

    for layer in range(N_LAYERS):
        for head in range(N_HEADS):
            row = {"layer": layer, "head": head, "baseline_loss": baseline_loss}

            for abl_type in ["zero", "mean"]:
                abl_losses = []
                with torch.no_grad():
                    for batch in batches:
                        input_ids = batch["input_ids"].to(DEVICE)
                        mean_act = mean_activations[(layer, head)]

                        def make_hook(h, atype, mean_val):
                            def hook_fn(activation, hook):
                                if atype == "zero":
                                    activation[:, :, h, :] = 0.0
                                else:
                                    activation[:, :, h, :] = mean_val
                                return activation
                            return hook_fn

                        logits = model.run_with_hooks(
                            input_ids,
                            fwd_hooks=[(f"blocks.{layer}.attn.hook_z",
                                       make_hook(head, abl_type, mean_act))],
                        )
                        labels = batch["labels"].to(DEVICE)
                        shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
                        shift_labels = labels[:, 1:].contiguous().view(-1)
                        loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
                        abl_losses.append(loss.item())

                avg_loss = sum(abl_losses) / len(abl_losses)
                row[f"{abl_type}_ablation_loss"] = avg_loss
                row[f"{abl_type}_delta"] = avg_loss - baseline_loss

            head_results.append(row)

    # Print summary table
    print(f"\n  {'Head':<8} {'Baseline':>12} {'Zero-Abl':>12} {'Mean-Abl':>12} {'Zero Δ':>10} {'Mean Δ':>10} {'Δ from logK':>12}")
    print(f"  {'-'*76}")
    for r in head_results:
        tag = " <-- DISAM" if (r["layer"] == DISAM_LAYER and r["head"] == DISAM_HEAD) else ""
        print(f"  L{r['layer']}H{r['head']:<5} {r['baseline_loss']:>12.4f} "
              f"{r['zero_ablation_loss']:>12.4f} {r['mean_ablation_loss']:>12.4f} "
              f"{r['zero_delta']:>10.4f} {r['mean_delta']:>10.4f} "
              f"{abs(r['zero_ablation_loss'] - LOG_K):>12.4f}{tag}")

    # ── Plot 3: Bar chart — baseline vs zero vs mean ablation ──
    fig, ax = plt.subplots(figsize=(14, 6))
    head_labels = [f"L{r['layer']}H{r['head']}" for r in head_results]
    x = np.arange(len(head_labels))
    w = 0.25

    zero_losses = [r["zero_ablation_loss"] for r in head_results]
    mean_losses = [r["mean_ablation_loss"] for r in head_results]

    ax.bar(x - w, [baseline_loss]*len(head_results), w, label="Baseline", color="tab:green", alpha=0.7)
    ax.bar(x, zero_losses, w, label="Zero ablation", color="tab:red", alpha=0.7)
    ax.bar(x + w, mean_losses, w, label="Mean ablation", color="tab:orange", alpha=0.7)
    ax.axhline(y=LOG_K, color="black", linestyle="--", linewidth=1, alpha=0.6, label=f"log K = {LOG_K:.2f}")

    # Highlight L0H3
    disam_idx = DISAM_LAYER * N_HEADS + DISAM_HEAD
    ax.annotate("L0H3", xy=(x[disam_idx], zero_losses[disam_idx]),
                xytext=(x[disam_idx]+0.5, zero_losses[disam_idx]+0.3),
                fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="black"))

    ax.set_xticks(x)
    ax.set_xticklabels(head_labels, fontsize=9)
    ax.set_ylabel("Loss (nats)", fontsize=12)
    ax.set_title(f"Per-Head Ablation at Final Checkpoint (K={K}, step {final_step})", fontsize=13)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "plot3_ablation_bar_chart.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved plot3_ablation_bar_chart.png")

    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # ── Part D: Zero-ablation of L0H3 across training ──
    print(f"\n  Part D: L0H3 zero-ablation across {len(checkpoint_steps_for_sweep)} checkpoints...")

    ablation_sweep = {}
    for i, step in enumerate(checkpoint_steps_for_sweep):
        if i % 20 == 0:
            print(f"    Checkpoint {i+1}/{len(checkpoint_steps_for_sweep)}: step {step}")

        model = load_model_at_step(cfg, tokenizer, step)

        # Baseline
        bl = []
        with torch.no_grad():
            for batch in batches:
                bl.append(compute_loss(model, batch))
        bl_avg = sum(bl) / len(bl)

        # Zero-ablation of L0H3
        abl = []
        with torch.no_grad():
            for batch in batches:
                input_ids = batch["input_ids"].to(DEVICE)

                def zero_hook(activation, hook):
                    activation[:, :, DISAM_HEAD, :] = 0.0
                    return activation

                logits = model.run_with_hooks(
                    input_ids,
                    fwd_hooks=[(f"blocks.{DISAM_LAYER}.attn.hook_z", zero_hook)],
                )
                labels = batch["labels"].to(DEVICE)
                shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
                shift_labels = labels[:, 1:].contiguous().view(-1)
                loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
                abl.append(loss.item())
        abl_avg = sum(abl) / len(abl)

        ablation_sweep[step] = {
            "baseline_loss": bl_avg,
            "ablated_loss": abl_avg,
            "delta": abl_avg - bl_avg,
        }

        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # ── Plot 4: Ablated loss vs training step ──
    fig, ax = plt.subplots(figsize=(10, 5))

    sweep_steps = sorted(ablation_sweep.keys())
    bl_vals = [ablation_sweep[s]["baseline_loss"] for s in sweep_steps]
    abl_vals = [ablation_sweep[s]["ablated_loss"] for s in sweep_steps]
    delta_vals = [ablation_sweep[s]["delta"] for s in sweep_steps]

    ax.plot(sweep_steps, bl_vals, "-", color="tab:green", linewidth=1.5, label="Baseline loss")
    ax.plot(sweep_steps, abl_vals, "-", color="tab:red", linewidth=1.5, label="L0H3 zero-ablated loss")
    ax.fill_between(sweep_steps, bl_vals, abl_vals, color="tab:red", alpha=0.1)
    ax.axhline(y=LOG_K, color="black", linestyle="--", alpha=0.4, label=f"log K = {LOG_K:.2f}")

    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("Loss (nats)", fontsize=12)
    ax.set_title(f"L0H3 Zero-Ablation Effect Across Training (K={K})", fontsize=13)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "plot4_ablation_across_training.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot4_ablation_across_training.png")

    # ── Plot 5: Delta (ablation impact) vs training step ──
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(sweep_steps, delta_vals, "-o", color="tab:purple", markersize=2, linewidth=1.5, label="Ablation Δ (L0H3)")
    ax1.set_xlabel("Training step", fontsize=12)
    ax1.set_ylabel("Loss increase from L0H3 ablation (nats)", fontsize=12, color="tab:purple")
    ax1.tick_params(axis="y", labelcolor="tab:purple")

    ax2 = ax1.twinx()
    ax2.plot(sweep_steps, bl_vals, "-", color="tab:gray", linewidth=1, alpha=0.5, label="Baseline loss")
    ax2.set_ylabel("Baseline loss (nats)", fontsize=12, color="tab:gray")
    ax2.tick_params(axis="y", labelcolor="tab:gray")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="upper left")
    ax1.set_title(f"Causal Importance of L0H3 Across Training (K={K})", fontsize=13)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "plot5_ablation_delta_vs_step.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot5_ablation_delta_vs_step.png")

    return head_results, ablation_sweep


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print(f"Mechanistic Interpretability Analysis")
    print(f"Experiment: {EXPERIMENT} (K={K})")
    print(f"Device: {DEVICE}")
    print(f"Disambiguation head: L{DISAM_LAYER}H{DISAM_HEAD}")
    print(f"log K = {LOG_K:.4f}")

    loader, tokenizer, cfg, mapping_data = build_eval_data(n_examples=512)
    print(f"Eval examples: {len(loader.dataset)}")

    # Analysis 1
    attn_results, step_subset = analysis1_attention_across_training(loader, tokenizer, cfg)

    # Analysis 2 (use same step subset for sweep)
    head_results, ablation_sweep = analysis2_ablation(loader, tokenizer, cfg, step_subset)

    # Save all raw data
    raw_data = {
        "experiment": EXPERIMENT,
        "K": K,
        "log_K": LOG_K,
        "disam_head": f"L{DISAM_LAYER}H{DISAM_HEAD}",
        "attention_results": {str(k): v for k, v in attn_results.items()},
        "head_ablation_final": head_results,
        "ablation_sweep": {str(k): v for k, v in ablation_sweep.items()},
    }
    with open(ANALYSIS_DIR / "raw_results.json", "w") as f:
        json.dump(raw_data, f, indent=2, default=str)
    print(f"\nSaved raw_results.json")

    print("\nDone! All plots saved to analysis_outputs/")


if __name__ == "__main__":
    main()
