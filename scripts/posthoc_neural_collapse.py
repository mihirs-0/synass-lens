#!/usr/bin/env python3
"""
Experiment 5: Neural Collapse Proxy — Representation Geometry (Paper: 2509.20829)

Tests whether within-B variance of last-layer representations changes at the
transition. Before the transition (model ignores z), representations for all
(B, z_i) examples should be similar. After (model uses z), they should separate.

Requires forward passes — samples checkpoints every 2000 steps.
"""

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf
from src.data import create_tokenizer_from_config, create_datasets_from_config, collate_fn
from src.model import create_model_from_config

OUTPUTS = ROOT / "outputs"
SAVE_DIR = OUTPUTS / "paper_figures"
SAVE_DIR.mkdir(exist_ok=True)


def compute_tau(run_name, k, threshold_frac=0.5):
    p = OUTPUTS / run_name / "training_history.json"
    if not p.exists():
        return None
    with open(p) as f:
        h = json.load(f)
    log_k = math.log(k)
    for s, l in zip(h["steps"], h["first_target_loss"]):
        if l < threshold_frac * log_k:
            return s
    return None


@torch.no_grad()
def compute_representation_geometry(model, dataset, batch_size=64, device="cpu"):
    """Compute within-B and between-B variance of last-layer representations.

    Groups examples by base string B. For each B, there are K examples with
    different z values. Before the transition, all K representations are similar
    (model ignores z). After, they differentiate.

    Returns:
        within_var: mean within-B variance (averaged over all B groups)
        between_var: variance of B-group centroids
        total_var: total variance of all representations
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Collect representations grouped by base string
    base_to_reps = {}

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        target_starts = batch["target_start_positions"]
        base_strings = batch["base_strings"]

        # Forward pass — get residual stream at last layer
        # HookedTransformer returns logits; use run_with_cache for internals
        _, cache = model.run_with_cache(input_ids, names_filter="blocks.3.hook_resid_post")
        resid = cache["blocks.3.hook_resid_post"]  # [batch, seq, d_model]

        # Extract representation at the target position (first target token)
        for i in range(len(base_strings)):
            pos = target_starts[i].item()
            if pos < resid.shape[1]:
                rep = resid[i, pos].cpu().numpy()
                b_str = base_strings[i]
                if b_str not in base_to_reps:
                    base_to_reps[b_str] = []
                base_to_reps[b_str].append(rep)

    # Compute variances
    all_reps = []
    centroids = []
    within_vars = []

    for b_str, reps in base_to_reps.items():
        reps_arr = np.array(reps)  # [K, d_model]
        all_reps.append(reps_arr)

        centroid = reps_arr.mean(axis=0)
        centroids.append(centroid)

        # Within-B variance: mean squared distance from centroid
        if len(reps_arr) > 1:
            within_var = ((reps_arr - centroid) ** 2).mean()
            within_vars.append(within_var)

    all_reps = np.concatenate(all_reps, axis=0)
    centroids = np.array(centroids)

    # Total variance
    total_var = ((all_reps - all_reps.mean(axis=0)) ** 2).mean()

    # Between-B variance
    between_var = ((centroids - centroids.mean(axis=0)) ** 2).mean()

    # Within-B variance (averaged)
    within_var = np.mean(within_vars) if within_vars else 0.0

    return {
        "within_var": float(within_var),
        "between_var": float(between_var),
        "total_var": float(total_var),
        "n_groups": len(base_to_reps),
        "n_examples": len(all_reps),
    }


# ── Main analysis ────────────────────────────────────────────────────────────

print("=" * 80)
print("Experiment 5: Neural Collapse Proxy — Representation Geometry")
print("=" * 80)

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

SAMPLE_EVERY = 2000  # Only sample every 2000 steps to keep computation manageable

k_runs = {
    20: "landauer_dense_k20",
}

all_data = {}

for k, run_name in k_runs.items():
    ckpt_dir = OUTPUTS / run_name / "checkpoints"
    config_path = OUTPUTS / run_name / "config.yaml"

    if not ckpt_dir.exists() or not config_path.exists():
        print(f"  K={k}: missing checkpoints or config")
        continue

    tau = compute_tau(run_name, k)
    cfg = OmegaConf.load(config_path)

    # Create tokenizer and dataset
    tokenizer = create_tokenizer_from_config(cfg)
    train_dataset, probe_dataset, mapping_data = create_datasets_from_config(cfg, tokenizer)

    # Use probe dataset if available, else train (subsampled)
    dataset = probe_dataset if len(probe_dataset) > 0 else train_dataset
    # Subsample if too large — we need enough per B group (K examples each)
    # but don't need all 1000 B groups. Use first 200 B groups = 200*K examples.
    from torch.utils.data import Subset
    max_examples = 200 * k
    if len(dataset) > max_examples:
        dataset = Subset(dataset, list(range(max_examples)))
    print(f"  K={k}: {len(dataset)} examples, τ={tau}")

    # Create model factory
    def make_model():
        return create_model_from_config(cfg, tokenizer)

    step_dirs = sorted([d for d in ckpt_dir.iterdir()
                        if d.is_dir() and d.name.startswith("step_")])

    steps = []
    within_vars = []
    between_vars = []
    total_vars = []
    var_ratios = []

    for d in step_dirs:
        step = int(d.name.split("_")[1])
        if step % SAMPLE_EVERY != 0 and step != 100:
            continue

        model_path = d / "model.pt"
        if not model_path.exists():
            continue

        # Load model
        model = make_model()
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)
        model.to(device)

        # Compute geometry
        geom = compute_representation_geometry(model, dataset, batch_size=64, device=device)

        steps.append(step)
        within_vars.append(geom["within_var"])
        between_vars.append(geom["between_var"])
        total_vars.append(geom["total_var"])
        ratio = geom["within_var"] / geom["between_var"] if geom["between_var"] > 0 else float("inf")
        var_ratios.append(ratio)

        print(f"    step {step}: within={geom['within_var']:.4f}, "
              f"between={geom['between_var']:.4f}, ratio={ratio:.4f}")

        # Free memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    all_data[k] = {
        "steps": np.array(steps),
        "within_var": np.array(within_vars),
        "between_var": np.array(between_vars),
        "total_var": np.array(total_vars),
        "var_ratio": np.array(var_ratios),
        "tau": tau,
    }

# ── Print summary ────────────────────────────────────────────────────────────

for k in sorted(all_data.keys()):
    d = all_data[k]
    tau = d["tau"]
    if tau is None:
        continue

    print(f"\nK={k} (τ={tau}):")

    plateau_mask = d["steps"] < tau * 0.5
    posttrans_mask = d["steps"] > tau * 1.5

    if plateau_mask.sum() > 0:
        print(f"  Plateau (steps < {tau//2}):")
        print(f"    within_var: {d['within_var'][plateau_mask].mean():.6f}")
        print(f"    between_var: {d['between_var'][plateau_mask].mean():.6f}")

    if posttrans_mask.sum() > 0:
        print(f"  Post-transition (steps > {int(tau*1.5)}):")
        print(f"    within_var: {d['within_var'][posttrans_mask].mean():.6f}")
        print(f"    between_var: {d['between_var'][posttrans_mask].mean():.6f}")

    if plateau_mask.sum() > 0 and posttrans_mask.sum() > 0:
        within_change = d['within_var'][posttrans_mask].mean() / d['within_var'][plateau_mask].mean()
        between_change = d['between_var'][posttrans_mask].mean() / d['between_var'][plateau_mask].mean()
        print(f"  Within-var change: {within_change:.2f}×")
        print(f"  Between-var change: {between_change:.2f}×")

# ── Figure ───────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

d20 = all_data.get(20)
if d20 is not None and len(d20["steps"]) > 0:
    tau = d20["tau"]

    # Panel (a): Within-B and between-B variance
    ax = axes[0]
    ax.plot(d20["steps"], d20["within_var"], "o-", color="#E74C3C",
            label="Within-B var", markersize=3, linewidth=1)
    ax.plot(d20["steps"], d20["between_var"], "s-", color="#3498DB",
            label="Between-B var", markersize=3, linewidth=1)
    if tau:
        ax.axvline(tau, color="black", linestyle="--", alpha=0.5,
                   label=rf"$\tau$={tau}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Variance")
    ax.set_title("(a) Representation variance (K=20)")
    ax.legend(fontsize=7)

    # Panel (b): Within/between ratio
    ax = axes[1]
    ax.plot(d20["steps"], d20["var_ratio"], "o-", color="#27AE60",
            markersize=3, linewidth=1)
    if tau:
        ax.axvline(tau, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Within-B / Between-B variance")
    ax.set_title("(b) Variance ratio (K=20)")
    ax.set_yscale("log")

    # Panel (c): Total variance
    ax = axes[2]
    ax.plot(d20["steps"], d20["total_var"], "o-", color="#9B59B6",
            markersize=3, linewidth=1)
    if tau:
        ax.axvline(tau, color="black", linestyle="--", alpha=0.5,
                   label=rf"$\tau$={tau}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Total variance")
    ax.set_title("(c) Total representation variance (K=20)")
    ax.legend(fontsize=7)

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(SAVE_DIR / f"fig_neural_collapse.{ext}", dpi=300, bbox_inches="tight")
plt.close()

print(f"\nFigure saved: {SAVE_DIR / 'fig_neural_collapse.pdf'}")
