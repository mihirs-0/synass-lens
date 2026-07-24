#!/usr/bin/env python
"""
Experiment 3: Per-Head Causal Ablation of L1H3

Reviewer concern: L1H3 attention patterns are correlational.
Need causal evidence (necessity + sufficiency).

Design: For each head (L, H):
  - Zero ablation (necessity): zero out head output, measure loss increase
  - Mean ablation: replace with mean activation across batch
  - Sufficiency: ablate ALL heads except one, check if model still uses z

Run at 3 checkpoints per K=10,20: pre-transition (τ/4), mid-transition (τ),
post-transition (2τ).

Expected output: Table showing each head's causal contribution at each phase.
L1H3 should show largest effect post-transition.
"""

import sys
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader, Subset
from omegaconf import OmegaConf

from src.data.tokenizer import create_tokenizer_from_config
from src.data.dataset import create_datasets_from_config, collate_fn
from src.model.hooked_transformer import create_model_from_config
from src.probes.head_ablation import HeadAblationProbe
from scripts.experiment_helpers import load_history, detect_tau

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Config ─────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("outputs")
FIG_DIR = OUTPUT_DIR / "paper_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

K_VALUES = [10, 20]
RUN_PATTERN = "landauer_dense_k{k}"
PROBE_BATCH_SIZE = 256
N_PROBE_BATCHES = 8


def select_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def find_checkpoint(ckpt_dir, target_step):
    """Find the closest available checkpoint to target_step."""
    if not ckpt_dir.exists():
        return None
    all_steps = sorted([
        int(d.name.split("_")[1])
        for d in ckpt_dir.iterdir()
        if d.is_dir() and d.name.startswith("step_")
    ])
    if not all_steps:
        return None
    closest = min(all_steps, key=lambda s: abs(s - target_step))
    return closest


# ── Main ───────────────────────────────────────────────────────────────────

device = select_device()
print(f"Device: {device}")

print("=" * 70)
print("EXPERIMENT 3: PER-HEAD CAUSAL ABLATION")
print("=" * 70)

all_results = {}

for k in K_VALUES:
    run_name = RUN_PATTERN.format(k=k)
    print(f"\n{'─' * 50}")
    print(f"K = {k}  (run: {run_name})")

    config_path = OUTPUT_DIR / run_name / "config.yaml"
    if not config_path.exists():
        print(f"  No config found, skipping.")
        continue

    cfg = OmegaConf.load(config_path)
    h = load_history(run_name, str(OUTPUT_DIR))
    if h is None:
        print(f"  No history found, skipping.")
        continue

    log_k = math.log(k)
    tau = detect_tau(h, log_k)
    if tau is None:
        print(f"  τ not detected, skipping.")
        continue
    print(f"  τ = {tau}")

    # Create dataset and model
    tokenizer = create_tokenizer_from_config(cfg)
    train_ds, probe_ds, _ = create_datasets_from_config(cfg, tokenizer)
    # Use a subset for probing
    n_samples = min(PROBE_BATCH_SIZE * N_PROBE_BATCHES, len(train_ds))
    torch.manual_seed(42)
    indices = torch.randperm(len(train_ds))[:n_samples].tolist()
    subset = Subset(train_ds, indices)
    loader = DataLoader(subset, batch_size=PROBE_BATCH_SIZE, shuffle=False,
                        collate_fn=collate_fn)

    model = create_model_from_config(cfg, tokenizer).to(device)

    # Checkpoints: pre-transition, mid-transition, post-transition
    ckpt_dir = OUTPUT_DIR / run_name / "checkpoints"
    phases = {
        "pre": tau // 4,
        "mid": tau,
        "post": min(tau * 2, h["steps"][-1] if h["steps"] else tau * 2),
    }

    k_results = {}
    for phase_name, target_step in phases.items():
        actual_step = find_checkpoint(ckpt_dir, target_step)
        if actual_step is None:
            print(f"  No checkpoint near step {target_step} ({phase_name})")
            continue

        print(f"  Phase: {phase_name} (step {actual_step})")

        # Load checkpoint
        model_path = ckpt_dir / f"step_{actual_step:06d}" / "model.pt"
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)

        # Run probe
        probe = HeadAblationProbe(config={
            "n_batches": N_PROBE_BATCHES,
            "ablation_types": ["zero", "mean"],
        })
        result = probe.run(model, loader, device=device)

        k_results[phase_name] = {
            "step": actual_step,
            "target_step": target_step,
            **probe._to_json_serializable(result),
        }

        # Print summary for this phase
        if "head_effects" in result:
            for abl_type in ["zero", "mean"]:
                if abl_type not in result["head_effects"]:
                    continue
                effects = result["head_effects"][abl_type]
                # Sort by loss_delta descending
                ranked = sorted(effects, key=lambda x: x["loss_delta"], reverse=True)
                top3 = ranked[:3]
                print(f"    [{abl_type}] Top 3 heads by loss delta:")
                for h_eff in top3:
                    print(f"      L{h_eff['layer']}H{h_eff['head']}: "
                          f"Δloss={h_eff['loss_delta']:+.4f}, "
                          f"Δz_gap={h_eff['z_gap_delta']:+.4f}")

        if "sufficiency" in result:
            suff = result["sufficiency"]
            best = min(suff, key=lambda x: x["loss_only_this_head"])
            print(f"    [sufficiency] Best single head: "
                  f"L{best['layer']}H{best['head']} "
                  f"(loss={best['loss_only_this_head']:.4f})")

    all_results[k] = {
        "tau": tau,
        "log_k": log_k,
        "phases": k_results,
    }

# ── Figure: heatmap of head contributions ──────────────────────────────────

for k in K_VALUES:
    if k not in all_results:
        continue
    kr = all_results[k]
    phase_names = sorted(kr["phases"].keys())

    if not phase_names:
        continue

    # Get n_layers, n_heads from first available phase
    first_phase = kr["phases"][phase_names[0]]
    n_layers = first_phase.get("n_layers", 4)
    n_heads = first_phase.get("n_heads", 4)

    fig, axes = plt.subplots(1, len(phase_names), figsize=(5 * len(phase_names), 4))
    if len(phase_names) == 1:
        axes = [axes]

    for idx, phase_name in enumerate(phase_names):
        phase_data = kr["phases"][phase_name]
        if "head_effects" not in phase_data or "zero" not in phase_data["head_effects"]:
            continue

        effects = phase_data["head_effects"]["zero"]
        grid = np.zeros((n_layers, n_heads))
        for eff in effects:
            grid[eff["layer"], eff["head"]] = eff["loss_delta"]

        ax = axes[idx]
        im = ax.imshow(grid, cmap="RdBu_r", aspect="auto",
                       vmin=-max(abs(grid.min()), abs(grid.max())),
                       vmax=max(abs(grid.min()), abs(grid.max())))
        ax.set_xlabel("Head", fontsize=9)
        ax.set_ylabel("Layer", fontsize=9)
        ax.set_title(f"{phase_name} (step {phase_data['step']})", fontsize=10)
        ax.set_xticks(range(n_heads))
        ax.set_yticks(range(n_layers))

        # Annotate cells
        for l in range(n_layers):
            for h in range(n_heads):
                ax.text(h, l, f"{grid[l, h]:.3f}", ha="center", va="center",
                        fontsize=7, color="black" if abs(grid[l, h]) < grid.max() * 0.5 else "white")

        plt.colorbar(im, ax=ax, shrink=0.8, label="Δloss (zero ablation)")

    fig.suptitle(f"K={k}: Per-head ablation (necessity)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(FIG_DIR / f"fig_head_ablation_k{k}.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved: fig_head_ablation_k{k}")

# ── Save summary ───────────────────────────────────────────────────────────

summary = {
    "experiment": "head_ablation",
    "description": "Per-head causal ablation (Exp 3)",
    "results": {str(k): v for k, v in all_results.items()},
}
summary_path = OUTPUT_DIR / "head_ablation_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary saved: {summary_path}")
