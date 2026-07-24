#!/usr/bin/env python
"""
Experiment 4: Hessian Robustness Checks

Reviewer concern: Power iteration is noisy. Need to verify eigenvalue
estimates are stable across batch sizes and iteration counts.

Design: Re-run Hessian eigenvalue computation at K=20 with varied settings:
- Batch sizes: 256, 512 (existing), 1024, 2048
- Power iteration counts: 25, 50 (existing), 100, 200
- Run at 5 checkpoints spanning plateau and transition
- Total: 4 × 4 = 16 settings × 5 checkpoints = 80 computations

Expected output: Table showing λ_min/λ_max ± std across settings.
If CV < 0.1, estimates are robust.
"""

import sys
import json
import math
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import Subset, DataLoader

from src.data.tokenizer import create_tokenizer_from_config
from src.data.dataset import create_datasets_from_config, collate_fn as dataset_collate_fn
from src.model.hooked_transformer import create_model_from_config
from scripts.experiment_helpers import load_history, detect_tau

# Import Hessian functions from existing script
from scripts.posthoc_hessian_eigenvalue import (
    hessian_vector_product, power_iteration_max, power_iteration_min,
    select_device,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ─────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("outputs")
FIG_DIR = OUTPUT_DIR / "paper_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

K = 20
RUN_NAME = f"landauer_dense_k{K}"
BATCH_SIZES = [256, 512, 1024, 2048]
N_POWER_ITERS = [25, 50, 100, 200]
N_CHECKPOINTS = 5  # Evenly spaced across plateau + transition


# ── Select checkpoints ─────────────────────────────────────────────────────

def select_analysis_checkpoints(ckpt_dir, tau, n=5):
    """Select n checkpoints spanning plateau and transition."""
    all_steps = sorted([
        int(d.name.split("_")[1])
        for d in ckpt_dir.iterdir()
        if d.is_dir() and d.name.startswith("step_")
    ])
    if not all_steps:
        return []

    if tau is None:
        # Evenly space through all checkpoints
        indices = np.linspace(0, len(all_steps) - 1, n, dtype=int)
        return [all_steps[i] for i in indices]

    # Select: 2 pre-τ, 1 at τ, 2 post-τ
    pre = [s for s in all_steps if s < tau * 0.8]
    at_tau = min(all_steps, key=lambda s: abs(s - tau))
    post = [s for s in all_steps if s > tau * 1.5]

    selected = []
    if pre:
        # Early plateau + late plateau
        selected.append(pre[len(pre) // 4])
        selected.append(pre[-1])
    selected.append(at_tau)
    if post:
        selected.append(post[min(2, len(post) - 1)])
        if len(post) > 5:
            selected.append(post[-1])

    return sorted(set(selected))[:n]


# ── Main ───────────────────────────────────────────────────────────────────

device = select_device()
print(f"Device: {device}")

print("=" * 70)
print("EXPERIMENT 4: HESSIAN ROBUSTNESS CHECKS")
print("=" * 70)

config_path = OUTPUT_DIR / RUN_NAME / "config.yaml"
if not config_path.exists():
    print(f"ERROR: No config found at {config_path}")
    sys.exit(1)

cfg = OmegaConf.load(config_path)
h = load_history(RUN_NAME, str(OUTPUT_DIR))
if h is None:
    print("ERROR: No training history found")
    sys.exit(1)

log_k = math.log(K)
tau = detect_tau(h, log_k)
print(f"K = {K}, τ = {tau}, log K = {log_k:.3f}")

# Create model and dataset
tokenizer = create_tokenizer_from_config(cfg)
train_ds, _, _ = create_datasets_from_config(cfg, tokenizer)
model = create_model_from_config(cfg, tokenizer).to(device)

ckpt_dir = OUTPUT_DIR / RUN_NAME / "checkpoints"
ckpt_steps = select_analysis_checkpoints(ckpt_dir, tau, N_CHECKPOINTS)
print(f"Selected checkpoints: {ckpt_steps}")

# ── Run grid ───────────────────────────────────────────────────────────────

results = []

for step in ckpt_steps:
    model_path = ckpt_dir / f"step_{step:06d}" / "model.pt"
    if not model_path.exists():
        print(f"  Checkpoint step_{step:06d} not found, skipping")
        continue

    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    step_results = []

    for bs in BATCH_SIZES:
        # Create batch of given size
        n_samples = min(bs, len(train_ds))
        torch.manual_seed(42)
        indices = torch.randperm(len(train_ds))[:n_samples].tolist()
        subset = Subset(train_ds, indices)
        loader = DataLoader(subset, batch_size=n_samples, shuffle=False,
                            collate_fn=dataset_collate_fn)
        batch = next(iter(loader))
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        for n_iter in N_POWER_ITERS:
            t0 = time.time()

            lambda_max, loss, grad_norm, iters_max, _ = power_iteration_max(
                model, input_ids, labels, n_iter=n_iter)
            lambda_min, iters_min = power_iteration_min(
                model, input_ids, labels, lambda_max, n_iter=n_iter)

            dt = time.time() - t0

            result = {
                "step": step,
                "batch_size": bs,
                "n_iter": n_iter,
                "lambda_max": float(lambda_max),
                "lambda_min": float(lambda_min),
                "condition_ratio": float(abs(lambda_max / lambda_min)) if abs(lambda_min) > 1e-10 else float("inf"),
                "loss": float(loss),
                "grad_norm": float(grad_norm),
                "iters_max": iters_max,
                "iters_min": iters_min,
                "time_s": round(dt, 1),
            }
            step_results.append(result)
            results.append(result)

            print(f"  step={step:6d} bs={bs:5d} iter={n_iter:4d}  "
                  f"λ_max={lambda_max:+10.4f}  λ_min={lambda_min:+10.4f}  "
                  f"({dt:.1f}s)")

# ── Compute robustness stats ───────────────────────────────────────────────

print(f"\n{'Step':<8} {'λ_max mean':>12} {'λ_max CV':>10} {'λ_min mean':>12} {'λ_min CV':>10}")
print("-" * 56)

robustness = {}
for step in ckpt_steps:
    step_data = [r for r in results if r["step"] == step]
    if not step_data:
        continue
    lmax_vals = np.array([r["lambda_max"] for r in step_data])
    lmin_vals = np.array([r["lambda_min"] for r in step_data])

    lmax_mean = np.mean(lmax_vals)
    lmax_std = np.std(lmax_vals)
    lmax_cv = lmax_std / abs(lmax_mean) if abs(lmax_mean) > 1e-10 else float("inf")

    lmin_mean = np.mean(lmin_vals)
    lmin_std = np.std(lmin_vals)
    lmin_cv = lmin_std / abs(lmin_mean) if abs(lmin_mean) > 1e-10 else float("inf")

    robustness[step] = {
        "lambda_max": {"mean": float(lmax_mean), "std": float(lmax_std), "cv": float(lmax_cv)},
        "lambda_min": {"mean": float(lmin_mean), "std": float(lmin_std), "cv": float(lmin_cv)},
    }

    print(f"  {step:<8} {lmax_mean:>+12.4f} {lmax_cv:>10.4f} "
          f"{lmin_mean:>+12.4f} {lmin_cv:>10.4f}")

# Overall assessment
all_cv_max = [v["lambda_max"]["cv"] for v in robustness.values()]
all_cv_min = [v["lambda_min"]["cv"] for v in robustness.values()]
if all_cv_max and all_cv_min:
    max_cv = max(max(all_cv_max), max(all_cv_min))
    print(f"\nMax CV across all checkpoints: {max_cv:.4f}")
    if max_cv < 0.1:
        print("→ ROBUST: All CVs < 0.1")
    elif max_cv < 0.2:
        print("→ MODERATELY ROBUST: Max CV < 0.2")
    else:
        print("→ SENSITIVE: Max CV ≥ 0.2 — consider increasing iterations")

# ── Figure ─────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Panel A: λ_max sensitivity
ax = axes[0]
for bs in BATCH_SIZES:
    bs_data = [r for r in results if r["batch_size"] == bs]
    for n_iter in N_POWER_ITERS:
        subset = [r for r in bs_data if r["n_iter"] == n_iter]
        if not subset:
            continue
        steps = [r["step"] for r in subset]
        lmax = [r["lambda_max"] for r in subset]
        ax.plot(steps, lmax, "o-", markersize=3, linewidth=0.8, alpha=0.6,
                label=f"bs={bs},iter={n_iter}" if n_iter == N_POWER_ITERS[0] and bs == BATCH_SIZES[0] else None)

if tau:
    ax.axvline(tau, color="green", linestyle="--", alpha=0.6, label=f"τ={tau}")
ax.set_xlabel("Step", fontsize=10)
ax.set_ylabel("λ_max", fontsize=10)
ax.set_title("λ_max across settings", fontsize=11, fontweight="bold")
ax.tick_params(labelsize=8)

# Panel B: λ_min sensitivity
ax = axes[1]
for bs in BATCH_SIZES:
    bs_data = [r for r in results if r["batch_size"] == bs]
    for n_iter in N_POWER_ITERS:
        subset = [r for r in bs_data if r["n_iter"] == n_iter]
        if not subset:
            continue
        steps = [r["step"] for r in subset]
        lmin = [r["lambda_min"] for r in subset]
        ax.plot(steps, lmin, "o-", markersize=3, linewidth=0.8, alpha=0.6)

if tau:
    ax.axvline(tau, color="green", linestyle="--", alpha=0.6, label=f"τ={tau}")
ax.axhline(0, color="red", linestyle="-", alpha=0.3)
ax.set_xlabel("Step", fontsize=10)
ax.set_ylabel("λ_min", fontsize=10)
ax.set_title("λ_min across settings", fontsize=11, fontweight="bold")
ax.tick_params(labelsize=8)

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(FIG_DIR / f"fig_hessian_robustness.{ext}", dpi=300,
                bbox_inches="tight")
plt.close()
print(f"\nFigure saved: {FIG_DIR}/fig_hessian_robustness")

# ── Save summary ───────────────────────────────────────────────────────────

summary = {
    "experiment": "hessian_robustness",
    "description": "Hessian robustness checks (Exp 4)",
    "k": K,
    "tau": tau,
    "batch_sizes": BATCH_SIZES,
    "n_power_iters": N_POWER_ITERS,
    "checkpoints": ckpt_steps,
    "results": results,
    "robustness": {str(k): v for k, v in robustness.items()},
}
summary_path = OUTPUT_DIR / "hessian_robustness_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary saved: {summary_path}")
