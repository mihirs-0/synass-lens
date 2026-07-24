#!/usr/bin/env python3
"""
Hessian Anisotropy vs K: The Closing Experiment

Tests whether the Hessian anisotropy ratio |λ_max/λ_min| scales with K,
and whether this scaling matches the plateau duration scaling τ ∝ K^1.3.

Causal chain: K → anisotropy K^γ → plateau duration K^α
If γ ≈ α ≈ 1.3, then geometric drowning quantitatively explains the lag.

Pipeline:
  Phase 1: Train models for each K (skip if outputs already exist)
  Phase 2: Compute Hessian eigenvalues at selected checkpoints
  Phase 3: Analyze scaling laws and generate figures
"""

import sys
import json
import math
import time
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from src.data import (
    create_tokenizer_from_config,
    create_datasets_from_config,
    collate_fn as dataset_collate_fn,
)
from src.model import create_model_from_config
from src.training import train

OUTPUTS = ROOT / "outputs"
SAVE_DIR = OUTPUTS / "paper_figures"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ── Configuration ────────────────────────────────────────────────────────────
K_VALUES = [3, 5, 7, 10, 13, 17, 20, 25, 30, 36]
HESSIAN_BATCH_SIZE = 512
N_POWER_ITER = 50

# Approximate τ values to set efficient training budgets
# Train to ~3.5× estimated τ, with early stopping at 5% of log(K)
APPROX_TAU = {3: 300, 5: 800, 7: 1200, 10: 2000, 13: 3000,
              17: 4500, 20: 5500, 25: 8000, 30: 10000, 36: 13000}

# ── Device selection ─────────────────────────────────────────────────────────

def select_device():
    """Pick device, testing that HVP with create_graph works."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            x = torch.randn(4, requires_grad=True, device="mps")
            y = (x ** 2).sum()
            g = torch.autograd.grad(y, x, create_graph=True)[0]
            g2 = torch.autograd.grad(g.sum(), x)[0]
            if g2 is not None:
                return "mps"
        except Exception:
            pass
    return "cpu"


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Training
# ═══════════════════════════════════════════════════════════════════════════════

def load_config(k):
    """Load the landauer_dense config for a given K."""
    config_path = ROOT / "configs" / "experiments" / f"landauer_dense_k{k}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    cfg = OmegaConf.load(config_path)

    # Ensure base config values are present (merge with base.yaml defaults)
    base_path = ROOT / "configs" / "base.yaml"
    if base_path.exists():
        base = OmegaConf.load(base_path)
        cfg = OmegaConf.merge(base, cfg)

    return cfg


def train_model_for_k(k, device):
    """Train a model for the given K if outputs don't already exist."""
    run_name = f"landauer_dense_k{k}"
    output_dir = OUTPUTS
    history_path = output_dir / run_name / "training_history.json"

    # Check if already trained
    if history_path.exists():
        print(f"  K={k}: Already trained, skipping.")
        return

    print(f"\n{'='*70}")
    print(f"  Training K={k}")
    print(f"{'='*70}")

    cfg = load_config(k)

    # Override max_steps for efficient training: ~3.5× estimated τ
    approx_tau = APPROX_TAU.get(k, 5000)
    efficient_max = max(2000, int(approx_tau * 3.5))
    # Cap at original max_steps
    efficient_max = min(efficient_max, cfg.training.max_steps)
    print(f"  Efficient max_steps: {efficient_max} (approx τ ≈ {approx_tau})")
    cfg.training.max_steps = efficient_max

    # Enable early stopping at 5% of log(K) for 20 consecutive evals
    cfg.training.early_stop_convergence_frac = 0.05

    torch.manual_seed(cfg.experiment.seed)

    tokenizer = create_tokenizer_from_config(cfg)
    train_dataset, probe_dataset, mapping_data = create_datasets_from_config(cfg, tokenizer)
    print(f"  Train examples: {len(train_dataset)}, Vocab: {tokenizer.vocab_size}")

    model = create_model_from_config(cfg, tokenizer)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} parameters on {device}")

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.training.batch_size,
        shuffle=True, collate_fn=dataset_collate_fn, num_workers=0,
    )
    probe_loader = DataLoader(
        probe_dataset, batch_size=cfg.training.batch_size,
        shuffle=False, collate_fn=dataset_collate_fn, num_workers=0,
    )

    # Save config
    config_save_path = output_dir / run_name / "config.yaml"
    config_save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_save_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # Disable gradient clipping for constant-LR experiments
    grad_clip = None if getattr(cfg.training, "scheduler", "cosine") == "constant" else 1.0

    history = train(
        model=model,
        train_loader=train_loader,
        probe_loader=probe_loader,
        cfg=cfg,
        output_dir=output_dir,
        grad_clip=grad_clip,
        mapping_data=mapping_data,
        tokenizer=tokenizer,
    )

    print(f"  K={k}: Training complete. Final loss: {history['train_loss'][-1]:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Hessian Eigenvalue Computation
# ═══════════════════════════════════════════════════════════════════════════════

def find_tau(k, threshold_frac=0.5):
    """Find the transition step τ where candidate_loss first drops below threshold."""
    run_name = f"landauer_dense_k{k}"
    hist_path = OUTPUTS / run_name / "training_history.json"
    if not hist_path.exists():
        return None

    with open(hist_path) as f:
        h = json.load(f)

    log_k = math.log(k)
    # Use first_target_loss (or candidate_loss if available)
    loss_key = "candidate_loss" if "candidate_loss" in h else "first_target_loss"
    for s, l in zip(h["steps"], h[loss_key]):
        if l < threshold_frac * log_k:
            return int(s)

    # If never crosses threshold, return last step
    return int(h["steps"][-1])


def get_final_step(k):
    """Get the final training step from history."""
    hist_path = OUTPUTS / f"landauer_dense_k{k}" / "training_history.json"
    if not hist_path.exists():
        return None
    with open(hist_path) as f:
        h = json.load(f)
    return int(h["steps"][-1]) if h["steps"] else None


def select_hessian_checkpoints(k, tau):
    """Select checkpoints for Hessian computation: plateau, transition, post-convergence."""
    ckpt_dir = OUTPUTS / f"landauer_dense_k{k}" / "checkpoints"
    if not ckpt_dir.exists():
        return []

    all_steps = sorted([
        int(d.name.split("_")[1]) for d in ckpt_dir.iterdir()
        if d.is_dir() and d.name.startswith("step_")
    ])

    if not all_steps or tau is None:
        return all_steps[:10]

    final_step = all_steps[-1]
    target_steps = set()

    # PLATEAU: 5 evenly spaced between max(100, τ*0.2) and τ*0.8
    plateau_start = max(100, int(tau * 0.2))
    plateau_end = int(tau * 0.8)
    if plateau_end > plateau_start:
        for i in range(5):
            frac = i / 4.0
            s = int(plateau_start + frac * (plateau_end - plateau_start))
            target_steps.add(s)

    # TRANSITION: τ*0.9, τ*1.0, τ*1.2
    for frac in [0.9, 1.0, 1.2]:
        target_steps.add(int(tau * frac))

    # POST-CONVERGENCE: τ*2.0, min(τ*3.0, final_step)
    target_steps.add(int(tau * 2.0))
    target_steps.add(min(int(tau * 3.0), final_step))

    # Map each target to nearest available checkpoint
    selected = set()
    for target in target_steps:
        nearest = min(all_steps, key=lambda s: abs(s - target))
        selected.add(nearest)

    return sorted(selected)


def hessian_vector_product(model, input_ids, labels, v):
    """Compute H @ v where H is the Hessian of cross-entropy loss."""
    model.zero_grad()

    logits = model(input_ids)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )

    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, create_graph=True)
    flat_grad = torch.cat([g.contiguous().view(-1) for g in grads])
    grad_norm = flat_grad.detach().norm().item()

    # H @ v = d/dθ (g · v)
    gv = flat_grad.dot(v)
    hvp_grads = torch.autograd.grad(gv, params)
    flat_hvp = torch.cat([g.contiguous().view(-1) for g in hvp_grads])

    return flat_hvp.detach(), loss.item(), grad_norm


def power_iteration_max(model, input_ids, labels, n_iter=N_POWER_ITER):
    """Find largest eigenvalue via standard power iteration."""
    device = next(model.parameters()).device
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    torch.manual_seed(0)
    v = torch.randn(n_params, device=device)
    v = v / v.norm()

    eigenvalue = 0.0
    loss_val = 0.0
    grad_norm = 0.0

    for i in range(n_iter):
        Hv, loss_val, grad_norm = hessian_vector_product(model, input_ids, labels, v)
        eigenvalue = v.dot(Hv).item()
        Hv_norm = Hv.norm()
        if Hv_norm < 1e-12:
            break
        v_new = Hv / Hv_norm
        cos = v.dot(v_new).abs().item()
        v = v_new
        if cos > 0.9999 and i > 10:
            break

    return eigenvalue, loss_val, grad_norm, i + 1, v


def power_iteration_min(model, input_ids, labels, lambda_max, n_iter=N_POWER_ITER):
    """Find smallest eigenvalue via shifted power iteration on (H - σI)."""
    device = next(model.parameters()).device
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    sigma = lambda_max + 1.0

    torch.manual_seed(1)
    v = torch.randn(n_params, device=device)
    v = v / v.norm()

    eigenvalue_shifted = 0.0

    for i in range(n_iter):
        Hv, _, _ = hessian_vector_product(model, input_ids, labels, v)
        shifted = Hv - sigma * v

        eigenvalue_shifted = v.dot(shifted).item()
        s_norm = shifted.norm()
        if s_norm < 1e-12:
            break
        v_new = shifted / s_norm
        cos = v.dot(v_new).abs().item()
        v = v_new
        if cos > 0.9999 and i > 10:
            break

    lambda_min = eigenvalue_shifted + sigma
    return lambda_min, i + 1


def prepare_hessian_batch(k, device):
    """Prepare a fixed data batch for Hessian computation."""
    cfg = load_config(k)
    tokenizer = create_tokenizer_from_config(cfg)
    train_dataset, _, _ = create_datasets_from_config(cfg, tokenizer)

    n_samples = min(HESSIAN_BATCH_SIZE, len(train_dataset))
    torch.manual_seed(42)
    indices = torch.randperm(len(train_dataset))[:n_samples].tolist()
    subset = Subset(train_dataset, indices)
    loader = DataLoader(subset, batch_size=n_samples, shuffle=False,
                        collate_fn=dataset_collate_fn)
    batch = next(iter(loader))
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    return cfg, tokenizer, input_ids, labels


def compute_hessian_for_k(k, device):
    """Compute Hessian eigenvalues at selected checkpoints for a given K."""
    run_name = f"landauer_dense_k{k}"
    results_path = OUTPUTS / f"hessian_anisotropy_k{k}.json"

    tau = find_tau(k)
    log_k = math.log(k)
    print(f"\n{'='*70}")
    print(f"  K={k}, τ={tau}, log(K)={log_k:.3f}")
    print(f"{'='*70}")

    if tau is None:
        print(f"  Cannot determine τ, skipping.")
        return None

    steps = select_hessian_checkpoints(k, tau)
    if not steps:
        print(f"  No checkpoints found, skipping.")
        return None

    print(f"  {len(steps)} checkpoints: {steps}")

    # Load existing partial results
    existing = {}
    if results_path.exists():
        with open(results_path) as f:
            old = json.load(f)
        if "results" in old:
            valid = {r["step"]: r for r in old["results"]
                     if abs(r.get("lambda_min", 0) - r.get("lambda_max", 0)) > 0.01}
            if valid:
                existing = valid
                print(f"  Resuming: {len(existing)} checkpoints cached")

    # Prepare data and model
    cfg, tokenizer, input_ids, labels = prepare_hessian_batch(k, device)
    model = create_model_from_config(cfg, tokenizer).to(device)

    ckpt_dir = OUTPUTS / run_name / "checkpoints"
    results = []

    for i, step in enumerate(steps):
        if step in existing:
            results.append(existing[step])
            continue

        model_path = ckpt_dir / f"step_{step:06d}" / "model.pt"
        if not model_path.exists():
            print(f"  [{i+1}/{len(steps)}] step {step}: checkpoint missing, skip")
            continue

        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(True)

        t0 = time.time()
        lambda_max, loss, grad_norm, iters_max, _ = power_iteration_max(
            model, input_ids, labels)
        lambda_min, iters_min = power_iteration_min(
            model, input_ids, labels, lambda_max)
        dt = time.time() - t0

        ratio = abs(lambda_max) / max(abs(lambda_min), 1e-10)
        phase = "PLATEAU" if step < tau * 0.8 else ("TRANS" if step < tau * 1.2 else "CONV")

        result = {
            "step": step,
            "lambda_max": float(lambda_max),
            "lambda_min": float(lambda_min),
            "ratio": float(ratio),
            "loss": float(loss),
            "grad_norm": float(grad_norm),
            "iters_max": iters_max,
            "iters_min": iters_min,
            "time_s": round(dt, 1),
            "phase": phase,
        }
        results.append(result)

        sign = "+" if lambda_min > 0 else "-"
        print(f"  [{i+1:2d}/{len(steps)}] step {step:6d} [{phase:>7s}]  "
              f"lmax={lambda_max:+10.4f}  lmin={lambda_min:+10.4f}({sign})  "
              f"ratio={ratio:.1f}  loss={loss:.4f}  ({dt:.1f}s)")

        # Save incrementally
        out = {"k": k, "tau": tau, "log_k": log_k, "results": results}
        with open(results_path, "w") as f:
            json.dump(out, f, indent=2)

    return {"k": k, "tau": tau, "log_k": log_k, "results": results}


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: Analysis and Figures
# ═══════════════════════════════════════════════════════════════════════════════

def run_analysis(all_hessian_data):
    """Analyze Hessian anisotropy scaling with K and generate figures."""
    print(f"\n{'='*70}")
    print("  PHASE 3: Analysis")
    print(f"{'='*70}")

    # Extract plateau anisotropy for each K
    plateau_anisotropy = {}
    plateau_anisotropy_std = {}
    plateau_lambda_max = {}
    plateau_lambda_min = {}
    taus = {}

    for k, data in sorted(all_hessian_data.items()):
        tau = data["tau"]
        taus[k] = tau
        results = data["results"]

        # Plateau checkpoints: phase == "PLATEAU"
        plateau = [r for r in results if r.get("phase") == "PLATEAU"]
        if not plateau:
            # Fallback: steps < tau * 0.8
            plateau = [r for r in results if r["step"] < tau * 0.8]

        if not plateau:
            print(f"  K={k}: No plateau checkpoints, skipping")
            continue

        ratios = [abs(r["lambda_max"]) / max(abs(r["lambda_min"]), 1e-10)
                  for r in plateau]
        lmaxes = [r["lambda_max"] for r in plateau]
        lmins = [abs(r["lambda_min"]) for r in plateau]

        plateau_anisotropy[k] = np.mean(ratios)
        plateau_anisotropy_std[k] = np.std(ratios)
        plateau_lambda_max[k] = np.mean(lmaxes)
        plateau_lambda_min[k] = np.mean(lmins)

        print(f"  K={k}: ratio={np.mean(ratios):.1f} +/- {np.std(ratios):.1f}  "
              f"lmax={np.mean(lmaxes):.4f}  |lmin|={np.mean(lmins):.4f}  "
              f"({len(plateau)} plateau ckpts)")

    if len(plateau_anisotropy) < 3:
        print("\n  INSUFFICIENT DATA for scaling analysis (need >= 3 K values)")
        return

    # ── Scaling fits ──────────────────────────────────────────────────────

    Ks = np.array(sorted(plateau_anisotropy.keys()), dtype=float)
    ratios = np.array([plateau_anisotropy[int(k)] for k in Ks])
    ratio_errs = np.array([plateau_anisotropy_std[int(k)] for k in Ks])
    lmaxes = np.array([plateau_lambda_max[int(k)] for k in Ks])
    lmins = np.array([plateau_lambda_min[int(k)] for k in Ks])
    tau_vals = np.array([taus[int(k)] for k in Ks])

    # Fit: |λ_max/λ_min| ∝ K^γ
    log_K = np.log(Ks)
    log_ratio = np.log(ratios)
    gamma, c = np.polyfit(log_K, log_ratio, 1)
    pred_log_ratio = gamma * log_K + c
    ss_res = np.sum((log_ratio - pred_log_ratio) ** 2)
    ss_tot = np.sum((log_ratio - log_ratio.mean()) ** 2)
    r2_gamma = 1 - ss_res / max(ss_tot, 1e-30)

    # Fit: τ ∝ K^α
    log_tau = np.log(tau_vals)
    alpha, c_tau = np.polyfit(log_K, log_tau, 1)
    pred_log_tau = alpha * log_K + c_tau
    ss_res_tau = np.sum((log_tau - pred_log_tau) ** 2)
    ss_tot_tau = np.sum((log_tau - log_tau.mean()) ** 2)
    r2_alpha = 1 - ss_res_tau / max(ss_tot_tau, 1e-30)

    # Fit: τ ∝ ratio^β
    beta, d = np.polyfit(log_ratio, log_tau, 1)
    pred_log_tau_from_ratio = beta * log_ratio + d
    ss_res_beta = np.sum((log_tau - pred_log_tau_from_ratio) ** 2)
    r2_beta = 1 - ss_res_beta / max(ss_tot_tau, 1e-30)

    # Component scaling: λ_max ∝ K^? and |λ_min| ∝ K^?
    exp_lmax, c_lmax = np.polyfit(log_K, np.log(np.maximum(lmaxes, 1e-10)), 1)
    exp_lmin, c_lmin = np.polyfit(log_K, np.log(np.maximum(lmins, 1e-10)), 1)

    # ── Summary Table ─────────────────────────────────────────────────────

    print(f"\n{'='*70}")
    print("HESSIAN ANISOTROPY VS K — RESULTS")
    print(f"{'='*70}")
    print(f"{'K':>5s}  {'lmax':>10s}  {'|lmin|':>10s}  {'Ratio':>10s}  {'log(R)':>8s}  {'tau':>8s}")
    print("-" * 70)
    for k_val in Ks:
        ki = int(k_val)
        print(f"{ki:5d}  {plateau_lambda_max[ki]:10.4f}  {plateau_lambda_min[ki]:10.6f}  "
              f"{plateau_anisotropy[ki]:10.1f}  {np.log10(plateau_anisotropy[ki]):8.3f}  "
              f"{taus[ki]:8d}")

    print(f"\nScaling laws:")
    print(f"  |lmax/lmin| ~ K^{gamma:.3f}   (R^2 = {r2_gamma:.4f})")
    print(f"  tau         ~ K^{alpha:.3f}   (R^2 = {r2_alpha:.4f})")
    print(f"  lmax        ~ K^{exp_lmax:.3f}")
    print(f"  |lmin|      ~ K^{exp_lmin:.3f}")
    print(f"  tau ~ ratio^{beta:.3f}        (R^2 = {r2_beta:.4f})")

    match = "YES" if abs(gamma - alpha) < 0.3 else ("PARTIAL" if abs(gamma - alpha) < 0.6 else "NO")
    direct = "YES" if r2_beta > 0.9 else ("PARTIAL" if r2_beta > 0.7 else "NO")

    print(f"\nVERDICT:")
    print(f"  Does anisotropy explain plateau duration?")
    print(f"  - gamma ~ alpha ({gamma:.3f} vs {alpha:.3f})?  {match}")
    print(f"  - tau ~ ratio^beta with R^2 > 0.9?  {direct}  (R^2 = {r2_beta:.4f})")
    print(f"{'='*70}")

    # Save results
    summary = {
        "K_values": Ks.tolist(),
        "plateau_anisotropy": {int(k): float(v) for k, v in plateau_anisotropy.items()},
        "plateau_anisotropy_std": {int(k): float(v) for k, v in plateau_anisotropy_std.items()},
        "plateau_lambda_max": {int(k): float(v) for k, v in plateau_lambda_max.items()},
        "plateau_lambda_min": {int(k): float(v) for k, v in plateau_lambda_min.items()},
        "taus": {int(k): int(v) for k, v in taus.items()},
        "scaling": {
            "gamma": float(gamma),
            "r2_gamma": float(r2_gamma),
            "alpha": float(alpha),
            "r2_alpha": float(r2_alpha),
            "beta": float(beta),
            "r2_beta": float(r2_beta),
            "exp_lmax": float(exp_lmax),
            "exp_lmin": float(exp_lmin),
            "match": match,
            "direct": direct,
        },
    }
    with open(OUTPUTS / "hessian_anisotropy_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to outputs/hessian_anisotropy_summary.json")

    # ── Figures ───────────────────────────────────────────────────────────

    # Use a clean style
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 150,
    })

    # ── Figure 1: Anisotropy vs K ─────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))

    ax.errorbar(Ks, ratios, yerr=ratio_errs, fmt="o", color="#E74C3C",
                markersize=6, capsize=3, linewidth=1.5, label="Plateau anisotropy")

    # Fit line
    K_fit = np.linspace(Ks.min() * 0.8, Ks.max() * 1.2, 100)
    ratio_fit = np.exp(c) * K_fit ** gamma
    ax.plot(K_fit, ratio_fit, "--", color="#E74C3C", linewidth=1.2, alpha=0.7,
            label=rf"$|{{\lambda_{{\max}}}}/\lambda_{{\min}}| \propto K^{{{gamma:.2f}}}$  ($R^2={r2_gamma:.3f}$)")

    # Overlay τ vs K (secondary y-axis)
    ax2 = ax.twinx()
    ax2.plot(Ks, tau_vals, "s-", color="#3498DB", markersize=5, linewidth=1.2, alpha=0.8,
             label=rf"$\tau \propto K^{{{alpha:.2f}}}$")
    tau_fit = np.exp(c_tau) * K_fit ** alpha
    ax2.plot(K_fit, tau_fit, ":", color="#3498DB", linewidth=1.0, alpha=0.5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax2.set_yscale("log")
    ax.set_xlabel("K (number of candidates)")
    ax.set_ylabel(r"$|\lambda_{\max}/\lambda_{\min}|$ (anisotropy)", color="#E74C3C")
    ax2.set_ylabel(r"$\tau$ (plateau duration, steps)", color="#3498DB")
    ax.tick_params(axis="y", labelcolor="#E74C3C")
    ax2.tick_params(axis="y", labelcolor="#3498DB")

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    ax.set_title(r"Hessian Anisotropy vs Task Complexity $K$")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.get_major_formatter().set_scientific(False)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(SAVE_DIR / f"hessian_anisotropy_vs_K.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: hessian_anisotropy_vs_K")

    # ── Figure 2: Eigenvalue trajectories ─────────────────────────────────
    # Select a few K values for detail
    detail_Ks = [k for k in [5, 10, 20, 36] if k in all_hessian_data]
    if not detail_Ks:
        detail_Ks = sorted(all_hessian_data.keys())[:4]

    n_detail = len(detail_Ks)
    fig, axes = plt.subplots(1, n_detail, figsize=(4 * n_detail, 3.5), squeeze=False)

    for idx, k_val in enumerate(detail_Ks):
        data = all_hessian_data[k_val]
        tau = data["tau"]
        results = data["results"]

        steps_arr = np.array([r["step"] for r in results])
        lmax = np.array([r["lambda_max"] for r in results])
        lmin_abs = np.array([abs(r["lambda_min"]) for r in results])

        # Normalize steps by τ
        steps_norm = steps_arr / tau

        ax = axes[0][idx]
        ax.semilogy(steps_norm, np.maximum(lmax, 1e-10), "o-", color="#E74C3C",
                    markersize=3, linewidth=1.2, label=r"$\lambda_{\max}$")
        ax.semilogy(steps_norm, np.maximum(lmin_abs, 1e-10), "s-", color="#3498DB",
                    markersize=3, linewidth=1.2, label=r"$|\lambda_{\min}|$")
        ax.axvline(1.0, color="green", linestyle="--", alpha=0.5, linewidth=0.8,
                   label=r"$\tau$")

        ax.set_xlabel(r"Step / $\tau$")
        if idx == 0:
            ax.set_ylabel("Eigenvalue magnitude")
        ax.set_title(f"K = {k_val}", fontweight="bold")
        ax.legend(fontsize=7)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(SAVE_DIR / f"hessian_eigenvalue_trajectories.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: hessian_eigenvalue_trajectories")

    # ── Figure 3: Component scaling ───────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    ax.loglog(Ks, lmaxes, "o-", color="#E74C3C", markersize=6, linewidth=1.5,
              label=rf"$\lambda_{{\max}} \propto K^{{{exp_lmax:.2f}}}$")
    ax.loglog(Ks, lmins, "s-", color="#3498DB", markersize=6, linewidth=1.5,
              label=rf"$|\lambda_{{\min}}| \propto K^{{{exp_lmin:.2f}}}$")

    # Fit lines
    lmax_fit = np.exp(c_lmax) * K_fit ** exp_lmax
    lmin_fit = np.exp(c_lmin) * K_fit ** exp_lmin
    ax.loglog(K_fit, lmax_fit, "--", color="#E74C3C", alpha=0.4, linewidth=1.0)
    ax.loglog(K_fit, lmin_fit, "--", color="#3498DB", alpha=0.4, linewidth=1.0)

    ax.set_xlabel("K (number of candidates)")
    ax.set_ylabel("Eigenvalue magnitude (plateau mean)")
    ax.set_title("Hessian Eigenvalue Components vs K")
    ax.legend()
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.get_major_formatter().set_scientific(False)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(SAVE_DIR / f"hessian_components_vs_K.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: hessian_components_vs_K")

    # ── Figure 4: τ vs anisotropy (closing the loop) ─────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    ax.loglog(ratios, tau_vals, "o", color="#2C3E50", markersize=8, linewidth=1.5,
              zorder=5)

    # Fit line
    ratio_fit_range = np.linspace(ratios.min() * 0.8, ratios.max() * 1.2, 100)
    tau_from_ratio_fit = np.exp(d) * ratio_fit_range ** beta
    ax.loglog(ratio_fit_range, tau_from_ratio_fit, "--", color="#E67E22",
              linewidth=1.5, alpha=0.7,
              label=rf"$\tau \propto R^{{{beta:.2f}}}$  ($R^2={r2_beta:.3f}$)")

    # Label each point with K value
    for ki, ri, ti in zip(Ks, ratios, tau_vals):
        ax.annotate(f"K={int(ki)}", (ri, ti), textcoords="offset points",
                    xytext=(5, 5), fontsize=7, alpha=0.7)

    ax.set_xlabel(r"Plateau anisotropy $|\lambda_{\max}/\lambda_{\min}|$")
    ax.set_ylabel(r"Plateau duration $\tau$ (steps)")
    ax.set_title(r"$\tau$ vs Hessian Anisotropy: Closing the Causal Loop")
    ax.legend(fontsize=9)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(SAVE_DIR / f"tau_vs_anisotropy.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: tau_vs_anisotropy")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    device = select_device()
    print(f"Device: {device}")
    print(f"K values: {K_VALUES}")
    print(f"Output: {OUTPUTS}")

    # ── Phase 1: Train models ────────────────────────────────────────────
    print(f"\n{'#'*70}")
    print("  PHASE 1: Training models")
    print(f"{'#'*70}")

    for k in K_VALUES:
        t0 = time.time()
        train_model_for_k(k, device)
        dt = time.time() - t0
        if dt > 1:
            print(f"  K={k}: {dt:.0f}s")

    # ── Phase 2: Hessian eigenvalue computation ──────────────────────────
    print(f"\n{'#'*70}")
    print("  PHASE 2: Hessian Eigenvalue Computation")
    print(f"{'#'*70}")

    all_hessian_data = {}
    for k in K_VALUES:
        data = compute_hessian_for_k(k, device)
        if data is not None:
            all_hessian_data[k] = data

    # ── Phase 3: Analysis and figures ────────────────────────────────────
    print(f"\n{'#'*70}")
    print("  PHASE 3: Analysis and Figures")
    print(f"{'#'*70}")

    if all_hessian_data:
        run_analysis(all_hessian_data)
    else:
        print("No Hessian data computed. Cannot analyze.")

    print("\nDone.")
