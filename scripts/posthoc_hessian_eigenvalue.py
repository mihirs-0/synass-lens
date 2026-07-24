#!/usr/bin/env python3
"""
Post-hoc Experiment 8: Hessian Eigenvalue Tracking (Spinodal Test)

Tests the spinodal decomposition prediction: during the metastable plateau,
the minimum Hessian eigenvalue should be positive (locally stable uniform
solution). At the transition τ, λ_min should cross zero — the uniform-over-K
solution becomes linearly unstable, triggering irreversible decomposition.

  Spinodal prediction:  λ_min > 0 (plateau) → λ_min = 0 (at τ) → λ_min < 0 (post-τ)
  Nucleation (ruled out): λ_min > 0 throughout, transition via stochastic barrier hopping

Method: Power iteration for λ_max and λ_min via Hessian-vector products.
Each HVP = 2 backward passes. ~50 iterations per eigenvalue.
"""

import sys
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from omegaconf import OmegaConf
from src.data.tokenizer import create_tokenizer_from_config
from src.data.dataset import create_datasets_from_config, collate_fn as dataset_collate_fn
from src.model.hooked_transformer import create_model_from_config

OUTPUTS = ROOT / "outputs"
SAVE_DIR = OUTPUTS / "paper_figures"
SAVE_DIR.mkdir(exist_ok=True)

# ── Config ──────────────────────────────────────────────────────────────────
K_VALUES = [10, 20, 36]
HESSIAN_BATCH_SIZE = 512
N_POWER_ITER = 50
CHECKPOINT_STRIDE = 1000  # Sparse sampling stride
DENSE_STRIDE = 200        # Dense sampling around τ
DENSE_WINDOW = 2000       # ±2000 steps around τ


# ── Helpers ─────────────────────────────────────────────────────────────────

def load_config(run_name):
    config_path = OUTPUTS / run_name / "config.yaml"
    if not config_path.exists():
        return None
    return OmegaConf.load(config_path)


def find_tau(run_name, k, threshold_frac=0.5):
    hist_path = OUTPUTS / run_name / "training_history.json"
    if not hist_path.exists():
        return None
    with open(hist_path) as f:
        h = json.load(f)
    log_k = math.log(k)
    for s, l in zip(h["steps"], h["first_target_loss"]):
        if l < threshold_frac * log_k:
            return int(s)
    return None


def select_checkpoints(ckpt_dir, tau):
    """Sparse globally, dense around τ."""
    all_steps = sorted([
        int(d.name.split("_")[1]) for d in ckpt_dir.iterdir()
        if d.is_dir() and d.name.startswith("step_")
    ])

    if tau is None:
        return [s for s in all_steps if s % CHECKPOINT_STRIDE == 0]

    selected = set()
    for s in all_steps:
        if abs(s - tau) <= DENSE_WINDOW and s % DENSE_STRIDE == 0:
            selected.add(s)
        elif s % CHECKPOINT_STRIDE == 0:
            selected.add(s)
    return sorted(selected)


# ── Hessian-Vector Product ──────────────────────────────────────────────────

def hessian_vector_product(model, input_ids, labels, v):
    """
    Compute H @ v where H is the Hessian of the cross-entropy loss
    w.r.t. all model parameters.

    Returns: (Hv, loss_value, gradient_norm)
    """
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
    """
    Find smallest eigenvalue via shifted power iteration on (H - σI).

    With σ = λ_max + 1, all eigenvalues of (H - σI) are negative.
    The largest-magnitude eigenvalue of (H - σI) corresponds to the
    smallest eigenvalue of H. Power iteration converges to it.
    """
    device = next(model.parameters()).device
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    sigma = lambda_max + 1.0  # Shift: ensures all shifted eigenvalues < 0

    torch.manual_seed(1)  # Different seed from λ_max
    v = torch.randn(n_params, device=device)
    v = v / v.norm()

    eigenvalue_shifted = 0.0

    for i in range(n_iter):
        Hv, _, _ = hessian_vector_product(model, input_ids, labels, v)
        shifted = Hv - sigma * v  # (H - σI)v

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


# ── Device selection ────────────────────────────────────────────────────────

def select_device():
    """Pick device, testing that HVP with create_graph works."""
    if torch.cuda.is_available():
        return "cuda"

    # MPS may not support create_graph=True — test it
    if torch.backends.mps.is_available():
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


# ── Per-K runner ────────────────────────────────────────────────────────────

def run_for_k(k, device):
    run_name = f"landauer_dense_k{k}"
    print(f"\n{'=' * 80}")
    print(f"K = {k}  (run: {run_name})")
    print(f"{'=' * 80}")

    cfg = load_config(run_name)
    if cfg is None:
        print(f"  No config found, skipping.")
        return None

    tokenizer = create_tokenizer_from_config(cfg)
    train_dataset, _, _ = create_datasets_from_config(cfg, tokenizer)
    model = create_model_from_config(cfg, tokenizer).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} parameters on {device}")

    # Fixed subsample
    from torch.utils.data import Subset, DataLoader
    n_samples = min(HESSIAN_BATCH_SIZE, len(train_dataset))
    torch.manual_seed(42)
    indices = torch.randperm(len(train_dataset))[:n_samples].tolist()
    subset = Subset(train_dataset, indices)
    loader = DataLoader(subset, batch_size=n_samples, shuffle=False,
                        collate_fn=dataset_collate_fn)
    batch = next(iter(loader))
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    tau = find_tau(run_name, k)
    log_k = math.log(k)
    print(f"  τ = {tau}, log K = {log_k:.3f}")

    ckpt_dir = OUTPUTS / run_name / "checkpoints"
    if not ckpt_dir.exists():
        print(f"  No checkpoints, skipping.")
        return None

    steps = select_checkpoints(ckpt_dir, tau)
    print(f"  {len(steps)} checkpoints selected")

    # Check for existing partial results (only use if λ_min ≠ λ_max, i.e. not buggy)
    results_path = OUTPUTS / f"hessian_eigenvalues_k{k}.json"
    existing = {}
    if results_path.exists():
        with open(results_path) as f:
            old = json.load(f)
        if "results" in old:
            valid = {r["step"]: r for r in old["results"]
                     if abs(r["lambda_min"] - r["lambda_max"]) > 0.01}
            if valid:
                existing = valid
                print(f"  Resuming: {len(existing)} valid checkpoints cached")

    results = []
    for i, step in enumerate(steps):
        # Skip already-computed
        if step in existing:
            results.append(existing[step])
            continue

        model_path = ckpt_dir / f"step_{step:06d}" / "model.pt"
        if not model_path.exists():
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

        result = {
            "step": step,
            "lambda_max": float(lambda_max),
            "lambda_min": float(lambda_min),
            "loss": float(loss),
            "grad_norm": float(grad_norm),
            "iters_max": iters_max,
            "iters_min": iters_min,
            "time_s": round(dt, 1),
        }
        results.append(result)

        phase = "PLATEAU" if loss > 0.5 * log_k else ("TRANS" if loss > 0.05 * log_k else "CONV")
        sign = "+" if lambda_min > 0 else "−"
        print(f"  [{i+1:3d}/{len(steps)}] step {step:6d} [{phase:>7s}]  "
              f"λ_max={lambda_max:+10.4f}  λ_min={lambda_min:+10.4f}({sign})  "
              f"loss={loss:.4f}  |∇|={grad_norm:.4f}  ({dt:.1f}s)")

        # Save incrementally
        out = {"k": k, "tau": tau, "log_k": log_k, "results": results}
        with open(results_path, "w") as f:
            json.dump(out, f, indent=2)

    return {"k": k, "tau": tau, "log_k": log_k, "results": results}


# ── Main ────────────────────────────────────────────────────────────────────

device = select_device()
print(f"Device: {device}")

all_results = {}
for k in K_VALUES:
    data = run_for_k(k, device)
    if data is not None:
        all_results[k] = data


# ── Figure 1: Eigenvalue trajectories ───────────────────────────────────────

n_k = len(all_results)
fig, axes = plt.subplots(2, n_k, figsize=(5 * n_k, 7), squeeze=False)

for idx, (k, data) in enumerate(sorted(all_results.items())):
    results = data["results"]
    tau = data["tau"]
    log_k = data["log_k"]

    steps = np.array([r["step"] for r in results])
    lmax = np.array([r["lambda_max"] for r in results])
    lmin = np.array([r["lambda_min"] for r in results])
    losses = np.array([r["loss"] for r in results])
    gnorms = np.array([r["grad_norm"] for r in results])

    # Top row: eigenvalues
    ax = axes[0][idx]
    ax.plot(steps, lmax, color="#E74C3C", linewidth=1.2, label=r"$\lambda_{\max}$")
    ax.plot(steps, lmin, color="#3498DB", linewidth=1.2, label=r"$\lambda_{\min}$")
    ax.axhline(0, color="gray", linestyle="-", alpha=0.3, linewidth=0.8)
    if tau:
        ax.axvline(tau, color="green", linestyle="--", alpha=0.6,
                   label=f"$\\tau = {tau}$")
    ax.set_xlabel("Step", fontsize=9)
    ax.set_ylabel("Hessian eigenvalue", fontsize=9)
    ax.set_title(f"K = {k}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, loc="best")
    ax.tick_params(labelsize=7)

    # Bottom row: loss + grad norm
    ax2 = axes[1][idx]
    color_loss = "#2C3E50"
    color_grad = "#E67E22"
    ln1 = ax2.plot(steps, losses, color=color_loss, linewidth=1.2, label="Loss")
    ax2.axhline(log_k, color=color_loss, linestyle=":", alpha=0.4)
    ax2.set_xlabel("Step", fontsize=9)
    ax2.set_ylabel("Loss", fontsize=9, color=color_loss)
    ax2.tick_params(labelsize=7)
    if tau:
        ax2.axvline(tau, color="green", linestyle="--", alpha=0.6)

    ax2r = ax2.twinx()
    ln2 = ax2r.plot(steps, gnorms, color=color_grad, linewidth=0.8, alpha=0.7,
                    label=r"$\|\nabla L\|$")
    ax2r.set_ylabel(r"$\|\nabla L\|$", fontsize=9, color=color_grad)
    ax2r.tick_params(labelsize=7)

    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, fontsize=7, loc="best")

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(SAVE_DIR / f"fig_hessian_eigenvalues.{ext}", dpi=300,
                bbox_inches="tight")
plt.close()
print(f"\nFigure saved: fig_hessian_eigenvalues")


# ── Figure 2: λ_min zoom around τ ──────────────────────────────────────────

fig, axes = plt.subplots(1, n_k, figsize=(5 * n_k, 3.5), squeeze=False)

for idx, (k, data) in enumerate(sorted(all_results.items())):
    results = data["results"]
    tau = data["tau"]
    log_k = data["log_k"]
    if tau is None:
        continue

    steps = np.array([r["step"] for r in results])
    lmin = np.array([r["lambda_min"] for r in results])

    # Window: τ ± 3× dense_window
    window = 3 * DENSE_WINDOW
    mask = np.abs(steps - tau) <= window
    if mask.sum() < 3:
        mask = np.ones(len(steps), dtype=bool)

    ax = axes[0][idx]
    ax.plot(steps[mask], lmin[mask], "o-", color="#3498DB", markersize=3,
            linewidth=1.2)
    ax.axhline(0, color="red", linestyle="-", linewidth=1.0, alpha=0.6)
    ax.axvline(tau, color="green", linestyle="--", alpha=0.6,
               label=f"$\\tau = {tau}$")

    # Shade positive/negative regions
    ax.fill_between(steps[mask], 0, lmin[mask],
                    where=lmin[mask] > 0, alpha=0.15, color="blue",
                    label=r"$\lambda_{\min} > 0$ (stable)")
    ax.fill_between(steps[mask], 0, lmin[mask],
                    where=lmin[mask] < 0, alpha=0.15, color="red",
                    label=r"$\lambda_{\min} < 0$ (unstable)")

    ax.set_xlabel("Step", fontsize=9)
    ax.set_ylabel(r"$\lambda_{\min}$", fontsize=9)
    ax.set_title(f"K = {k}: Stability test around $\\tau$",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(SAVE_DIR / f"fig_hessian_lambda_min_zoom.{ext}", dpi=300,
                bbox_inches="tight")
plt.close()
print(f"Figure saved: fig_hessian_lambda_min_zoom")


# ── Summary ─────────────────────────────────────────────────────────────────

print(f"\n{'=' * 80}")
print("SPINODAL DECOMPOSITION TEST")
print(f"{'=' * 80}")
print(f"Prediction: λ_min > 0 during plateau, crosses zero at τ, < 0 post-τ")
print()

for k, data in sorted(all_results.items()):
    results = data["results"]
    tau = data["tau"]
    log_k = data["log_k"]

    if not results or tau is None:
        print(f"K={k}: insufficient data (tau={tau})")
        continue

    pre = [r for r in results if r["step"] < tau * 0.8]
    at_tau = [r for r in results if abs(r["step"] - tau) <= DENSE_STRIDE * 2]
    post = [r for r in results if r["step"] > tau * 1.2]

    if pre:
        pre_lmin = [r["lambda_min"] for r in pre]
        pre_mean = np.mean(pre_lmin)
        pre_std = np.std(pre_lmin)
        pre_all_pos = all(x > 0 for x in pre_lmin[-5:])
    else:
        pre_mean = pre_std = float("nan")
        pre_all_pos = False

    if at_tau:
        tau_lmin = [r["lambda_min"] for r in at_tau]
        tau_mean = np.mean(tau_lmin)
    else:
        tau_mean = float("nan")

    if post:
        post_lmin = [r["lambda_min"] for r in post[:10]]
        post_mean = np.mean(post_lmin)
        post_any_neg = any(x < 0 for x in post_lmin)
    else:
        post_mean = float("nan")
        post_any_neg = False

    # Trend analysis: does λ_min decrease toward τ?
    if len(pre) >= 6:
        early = np.mean([r["lambda_min"] for r in pre[:len(pre)//3]])
        late = np.mean([r["lambda_min"] for r in pre[-len(pre)//3:]])
        trending_down = late < early
    else:
        trending_down = False

    print(f"K={k} (τ={tau}, log K={log_k:.3f}):")
    print(f"  Pre-τ  λ_min:  {pre_mean:+.4f} ± {pre_std:.4f}  "
          f"(last 5 all positive: {pre_all_pos})")
    print(f"  At τ   λ_min:  {tau_mean:+.4f}")
    print(f"  Post-τ λ_min:  {post_mean:+.4f}  "
          f"(any negative: {post_any_neg})")
    print(f"  Trending toward zero pre-τ: {trending_down}")

    if pre_all_pos and post_any_neg:
        print(f"  → SPINODAL CONFIRMED: λ_min crosses from + to − at τ")
    elif trending_down and post_any_neg:
        print(f"  → PARTIAL SUPPORT: λ_min trends down and goes negative post-τ")
    elif trending_down:
        print(f"  → WEAK SUPPORT: λ_min trends down but doesn't cross zero")
    elif post_any_neg:
        print(f"  → MIXED: λ_min negative post-τ but no clear pre-τ trend")
    else:
        print(f"  → NOT CONFIRMED: no clear sign change at τ")
    print()

print(f"{'=' * 80}")
print("INTERPRETATION")
print(f"{'=' * 80}")
print("If λ_min crosses zero at τ → spinodal decomposition: the uniform")
print("solution becomes linearly unstable, triggering irreversible transition.")
print("If λ_min stays positive → the metastable phase remains locally stable;")
print("the transition is driven by finite-amplitude perturbations, not instability.")
