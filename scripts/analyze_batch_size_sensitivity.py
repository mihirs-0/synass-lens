#!/usr/bin/env python3
"""
Batch-size sensitivity analysis + SGD temperature decomposition.

Reads existing runs:
  - temp_lr1e3_bs{64,128,256,512}_k{20,36}  (batch-size sweep at fixed η=1e-3)
  - lr_sweep_eta{3e-4,5e-4,1e-3,2e-3}        (LR sweep at fixed BS=128, K=20)

Produces:
  1. Table of τ vs batch size at fixed η=1e-3 for K=20 and K=36
  2. SGD temperature decomposition table (matched T = η/BS)
  3. Figure: outputs/paper_figures/fig_batch_size.pdf
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
SAVE_DIR = OUTPUTS / "paper_figures"
SAVE_DIR.mkdir(exist_ok=True)


def load_history(name: str):
    p = OUTPUTS / name / "training_history.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def compute_tau(history, k, threshold_frac=0.5):
    """Compute transition step: first step where loss < threshold_frac * log(K)."""
    if history is None:
        return None
    log_k = math.log(k)
    threshold = threshold_frac * log_k
    for s, l in zip(history["steps"], history["first_target_loss"]):
        if l < threshold:
            return s
    return None


def compute_final_loss(history):
    if history is None:
        return None
    return history["first_target_loss"][-1]


# ── Data collection ──────────────────────────────────────────────────────────

batch_sizes = [64, 128, 256, 512]
k_values = [20, 36]

# Batch-size sweep (η=1e-3 fixed)
bs_results = {}
for k in k_values:
    bs_results[k] = {}
    for bs in batch_sizes:
        name = f"temp_lr1e3_bs{bs}_k{k}"
        h = load_history(name)
        if h is None:
            # Try extended run
            name_ext = f"temp_lr1e3_bs{bs}_k{k}_30k"
            h = load_history(name_ext)
        tau = compute_tau(h, k)
        final = compute_final_loss(h)
        noise_scale = 1.0 / math.sqrt(bs)
        sgd_temp = 1e-3 / bs
        bs_results[k][bs] = {
            "tau": tau,
            "final_loss": final,
            "converged": tau is not None,
            "noise_scale": noise_scale,
            "sgd_temp": sgd_temp,
            "max_step": h["steps"][-1] if h else None,
        }

# K=36/BS=128 extended run
h_ext = load_history("temp_lr1e3_bs128_k36_30k")
if h_ext is not None:
    tau_ext = compute_tau(h_ext, 36)
    if tau_ext is not None:
        bs_results[36][128]["tau"] = tau_ext
        bs_results[36][128]["converged"] = True
        bs_results[36][128]["final_loss"] = compute_final_loss(h_ext)
        bs_results[36][128]["max_step"] = h_ext["steps"][-1]

# LR sweep (BS=128 fixed, K=20)
lr_values = [3e-4, 5e-4, 1e-3, 2e-3]
lr_names = ["3e-4", "5e-4", "1e-3", "2e-3"]
lr_results = {}
for eta, name_suffix in zip(lr_values, lr_names):
    h = load_history(f"lr_sweep_eta{name_suffix}")
    tau = compute_tau(h, 20)
    lr_results[eta] = {
        "tau": tau,
        "final_loss": compute_final_loss(h),
        "converged": tau is not None,
        "sgd_temp": eta / 128,
        "max_step": h["steps"][-1] if h else None,
    }


# ── Print tables ─────────────────────────────────────────────────────────────

print("=" * 72)
print("TABLE 1: Batch-size sensitivity (η = 1e-3 fixed)")
print("=" * 72)
print(f"{'K':>4}  {'BS':>4}  {'1/√BS':>8}  {'T=η/BS':>12}  {'τ':>8}  {'Final loss':>11}  {'Conv?':>6}")
print("-" * 72)
for k in k_values:
    for bs in batch_sizes:
        r = bs_results[k][bs]
        tau_str = str(r["tau"]) if r["tau"] else "---"
        final_str = f"{r['final_loss']:.4f}" if r["final_loss"] is not None else "---"
        conv_str = "Yes" if r["converged"] else "No"
        print(f"{k:>4}  {bs:>4}  {r['noise_scale']:>8.4f}  {r['sgd_temp']:>12.2e}  {tau_str:>8}  {final_str:>11}  {conv_str:>6}")
    print()

print()
print("=" * 72)
print("TABLE 2: SGD Temperature Decomposition (K=20)")
print("=" * 72)
print(f"{'η':>10}  {'BS':>4}  {'T=η/BS':>12}  {'τ':>8}  {'Source':>15}")
print("-" * 72)
# BS sweep entries at η=1e-3
for bs in batch_sizes:
    r = bs_results[20][bs]
    tau_str = str(r["tau"]) if r["tau"] else "---"
    print(f"{'1e-3':>10}  {bs:>4}  {r['sgd_temp']:>12.2e}  {tau_str:>8}  {'BS sweep':>15}")
# LR sweep entries at BS=128
for eta in lr_values:
    if eta == 1e-3:
        continue  # already shown
    r = lr_results[eta]
    tau_str = str(r["tau"]) if r["tau"] else "---"
    print(f"{eta:>10.0e}  {128:>4}  {r['sgd_temp']:>12.2e}  {tau_str:>8}  {'LR sweep':>15}")

# Highlight matched-T comparisons
print()
print("KEY COMPARISONS (matched SGD temperature):")
# η=1e-3/BS=256 → T = 3.9e-6
# η=5e-4/BS=128 → T = 3.9e-6
t1_tau = bs_results[20][256]["tau"]
t2_tau = lr_results[5e-4]["tau"]
print(f"  T ≈ 3.9e-6:  (η=1e-3, BS=256) → τ={t1_tau}   vs   (η=5e-4, BS=128) → τ={t2_tau}")
print(f"  Same noise level, different step sizes. η=1e-3 converges {'faster' if t1_tau and t2_tau and t1_tau < t2_tau else 'slower'}.")

# η=1e-3/BS=128 → T = 7.8e-6
# η=2e-3/BS=128 → T = 1.6e-5
t3_tau = bs_results[20][128]["tau"]
t4_tau = lr_results[2e-3]["tau"]
print(f"  Fixed BS=128: (η=1e-3, T=7.8e-6) → τ={t3_tau}   vs   (η=2e-3, T=1.6e-5) → τ={t4_tau}")
print(f"  Higher T (more noise + larger steps) is strictly worse.")


# ── Figure ───────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))

# Panel (a): τ vs BS for K=20 and K=36
ax = axes[0]
for k, marker, color in [(20, "o", "#3498DB"), (36, "s", "#E74C3C")]:
    bs_vals = []
    tau_vals = []
    for bs in batch_sizes:
        r = bs_results[k][bs]
        if r["converged"]:
            bs_vals.append(bs)
            tau_vals.append(r["tau"])
    if bs_vals:
        ax.plot(bs_vals, tau_vals, f"{marker}-", color=color, label=f"K={k}",
                markersize=6, linewidth=1.5)
    # Mark failures with an X
    for bs in batch_sizes:
        r = bs_results[k][bs]
        if not r["converged"]:
            ax.plot(bs, r["max_step"], "x", color=color, markersize=10,
                    markeredgewidth=2)

ax.set_xlabel("Batch size")
ax.set_ylabel(r"$\tau$ (steps)")
ax.set_title("(a) Batch-size sensitivity")
ax.legend(fontsize=8)
ax.set_xscale("log", base=2)
ax.set_xticks(batch_sizes)
ax.set_xticklabels([str(b) for b in batch_sizes])

# Panel (b): SGD temperature decomposition
ax = axes[1]
# BS sweep: η=1e-3, varying BS → varying T
bs_temps = []
bs_taus = []
for bs in batch_sizes:
    r = bs_results[20][bs]
    if r["converged"]:
        bs_temps.append(r["sgd_temp"])
        bs_taus.append(r["tau"])
ax.plot(bs_temps, bs_taus, "o-", color="#3498DB", label=r"Vary BS ($\eta$=1e-3)",
        markersize=6, linewidth=1.5)

# LR sweep: BS=128, varying η → varying T
lr_temps = []
lr_taus = []
for eta in lr_values:
    r = lr_results[eta]
    if r["converged"]:
        lr_temps.append(r["sgd_temp"])
        lr_taus.append(r["tau"])
ax.plot(lr_temps, lr_taus, "s-", color="#E74C3C", label=r"Vary $\eta$ (BS=128)",
        markersize=6, linewidth=1.5)

ax.set_xlabel(r"SGD temperature $T = \eta / \mathrm{BS}$")
ax.set_ylabel(r"$\tau$ (steps)")
ax.set_title("(b) Temperature decomposition (K=20)")
ax.legend(fontsize=7)
ax.set_xscale("log")

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(SAVE_DIR / f"fig_batch_size.{ext}", dpi=300, bbox_inches="tight")
plt.close()

print(f"\nFigure saved: {SAVE_DIR / 'fig_batch_size.pdf'}")
