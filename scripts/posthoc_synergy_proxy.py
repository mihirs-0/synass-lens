#!/usr/bin/env python3
"""
Experiment 7: Z-Dependence as Synergy Proxy (Paper: 2408.08954)

Tests whether the z-shuffle gap Δ_z (a proxy for synergistic information from z)
rises BEFORE the loss transition, consistent with synergy as an order parameter
for the phase transition.

No checkpoint loading — uses existing training histories.
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

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


def find_onset(values, steps, threshold):
    """Find first step where values exceed threshold."""
    for s, v in zip(steps, values):
        if v > threshold:
            return s
    return None


# ── Analyze z-shuffle gap dynamics ───────────────────────────────────────────

print("=" * 80)
print("Experiment 7: Z-Dependence as Synergy Proxy")
print("=" * 80)

# Landauer dense K-sweep (best data: every 100 steps for 50k steps)
k_values = [3, 5, 7, 10, 13, 17, 20, 25, 30, 36]
results = []

print(f"\n{'K':>4}  {'log K':>6}  {'τ_loss':>8}  {'τ_synergy':>10}  {'lead':>8}  {'lead/τ':>8}")
print("-" * 60)

for k in k_values:
    h = load_history(f"landauer_dense_k{k}")
    if h is None:
        continue

    steps = np.array(h["steps"])
    loss = np.array(h["first_target_loss"])
    z_shuf = np.array(h["loss_z_shuffled"])
    log_k = math.log(k)

    # Compute z-gap
    z_gap = z_shuf - loss

    # Loss onset: first step where loss < 90% of log(K) (early transition)
    loss_onset = find_onset(-loss, steps, -0.9 * log_k)  # negate for "less than"
    # Recompute properly
    loss_onset = None
    for s, l in zip(steps, loss):
        if l < 0.9 * log_k:
            loss_onset = int(s)
            break

    # Synergy onset: first step where z_gap > 0.1 nats
    synergy_onset = find_onset(z_gap, steps, 0.1)

    # Also try a more sensitive threshold
    synergy_onset_sensitive = find_onset(z_gap, steps, 0.05)

    lead = None
    lead_frac = None
    if loss_onset and synergy_onset:
        lead = loss_onset - synergy_onset
        lead_frac = lead / loss_onset if loss_onset > 0 else None

    results.append({
        "K": k,
        "log_k": log_k,
        "loss_onset": loss_onset,
        "synergy_onset": synergy_onset,
        "synergy_onset_sensitive": synergy_onset_sensitive,
        "lead": lead,
        "lead_frac": lead_frac,
        "steps": steps,
        "loss": loss,
        "z_gap": z_gap,
    })

    loss_str = str(loss_onset) if loss_onset else "---"
    syn_str = str(synergy_onset) if synergy_onset else "---"
    lead_str = str(lead) if lead is not None else "---"
    frac_str = f"{lead_frac:.2f}" if lead_frac is not None else "---"
    print(f"{k:>4}  {log_k:>6.2f}  {loss_str:>8}  {syn_str:>10}  {lead_str:>8}  {frac_str:>8}")

# ── Scaling analysis ─────────────────────────────────────────────────────────

valid = [r for r in results if r["lead"] is not None and r["lead"] > 0]
if len(valid) >= 3:
    ks = np.array([r["K"] for r in valid])
    leads = np.array([r["lead"] for r in valid])
    slope, intercept, r_val, _, _ = stats.linregress(np.log(ks), np.log(leads))
    print(f"\nLead time scaling: Δt_lead ∝ K^{slope:.2f} (R² = {r_val**2:.3f})")

    loss_onsets = np.array([r["loss_onset"] for r in valid])
    lead_fracs = leads / loss_onsets
    print(f"Mean lead fraction: {lead_fracs.mean():.2f} ± {lead_fracs.std():.2f}")

# ── Per-head probe analysis ──────────────────────────────────────────────────

print("\n" + "=" * 80)
print("Per-head attention onset vs aggregate synergy onset (K=20)")
print("=" * 80)

probe_path = OUTPUTS / "temp_lr1e3_bs128_k20" / "probe_results" / "all_probes.json"
if probe_path.exists():
    with open(probe_path) as f:
        probe_data = json.load(f)

    attn_data = probe_data["probe_results"]["attention_to_z"]
    probe_steps = sorted(int(s) for s in attn_data.keys())

    # Compute baseline (first 5 steps)
    baselines = {}
    for layer in range(4):
        for head in range(4):
            vals = [attn_data[str(s)]["attention_to_z"][layer][head] for s in probe_steps[:5]]
            baselines[(layer, head)] = np.mean(vals)

    # Find when each head exceeds 1.5x baseline
    head_onsets = {}
    for layer in range(4):
        for head in range(4):
            bl = baselines[(layer, head)]
            for s in probe_steps:
                val = attn_data[str(s)]["attention_to_z"][layer][head]
                if val > 1.5 * bl:
                    head_onsets[(layer, head)] = s
                    break

    # Find corresponding z-gap onset for this run
    h20 = load_history("temp_lr1e3_bs128_k20")
    if h20:
        steps20 = np.array(h20["steps"])
        loss20 = np.array(h20["first_target_loss"])
        zshuf20 = np.array(h20["loss_z_shuffled"])
        zgap20 = zshuf20 - loss20
        zgap_onset = find_onset(zgap20, steps20, 0.1)

        print(f"\nZ-gap onset (Δ_z > 0.1): step {zgap_onset}")
        print(f"\nPer-head attention onset (> 1.5× baseline):")
        for (l, h), onset in sorted(head_onsets.items()):
            is_lead = " ← LEAD" if (l, h) == (1, 3) else ""
            delta = f"  ({zgap_onset - onset:+d} vs z-gap)" if zgap_onset else ""
            print(f"  L{l}H{h}: step {onset}{delta}{is_lead}")

# ── Figure ───────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

# Panel (a): z-gap and loss overlay for selected K values
ax = axes[0]
colors_k = {5: "#95A5A6", 10: "#3498DB", 20: "#E67E22", 36: "#27AE60"}
for r in results:
    if r["K"] not in colors_k:
        continue
    color = colors_k[r["K"]]
    # Normalize loss by log(K) for comparison
    norm_loss = r["loss"] / r["log_k"]
    # Normalize z-gap by log(K)
    norm_zgap = r["z_gap"] / r["log_k"]
    ax.plot(r["steps"], norm_loss, "-", color=color, alpha=0.4, linewidth=0.8)
    ax.plot(r["steps"], norm_zgap, "--", color=color, alpha=0.8, linewidth=1.2,
            label=f"K={r['K']}")
ax.set_xlabel("Training step")
ax.set_ylabel(r"Normalized ($\div \log K$)")
ax.set_title(r"(a) Loss (solid) vs $\Delta_z$ (dashed)")
ax.legend(fontsize=7)
ax.set_xlim(0, 35000)

# Panel (b): Synergy onset vs loss onset
ax = axes[1]
valid_plot = [r for r in results if r["loss_onset"] and r["synergy_onset"]]
if valid_plot:
    loss_ons = [r["loss_onset"] for r in valid_plot]
    syn_ons = [r["synergy_onset"] for r in valid_plot]
    ks_plot = [r["K"] for r in valid_plot]
    ax.scatter(loss_ons, syn_ons, c=[math.log(k) for k in ks_plot],
               cmap="viridis", s=50, zorder=3)
    # Identity line
    max_val = max(max(loss_ons), max(syn_ons)) * 1.1
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3, label="$t_{syn}=t_{loss}$")
    # Annotate K values
    for r in valid_plot:
        ax.annotate(f"K={r['K']}", (r["loss_onset"], r["synergy_onset"]),
                    fontsize=6, ha="left", va="bottom")
    ax.set_xlabel("Loss onset step")
    ax.set_ylabel("Synergy onset step")
    ax.set_title("(b) Synergy leads loss")
    ax.legend(fontsize=7)

# Panel (c): Lead time vs K
ax = axes[2]
if len(valid) >= 3:
    ks_v = [r["K"] for r in valid]
    leads_v = [r["lead"] for r in valid]
    ax.scatter(ks_v, leads_v, color="#3498DB", s=50, zorder=3)
    # Fit line
    k_fit = np.linspace(min(ks_v) * 0.8, max(ks_v) * 1.2, 100)
    lead_fit = np.exp(intercept) * k_fit ** slope
    ax.plot(k_fit, lead_fit, "--", color="#E74C3C", alpha=0.6,
            label=rf"$\Delta t \propto K^{{{slope:.2f}}}$ ($R^2$={r_val**2:.2f})")
    ax.set_xlabel("Ambiguity $K$")
    ax.set_ylabel("Lead time (steps)")
    ax.set_title("(c) Synergy lead time vs $K$")
    ax.legend(fontsize=7)
    ax.set_xscale("log")
    ax.set_yscale("log")

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(SAVE_DIR / f"fig_synergy_proxy.{ext}", dpi=300, bbox_inches="tight")
plt.close()

print(f"\nFigure saved: {SAVE_DIR / 'fig_synergy_proxy.pdf'}")
