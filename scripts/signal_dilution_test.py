#!/usr/bin/env python3
"""
Signal Dilution Hypothesis: Is the Plateau Just a Coverage Timescale?

Four tests:
  1. τ × grad_norm_early ∝ K^β  (expect β ≈ 1.0 for pure coverage)
  2. Candidate loss lead time analysis
  3. Per-group learning curves (THE KEY TEST)
  4. τ prediction from first principles

If all four pass → coverage confirmed, phenomenon is simpler than we thought.
If Test 3 fails → genuine phase transition at the individual-group level.
"""

import sys
import json
import math
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

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

K_VALUES = [3, 5, 7, 10, 13, 17, 20, 25, 30, 36]
EVAL_EVERY = 50  # steps between eval points


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_history(k):
    p = OUTPUTS / f"landauer_dense_k{k}" / "training_history.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def find_tau_idx(cand_loss, log_k, threshold=0.5):
    """Find first index where candidate_loss < threshold * log(K)."""
    below = np.where(np.array(cand_loss) < threshold * log_k)[0]
    return int(below[0]) if len(below) > 0 else None


def select_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1: τ × grad_norm_early ∝ K^β
# ══════════════════════════════════════════════════════════════════════════════

def run_test1():
    print("=" * 80)
    print("TEST 1: τ × grad_norm_early ∝ K^β  (expect β ≈ 1.0 for pure coverage)")
    print("=" * 80)

    results = {}
    for K in K_VALUES:
        h = load_history(K)
        if h is None:
            print(f"  K={K}: no data, skipping")
            continue

        log_k = math.log(K)
        cand = np.array(h["candidate_loss"])
        steps = np.array(h["steps"])
        gnorm_sq = np.array(h["grad_norm_sq"])
        gnorm = np.sqrt(gnorm_sq)

        tau_idx = find_tau_idx(cand, log_k)
        if tau_idx is None:
            print(f"  K={K}: never reaches 50% log K, skipping")
            continue

        tau_step = steps[tau_idx]

        # Gradient norm during early plateau (20%-60% of τ)
        early_start = max(1, int(tau_idx * 0.2))
        early_end = int(tau_idx * 0.6)
        if early_end <= early_start:
            early_end = early_start + 1
        grad_norm_early = gnorm[early_start:early_end].mean()

        results[K] = {
            "tau_idx": tau_idx,
            "tau_step": int(tau_step),
            "grad_norm_early": float(grad_norm_early),
            "product": float(tau_step * grad_norm_early),
        }

    if len(results) < 3:
        print("  Not enough data points for regression")
        return results

    Ks = np.array(sorted(results.keys()))
    products = np.array([results[k]["product"] for k in Ks])
    taus = np.array([results[k]["tau_step"] for k in Ks])
    gnorms = np.array([results[k]["grad_norm_early"] for k in Ks])

    # Fit τ × grad_norm ∝ K^β
    beta, c = np.polyfit(np.log(Ks), np.log(products), 1)
    predicted_log = beta * np.log(Ks) + c
    ss_res = np.sum((np.log(products) - predicted_log) ** 2)
    ss_tot = np.sum((np.log(products) - np.log(products).mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Also fit grad_norm_early ∝ K^α
    alpha_g, c_g = np.polyfit(np.log(Ks), np.log(gnorms), 1)
    # And τ ∝ K^α_tau
    alpha_tau, c_tau = np.polyfit(np.log(Ks), np.log(taus), 1)

    print(f"\n  τ ∝ K^{alpha_tau:.3f}")
    print(f"  grad_norm_early ∝ K^{alpha_g:.3f}")
    print(f"  τ × grad_norm ∝ K^{beta:.3f}  (R² = {r2:.4f})")
    print(f"  Expected β ≈ 1.0 for pure coverage")
    verdict = abs(beta - 1.0) < 0.3 and r2 > 0.9
    print(f"  VERDICT: {'PASS' if verdict else 'FAIL'} (β = {beta:.3f})")

    print(f"\n  {'K':>4s} {'τ':>8s} {'ḡ_early':>10s} {'τ×ḡ':>12s} {'τ×ḡ/K':>10s}")
    print("  " + "-" * 50)
    for K in sorted(results):
        r = results[K]
        print(f"  {K:4d} {r['tau_step']:8d} {r['grad_norm_early']:10.4f} "
              f"{r['product']:12.2f} {r['product']/K:10.2f}")

    # ── Figure ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: τ × grad_norm vs K (log-log)
    ax1.scatter(Ks, products, s=60, c="#2C3E50", zorder=5)
    K_fit = np.linspace(Ks.min(), Ks.max(), 100)
    ax1.plot(K_fit, np.exp(c) * K_fit ** beta, "--", color="#E74C3C",
             label=f"$\\beta = {beta:.2f}$, $R^2 = {r2:.3f}$")
    # Reference: β = 1 line
    c_ref = np.mean(np.log(products) - np.log(Ks))
    ax1.plot(K_fit, np.exp(c_ref) * K_fit ** 1.0, ":", color="gray",
             alpha=0.5, label="$\\beta = 1.0$ (pure coverage)")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("K", fontsize=11)
    ax1.set_ylabel(r"$\tau \times \bar{g}_{\mathrm{early}}$", fontsize=11)
    ax1.set_title(r"Test 1: $\tau \times \bar{g} \propto K^{\beta}$", fontsize=12)
    ax1.legend(fontsize=9)

    # Right: τ × grad_norm / K vs K (should be flat)
    ratios = products / Ks
    ax2.scatter(Ks, ratios, s=60, c="#2C3E50", zorder=5)
    ax2.axhline(ratios.mean(), linestyle="--", color="#E74C3C", alpha=0.5,
                label=f"Mean = {ratios.mean():.1f}")
    ax2.set_xlabel("K", fontsize=11)
    ax2.set_ylabel(r"$\tau \times \bar{g}_{\mathrm{early}} / K$", fontsize=11)
    ax2.set_title("Should be flat if pure coverage", fontsize=12)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(SAVE_DIR / f"signal_dilution_test1.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved: signal_dilution_test1")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2: Candidate Loss Leads Population Loss — How Much?
# ══════════════════════════════════════════════════════════════════════════════

def run_test2():
    print("\n" + "=" * 80)
    print("TEST 2: Candidate loss onset timing (as fraction of τ)")
    print("=" * 80)

    results = {}
    for K in K_VALUES:
        h = load_history(K)
        if h is None:
            continue

        log_k = math.log(K)
        cand = np.array(h["candidate_loss"])
        steps = np.array(h["steps"])

        tau_idx = find_tau_idx(cand, log_k, threshold=0.5)
        if tau_idx is None:
            continue

        # Onset thresholds (as fraction of τ index)
        onset_95 = np.where(cand < 0.95 * log_k)[0]
        onset_90 = np.where(cand < 0.90 * log_k)[0]
        onset_80 = np.where(cand < 0.80 * log_k)[0]
        onset_50 = np.where(cand < 0.50 * log_k)[0]
        onset_10 = np.where(cand < 0.10 * log_k)[0]

        results[K] = {
            "tau_idx": tau_idx,
            "tau_step": int(steps[tau_idx]),
            "onset_95": onset_95[0] / tau_idx if len(onset_95) > 0 else None,
            "onset_90": onset_90[0] / tau_idx if len(onset_90) > 0 else None,
            "onset_80": onset_80[0] / tau_idx if len(onset_80) > 0 else None,
            "onset_50": 1.0,  # by definition
            "onset_10": onset_10[0] / tau_idx if len(onset_10) > 0 else None,
        }

    print(f"\n  {'K':>4s} {'τ':>8s} {'95% onset':>10s} {'90% onset':>10s} "
          f"{'80% onset':>10s} {'50% (=τ)':>10s} {'10% onset':>10s}")
    print("  " + "-" * 65)
    for K in sorted(results):
        r = results[K]
        def fmt(v):
            return f"{v:10.2%}" if v is not None else "       N/A"
        print(f"  {K:4d} {r['tau_step']:8d} {fmt(r['onset_95'])} "
              f"{fmt(r['onset_90'])} {fmt(r['onset_80'])} "
              f"{fmt(r['onset_50'])} {fmt(r['onset_10'])}")

    # Check: does 90% onset fraction increase, decrease, or stay flat with K?
    Ks_with_90 = [K for K in sorted(results) if results[K]["onset_90"] is not None]
    if len(Ks_with_90) >= 3:
        fracs = [results[K]["onset_90"] for K in Ks_with_90]
        corr = np.corrcoef(Ks_with_90, fracs)[0, 1]
        slope, _ = np.polyfit(np.log(Ks_with_90), fracs, 1)
        print(f"\n  90% onset fraction vs K correlation: r = {corr:.3f}")
        if corr > 0.3:
            print("  Pattern: onset fraction INCREASES with K → model takes relatively "
                  "longer to start. AGAINST pure coverage.")
        elif corr < -0.3:
            print("  Pattern: onset fraction DECREASES with K → model starts earlier "
                  "but takes longer to finish. CONSISTENT with coverage.")
        else:
            print("  Pattern: onset fraction roughly CONSTANT across K. "
                  "CONSISTENT with coverage.")

    # ── Figure: normalized loss trajectories ──
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(results)))
    for idx, K in enumerate(sorted(results)):
        h = load_history(K)
        log_k = math.log(K)
        cand = np.array(h["candidate_loss"])
        steps = np.array(h["steps"])
        tau_idx = results[K]["tau_idx"]
        tau_step = results[K]["tau_step"]

        # Normalize: y = loss/log(K), x = step/τ
        x = steps / tau_step
        y = cand / log_k
        # Only show up to 1.5× τ
        mask = x <= 1.5
        ax.plot(x[mask], y[mask], color=colors[idx], linewidth=1.2,
                label=f"K={K}", alpha=0.8)

    ax.axhline(1.0, color="red", linestyle=":", alpha=0.4, label="log K")
    ax.axhline(0.5, color="orange", linestyle=":", alpha=0.4, label="50% log K")
    ax.axvline(1.0, color="green", linestyle="--", alpha=0.4, label="τ")
    ax.set_xlabel("Step / τ", fontsize=11)
    ax.set_ylabel("Candidate Loss / log K", fontsize=11)
    ax.set_title("Test 2: Normalized loss trajectories", fontsize=12)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.set_xlim(0, 1.5)
    ax.set_ylim(-0.05, 1.15)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(SAVE_DIR / f"signal_dilution_test2.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved: signal_dilution_test2")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3: Per-Group Learning Curves (THE KEY TEST)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_per_group(model, mapping_data, tokenizer, device, n_groups=100,
                       seed=42):
    """
    For each of n_groups B-groups, score the correct A among all K candidates.
    Returns fraction of groups where the model picks the correct A for ALL K
    z values (fully solved) and fraction where it picks correct for >80%.
    """
    rng = random.Random(seed)
    all_b = list(mapping_data.mappings.keys())
    sample_b = rng.sample(all_b, min(n_groups, len(all_b)))

    group_accs = {}
    model.eval()

    for b_str in sample_b:
        entries = mapping_data.mappings[b_str]  # list of (z, a) tuples
        K = len(entries)
        candidate_a_strings = [a for _, a in entries]

        correct_count = 0
        for correct_idx, (z_str, a_str) in enumerate(entries):
            # Encode all K candidates with this z
            encoded_list = []
            for cand_a in candidate_a_strings:
                enc = tokenizer.encode_sequence(b_str, z_str, cand_a,
                                                task="bz_to_a")
                encoded_list.append(enc)

            input_ids = torch.stack([e["input_ids"] for e in encoded_list]).to(device)
            labels = torch.stack([e["labels"] for e in encoded_list]).to(device)

            with torch.no_grad():
                logits = model(input_ids)

            # Compute sequence log-prob for each candidate
            log_probs = []
            for c in range(K):
                total_lp = 0.0
                start = int(encoded_list[c]["target_start_position"])
                end = int(encoded_list[c]["target_end_position"])
                for pos in range(start, end + 1):
                    target_token = labels[c, pos].item()
                    if target_token == -100:
                        continue
                    lp = F.log_softmax(logits[c, pos - 1], dim=-1)
                    total_lp += lp[target_token].item()
                log_probs.append(total_lp)

            if np.argmax(log_probs) == correct_idx:
                correct_count += 1

        group_accs[b_str] = correct_count / K

    return group_accs


def run_test3(test1_results):
    print("\n" + "=" * 80)
    print("TEST 3: Per-Group Learning Curves (THE KEY TEST)")
    print("=" * 80)

    device = select_device()
    print(f"  Device: {device}")

    # Test K values — pick a few representative ones
    test_ks = [10, 20]
    # Add K=36 only if it eventually converges (it doesn't in our data, but
    # we can still look at partial progress)
    available_ks = []
    for K in test_ks:
        run_name = f"landauer_dense_k{K}"
        cfg_path = OUTPUTS / run_name / "config.yaml"
        ckpt_dir = OUTPUTS / run_name / "checkpoints"
        if cfg_path.exists() and ckpt_dir.exists():
            available_ks.append(K)

    if not available_ks:
        print("  No checkpoint data available for per-group evaluation")
        return {}

    all_results = {}
    n_groups = 200  # sample 200 B-groups per checkpoint

    for K in available_ks:
        run_name = f"landauer_dense_k{K}"
        print(f"\n  K = {K}")

        cfg = OmegaConf.load(OUTPUTS / run_name / "config.yaml")
        tokenizer = create_tokenizer_from_config(cfg)
        _, _, mapping_data = create_datasets_from_config(cfg, tokenizer)
        model = create_model_from_config(cfg, tokenizer).to(device)

        log_k = math.log(K)
        h = load_history(K)
        cand = np.array(h["candidate_loss"])
        steps_arr = np.array(h["steps"])
        tau_idx = find_tau_idx(cand, log_k)

        if tau_idx is not None:
            tau_step = int(steps_arr[tau_idx])
        else:
            tau_step = int(steps_arr[-1])

        # Select checkpoints: evenly spaced from early to 1.3× τ
        ckpt_dir = OUTPUTS / run_name / "checkpoints"
        all_ckpt_steps = sorted([
            int(d.name.split("_")[1])
            for d in ckpt_dir.iterdir()
            if d.is_dir() and d.name.startswith("step_")
        ])

        # Pick ~20 checkpoints spanning the plateau and transition
        max_step = min(int(tau_step * 1.5), all_ckpt_steps[-1])
        target_steps = np.linspace(100, max_step, 20).astype(int)
        # Snap to nearest available
        selected = []
        for t in target_steps:
            nearest = min(all_ckpt_steps, key=lambda s: abs(s - t))
            if nearest not in selected:
                selected.append(nearest)
        selected.sort()

        print(f"    τ = {tau_step}, evaluating {len(selected)} checkpoints, "
              f"{n_groups} B-groups each")

        k_results = {"steps": [], "fraction_solved": [], "fraction_80": [],
                      "mean_acc": [], "acc_distribution": []}

        for ckpt_step in selected:
            model_path = ckpt_dir / f"step_{ckpt_step:06d}" / "model.pt"
            if not model_path.exists():
                continue

            state_dict = torch.load(model_path, map_location=device,
                                    weights_only=False)
            model.load_state_dict(state_dict)

            group_accs = evaluate_per_group(model, mapping_data, tokenizer,
                                            device, n_groups=n_groups)

            accs = list(group_accs.values())
            frac_solved = sum(1 for a in accs if a >= 1.0) / len(accs)
            frac_80 = sum(1 for a in accs if a >= 0.8) / len(accs)
            mean_acc = np.mean(accs)

            k_results["steps"].append(ckpt_step)
            k_results["fraction_solved"].append(frac_solved)
            k_results["fraction_80"].append(frac_80)
            k_results["mean_acc"].append(mean_acc)
            k_results["acc_distribution"].append(accs)

            step_frac = ckpt_step / tau_step if tau_step > 0 else 0
            print(f"    step {ckpt_step:6d} ({step_frac:.2f}τ): "
                  f"solved={frac_solved:.1%}, ≥80%={frac_80:.1%}, "
                  f"mean_acc={mean_acc:.3f}")

        all_results[K] = k_results
        k_results["tau_step"] = tau_step

    # ── KEY ANALYSIS ──
    print("\n  " + "-" * 70)
    print("  KEY ANALYSIS: Fraction of groups solved at different stages")
    print("  " + "-" * 70)

    for K in available_ks:
        r = all_results.get(K)
        if not r or not r["steps"]:
            continue
        tau = r["tau_step"]
        steps = np.array(r["steps"])
        fracs = np.array(r["fraction_solved"])
        fracs_80 = np.array(r["fraction_80"])

        # Find fraction solved at τ/4, τ/2, 3τ/4, τ
        for frac_tau in [0.25, 0.5, 0.75, 1.0]:
            target = frac_tau * tau
            closest_idx = np.argmin(np.abs(steps - target))
            actual_step = steps[closest_idx]
            print(f"    K={K} at {frac_tau:.0%}τ (step {actual_step}): "
                  f"solved={fracs[closest_idx]:.1%}, "
                  f"≥80%={fracs_80[closest_idx]:.1%}")

    # ── Determine curve shape ──
    for K in available_ks:
        r = all_results.get(K)
        if not r or not r["steps"]:
            continue
        tau = r["tau_step"]
        steps = np.array(r["steps"])
        fracs_80 = np.array(r["fraction_80"])

        # At τ/2, what fraction is ≥80% solved?
        mid_idx = np.argmin(np.abs(steps - 0.5 * tau))
        mid_frac = fracs_80[mid_idx]

        if mid_frac > 0.2:
            shape = "SIGMOID (gradual) → COVERAGE"
        elif mid_frac > 0.05:
            shape = "MIXED (some gradual, some snap)"
        else:
            shape = "STEP FUNCTION (snap) → PHASE TRANSITION"
        print(f"\n    K={K}: At τ/2, {mid_frac:.1%} of groups ≥80% → {shape}")

    # ── Figure: THE KEY FIGURE ──
    fig, axes = plt.subplots(1, len(available_ks), figsize=(7 * len(available_ks), 5),
                             squeeze=False)

    for idx, K in enumerate(available_ks):
        r = all_results.get(K)
        if not r or not r["steps"]:
            continue
        ax = axes[0][idx]
        tau = r["tau_step"]
        steps = np.array(r["steps"]) / tau  # normalize by τ

        ax.plot(steps, r["fraction_solved"], "o-", color="#E74C3C",
                markersize=4, linewidth=1.5, label="100% correct")
        ax.plot(steps, r["fraction_80"], "s-", color="#3498DB",
                markersize=4, linewidth=1.5, label="≥80% correct")
        ax.plot(steps, r["mean_acc"], "^-", color="#2ECC71",
                markersize=4, linewidth=1.5, label="Mean accuracy")

        ax.axvline(1.0, color="green", linestyle="--", alpha=0.4, label="τ")
        ax.axvline(0.5, color="orange", linestyle=":", alpha=0.3, label="τ/2")

        # Sigmoid reference (gradual coverage)
        x_ref = np.linspace(0, 1.5, 100)
        sigmoid = 1 / (1 + np.exp(-8 * (x_ref - 0.7)))
        ax.plot(x_ref, sigmoid, "--", color="gray", alpha=0.3,
                label="Sigmoid ref (coverage)")

        # Step reference (phase transition)
        step_ref = np.where(x_ref > 0.95, 1.0, 0.0)
        ax.plot(x_ref, step_ref, ":", color="gray", alpha=0.3,
                label="Step ref (transition)")

        ax.set_xlabel("Step / τ", fontsize=11)
        ax.set_ylabel("Fraction of B-groups", fontsize=11)
        ax.set_title(f"K = {K}: Per-group learning curves", fontsize=12,
                     fontweight="bold")
        ax.legend(fontsize=7, loc="upper left")
        ax.set_xlim(0, 1.5)
        ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(SAVE_DIR / f"per_group_learning_curves.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved: per_group_learning_curves")

    # ── Bonus: histogram of per-group accuracy at selected checkpoints ──
    fig, axes = plt.subplots(len(available_ks), 4, figsize=(16, 4 * len(available_ks)),
                             squeeze=False)

    for k_idx, K in enumerate(available_ks):
        r = all_results.get(K)
        if not r or not r["steps"]:
            continue
        tau = r["tau_step"]
        steps = np.array(r["steps"])

        for panel, frac_tau in enumerate([0.25, 0.5, 0.75, 1.0]):
            target = frac_tau * tau
            closest_idx = np.argmin(np.abs(steps - target))
            accs = r["acc_distribution"][closest_idx]

            ax = axes[k_idx][panel]
            ax.hist(accs, bins=20, range=(0, 1), color="#3498DB",
                    edgecolor="white", alpha=0.8)
            ax.axvline(0.8, color="red", linestyle="--", alpha=0.5)

            n_solved = sum(1 for a in accs if a >= 0.8)
            ax.set_title(f"K={K}, step={int(steps[closest_idx])} "
                         f"({frac_tau:.0%}τ)\n"
                         f"≥80%: {n_solved}/{len(accs)} = {n_solved/len(accs):.0%}",
                         fontsize=9)
            ax.set_xlabel("Group accuracy", fontsize=8)
            ax.set_ylabel("Count", fontsize=8)
            ax.tick_params(labelsize=7)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(SAVE_DIR / f"per_group_histograms.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: per_group_histograms")

    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4: Can We PREDICT τ from First Principles?
# ══════════════════════════════════════════════════════════════════════════════

def run_test4(test1_results):
    print("\n" + "=" * 80)
    print("TEST 4: τ_predicted = C × K / grad_norm_early")
    print("=" * 80)

    if len(test1_results) < 3:
        print("  Not enough data from Test 1")
        return

    Ks = np.array(sorted(test1_results.keys()))
    taus = np.array([test1_results[k]["tau_step"] for k in Ks])
    gnorms = np.array([test1_results[k]["grad_norm_early"] for k in Ks])

    # C = τ × gnorm / K → fit one global constant
    C_values = taus * gnorms / Ks
    C = C_values.mean()
    C_std = C_values.std()
    C_cv = C_std / C  # coefficient of variation

    tau_pred = C * Ks / gnorms
    residuals = (taus - tau_pred) / taus

    ss_res = np.sum((taus - tau_pred) ** 2)
    ss_tot = np.sum((taus - taus.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    print(f"\n  C = {C:.2f} ± {C_std:.2f}  (CV = {C_cv:.1%})")
    print(f"  R² = {r2:.4f}")
    verdict = r2 > 0.9
    print(f"  VERDICT: {'PASS' if verdict else 'FAIL'}")

    print(f"\n  {'K':>4s} {'τ_actual':>8s} {'τ_pred':>8s} {'error':>8s} {'C_i':>8s}")
    print("  " + "-" * 40)
    for i, K in enumerate(Ks):
        print(f"  {K:4d} {taus[i]:8.0f} {tau_pred[i]:8.0f} "
              f"{residuals[i]:8.1%} {C_values[i]:8.2f}")

    # ── Figure ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: actual vs predicted
    ax1.scatter(tau_pred, taus, s=60, c="#2C3E50", zorder=5)
    for i, K in enumerate(Ks):
        ax1.annotate(f"K={K}", (tau_pred[i], taus[i]), fontsize=7,
                     xytext=(5, 5), textcoords="offset points")
    lims = [0, max(taus.max(), tau_pred.max()) * 1.1]
    ax1.plot(lims, lims, "--", color="#E74C3C", alpha=0.5, label="Perfect prediction")
    ax1.set_xlabel(r"$\tau_{\mathrm{predicted}} = C \cdot K / \bar{g}_{\mathrm{early}}$",
                   fontsize=11)
    ax1.set_ylabel(r"$\tau_{\mathrm{actual}}$", fontsize=11)
    ax1.set_title(f"Test 4: $R^2 = {r2:.3f}$", fontsize=12)
    ax1.legend(fontsize=9)

    # Right: C values per K (should be flat)
    ax2.scatter(Ks, C_values, s=60, c="#2C3E50", zorder=5)
    ax2.axhline(C, linestyle="--", color="#E74C3C", alpha=0.5,
                label=f"C = {C:.1f} ± {C_std:.1f}")
    ax2.fill_between([Ks.min() * 0.8, Ks.max() * 1.2],
                     C - C_std, C + C_std, alpha=0.1, color="#E74C3C")
    ax2.set_xlabel("K", fontsize=11)
    ax2.set_ylabel(r"$C_K = \tau \cdot \bar{g} / K$", fontsize=11)
    ax2.set_title("Per-K constant (should be flat for coverage)", fontsize=12)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(SAVE_DIR / f"signal_dilution_test4.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved: signal_dilution_test4")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("SIGNAL DILUTION HYPOTHESIS — COMPREHENSIVE TEST")
    print("=" * 80)

    # Tests 1, 2, 4 are fast (aggregate stats only)
    test1_results = run_test1()
    test2_results = run_test2()
    test3_results = run_test3(test1_results)
    run_test4(test1_results)

    # ── Summary verdict ──
    print("\n\n" + "=" * 80)
    print("SIGNAL DILUTION HYPOTHESIS — SUMMARY")
    print("=" * 80)

    # Test 1 verdict
    if test1_results:
        Ks = np.array(sorted(test1_results.keys()))
        products = np.array([test1_results[k]["product"] for k in Ks])
        beta, _ = np.polyfit(np.log(Ks), np.log(products), 1)
        t1_pass = abs(beta - 1.0) < 0.3
        print(f"\nTEST 1: τ × grad_norm ∝ K^{beta:.2f}")
        print(f"  VERDICT: {'PASS' if t1_pass else 'FAIL'} "
              f"(expect β ≈ 1.0, got {beta:.2f})")
    else:
        t1_pass = False
        print("\nTEST 1: INSUFFICIENT DATA")

    # Test 2 verdict
    print(f"\nTEST 2: Candidate loss onset timing")
    if test2_results:
        Ks_90 = [K for K in sorted(test2_results)
                 if test2_results[K].get("onset_90") is not None]
        if Ks_90:
            fracs = [test2_results[K]["onset_90"] for K in Ks_90]
            corr = np.corrcoef(Ks_90, fracs)[0, 1]
            if corr > 0.3:
                print(f"  Onset fraction increases with K (r={corr:.2f})")
                print("  VERDICT: INCONSISTENT with coverage "
                      "(harder K → relatively slower start)")
                t2_pass = False
            elif corr < -0.3:
                print(f"  Onset fraction decreases with K (r={corr:.2f})")
                print("  VERDICT: CONSISTENT with coverage")
                t2_pass = True
            else:
                print(f"  Onset fraction roughly constant (r={corr:.2f})")
                print("  VERDICT: CONSISTENT with coverage")
                t2_pass = True
        else:
            t2_pass = False
    else:
        t2_pass = False

    # Test 3 verdict
    print(f"\nTEST 3: Per-group learning curves (THE KEY TEST)")
    t3_verdict = None
    if test3_results:
        for K in sorted(test3_results):
            r = test3_results[K]
            if not r.get("steps"):
                continue
            tau = r["tau_step"]
            steps = np.array(r["steps"])
            fracs_80 = np.array(r["fraction_80"])
            mid_idx = np.argmin(np.abs(steps - 0.5 * tau))
            mid_frac = fracs_80[mid_idx]

            if mid_frac > 0.2:
                shape = "SIGMOID → COVERAGE"
                t3_verdict = "COVERAGE" if t3_verdict != "PHASE TRANSITION" else "MIXED"
            elif mid_frac > 0.05:
                shape = "MIXED"
                t3_verdict = "MIXED"
            else:
                shape = "STEP FUNCTION → PHASE TRANSITION"
                t3_verdict = "PHASE TRANSITION" if t3_verdict != "COVERAGE" else "MIXED"

            print(f"  K={K}: at τ/2, {mid_frac:.1%} groups ≥80% → {shape}")

        print(f"  VERDICT: {t3_verdict}")
    else:
        print("  INSUFFICIENT DATA")
        t3_verdict = "UNKNOWN"

    # Test 4 verdict
    if test1_results and len(test1_results) >= 3:
        Ks = np.array(sorted(test1_results.keys()))
        taus = np.array([test1_results[k]["tau_step"] for k in Ks])
        gnorms = np.array([test1_results[k]["grad_norm_early"] for k in Ks])
        C_values = taus * gnorms / Ks
        C = C_values.mean()
        tau_pred = C * Ks / gnorms
        ss_res = np.sum((taus - tau_pred) ** 2)
        ss_tot = np.sum((taus - taus.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        t4_pass = r2 > 0.9
        print(f"\nTEST 4: τ prediction R² = {r2:.3f}")
        print(f"  VERDICT: {'PASS' if t4_pass else 'FAIL'}")
    else:
        t4_pass = False

    # Overall
    print("\n" + "=" * 80)
    if t3_verdict == "COVERAGE" and t1_pass and t4_pass:
        overall = "COVERAGE CONFIRMED"
    elif t3_verdict == "PHASE TRANSITION":
        overall = "COVERAGE REJECTED"
    elif t3_verdict == "MIXED" or (t1_pass and not t4_pass):
        overall = "COVERAGE PARTIAL"
    else:
        overall = "INCONCLUSIVE"

    print(f"OVERALL VERDICT: {overall}")
    print("=" * 80)

    if overall == "COVERAGE CONFIRMED":
        print("The plateau is a coverage timescale, not a phase transition.")
        print("Individual bindings are learned gradually; the population loss")
        print("only drops when enough are covered for the aggregate to move.")
    elif overall == "COVERAGE REJECTED":
        print("The phenomenon is NOT trivial. Groups transition simultaneously,")
        print("not one at a time. A deeper mechanism is needed — the phase")
        print("transition is real at the individual-group level.")
    elif overall == "COVERAGE PARTIAL":
        print("Coverage explains part of the variance but not all. There may")
        print("be both a coverage component and a genuine transition component.")
    else:
        print("Insufficient data to reach a conclusion.")
