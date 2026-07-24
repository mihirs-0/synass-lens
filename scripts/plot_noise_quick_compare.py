#!/usr/bin/env python
"""
Quick comparison plot: K=20 baseline (noise=0.0) vs noise=0.05.

Uses training history (first_target_loss, loss_z_shuffled) and gradient
norms to compare the learning dynamics with and without label noise.
No need for candidate_eval to be complete.

Usage:
    python scripts/plot_noise_quick_compare.py
"""

import json
import math
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load_training_history(exp_dir):
    path = Path(exp_dir) / "training_history.json"
    with open(path) as f:
        return json.load(f)


def load_gradient_norms(exp_dir):
    path = Path(exp_dir) / "gradient_norm_results.json"
    with open(path) as f:
        return json.load(f)


def load_candidate_eval(exp_dir):
    path = Path(exp_dir) / "candidate_eval_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def smooth(arr, window=20):
    """Simple moving average."""
    arr = np.array(arr, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def main():
    output_dir = Path("outputs")
    K = 20
    log_K = math.log(K)

    # ── Load data ──
    experiments = {}
    for label, dirname in [
        ("p=0.00 (baseline)", "landauer_k20"),
        ("p=0.05", "landauer_k20_noise0.05"),
        ("p=0.10", "landauer_k20_noise0.10"),
        ("p=0.20", "landauer_k20_noise0.20"),
    ]:
        exp_path = output_dir / dirname
        if not exp_path.exists():
            continue
        th_path = exp_path / "training_history.json"
        gn_path = exp_path / "gradient_norm_results.json"
        if not th_path.exists():
            continue
        th = load_training_history(exp_path)
        gn = load_gradient_norms(exp_path) if gn_path.exists() else None
        ce = load_candidate_eval(exp_path)
        experiments[label] = {"th": th, "gn": gn, "ce": ce}

    if not experiments:
        print("No experiments found!")
        return

    print(f"Found {len(experiments)} experiments: {list(experiments.keys())}")

    n_exps = len(experiments)
    colors = {
        "p=0.00 (baseline)": "#1f77b4",
        "p=0.05": "#ff7f0e",
        "p=0.10": "#2ca02c",
        "p=0.20": "#d62728",
    }

    fig = plt.figure(figsize=(18, 18))
    gs = gridspec.GridSpec(4, 2, hspace=0.32, wspace=0.25)

    # ── Panel A: First-target token loss vs step (proxy for candidate loss) ──
    ax = fig.add_subplot(gs[0, 0])
    for label, data in experiments.items():
        th = data["th"]
        steps = th["steps"]
        ftl = th["first_target_loss"]
        c = colors.get(label, "gray")
        ax.plot(steps, smooth(ftl, 15), "-", color=c, label=label, alpha=0.85, linewidth=1.5)
    ax.axhline(log_K, color="red", ls=":", alpha=0.4, label=f"log({K})={log_K:.2f}")
    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel("First-Target Token Loss", fontsize=11)
    ax.set_title("(A) First-Target Loss — Plateau & Transition", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(-0.2, log_K + 0.5)

    # ── Panel B: z-gap vs step ──
    ax = fig.add_subplot(gs[0, 1])
    for label, data in experiments.items():
        th = data["th"]
        steps = th["steps"]
        z_shuf = th["loss_z_shuffled"]
        ftl = th["first_target_loss"]
        z_gap = [zs - ft for zs, ft in zip(z_shuf, ftl)]
        c = colors.get(label, "gray")
        ax.plot(steps, smooth(z_gap, 15), "-", color=c, label=label, alpha=0.85, linewidth=1.5)
    ax.axhline(0, color="gray", ls="-", alpha=0.3)
    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel("z-gap (shuffled − clean)", fontsize=11)
    ax.set_title("(B) z-gap — When Does z-Dependence Emerge?", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    # ── Panel C: Gradient norm² vs step ──
    ax = fig.add_subplot(gs[1, 0])
    for label, data in experiments.items():
        gn = data["gn"]
        if gn is None:
            continue
        c = colors.get(label, "gray")
        ax.plot(gn["steps"], smooth(gn["total_grad_norm_sq"], 10),
                "-", color=c, label=label, alpha=0.85, linewidth=1.5)
    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel("||∇L||²", fontsize=11)
    ax.set_title("(C) Gradient Norm² — Dissipation Proxy", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    # ── Panel D: Training accuracy vs step ──
    ax = fig.add_subplot(gs[1, 1])
    for label, data in experiments.items():
        th = data["th"]
        steps = th["steps"]
        acc = th["train_accuracy"]
        c = colors.get(label, "gray")
        ax.plot(steps, acc, "-", color=c, label=label, alpha=0.85, linewidth=1.5)
    ax.axhline(1.0, color="gray", ls=":", alpha=0.3)
    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel("Train Accuracy", fontsize=11)
    ax.set_title("(D) Training Accuracy", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.05)

    # ── Panel E: Candidate eval (if available) ──
    ax = fig.add_subplot(gs[2, 0])
    has_any_ce = False
    for label, data in experiments.items():
        ce = data["ce"]
        if ce is None:
            continue
        has_any_ce = True
        c = colors.get(label, "gray")
        ax.plot(ce["steps"], ce["candidate_loss"],
                "-", color=c, label=label, alpha=0.85, linewidth=1.5)
    if has_any_ce:
        ax.axhline(log_K, color="red", ls=":", alpha=0.4, label=f"log({K})")
        ax.set_xlabel("Training Step", fontsize=11)
        ax.set_ylabel("Candidate Loss", fontsize=11)
        ax.set_title("(E) Candidate Loss (post-hoc eval)", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "Candidate eval\nnot yet available",
                ha="center", va="center", transform=ax.transAxes, fontsize=14, color="gray")
        ax.set_title("(E) Candidate Loss — pending", fontsize=12, fontweight="bold")

    # ── Panel F: Candidate eval z-gap (if available) ──
    ax = fig.add_subplot(gs[2, 1])
    has_any_zgap = False
    for label, data in experiments.items():
        ce = data["ce"]
        if ce is None or "z_gap" not in ce:
            continue
        has_any_zgap = True
        c = colors.get(label, "gray")
        ax.plot(ce["steps"], ce["z_gap"],
                "-", color=c, label=label, alpha=0.85, linewidth=1.5)
    if has_any_zgap:
        ax.axhline(0, color="gray", ls="-", alpha=0.3)
        ax.set_xlabel("Training Step", fontsize=11)
        ax.set_ylabel("z-gap (candidate eval)", fontsize=11)
        ax.set_title("(F) z-gap from Candidate Eval", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "Candidate eval z-gap\nnot yet available",
                ha="center", va="center", transform=ax.transAxes, fontsize=14, color="gray")
        ax.set_title("(F) z-gap (candidate eval) — pending", fontsize=12, fontweight="bold")

    # ── Panel G: Q_zgap bar chart (from analyze_label_noise.py output) ──
    ax = fig.add_subplot(gs[3, 0])
    q_zgap_path = output_dir / "label_noise_analysis.json"
    if q_zgap_path.exists():
        q_data = json.loads(q_zgap_path.read_text())
        noise_vals = q_data.get("landauer_constant_vs_noise", {}).get("p_noise", [])
        # Get Q_zgap for K=20 from results_by_noise
        q_vals = []
        q_labels = []
        q_colors_list = []
        for p_noise_val in noise_vals:
            p_str = str(p_noise_val)
            rbn = q_data.get("results_by_noise", {}).get(p_str, {})
            Q_list = rbn.get("Q_zgap", [])
            K_list = rbn.get("K_values", [])
            if 20 in K_list:
                idx = K_list.index(20)
                q = Q_list[idx]
                if q is not None:
                    q_vals.append(q)
                    q_labels.append(f"p={p_noise_val}")
                    lbl = f"p={p_noise_val:.2f}" if p_noise_val > 0 else "p=0.00 (baseline)"
                    q_colors_list.append(colors.get(lbl, "gray"))
        if q_vals:
            bars = ax.bar(range(len(q_vals)), q_vals, color=q_colors_list, alpha=0.85,
                          edgecolor="black", linewidth=0.5)
            ax.set_xticks(range(len(q_vals)))
            ax.set_xticklabels(q_labels, fontsize=10)
            for bar, val in zip(bars, q_vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
            ax.set_ylabel("Q_zgap", fontsize=11)
            ax.set_title("(G) Q_zgap vs Label Noise — Key Metric",
                         fontsize=12, fontweight="bold")
            # baseline reference line
            ax.axhline(q_vals[0], color=colors["p=0.00 (baseline)"], ls="--", alpha=0.4)
        else:
            ax.text(0.5, 0.5, "Q_zgap not computed yet",
                    ha="center", va="center", transform=ax.transAxes, fontsize=14, color="gray")
            ax.set_title("(G) Q_zgap — pending", fontsize=12, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "Run analyze_label_noise.py first",
                ha="center", va="center", transform=ax.transAxes, fontsize=14, color="gray")
        ax.set_title("(G) Q_zgap — pending", fontsize=12, fontweight="bold")

    # ── Panel H: Plateau exit step vs noise ──
    ax = fig.add_subplot(gs[3, 1])
    p_noise_labels = []
    exit_steps = []
    exit_colors = []
    for label, data in experiments.items():
        th = data["th"]
        steps_arr = th["steps"]
        ftl = th["first_target_loss"]
        for s, f in zip(steps_arr, ftl):
            if f < 0.8 * log_K:
                exit_steps.append(s)
                break
        else:
            exit_steps.append(None)
        p_noise_labels.append(label.split("(")[0].strip())
        exit_colors.append(colors.get(label, "gray"))
    valid_exits = [(l, s, c) for l, s, c in zip(p_noise_labels, exit_steps, exit_colors) if s is not None]
    if valid_exits:
        labels, steps_vals, cols = zip(*valid_exits)
        bars = ax.bar(range(len(steps_vals)), steps_vals, color=cols, alpha=0.85,
                      edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(steps_vals)))
        ax.set_xticklabels(labels, fontsize=10)
        for bar, val in zip(bars, steps_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                    f"{val:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_ylabel("Plateau Exit Step", fontsize=11)
        ax.set_title("(H) Plateau Duration — Noise Shortens Plateau",
                     fontsize=12, fontweight="bold")

    # ── Suptitle ──
    available = ", ".join(experiments.keys())
    fig.suptitle(
        f"Label Noise Experiment: K={K} — Baseline vs Noise\n"
        f"Available: {available}",
        fontsize=13, fontweight="bold", y=1.01
    )

    # ── Numerical summary ──
    print("\n" + "=" * 70)
    print(f"Label Noise Quick Comparison — K={K}")
    print("=" * 70)
    for label, data in experiments.items():
        th = data["th"]
        gn = data["gn"]
        steps = th["steps"]
        ftl = th["first_target_loss"]
        z_shuf = th["loss_z_shuffled"]
        z_gap = [zs - ft for zs, ft in zip(z_shuf, ftl)]
        acc = th["train_accuracy"]

        # Find plateau exit: first step where first_target_loss < 0.8 * log_K
        plateau_exit = None
        for s, f in zip(steps, ftl):
            if f < 0.8 * log_K:
                plateau_exit = s
                break

        # Max z-gap
        max_zgap = max(z_gap) if z_gap else 0

        # Grad norm peak
        gn_peak = None
        gn_peak_step = None
        if gn:
            gn_sq = gn["total_grad_norm_sq"]
            gn_peak = max(gn_sq)
            gn_peak_step = gn["steps"][gn_sq.index(gn_peak)]

        # Cumulative dissipation (Q = η × Σ ||∇L||² × Δs)
        Q_total = None
        if gn:
            lr = 1e-3
            gn_steps = gn["steps"]
            gn_sq = gn["total_grad_norm_sq"]
            Q_total = 0
            for i in range(1, len(gn_steps)):
                ds = gn_steps[i] - gn_steps[i-1]
                Q_total += lr * gn_sq[i] * ds

        print(f"\n  {label}:")
        print(f"    Final accuracy:    {acc[-1]:.4f}")
        print(f"    Final first_loss:  {ftl[-1]:.4f}")
        print(f"    Plateau exit step: {plateau_exit}")
        print(f"    Max z-gap:         {max_zgap:.4f}")
        if gn_peak is not None:
            print(f"    Peak ||∇L||²:      {gn_peak:.4f} (step {gn_peak_step})")
        if Q_total is not None:
            print(f"    Total Q (η·Σ||∇L||²·Δs): {Q_total:.2f}")

    print("=" * 70)

    fig_path = output_dir / "figures" / "noise_quick_compare.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(fig_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {fig_path}")


if __name__ == "__main__":
    main()
