#!/usr/bin/env python
"""
Post-training analysis for overnight experiment suites.

Loads all results, computes summary tables, generates figures.
Designed to run after overnight_master.sh completes.
"""

import sys
import math
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.experiment_helpers import (
    load_history, detect_tau, detect_convergence,
)

OUTPUT_DIR = "outputs"
FIG_DIR = Path(OUTPUT_DIR) / "paper_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

K_VALUES = [5, 10, 20, 36]
SEED = 42


# ═══════════════════════════════════════════════════════════
# SUITE 1: REVERSAL ANALYSIS
# ═══════════════════════════════════════════════════════════

def analyze_suite1():
    """Analyze reversal experiment results."""
    print("\n" + "=" * 60)
    print("SUITE 1 ANALYSIS: REVERSAL CURSE")
    print("=" * 60)

    table = []
    histories = {}

    for K in K_VALUES:
        log_k = math.log(K)
        row = {"K": K, "log_K": f"{log_k:.2f}"}

        # Task F
        name_f = f"reversal_F_K{K}_s{SEED}"
        h_f = load_history(name_f, OUTPUT_DIR)
        if h_f:
            tau_f = detect_convergence(h_f, threshold=0.1, key="first_target_loss")
            row["Task_F_tau"] = tau_f
            histories[f"F_K{K}"] = h_f
        else:
            row["Task_F_tau"] = None

        # Task Bz
        name_bz = f"reversal_Bz_K{K}_s{SEED}"
        h_bz = load_history(name_bz, OUTPUT_DIR)
        if h_bz:
            tau_bz = detect_tau(h_bz, log_k, key="candidate_loss", threshold_frac=0.5)
            if tau_bz is None:
                tau_bz = detect_tau(h_bz, log_k, key="first_target_loss", threshold_frac=0.5)
            row["Task_Bz_tau"] = tau_bz
            histories[f"Bz_K{K}"] = h_bz
        else:
            row["Task_Bz_tau"] = None

        # Task B
        name_b = f"reversal_B_K{K}_s{SEED}"
        h_b = load_history(name_b, OUTPUT_DIR)
        if h_b:
            final_loss = h_b["first_target_loss"][-1] if h_b.get("first_target_loss") else None
            row["Task_B_final"] = final_loss
            row["Task_B_logK_ratio"] = final_loss / log_k if final_loss else None
            histories[f"B_K{K}"] = h_b
        else:
            row["Task_B_final"] = None
            row["Task_B_logK_ratio"] = None

        # Transfer
        name_tr = f"reversal_Transfer_K{K}_s{SEED}"
        h_tr = load_history(name_tr, OUTPUT_DIR)
        if h_tr:
            tau_tr = detect_tau(h_tr, log_k, key="candidate_loss", threshold_frac=0.5)
            if tau_tr is None:
                tau_tr = detect_tau(h_tr, log_k, key="first_target_loss", threshold_frac=0.5)
            row["Transfer_tau"] = tau_tr
            histories[f"Transfer_K{K}"] = h_tr
        else:
            row["Transfer_tau"] = None

        # Speedup
        if row.get("Task_Bz_tau") and row.get("Transfer_tau") and row["Transfer_tau"] > 0:
            row["Speedup"] = row["Task_Bz_tau"] / row["Transfer_tau"]
        else:
            row["Speedup"] = None

        table.append(row)

    # Print summary table
    print(f"\n{'K':>4}  {'F_tau':>8}  {'Bz_tau':>8}  {'B_final':>8}  {'B/logK':>6}  "
          f"{'Tr_tau':>8}  {'Speedup':>7}")
    print("-" * 70)
    for r in table:
        f_tau = f"{r['Task_F_tau']}" if r['Task_F_tau'] else "---"
        bz_tau = f"{r['Task_Bz_tau']}" if r['Task_Bz_tau'] else "---"
        b_final = f"{r['Task_B_final']:.3f}" if r['Task_B_final'] else "---"
        b_ratio = f"{r['Task_B_logK_ratio']:.2f}" if r['Task_B_logK_ratio'] else "---"
        tr_tau = f"{r['Transfer_tau']}" if r['Transfer_tau'] else "---"
        speedup = f"{r['Speedup']:.2f}x" if r['Speedup'] else "---"
        print(f"{r['K']:>4}  {f_tau:>8}  {bz_tau:>8}  {b_final:>8}  {b_ratio:>6}  "
              f"{tr_tau:>8}  {speedup:>7}")

    # Save table
    with open(Path(OUTPUT_DIR) / "reversal_summary.json", "w") as f:
        json.dump(table, f, indent=2, default=str)

    # ---- Figure: Directional Asymmetry (4 panels) ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, K in enumerate(K_VALUES):
        ax = axes[idx // 2, idx % 2]
        log_k = math.log(K)

        for task_key, label, color, ls in [
            (f"F_K{K}", "Task F: A→B", "tab:green", "-"),
            (f"Bz_K{K}", "Task Bz: (B,z)→A", "tab:blue", "-"),
            (f"B_K{K}", "Task B: B→A (no z)", "tab:red", "--"),
            (f"Transfer_K{K}", "Transfer: F→Bz", "tab:purple", "-."),
        ]:
            h = histories.get(task_key)
            if h and h.get("first_target_loss"):
                ax.plot(h["steps"], h["first_target_loss"],
                        label=label, color=color, ls=ls, alpha=0.8)

        ax.axhline(y=log_k, color="gray", ls=":", alpha=0.5, label=f"log({K})={log_k:.2f}")
        ax.set_xlabel("Step")
        ax.set_ylabel("First-target loss")
        ax.set_title(f"K = {K}")
        ax.legend(fontsize=7)
        ax.set_ylim(bottom=-0.1)

    fig.suptitle("Directional Asymmetry: All Tasks", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_directional_asymmetry.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_directional_asymmetry.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved fig_directional_asymmetry")

    # ---- Figure: Transfer Comparison ----
    fig, axes = plt.subplots(1, len(K_VALUES), figsize=(4 * len(K_VALUES), 4))
    if len(K_VALUES) == 1:
        axes = [axes]
    for idx, K in enumerate(K_VALUES):
        ax = axes[idx]
        log_k = math.log(K)

        h_bz = histories.get(f"Bz_K{K}")
        h_tr = histories.get(f"Transfer_K{K}")

        if h_bz and h_bz.get("first_target_loss"):
            ax.plot(h_bz["steps"], h_bz["first_target_loss"],
                    label="From scratch", color="tab:blue", alpha=0.8)
        if h_tr and h_tr.get("first_target_loss"):
            ax.plot(h_tr["steps"], h_tr["first_target_loss"],
                    label="After F training", color="tab:purple", alpha=0.8)

        ax.axhline(y=log_k, color="gray", ls=":", alpha=0.5)
        ax.set_xlabel("Step")
        ax.set_ylabel("First-target loss")
        ax.set_title(f"K = {K}")
        ax.legend(fontsize=8)

    fig.suptitle("Transfer vs Scratch: Does Forward Training Help?", fontsize=12)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_transfer_comparison.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_transfer_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved fig_transfer_comparison")

    # ---- Figure: Backward Representations (Task B z-shuffle) ----
    fig, axes = plt.subplots(1, len(K_VALUES), figsize=(4 * len(K_VALUES), 4))
    if len(K_VALUES) == 1:
        axes = [axes]
    for idx, K in enumerate(K_VALUES):
        ax = axes[idx]
        h_b = histories.get(f"B_K{K}")
        log_k = math.log(K)

        if h_b:
            if h_b.get("first_target_loss"):
                ax.plot(h_b["steps"], h_b["first_target_loss"],
                        label="Loss", color="tab:red")
            if h_b.get("loss_z_shuffled"):
                ax.plot(h_b["steps"], h_b["loss_z_shuffled"],
                        label="z-shuffled loss", color="tab:orange", ls="--")
            ax.axhline(y=log_k, color="gray", ls=":", alpha=0.5,
                       label=f"log({K})")

        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title(f"Task B: K={K}")
        ax.legend(fontsize=8)

    fig.suptitle("Task B (no z): Loss Should Plateau at log(K)", fontsize=12)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_backward_representations.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_backward_representations.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved fig_backward_representations")

    return table


# ═══════════════════════════════════════════════════════════
# SUITE 2A: LABEL NOISE ANALYSIS
# ═══════════════════════════════════════════════════════════

def analyze_suite2a():
    """Analyze label noise sweep results."""
    print("\n" + "=" * 60)
    print("SUITE 2A ANALYSIS: LABEL NOISE SWEEP")
    print("=" * 60)

    noise_levels = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]
    k_values = [10, 20, 36]

    # Aliases for existing runs
    aliases = {
        (20, 0.0): "landauer_dense_k20",
        (20, 0.05): "landauer_k20_noise0.05",
        (20, 0.10): "landauer_k20_noise0.10",
        (20, 0.20): "landauer_k20_noise0.20",
        (10, 0.0): "landauer_dense_k10",
        (36, 0.0): "landauer_dense_k36",
    }

    results = []
    histories = {}

    for K in k_values:
        log_k = math.log(K)
        for noise in noise_levels:
            # Try canonical name first, then alias
            name = f"ziyin_noise_K{K}_p{noise:.2f}"
            h = load_history(name, OUTPUT_DIR)
            if h is None:
                alias = aliases.get((K, noise))
                if alias:
                    h = load_history(alias, OUTPUT_DIR)
                    if h:
                        name = alias

            if h is None:
                print(f"  K={K}, noise={noise:.2f}: NOT FOUND")
                continue

            tau = detect_tau(h, log_k)
            final_loss = h.get("first_target_loss", [None])[-1]
            converged = tau is not None

            # Q_transition
            q_trans = None
            if h.get("grad_norm_sq") and h.get("steps"):
                eta = 1e-3
                gnorm_sq = h["grad_norm_sq"]
                steps = h["steps"]
                eval_every = steps[1] - steps[0] if len(steps) > 1 else 50
                if tau:
                    tau_idx = next((i for i, s in enumerate(steps) if s >= tau), len(steps))
                    q_trans = sum(eta * g * eval_every for g in gnorm_sq[:tau_idx])

            results.append({
                "K": K, "noise": noise, "tau": tau, "q_trans": q_trans,
                "final_loss": final_loss, "converged": converged, "name": name,
            })
            histories[(K, noise)] = h

    # Print table
    print(f"\n{'K':>4}  {'noise':>6}  {'tau':>8}  {'Q_trans':>10}  {'final':>8}  {'conv':>5}")
    print("-" * 55)
    for r in results:
        tau_s = f"{r['tau']}" if r['tau'] else "---"
        q_s = f"{r['q_trans']:.2f}" if r['q_trans'] else "---"
        fl = f"{r['final_loss']:.4f}" if r['final_loss'] else "---"
        cv = "yes" if r['converged'] else "no"
        print(f"{r['K']:>4}  {r['noise']:>6.2f}  {tau_s:>8}  {q_s:>10}  {fl:>8}  {cv:>5}")

    with open(Path(OUTPUT_DIR) / "ziyin_noise_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # ---- Figure: Label Noise Sweep ----
    fig, axes = plt.subplots(1, len(k_values), figsize=(5 * len(k_values), 4))
    if len(k_values) == 1:
        axes = [axes]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(noise_levels)))

    for idx, K in enumerate(k_values):
        ax = axes[idx]
        log_k = math.log(K)
        for ni, noise in enumerate(noise_levels):
            h = histories.get((K, noise))
            if h and h.get("first_target_loss"):
                label = f"{noise:.0%}" if noise > 0 else "clean"
                ax.plot(h["steps"], h["first_target_loss"],
                        label=label, color=colors[ni], alpha=0.8)
        ax.axhline(y=log_k, color="gray", ls=":", alpha=0.5)
        ax.set_xlabel("Step")
        ax.set_ylabel("First-target loss")
        ax.set_title(f"K = {K}")
        ax.legend(fontsize=7, title="Noise")

    fig.suptitle("Label Noise Sweep", fontsize=12)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_label_noise_sweep.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_label_noise_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved fig_label_noise_sweep")

    return results


# ═══════════════════════════════════════════════════════════
# SUITE 2B: Q SLOPE ANALYSIS
# ═══════════════════════════════════════════════════════════

def analyze_suite2b():
    """Analyze Q vs logK slope results."""
    print("\n" + "=" * 60)
    print("SUITE 2B ANALYSIS: Q vs log(K) SLOPE")
    print("=" * 60)

    # Load pre-computed results if available
    results_path = Path(OUTPUT_DIR) / "ziyin_q_slope_results.json"
    if results_path.exists():
        with open(results_path) as f:
            data = json.load(f)
        slopes = data.get("slopes", [])
        results = data.get("results", {})

        if slopes:
            print(f"\n{'eta':>10}  {'slope':>10}  {'R2':>6}  {'n':>3}")
            print("-" * 35)
            for s in slopes:
                print(f"{s['eta']:>10.0e}  {s['slope']:>10.4f}  "
                      f"{s['r2']:>6.3f}  {s['n']:>3}")

        # ---- Figure: Q vs logK by eta ----
        if results:
            fig, ax = plt.subplots(figsize=(7, 5))
            colors = ["tab:blue", "tab:orange", "tab:green"]
            for i, (eta_str, q_vals) in enumerate(results.items()):
                if not q_vals:
                    continue
                ks = sorted(int(k) for k in q_vals.keys())
                log_ks = [math.log(k) for k in ks]
                qs = [q_vals[str(k)] for k in ks]
                c = colors[i % len(colors)]
                ax.plot(log_ks, qs, "o-", color=c, label=f"eta={eta_str}")

            ax.set_xlabel("log(K)")
            ax.set_ylabel("Q_transition")
            ax.set_title("Q_transition vs log(K) by Learning Rate")
            ax.legend()
            plt.tight_layout()
            fig.savefig(FIG_DIR / "fig_Q_vs_logK_by_eta.pdf", bbox_inches="tight")
            fig.savefig(FIG_DIR / "fig_Q_vs_logK_by_eta.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"\nSaved fig_Q_vs_logK_by_eta")
    else:
        print("  No Q slope results found. Run ziyin_q_slope.py first.")


# ═══════════════════════════════════════════════════════════
# SUITE 2C: BATCH SIZE ANALYSIS
# ═══════════════════════════════════════════════════════════

def analyze_suite2c():
    """Analyze batch size sweep results."""
    print("\n" + "=" * 60)
    print("SUITE 2C ANALYSIS: BATCH SIZE SWEEP")
    print("=" * 60)

    K = 20
    log_k = math.log(K)
    batch_sizes = [32, 64, 128, 256, 512]
    aliases = {
        64: "temp_lr1e3_bs64",
        128: "temp_lr1e3_bs128",
        256: "temp_lr1e3_bs256",
        512: "temp_lr1e3_bs512",
    }

    results = []
    histories = {}

    for bs in batch_sizes:
        # Try canonical name, then alias
        name = f"ziyin_batch_K{K}_bs{bs}"
        h = load_history(name, OUTPUT_DIR)
        if h is None:
            alias = aliases.get(bs)
            if alias:
                h = load_history(alias, OUTPUT_DIR)
                if h:
                    name = alias

        if h is None:
            print(f"  BS={bs}: NOT FOUND")
            continue

        tau = detect_tau(h, log_k)

        # Gradient norm during plateau
        gnorm_plateau = None
        if h.get("grad_norm_sq") and h.get("steps") and tau:
            steps = h["steps"]
            gnorm_sq = h["grad_norm_sq"]
            # Plateau = [20%, 60%] of tau
            start = int(0.2 * tau)
            end = int(0.6 * tau)
            plateau_gnorms = [
                math.sqrt(g) for s, g in zip(steps, gnorm_sq) if start <= s <= end
            ]
            if plateau_gnorms:
                gnorm_plateau = np.mean(plateau_gnorms)

        results.append({
            "BS": bs, "tau": tau, "gnorm_plateau": gnorm_plateau, "name": name,
        })
        histories[bs] = h

    # Print table
    print(f"\n{'BS':>5}  {'tau':>8}  {'gnorm_plat':>10}")
    print("-" * 30)
    for r in results:
        tau_s = f"{r['tau']}" if r['tau'] else "---"
        gn = f"{r['gnorm_plateau']:.4f}" if r['gnorm_plateau'] else "---"
        print(f"{r['BS']:>5}  {tau_s:>8}  {gn:>10}")

    with open(Path(OUTPUT_DIR) / "ziyin_batch_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # ---- Figure: Batch Size Sweep ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel a: tau vs BS
    ax = axes[0]
    bs_vals = [r["BS"] for r in results if r["tau"]]
    tau_vals = [r["tau"] for r in results if r["tau"]]
    if bs_vals:
        ax.plot(bs_vals, tau_vals, "o-", color="tab:blue", markersize=8)
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("tau (steps)")
        ax.set_title("Plateau Duration vs Batch Size")

    # Panel b: Loss curves
    ax = axes[1]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(batch_sizes)))
    for i, bs in enumerate(batch_sizes):
        h = histories.get(bs)
        if h and h.get("first_target_loss"):
            ax.plot(h["steps"], h["first_target_loss"],
                    label=f"BS={bs}", color=colors[i], alpha=0.8)
    ax.axhline(y=log_k, color="gray", ls=":", alpha=0.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("First-target loss")
    ax.set_title(f"Loss Curves (K={K})")
    ax.legend(fontsize=8)

    fig.suptitle(f"Batch Size Sweep (K={K}, eta=1e-3)", fontsize=12)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_batch_size_sweep.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_batch_size_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved fig_batch_size_sweep")

    # ---- Figure: Entropic Force (gradient norm vs BS) ----
    fig, ax = plt.subplots(figsize=(7, 5))
    bs_vals = [r["BS"] for r in results if r["gnorm_plateau"]]
    gn_vals = [r["gnorm_plateau"] for r in results if r["gnorm_plateau"]]
    if bs_vals:
        ax.plot(bs_vals, gn_vals, "o-", color="tab:red", markersize=8)
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Mean ||grad|| during plateau")
        ax.set_title("Gradient Norm During Plateau vs Batch Size")
        plt.tight_layout()
        fig.savefig(FIG_DIR / "fig_entropic_force.pdf", bbox_inches="tight")
        fig.savefig(FIG_DIR / "fig_entropic_force.png", dpi=150, bbox_inches="tight")
        print(f"Saved fig_entropic_force")
    plt.close(fig)

    return results


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("OVERNIGHT ANALYSIS")
    print("=" * 60)

    s1 = analyze_suite1()
    s2a = analyze_suite2a()
    analyze_suite2b()
    s2c = analyze_suite2c()

    print("\n" + "=" * 60)
    print("ALL ANALYSIS COMPLETE")
    print(f"Figures saved to: {FIG_DIR}")
    print(f"Summary JSONs saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
