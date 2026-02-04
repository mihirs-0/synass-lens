#!/usr/bin/env python
"""
Analyze K-sweep experiments: compare Bz->A vs Az->B across K values.

Generates:
1. Per-K overlay: Bz->A vs Az->B training dynamics
2. K-dependence plot: how lag changes with K
"""

import sys
from pathlib import Path
import argparse
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from src.analysis.visualize import load_results, load_training_history


def load_experiment_data(exp_dir: Path):
    """Load training history and probe results for an experiment."""
    history_path = exp_dir / "training_history.json"
    results_path = exp_dir / "probe_results" / "all_probes.json"
    
    history = load_training_history(history_path)
    results = None
    if results_path.exists():
        results = load_results(results_path)
    
    return history, results


def plot_k_comparison(
    k: int,
    bz_to_a_dir: Path,
    az_to_b_dir: Path,
    save_path: Path,
):
    """
    Plot Bz->A vs Az->B comparison for a single K value.
    
    This is THE key comparison showing the lag.
    """
    bz_history, bz_results = load_experiment_data(bz_to_a_dir)
    az_history, az_results = load_experiment_data(az_to_b_dir)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    log_k = np.log(k)
    
    # 1. First-token loss comparison (top left) - THE KEY PLOT
    ax = axes[0, 0]
    if bz_history and "first_target_loss" in bz_history:
        ax.plot(bz_history["steps"], bz_history["first_target_loss"], 
                label="Bz→A", linewidth=2, color="blue")
    if az_history and "first_target_loss" in az_history:
        ax.plot(az_history["steps"], az_history["first_target_loss"], 
                label="Az→B", linewidth=2, color="orange")
    ax.axhline(y=log_k, color="red", linestyle="--", alpha=0.7, label=f"log({k}) = {log_k:.2f}")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("First Token Loss")
    ax.set_title(f"A) First Token Loss (K={k})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=-0.1)
    
    # 2. Training loss comparison (top right)
    ax = axes[0, 1]
    if bz_history:
        ax.plot(bz_history["steps"], bz_history["train_loss"], 
                label="Bz→A", linewidth=2, color="blue")
    if az_history:
        ax.plot(az_history["steps"], az_history["train_loss"], 
                label="Az→B", linewidth=2, color="orange")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Training Loss")
    ax.set_title(f"B) Training Loss (K={k})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Training accuracy comparison (bottom left)
    ax = axes[1, 0]
    if bz_history and "train_accuracy" in bz_history:
        ax.plot(bz_history["steps"], bz_history["train_accuracy"], 
                label="Bz→A", linewidth=2, color="blue")
    if az_history and "train_accuracy" in az_history:
        ax.plot(az_history["steps"], az_history["train_accuracy"], 
                label="Az→B", linewidth=2, color="orange")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Training Accuracy")
    ax.set_title(f"C) Training Accuracy (K={k})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # 4. Z-dependence comparison (bottom right) - if probe results exist
    ax = axes[1, 1]
    has_z_dep = False
    if bz_results and "causal_patching" in bz_results.get("probe_results", {}):
        patch = bz_results["probe_results"]["causal_patching"]
        steps = bz_results["steps"]
        z_scores = [patch[str(s)].get("z_dependence_score", 0) for s in steps]
        ax.plot(steps, z_scores, label="Bz→A", linewidth=2, color="blue")
        has_z_dep = True
    if az_results and "causal_patching" in az_results.get("probe_results", {}):
        patch = az_results["probe_results"]["causal_patching"]
        steps = az_results["steps"]
        z_scores = [patch[str(s)].get("z_dependence_score", 0) for s in steps]
        ax.plot(steps, z_scores, label="Az→B", linewidth=2, color="orange")
        has_z_dep = True
    
    if has_z_dep:
        ax.set_xlabel("Training Step")
        ax.set_ylabel("z-Dependence Score")
        ax.set_title(f"D) Causal z-Dependence (K={k})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.1)
    else:
        ax.text(0.5, 0.5, "No probe results", ha="center", va="center")
        ax.set_title(f"D) Causal z-Dependence (K={k})")
    
    plt.suptitle(f"Bz→A vs Az→B Comparison (K={k})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def compute_lag_metric(history, threshold_frac=0.5):
    """
    Compute when first_loss crosses threshold.
    
    Returns step at which first_loss < threshold_frac * initial_loss.
    """
    if not history or "first_target_loss" not in history:
        return None
    
    losses = history["first_target_loss"]
    steps = history["steps"]
    
    if not losses:
        return None
    
    initial_loss = losses[0]
    threshold = threshold_frac * initial_loss
    
    for step, loss in zip(steps, losses):
        if loss < threshold:
            return step
    
    return None  # Never crossed


def plot_k_dependence(
    k_values: list,
    output_dir: Path,
    save_path: Path,
):
    """
    Plot how the lag depends on K.
    
    Shows:
    - Convergence step for Bz->A vs Az->B at each K
    - The gap (lag) vs K
    """
    bz_convergence = []
    az_convergence = []
    lags = []
    valid_k = []
    
    for k in k_values:
        bz_dir = output_dir / f"bz_to_a_k{k}"
        az_dir = output_dir / f"az_to_b_k{k}"
        
        bz_history, _ = load_experiment_data(bz_dir)
        az_history, _ = load_experiment_data(az_dir)
        
        bz_step = compute_lag_metric(bz_history, threshold_frac=0.1)
        az_step = compute_lag_metric(az_history, threshold_frac=0.1)
        
        if bz_step is not None and az_step is not None:
            valid_k.append(k)
            bz_convergence.append(bz_step)
            az_convergence.append(az_step)
            lags.append(bz_step - az_step)
    
    if not valid_k:
        print("Warning: No valid data for K-dependence plot")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Convergence steps
    ax = axes[0]
    ax.plot(valid_k, bz_convergence, "o-", label="Bz→A", linewidth=2, markersize=8)
    ax.plot(valid_k, az_convergence, "s-", label="Az→B", linewidth=2, markersize=8)
    ax.set_xlabel("K (targets per base)")
    ax.set_ylabel("Steps to 10% of initial loss")
    ax.set_title("A) Convergence Speed vs K")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Lag vs K
    ax = axes[1]
    ax.plot(valid_k, lags, "o-", linewidth=2, markersize=8, color="red")
    ax.set_xlabel("K (targets per base)")
    ax.set_ylabel("Lag (steps)")
    ax.set_title("B) Disambiguation Lag vs K")
    ax.grid(True, alpha=0.3)
    
    # 3. Lag vs log(K)
    ax = axes[2]
    log_k = [np.log(k) for k in valid_k]
    ax.plot(log_k, lags, "o-", linewidth=2, markersize=8, color="purple")
    ax.set_xlabel("log(K)")
    ax.set_ylabel("Lag (steps)")
    ax.set_title("C) Lag vs log(K)")
    ax.grid(True, alpha=0.3)
    
    plt.suptitle("K-Dependence of Late Disambiguation Lag", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_all_k_overlay(
    k_values: list,
    output_dir: Path,
    save_path: Path,
):
    """
    Overlay first_loss curves for all K values on a single plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(k_values)))
    
    # Left: Bz->A across K
    ax = axes[0]
    for i, k in enumerate(k_values):
        bz_dir = output_dir / f"bz_to_a_k{k}"
        history, _ = load_experiment_data(bz_dir)
        if history and "first_target_loss" in history:
            ax.plot(history["steps"], history["first_target_loss"], 
                    label=f"K={k}", linewidth=2, color=colors[i])
            # Add log(K) reference line
            log_k = np.log(k)
            ax.axhline(y=log_k, color=colors[i], linestyle="--", alpha=0.3)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("First Token Loss")
    ax.set_title("Bz→A: First Token Loss vs K")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Right: Az->B across K
    ax = axes[1]
    for i, k in enumerate(k_values):
        az_dir = output_dir / f"az_to_b_k{k}"
        history, _ = load_experiment_data(az_dir)
        if history and "first_target_loss" in history:
            ax.plot(history["steps"], history["first_target_loss"], 
                    label=f"K={k}", linewidth=2, color=colors[i])
    ax.set_xlabel("Training Step")
    ax.set_ylabel("First Token Loss")
    ax.set_title("Az→B: First Token Loss vs K")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle("K-Sweep: First Token Loss Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze K-sweep experiments")
    parser.add_argument("--k-values", nargs="+", type=int, default=[5, 10, 20],
                        help="K values that were tested")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Base output directory")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "k_sweep_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("K-Sweep Analysis")
    print("=" * 60)
    print(f"K values: {args.k_values}")
    print(f"Output: {figures_dir}")
    print("=" * 60)
    
    # Generate per-K comparison plots
    for k in args.k_values:
        bz_dir = output_dir / f"bz_to_a_k{k}"
        az_dir = output_dir / f"az_to_b_k{k}"
        
        if bz_dir.exists() and az_dir.exists():
            print(f"\nGenerating K={k} comparison...")
            plot_k_comparison(
                k=k,
                bz_to_a_dir=bz_dir,
                az_to_b_dir=az_dir,
                save_path=figures_dir / f"comparison_k{k}.png",
            )
        else:
            print(f"Warning: Missing data for K={k}")
            if not bz_dir.exists():
                print(f"  - Missing: {bz_dir}")
            if not az_dir.exists():
                print(f"  - Missing: {az_dir}")
    
    # Generate K-dependence plot
    print("\nGenerating K-dependence plot...")
    plot_k_dependence(
        k_values=args.k_values,
        output_dir=output_dir,
        save_path=figures_dir / "k_dependence.png",
    )
    
    # Generate all-K overlay
    print("\nGenerating all-K overlay...")
    plot_all_k_overlay(
        k_values=args.k_values,
        output_dir=output_dir,
        save_path=figures_dir / "all_k_overlay.png",
    )
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Figures saved to: {figures_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
