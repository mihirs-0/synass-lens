#!/usr/bin/env python
"""
Plot candidate evaluation results for single or multiple experiments.
"""

import sys
from pathlib import Path
import argparse
import json
from typing import Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.visualize import load_training_history


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _apply_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "lines.linewidth": 2,
    })


def _plot_panel_a(ax, history, steps, loss_clean, log_k, onset_step):
    if history and "first_target_loss" in history and history["first_target_loss"]:
        ax.plot(
            history["steps"],
            history["first_target_loss"],
            label="first_target_loss",
            alpha=0.5,
            linewidth=1,
        )
    else:
        ax.text(0.5, 0.5, "first_target_loss missing", ha="center", va="center",
                transform=ax.transAxes, fontsize=8)

    ax.plot(steps, loss_clean, label="loss_clean", color="black")
    ax.axhline(log_k, color="gray", linestyle="--", linewidth=1, label=f"log(K) = {log_k:.2f}")
    if onset_step is not None:
        ax.axvline(onset_step, color="green", linestyle="dashdot", linewidth=1.5)
    ax.set_title("Per-Token First-Target Loss")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss (nats)")
    ax.legend(fontsize=8)


def _plot_panel_b(ax, steps, candidate_loss, log_k, onset_step):
    ax.plot(steps, candidate_loss, label="candidate_loss", color="blue")
    ax.axhline(
        log_k,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"log(K) = {log_k:.2f} (floor if ignoring z)",
    )
    ax.axhline(0, color="gray", linestyle=":", linewidth=1)
    if onset_step is not None:
        ax.axvline(onset_step, color="green", linewidth=1.5)
    ax.set_title("Candidate-Set Normalized Loss")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Candidate Loss (nats)")
    ax.set_ylim(bottom=-0.1)
    ax.legend(fontsize=8)


def _plot_panel_c(ax, steps, z_gap, gap_threshold, onset_step):
    ax.plot(steps, z_gap, label="z_gap", color="purple")
    ax.axhline(gap_threshold, color="gray", linestyle=":", linewidth=1, label="gap threshold")
    if onset_step is not None:
        ax.axvline(onset_step, color="green", linewidth=1.5)
    ax.set_title("z-Usage Gap (Δ_z)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("z_gap (nats)")
    ax.legend(fontsize=8)


def _plot_panel_d(ax, steps, candidate_acc, candidate_top3, k, onset_step):
    ax.plot(steps, candidate_acc, label="top1", color="teal")
    ax.plot(steps, candidate_top3, label="top3", color="teal", linestyle="--")
    ax.axhline(1.0 / k, color="red", linestyle=":", linewidth=1, label=f"chance = {1.0/k:.2f}")
    if onset_step is not None:
        ax.axvline(onset_step, color="green", linewidth=1.5)
    ax.set_title("Candidate Selection Accuracy")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)


def plot_single_experiment(experiment_dir: Path):
    results_path = experiment_dir / "candidate_eval_results.json"
    if not results_path.exists():
        print(f"Error: Missing results at {results_path}")
        sys.exit(1)

    results = _load_json(results_path)
    history = load_training_history(experiment_dir / "training_history.json")

    steps = results["steps"]
    loss_clean = results["loss_clean"]
    candidate_loss = results["candidate_loss"]
    candidate_acc = results["candidate_accuracy"]
    candidate_top3 = results["candidate_top3_accuracy"]
    z_gap = results["z_gap"]
    k = results["k"]
    log_k = results["log_k"]
    onset_step = results.get("binding_onset_step")
    gap_threshold = results.get("binding_onset_config", {}).get("gap_threshold", 0.5)

    figures_dir = experiment_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    _plot_panel_a(axes[0, 0], history, steps, loss_clean, log_k, onset_step)
    _plot_panel_b(axes[0, 1], steps, candidate_loss, log_k, onset_step)
    _plot_panel_c(axes[1, 0], steps, z_gap, gap_threshold, onset_step)
    _plot_panel_d(axes[1, 1], steps, candidate_acc, candidate_top3, k, onset_step)

    exp_name = experiment_dir.name
    fig.suptitle(f"{exp_name}: Disambiguation Evaluation Dashboard (K={k})", fontsize=12)
    fig.tight_layout()
    dashboard_path = figures_dir / "candidate_eval_dashboard.png"
    fig.savefig(dashboard_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {dashboard_path}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _plot_panel_a(axes[0], history, steps, loss_clean, log_k, onset_step)
    _plot_panel_c(axes[1], steps, z_gap, gap_threshold, onset_step)
    fig.suptitle(f"{exp_name}: z-Usage Diagnostics (K={k})", fontsize=12)
    fig.tight_layout()
    z_usage_path = figures_dir / "candidate_eval_z_usage.png"
    fig.savefig(z_usage_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {z_usage_path}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _plot_panel_b(axes[0], steps, candidate_loss, log_k, onset_step)
    _plot_panel_d(axes[1], steps, candidate_acc, candidate_top3, k, onset_step)
    fig.suptitle(f"{exp_name}: Candidate Diagnostics (K={k})", fontsize=12)
    fig.tight_layout()
    candidate_path = figures_dir / "candidate_eval_candidate.png"
    fig.savefig(candidate_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {candidate_path}")


def plot_overlay(experiment_dirs: List[Path], output_dir: Path):
    runs = []
    for exp_dir in experiment_dirs:
        results_path = exp_dir / "candidate_eval_results.json"
        if not results_path.exists():
            print(f"Warning: Missing results at {results_path}, skipping.")
            continue
        results = _load_json(results_path)
        runs.append({
            "name": exp_dir.name,
            "results": results,
            "k": results["k"],
            "log_k": results["log_k"],
        })

    if not runs:
        print("Error: No valid candidate_eval_results.json files found.")
        sys.exit(1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cmap = plt.cm.viridis
    n = len(runs)

    for i, run in enumerate(runs):
        color = cmap(0.5 if n == 1 else i / (n - 1))
        steps = run["results"]["steps"]
        label = f"{run['name']} (K={run['k']})"

        axes[0].plot(steps, run["results"]["candidate_loss"], label=label, color=color)
        axes[0].axhline(run["log_k"], color=color, linestyle="--", alpha=0.4)

        axes[1].plot(steps, run["results"]["z_gap"], label=label, color=color)
        axes[2].plot(steps, run["results"]["candidate_accuracy"], label=label, color=color)

    axes[0].set_title("Candidate Loss")
    axes[0].set_xlabel("Training Step")
    axes[0].set_ylabel("Loss (nats)")

    axes[1].set_title("z-Usage Gap (Δ_z)")
    axes[1].set_xlabel("Training Step")
    axes[1].set_ylabel("z_gap (nats)")

    axes[2].set_title("Candidate Accuracy")
    axes[2].set_xlabel("Training Step")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_ylim(0, 1)

    for ax in axes:
        ax.legend(fontsize=8)

    fig.suptitle("Candidate Evaluation Overlay", fontsize=12)
    fig.tight_layout()

    overlay_dir = output_dir / "candidate_eval_overlay"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    save_path = overlay_dir / "candidate_eval_overlay.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot candidate evaluation results")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--experiment", type=str, help="Single experiment name")
    group.add_argument("--experiments", nargs="+", help="Multiple experiments for overlay")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Base output directory")
    args = parser.parse_args()

    _apply_style()
    output_dir = Path(args.output_dir)

    if args.experiment:
        experiment_dir = output_dir / args.experiment
        plot_single_experiment(experiment_dir)
    else:
        experiment_dirs = [output_dir / name for name in args.experiments]
        plot_overlay(experiment_dirs, output_dir)


if __name__ == "__main__":
    main()
