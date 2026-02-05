#!/usr/bin/env python
"""
Analyze temperature sweep experiments for disambiguation lag.

Loads training histories, computes lag metrics, and generates figures.
"""

import sys
from pathlib import Path
import argparse
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.visualize import load_training_history

K_VALUE = 10

SWEEP_CONFIGS = [
    {"name": "temp_lr1e3_bs64", "lr": 1e-3, "bs": 64},
    {"name": "temp_lr1e3_bs128", "lr": 1e-3, "bs": 128},
    {"name": "temp_lr1e3_bs256", "lr": 1e-3, "bs": 256},
    {"name": "temp_lr1e3_bs512", "lr": 1e-3, "bs": 512},
    {"name": "temp_lr5e4_bs128", "lr": 5e-4, "bs": 128},
    {"name": "temp_lr2e3_bs128", "lr": 2e-3, "bs": 128},
]

CONTROL_TEFF_NAMES = {"temp_lr1e3_bs256", "temp_lr5e4_bs128"}


def _format_teff(teff: float) -> str:
    """Format effective temperature for labels."""
    return f"{teff:.2e}"


def _first_step_where(steps: List[int], values: List[float], predicate) -> Optional[int]:
    """Return the first step where predicate(value) is True."""
    for step, value in zip(steps, values):
        if predicate(value):
            return step
    return None


def compute_lag_metrics(history: Dict[str, Any], k_value: int) -> Dict[str, Optional[int]]:
    """Compute lag metrics from a training history."""
    if not history:
        return {
            "steps_to_below_half_logK": None,
            "steps_to_z_divergence": None,
            "steps_to_90pct_accuracy": None,
        }

    steps = history.get("steps", [])
    first_target_loss = history.get("first_target_loss", [])
    train_accuracy = history.get("train_accuracy", [])
    loss_z_shuffled = history.get("loss_z_shuffled", [])

    log_k = np.log(k_value)
    lag_loss = None
    if first_target_loss:
        threshold = 0.5 * log_k
        lag_loss = _first_step_where(steps, first_target_loss, lambda v: v < threshold)

    lag_z_div = None
    if first_target_loss and loss_z_shuffled:
        gaps = [z - f for z, f in zip(loss_z_shuffled, first_target_loss)]
        lag_z_div = _first_step_where(steps, gaps, lambda v: v > 0.5)

    lag_acc = None
    if train_accuracy:
        lag_acc = _first_step_where(steps, train_accuracy, lambda v: v > 0.9)

    return {
        "steps_to_below_half_logK": lag_loss,
        "steps_to_z_divergence": lag_z_div,
        "steps_to_90pct_accuracy": lag_acc,
    }


def _collect_run_histories(output_dir: Path) -> List[Dict[str, Any]]:
    """Load histories and metadata for all sweep runs."""
    runs = []
    for cfg in SWEEP_CONFIGS:
        exp_dir = output_dir / cfg["name"]
        history_path = exp_dir / "training_history.json"
        history = load_training_history(history_path)
        if history is None:
            print(f"Warning: Missing history at {history_path}")
            continue
        runs.append({
            "name": cfg["name"],
            "lr": cfg["lr"],
            "bs": cfg["bs"],
            "teff": cfg["lr"] / cfg["bs"],
            "history": history,
        })
    return runs


def _get_color_map(runs: List[Dict[str, Any]]):
    """Return a colormap function keyed by T_eff."""
    teffs = [r["teff"] for r in runs]
    vmin, vmax = min(teffs), max(teffs)
    cmap = plt.cm.viridis

    def color_for(teff: float):
        if vmax == vmin:
            return cmap(0.5)
        return cmap((teff - vmin) / (vmax - vmin))

    return color_for


def _plot_loss_curves(runs: List[Dict[str, Any]], save_path: Path):
    """Plot first-target loss curves for all runs."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    color_for = _get_color_map(runs)

    for run in runs:
        history = run["history"]
        if "first_target_loss" not in history:
            continue
        label = f"LR={run['lr']}, BS={run['bs']}, T={_format_teff(run['teff'])}"
        ax.plot(
            history["steps"],
            history["first_target_loss"],
            label=label,
            linewidth=2,
            color=color_for(run["teff"]),
        )

    log_k = np.log(K_VALUE)
    ax.axhline(y=log_k, color="gray", linestyle="--", alpha=0.7, label=f"log({K_VALUE}) = {log_k:.2f}")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("First Target Loss")
    ax.set_title("Temperature Sweep: First Target Loss (K=10)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def _plot_z_usage(runs: List[Dict[str, Any]], save_path: Path):
    """Plot z-usage dynamics for all runs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    color_for = _get_color_map(runs)

    # Left: first_target_loss vs loss_z_shuffled
    ax = axes[0]
    for run in runs:
        history = run["history"]
        steps = history.get("steps", [])
        first_loss = history.get("first_target_loss", [])
        z_loss = history.get("loss_z_shuffled", [])
        if not steps or not first_loss:
            continue
        color = color_for(run["teff"])
        label = f"LR={run['lr']}, BS={run['bs']}"
        ax.plot(steps, first_loss, linewidth=2, color=color, label=label)
        if z_loss:
            ax.plot(steps, z_loss, linewidth=2, color=color, linestyle="--")
        else:
            print(f"Warning: Missing loss_z_shuffled for {run['name']}")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("First Target Loss")
    ax.set_title("First Target Loss (solid) vs z-shuffled (dashed)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Right: gap
    ax = axes[1]
    for run in runs:
        history = run["history"]
        steps = history.get("steps", [])
        first_loss = history.get("first_target_loss", [])
        z_loss = history.get("loss_z_shuffled", [])
        if not steps or not first_loss or not z_loss:
            continue
        gap = [z - f for z, f in zip(z_loss, first_loss)]
        ax.plot(
            steps,
            gap,
            linewidth=2,
            color=color_for(run["teff"]),
            label=f"LR={run['lr']}, BS={run['bs']}",
        )
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss Gap (z-shuffled - true)")
    ax.set_title("z-Usage Gap Over Time")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def _fit_regression(teffs: List[float], lags: List[float]) -> Optional[Tuple[float, float, float]]:
    """Fit linear regression in log10(T_eff) and return slope, intercept, r^2."""
    if len(teffs) < 2:
        return None
    x = np.log10(np.array(teffs, dtype=float))
    y = np.array(lags, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope, intercept, r2


def _plot_lag_vs_teff(
    runs: List[Dict[str, Any]],
    metric_key: str,
    metric_label: str,
    ax,
):
    """Plot lag metric vs T_eff on a log scale."""
    teffs = []
    lags = []
    labels = []
    control_teffs = []

    for run in runs:
        metrics = run["metrics"]
        lag = metrics.get(metric_key)
        if lag is None:
            continue
        teffs.append(run["teff"])
        lags.append(lag)
        labels.append(f"LR={run['lr']}, BS={run['bs']}")
        if run["name"] in CONTROL_TEFF_NAMES:
            control_teffs.append(run["teff"])

    if not teffs:
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center")
        ax.set_title(metric_label)
        return

    color_for = _get_color_map(runs)
    for teff, lag, label, run in zip(teffs, lags, labels, [r for r in runs if r["metrics"].get(metric_key) is not None]):
        is_control = run["name"] in CONTROL_TEFF_NAMES
        marker = "s" if is_control else "o"
        edgecolor = "black" if is_control else "none"
        ax.scatter(
            teff,
            lag,
            s=80,
            marker=marker,
            color=color_for(teff),
            edgecolor=edgecolor,
            linewidth=1,
        )
        ax.text(teff, lag, label, fontsize=8, ha="left", va="bottom")

    ax.set_xscale("log")
    ax.set_xlabel("T_eff = LR / BS (log scale)")
    ax.set_ylabel("Lag (steps)")
    ax.set_title(metric_label)
    ax.grid(True, alpha=0.3)

    reg = _fit_regression(teffs, lags)
    if reg is not None:
        slope, intercept, r2 = reg
        x_fit = np.logspace(np.log10(min(teffs)), np.log10(max(teffs)), 100)
        y_fit = slope * np.log10(x_fit) + intercept
        ax.plot(x_fit, y_fit, color="black", linestyle="--", linewidth=2)
        ax.text(
            0.02,
            0.95,
            f"R² = {r2:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )


def _plot_lag_panels(runs: List[Dict[str, Any]], save_path: Path):
    """Plot lag metrics vs T_eff in a 3-panel figure."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    _plot_lag_vs_teff(
        runs,
        "steps_to_z_divergence",
        "A) Lag to z-divergence",
        axes[0],
    )
    _plot_lag_vs_teff(
        runs,
        "steps_to_below_half_logK",
        "B) Lag to 0.5*log(K) loss",
        axes[1],
    )
    _plot_lag_vs_teff(
        runs,
        "steps_to_90pct_accuracy",
        "C) Lag to 90% accuracy",
        axes[2],
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def _plot_dashboard(runs: List[Dict[str, Any]], save_path: Path):
    """Create a 2x2 dashboard summarizing the sweep."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # Top-left: loss curves
    ax = axes[0]
    color_for = _get_color_map(runs)
    for run in runs:
        history = run["history"]
        if "first_target_loss" not in history:
            continue
        ax.plot(
            history["steps"],
            history["first_target_loss"],
            linewidth=2,
            color=color_for(run["teff"]),
            label=f"LR={run['lr']}, BS={run['bs']}",
        )
    log_k = np.log(K_VALUE)
    ax.axhline(y=log_k, color="gray", linestyle="--", alpha=0.7, label=f"log({K_VALUE})")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("First Target Loss")
    ax.set_title("A) First Target Loss")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Top-right: z-divergence gap
    ax = axes[1]
    for run in runs:
        history = run["history"]
        steps = history.get("steps", [])
        first_loss = history.get("first_target_loss", [])
        z_loss = history.get("loss_z_shuffled", [])
        if not steps or not first_loss or not z_loss:
            continue
        gap = [z - f for z, f in zip(z_loss, first_loss)]
        ax.plot(
            steps,
            gap,
            linewidth=2,
            color=color_for(run["teff"]),
            label=f"LR={run['lr']}, BS={run['bs']}",
        )
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss Gap")
    ax.set_title("B) z-Usage Gap")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Bottom-left: lag vs T_eff (z-divergence)
    _plot_lag_vs_teff(
        runs,
        "steps_to_z_divergence",
        "C) Lag vs T_eff (z-divergence)",
        axes[2],
    )

    # Bottom-right: table summary
    ax = axes[3]
    ax.axis("off")
    columns = ["Config", "LR", "BS", "T_eff", "Lag(z)", "Lag(loss)", "Lag(acc)"]
    cell_text = []
    for run in runs:
        metrics = run["metrics"]
        cell_text.append([
            run["name"],
            f"{run['lr']:.1e}",
            str(run["bs"]),
            _format_teff(run["teff"]),
            "NA" if metrics["steps_to_z_divergence"] is None else str(metrics["steps_to_z_divergence"]),
            "NA" if metrics["steps_to_below_half_logK"] is None else str(metrics["steps_to_below_half_logK"]),
            "NA" if metrics["steps_to_90pct_accuracy"] is None else str(metrics["steps_to_90pct_accuracy"]),
        ])
    table = ax.table(cellText=cell_text, colLabels=columns, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)
    ax.set_title("D) Sweep Summary Table")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def _print_summary_table(runs: List[Dict[str, Any]]):
    """Print a formatted summary table to stdout."""
    print("Temperature Sweep Results (K=10)")
    print("=" * 100)
    header = (
        f"{'Config':<22} {'LR':<8} {'BS':<6} {'T_eff':<10} "
        f"{'Lag(z_div)':<12} {'EffS(z_div)':<12} "
        f"{'Lag(loss)':<10} {'EffS(loss)':<12} "
        f"{'Lag(acc)':<10} {'EffS(acc)':<12}"
    )
    print(header)
    print("-" * 100)

    for run in runs:
        metrics = run["metrics"]
        def fmt_step(step):
            return "NA" if step is None else str(step)
        def fmt_eff(step, bs):
            return "NA" if step is None else str(step * bs)

        print(
            f"{run['name']:<22} {run['lr']:<8.1e} {run['bs']:<6} {run['teff']:<10.2e} "
            f"{fmt_step(metrics['steps_to_z_divergence']):<12} {fmt_eff(metrics['steps_to_z_divergence'], run['bs']):<12} "
            f"{fmt_step(metrics['steps_to_below_half_logK']):<10} {fmt_eff(metrics['steps_to_below_half_logK'], run['bs']):<12} "
            f"{fmt_step(metrics['steps_to_90pct_accuracy']):<10} {fmt_eff(metrics['steps_to_90pct_accuracy'], run['bs']):<12}"
        )

    print("=" * 100)

    teffs = []
    lags = []
    for run in runs:
        lag = run["metrics"].get("steps_to_z_divergence")
        if lag is not None:
            teffs.append(run["teff"])
            lags.append(lag)

    if len(teffs) >= 2:
        r = np.corrcoef(np.array(teffs, dtype=float), np.array(lags, dtype=float))[0, 1]
        print(f"Correlation(T_eff, lag_z_divergence): r = {r:.3f}")
    else:
        print("Correlation(T_eff, lag_z_divergence): r = NA (insufficient data)")


def main():
    parser = argparse.ArgumentParser(description="Analyze temperature sweep experiments")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Base output directory")
    args = parser.parse_args()

    sns.set_theme(style="whitegrid")
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "temperature_sweep_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Temperature Sweep Analysis")
    print("=" * 60)
    print(f"Output: {figures_dir}")
    print("=" * 60)

    runs = _collect_run_histories(output_dir)
    if not runs:
        print("No valid runs found. Exiting.")
        return

    for run in runs:
        run["metrics"] = compute_lag_metrics(run["history"], K_VALUE)

    _plot_loss_curves(runs, figures_dir / "temperature_sweep_loss_curves.png")
    _plot_z_usage(runs, figures_dir / "temperature_sweep_z_usage.png")
    _plot_lag_panels(runs, figures_dir / "temperature_sweep_lag_vs_teff.png")
    _plot_dashboard(runs, figures_dir / "temperature_sweep_dashboard.png")

    _print_summary_table(runs)

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Figures saved to: {figures_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
