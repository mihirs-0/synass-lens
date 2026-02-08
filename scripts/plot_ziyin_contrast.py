#!/usr/bin/env python
"""
Plot contrast between linear baselines and transformer plateau dynamics.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt


def _load_json(path: Path) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def _apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "axes.grid": False,
        "lines.linewidth": 2,
        "font.size": 10,
    })


def _normalize_curve(values: List[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    denom = arr[0] - arr[-1]
    if abs(denom) < 1e-12:
        return np.zeros_like(arr)
    return (arr - arr[-1]) / denom


def _normalize_to_initial(values: List[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    base = arr[0] if abs(arr[0]) > 1e-12 else 1e-12
    return arr / base


def _plot_panel_a(ax, transformer: Dict, linear: Dict) -> None:
    t_steps = transformer["steps"]
    t_loss = transformer["candidate_loss"]
    l_steps = linear["steps"]
    l_loss = linear["total_mse"]

    t_norm = _normalize_curve(t_loss)
    l_norm = _normalize_curve(l_loss)

    ax.plot(l_steps, l_norm, color="blue", label="Linear: normalized MSE")
    ax.plot(t_steps, t_norm, color="red", label="Transformer: normalized candidate loss")

    log_k = transformer.get("log_k")
    if log_k is not None and len(t_loss) >= 2:
        denom = t_loss[0] - t_loss[-1]
        if abs(denom) > 1e-12:
            log_k_norm = (log_k - t_loss[-1]) / denom
            ax.axhline(
                log_k_norm,
                color="red",
                linestyle="--",
                alpha=0.5,
                label=f"log(K)={log_k:.2f}",
            )

    ax.text(
        0.02,
        0.92,
        "Transformer: plateau then cliff",
        transform=ax.transAxes,
        color="red",
        fontsize=8,
    )
    ax.text(
        0.02,
        0.84,
        "Linear: smooth decay",
        transform=ax.transAxes,
        color="blue",
        fontsize=8,
    )

    ax.set_title("Loss Curve Shape Comparison")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Normalized Loss (fraction remaining)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)


def _plot_panel_b(ax, linear: Dict) -> None:
    steps = linear["component_steps"]
    components = linear["component_losses"]

    fast = _normalize_to_initial(components["fast_component"])
    mid = _normalize_to_initial(components["mid_component"])
    slow = _normalize_to_initial(components["slow_component"])

    ax.plot(steps, fast, label="fast component", color="green")
    ax.plot(steps, mid, label="mid component", color="orange")
    ax.plot(steps, slow, label="slow component", color="purple")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 1.0)
    ax.set_title("Per-Component Learning Dynamics (Linear)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Normalized Component Error")
    ax.legend(fontsize=8)
    ax.text(0.02, 0.05, "Every component decays from step 0", transform=ax.transAxes)


def _plot_panel_c(ax, transformer: Dict, planted: Dict) -> None:
    t_steps = transformer["steps"]
    t_gap = transformer["z_gap"]
    p_steps = planted["eval_steps"]
    p_gap = planted["selector_gap"]

    transformer_line = ax.plot(t_steps, t_gap, color="red", label="Transformer: Δ_z")[0]
    ax.set_ylabel("Transformer Δ_z (nats)", color="red")
    ax.tick_params(axis="y", labelcolor="red")

    ax_right = ax.twinx()
    linear_line = ax_right.plot(p_steps, p_gap, color="blue", label="Linear: selector gap")[0]
    ax_right.set_ylabel("Linear selector gap (MSE)", color="blue")
    ax_right.tick_params(axis="y", labelcolor="blue")

    ax.text(
        0.02,
        0.92,
        "Transformer: zero-usage plateau then spike",
        transform=ax.transAxes,
        color="red",
        fontsize=8,
    )
    ax.set_title("Selector Usage Comparison")
    ax.set_xlabel("Training Step")
    ax.legend([transformer_line, linear_line], ["Transformer: Δ_z", "Linear: selector gap"], fontsize=8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Ziyin contrast figure")
    parser.add_argument(
        "--transformer-run",
        type=str,
        required=True,
        help="Transformer experiment name in outputs/",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Base output directory",
    )
    args = parser.parse_args()

    _apply_style()
    output_dir = Path(args.output_dir)

    transformer_path = output_dir / args.transformer_run / "candidate_eval_results.json"
    linear_path = output_dir / "linear_baseline" / "illconditioned_results.json"
    planted_path = output_dir / "linear_baseline" / "planted_results.json"

    if not transformer_path.exists():
        raise FileNotFoundError(f"Missing transformer results: {transformer_path}")
    if not linear_path.exists():
        raise FileNotFoundError(f"Missing linear results: {linear_path}")
    if not planted_path.exists():
        raise FileNotFoundError(f"Missing planted results: {planted_path}")

    transformer = _load_json(transformer_path)
    linear = _load_json(linear_path)
    planted = _load_json(planted_path)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    _plot_panel_a(axes[0], transformer, linear)
    _plot_panel_b(axes[1], linear)
    _plot_panel_c(axes[2], transformer, planted)

    fig.suptitle(
        "Ill-Conditioning vs Entropic Barrier: Qualitative Shape Comparison",
        fontsize=12,
    )
    fig.tight_layout()

    figures_dir = output_dir / "linear_baseline" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    save_path = figures_dir / "ziyin_contrast.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to: {save_path}")


if __name__ == "__main__":
    main()
