"""
Visualization for Late Disambiguation Lag experiments.

Creates the key figures that tell the mechanistic story.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(results_path: Path) -> Dict[str, Any]:
    """Load probe results from JSON."""
    with open(results_path, "r") as f:
        return json.load(f)


def load_training_history(history_path: Path) -> Optional[Dict[str, Any]]:
    """Load training history from JSON (if present)."""
    if not history_path.exists():
        return None
    with open(history_path, "r") as f:
        return json.load(f)


def plot_training_curves(
    history: Dict[str, Any],
    save_path: Optional[Path] = None,
    title: str = "Training Progress",
):
    """Plot training loss and training accuracy."""
    has_first_loss = "first_target_loss" in history and history["first_target_loss"]
    n_cols = 3 if has_first_loss else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 4))
    if n_cols == 2:
        axes = [axes[0], axes[1]]
    else:
        axes = list(axes)
    
    steps = history["steps"]
    
    # Loss
    ax = axes[0]
    ax.plot(steps, history["train_loss"], label="Train", alpha=0.8)
    if has_first_loss:
        ax.plot(steps, history["first_target_loss"], label="First token", alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Accuracy
    ax = axes[1]
    if "train_accuracy" in history:
        ax.plot(steps, history["train_accuracy"], color="green", alpha=0.8)
        ax.set_title("Training Accuracy")
    else:
        ax.plot(steps, np.zeros_like(steps, dtype=float), color="green", alpha=0.4)
        ax.set_title("Accuracy (missing)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    if has_first_loss:
        ax = axes[2]
        ax.plot(steps, history["first_target_loss"], color="purple", alpha=0.8)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("First Target Loss")
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    
    return fig


def plot_attention_to_z_evolution(
    results: Dict[str, Any],
    save_path: Optional[Path] = None,
    title: str = "Attention to z Over Training",
):
    """
    Plot how attention to z evolves across training.
    
    This is THE key mechanistic plot showing when the model
    starts using the disambiguation signal.
    """
    steps = results["steps"]
    attn_results = results["probe_results"]["attention_to_z"]
    
    # Extract attention values: (n_steps, n_layers, n_heads)
    n_layers = len(list(attn_results.values())[0]["attention_to_z"])
    n_heads = len(list(attn_results.values())[0]["attention_to_z"][0])
    
    attn_matrix = np.zeros((len(steps), n_layers, n_heads))
    for i, step in enumerate(steps):
        attn_matrix[i] = np.array(attn_results[str(step)]["attention_to_z"])
    
    # Average across heads for main plot
    attn_avg = attn_matrix.mean(axis=-1)  # (n_steps, n_layers)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Line plot of average attention by layer
    ax = axes[0]
    for layer in range(n_layers):
        ax.plot(steps, attn_avg[:, layer], label=f"Layer {layer}", linewidth=2)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Attention to z (from A positions)")
    ax.set_title("Attention to z by Layer")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Right: Heatmap of per-head attention at final checkpoint
    ax = axes[1]
    final_attn = attn_matrix[-1]  # (n_layers, n_heads)
    sns.heatmap(
        final_attn,
        ax=ax,
        cmap="Blues",
        annot=True,
        fmt=".2f",
        xticklabels=[f"H{h}" for h in range(n_heads)],
        yticklabels=[f"L{l}" for l in range(n_layers)],
    )
    ax.set_title("Final Attention to z (by Layer/Head)")
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    
    return fig


def plot_logit_lens_evolution(
    results: Dict[str, Any],
    save_path: Optional[Path] = None,
    title: str = "Logit Lens: When Does Each Layer Know the Answer?",
):
    """Plot logit lens results showing when each layer knows the correct answer."""
    steps = results["steps"]
    ll_results = results["probe_results"]["logit_lens"]
    
    # Extract correct probability by layer: (n_steps, n_layers+1)
    n_layers_plus_one = len(list(ll_results.values())[0]["correct_prob_by_layer"])
    
    prob_matrix = np.zeros((len(steps), n_layers_plus_one))
    for i, step in enumerate(steps):
        prob_matrix[i] = np.array(ll_results[str(step)]["correct_prob_by_layer"])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Line plot over training
    ax = axes[0]
    for layer in range(n_layers_plus_one):
        label = "Embed" if layer == 0 else f"Layer {layer}"
        ax.plot(steps, prob_matrix[:, layer], label=label, linewidth=2)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("P(correct token)")
    ax.set_title("Correct Token Probability at Each Layer")
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Right: Heatmap (steps x layers)
    ax = axes[1]
    step_indices = np.linspace(0, len(steps) - 1, min(20, len(steps)), dtype=int)
    subsampled_probs = prob_matrix[step_indices]
    subsampled_steps = [steps[i] for i in step_indices]
    
    sns.heatmap(
        subsampled_probs.T,
        ax=ax,
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        xticklabels=[str(s) for s in subsampled_steps],
        yticklabels=["Embed"] + [f"L{l}" for l in range(n_layers_plus_one - 1)],
    )
    ax.set_title("P(correct) by Layer and Step")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Layer")
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    
    return fig


def plot_z_dependence_evolution(
    results: Dict[str, Any],
    save_path: Optional[Path] = None,
    title: str = "Causal z-Dependence Over Training",
):
    """
    Plot causal patching results showing when the model starts depending on z.
    This is the SMOKING GUN plot.
    """
    steps = results["steps"]
    patch_results = results["probe_results"]["causal_patching"]
    
    first_result = list(patch_results.values())[0]
    if "error" in first_result:
        print(f"Warning: Causal patching had error: {first_result['error']}")
        return None
    
    z_scores = []
    for step in steps:
        z_scores.append(patch_results[str(step)]["z_dependence_score"])
    
    has_layer_effects = "patching_effect_by_layer" in first_result
    
    if has_layer_effects:
        n_layers = len(first_result["patching_effect_by_layer"])
        layer_effects = np.zeros((len(steps), n_layers))
        for i, step in enumerate(steps):
            layer_effects[i] = np.array(patch_results[str(step)]["patching_effect_by_layer"])
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(8, 5))
        axes = [axes]
    
    # Main z-dependence plot
    ax = axes[0]
    ax.plot(steps, z_scores, linewidth=2, color="red", marker="o", markersize=4)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("z-Dependence Score")
    ax.set_title("Causal z-Dependence\n(0 = ignores z, 1 = uses z)")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)
    
    # Add horizontal line at 0.5 for reference
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    
    if has_layer_effects and len(axes) > 1:
        ax = axes[1]
        for layer in range(n_layers):
            ax.plot(steps, layer_effects[:, layer], label=f"Layer {layer}", linewidth=2)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Patching Effect")
        ax.set_title("Effect of Patching z at Each Layer")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    
    return fig


def plot_random_z_sensitivity_evolution(
    results: Dict[str, Any],
    save_path: Optional[Path] = None,
    title: str = "Random-z Sensitivity Over Training",
):
    """
    Plot how predictions change when z is swapped.
    """
    steps = results["steps"]
    rz_results = results["probe_results"]["random_z_eval"]
    first_result = list(rz_results.values())[0]
    if "error" in first_result:
        print(f"Warning: Random-z eval had error: {first_result['error']}")
        return None
    
    prob_drop = []
    change_rate = []
    for step in steps:
        prob_drop.append(rz_results[str(step)]["target_prob_drop"])
        change_rate.append(rz_results[str(step)]["argmax_change_rate"])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    ax.plot(steps, prob_drop, linewidth=2, color="purple", marker="o", markersize=3)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Target Prob Drop")
    ax.set_title("Target Prob Drop (z swapped)")
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.plot(steps, change_rate, linewidth=2, color="orange", marker="o", markersize=3)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Argmax Change Rate")
    ax.set_title("Argmax Change Rate (z swapped)")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    
    return fig


def plot_combined_dashboard(
    results: Dict[str, Any],
    history: Dict[str, Any],
    save_path: Optional[Path] = None,
    title: str = "Late Disambiguation Lag: Mechanistic Analysis",
):
    """
    Create a combined dashboard showing all key metrics.
    
    This is the PUBLICATION FIGURE.
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Grid: 3 rows, 2 columns
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    steps = results["steps"]
    
    # 1. Training loss (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history["steps"], history["train_loss"], color="blue", linewidth=2, label="Train loss")
    if "first_target_loss" in history and history["first_target_loss"]:
        ax1.plot(history["steps"], history["first_target_loss"], color="purple", linewidth=2, label="First target loss")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("A) Training Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Attention to z (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    attn_matrix = None
    if "attention_to_z" in results["probe_results"]:
        attn_results = results["probe_results"]["attention_to_z"]
        n_layers = len(list(attn_results.values())[0]["attention_to_z"])
        attn_matrix = np.zeros((len(steps), n_layers))
        for i, step in enumerate(steps):
            attn_matrix[i] = np.array(attn_results[str(step)]["attention_to_z"]).mean(axis=-1)
        for layer in range(n_layers):
            ax2.plot(steps, attn_matrix[:, layer], label=f"L{layer}", linewidth=2)
        ax2.legend()
        ax2.set_ylabel("Attention to z")
        ax2.set_title("B) Attention to Selector (z)")
    else:
        ax2.text(0.5, 0.5, "Attention-to-z probe missing", ha="center", va="center")
        ax2.set_title("B) Attention to Selector (z)")
    ax2.set_xlabel("Step")
    ax2.grid(True, alpha=0.3)
    
    # 3. Logit lens (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    prob_matrix = None
    if "logit_lens" in results["probe_results"]:
        ll_results = results["probe_results"]["logit_lens"]
        n_layers_plus_one = len(list(ll_results.values())[0]["correct_prob_by_layer"])
        prob_matrix = np.zeros((len(steps), n_layers_plus_one))
        for i, step in enumerate(steps):
            prob_matrix[i] = np.array(ll_results[str(step)]["correct_prob_by_layer"])
        # Just plot final layer and embedding
        ax3.plot(steps, prob_matrix[:, 0], label="Embed", linewidth=2, linestyle="--")
        ax3.plot(steps, prob_matrix[:, -1], label="Final", linewidth=2)
        ax3.legend()
        ax3.set_ylabel("P(correct)")
        ax3.set_ylim(0, 1)
    else:
        ax3.text(0.5, 0.5, "Logit-lens probe missing", ha="center", va="center")
    ax3.set_xlabel("Step")
    ax3.set_title("C) Logit Lens: P(correct)")
    ax3.grid(True, alpha=0.3)
    
    # 4. Causal patching (middle right)
    ax4 = fig.add_subplot(gs[1, 1])
    patch_results = results["probe_results"].get("causal_patching")
    z_scores = None
    first_result = None
    if patch_results:
        first_result = list(patch_results.values())[0]
        if "error" not in first_result:
            z_scores = [patch_results[str(step)]["z_dependence_score"] for step in steps]
            ax4.plot(steps, z_scores, color="red", linewidth=2, marker="o", markersize=3)
            ax4.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    else:
        ax4.text(0.5, 0.5, "Causal-patching probe missing", ha="center", va="center")
    ax4.set_xlabel("Step")
    ax4.set_ylabel("z-Dependence")
    ax4.set_title("D) Causal z-Dependence")
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-0.1, 1.1)
    
    # 5. Combined timeline (bottom, spans both columns)
    ax5 = fig.add_subplot(gs[2, :])
    
    # Normalize all metrics to [0, 1] for comparison
    if "train_accuracy" in history:
        acc = np.array(history["train_accuracy"])
        acc_label = "Train Accuracy"
        acc_interp = np.interp(steps, history["steps"], acc)
        acc_series = acc_interp
    else:
        loss = np.array(history["train_loss"])
        loss_interp = np.interp(steps, history["steps"], loss)
        acc_label = "Train Loss (inv)"
        acc_series = (loss_interp.max() - loss_interp) / (loss_interp.max() - loss_interp.min() + 1e-8)
    first_loss_series = None
    if "first_target_loss" in history and history["first_target_loss"]:
        first_loss = np.array(history["first_target_loss"])
        first_loss_interp = np.interp(steps, history["steps"], first_loss)
        first_loss_series = (first_loss_interp.max() - first_loss_interp) / (
            first_loss_interp.max() - first_loss_interp.min() + 1e-8
        )
    
    ax5.plot(steps, acc_series, label=acc_label, linewidth=2)
    if first_loss_series is not None:
        ax5.plot(steps, first_loss_series, label="First target loss (inv)", linewidth=2)
    if attn_matrix is not None:
        attn_avg = attn_matrix.mean(axis=-1)  # Average across layers
        attn_norm = (attn_avg - attn_avg.min()) / (attn_avg.max() - attn_avg.min() + 1e-8)
        ax5.plot(steps, attn_norm, label="Attention to z (norm)", linewidth=2)
    if prob_matrix is not None:
        prob_final = prob_matrix[:, -1]
        prob_norm = prob_final  # Already in [0, 1]
        ax5.plot(steps, prob_norm, label="P(correct) final layer", linewidth=2)
    if z_scores is not None:
        z_scores_arr = np.array(z_scores)
        ax5.plot(steps, z_scores_arr, label="z-Dependence", linewidth=2, color="red")
    
    if "random_z_eval" in results["probe_results"]:
        rz_results = results["probe_results"]["random_z_eval"]
        rz_first = list(rz_results.values())[0]
        if "error" not in rz_first:
            rz_change = [rz_results[str(step)]["argmax_change_rate"] for step in steps]
            ax5.plot(steps, rz_change, label="z-swap change rate", linewidth=2, color="orange")
    
    ax5.set_xlabel("Training Step")
    ax5.set_ylabel("Normalized Metric")
    ax5.set_title("E) Combined View: The Transition")
    ax5.legend(loc="center right")
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(-0.1, 1.1)
    
    plt.suptitle(title, fontsize=14, fontweight="bold")
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")
    
    return fig


def generate_all_figures(
    experiment_dir: Path,
    output_dir: Optional[Path] = None,
):
    """Generate all figures for an experiment."""
    if output_dir is None:
        output_dir = experiment_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    results_path = experiment_dir / "probe_results" / "all_probes.json"
    history_path = experiment_dir / "training_history.json"
    
    results = load_results(results_path)
    history = load_training_history(history_path)
    
    exp_name = experiment_dir.name
    
    # Generate figures
    if history is not None:
        plot_training_curves(
            history,
            save_path=output_dir / "training_curves.png",
            title=f"{exp_name}: Training Progress",
        )
    else:
        print(f"Warning: training history missing at {history_path}. Skipping training curves.")

    if "attention_to_z" in results["probe_results"]:
        plot_attention_to_z_evolution(
            results,
            save_path=output_dir / "attention_to_z.png",
            title=f"{exp_name}: Attention to z",
        )

    if "logit_lens" in results["probe_results"]:
        plot_logit_lens_evolution(
            results,
            save_path=output_dir / "logit_lens.png",
            title=f"{exp_name}: Logit Lens",
        )

    if "causal_patching" in results["probe_results"]:
        plot_z_dependence_evolution(
            results,
            save_path=output_dir / "z_dependence.png",
            title=f"{exp_name}: Causal z-Dependence",
        )

    if "random_z_eval" in results["probe_results"]:
        plot_random_z_sensitivity_evolution(
            results,
            save_path=output_dir / "random_z_sensitivity.png",
            title=f"{exp_name}: Random-z Sensitivity",
        )

    if history is not None:
        plot_combined_dashboard(
            results,
            history,
            save_path=output_dir / "dashboard.png",
            title=f"{exp_name}: Mechanistic Analysis Dashboard",
        )
    else:
        print("Warning: training history missing. Skipping combined dashboard.")
    
    print(f"\nAll figures saved to {output_dir}")


def _extract_attention_avg(results: Dict[str, Any]) -> Optional[Tuple[List[int], np.ndarray]]:
    steps = results["steps"]
    attn_results = results["probe_results"].get("attention_to_z")
    if not attn_results:
        return None
    n_layers = len(list(attn_results.values())[0]["attention_to_z"])
    attn_matrix = np.zeros((len(steps), n_layers))
    for i, step in enumerate(steps):
        attn_matrix[i] = np.array(attn_results[str(step)]["attention_to_z"]).mean(axis=-1)
    attn_avg = attn_matrix.mean(axis=-1)
    return steps, attn_avg


def _extract_logit_final(results: Dict[str, Any]) -> Optional[Tuple[List[int], np.ndarray]]:
    steps = results["steps"]
    ll_results = results["probe_results"].get("logit_lens")
    if not ll_results:
        return None
    n_layers_plus_one = len(list(ll_results.values())[0]["correct_prob_by_layer"])
    prob_matrix = np.zeros((len(steps), n_layers_plus_one))
    for i, step in enumerate(steps):
        prob_matrix[i] = np.array(ll_results[str(step)]["correct_prob_by_layer"])
    prob_final = prob_matrix[:, -1]
    return steps, prob_final


def _extract_z_dependence(results: Dict[str, Any]) -> Optional[Tuple[List[int], np.ndarray]]:
    steps = results["steps"]
    patch_results = results["probe_results"].get("causal_patching")
    if not patch_results:
        return None
    first_result = list(patch_results.values())[0]
    if "error" in first_result:
        return None
    z_scores = [patch_results[str(step)]["z_dependence_score"] for step in steps]
    return steps, np.array(z_scores)


def _extract_random_z_change(results: Dict[str, Any]) -> Optional[Tuple[List[int], np.ndarray]]:
    steps = results["steps"]
    rz_results = results["probe_results"].get("random_z_eval")
    if not rz_results:
        return None
    first_result = list(rz_results.values())[0]
    if "error" in first_result:
        return None
    change_rate = [rz_results[str(step)]["argmax_change_rate"] for step in steps]
    return steps, np.array(change_rate)


def plot_overlay_dashboard(
    experiment_dirs: List[Path],
    save_path: Optional[Path] = None,
    title: str = "Overlay: Training vs Probes",
):
    """
    Overlay key metrics across multiple experiments.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # A) Training loss
    ax = axes[0]
    for exp_dir in experiment_dirs:
        history = load_training_history(exp_dir / "training_history.json")
        ax.plot(history["steps"], history["train_loss"], label=exp_dir.name, linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Training Loss")
    ax.set_title("A) Training Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # B) Attention to z (avg)
    ax = axes[1]
    for exp_dir in experiment_dirs:
        results = load_results(exp_dir / "probe_results" / "all_probes.json")
        extracted = _extract_attention_avg(results)
        if extracted is None:
            continue
        steps, attn_avg = extracted
        ax.plot(steps, attn_avg, label=exp_dir.name, linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Attention to z (avg)")
    ax.set_title("B) Attention to z")
    ax.grid(True, alpha=0.3)
    if ax.lines:
        ax.legend()
    
    # C) Logit lens (final layer)
    ax = axes[2]
    for exp_dir in experiment_dirs:
        results = load_results(exp_dir / "probe_results" / "all_probes.json")
        extracted = _extract_logit_final(results)
        if extracted is None:
            continue
        steps, prob_final = extracted
        ax.plot(steps, prob_final, label=exp_dir.name, linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("P(correct) final layer")
    ax.set_title("C) Logit Lens (final layer)")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    if ax.lines:
        ax.legend()
    
    # D) Causal z-dependence
    ax = axes[3]
    has_any = False
    for exp_dir in experiment_dirs:
        results = load_results(exp_dir / "probe_results" / "all_probes.json")
        extracted = _extract_z_dependence(results)
        if extracted is None:
            continue
        steps, z_scores = extracted
        ax.plot(steps, z_scores, label=exp_dir.name, linewidth=2)
        has_any = True
    ax.set_xlabel("Step")
    ax.set_ylabel("z-Dependence")
    ax.set_title("D) Causal z-Dependence")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)
    if has_any:
        ax.legend()
    
    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")
    
    return fig


def generate_overlay_figures(
    experiment_dirs: List[Path],
    output_dir: Path,
):
    """Generate overlay figures for multiple experiments."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_overlay_dashboard(
        experiment_dirs,
        save_path=output_dir / "overlay_dashboard.png",
        title="Overlay: Training vs Probes",
    )
    
    # Random-z overlay (if available)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    has_any = False
    for exp_dir in experiment_dirs:
        results = load_results(exp_dir / "probe_results" / "all_probes.json")
        extracted = _extract_random_z_change(results)
        if extracted is None:
            continue
        steps, change_rate = extracted
        ax.plot(steps, change_rate, label=exp_dir.name, linewidth=2)
        has_any = True
    ax.set_xlabel("Step")
    ax.set_ylabel("Argmax Change Rate")
    ax.set_title("Overlay: Random-z Sensitivity")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    if has_any:
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "overlay_random_z.png", dpi=200, bbox_inches="tight")
        print(f"Saved: {output_dir / 'overlay_random_z.png'}")
