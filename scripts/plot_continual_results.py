#!/usr/bin/env python
"""
Multi-panel analysis figure for continual learning experiments.

Panels:
  A: Candidate loss trajectories for each reassignment fraction
  B: Q_transition vs. f  (the main result)
  C: Q_continual vs. Q_scratch comparison
  D: Forgetting dynamics (old_data_candidate_loss over time)

Usage:
    python scripts/plot_continual_results.py \
        --experiments continual_reassign_f0.0 continual_reassign_f0.1 \
                      continual_reassign_f0.25 continual_reassign_f0.5 \
                      continual_reassign_f0.75 continual_reassign_f1.0 \
        --output outputs/continual_analysis/
"""

import sys
import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not available. Skipping plots.")


# Fraction → color for consistent styling
FRAC_COLORS = {
    0.0: "#888888",
    0.1: "#2196F3",
    0.25: "#4CAF50",
    0.5: "#FF9800",
    0.75: "#F44336",
    1.0: "#9C27B0",
}


def load_experiment(exp_name: str, output_dir: str):
    """Load all available data for an experiment."""
    exp_dir = Path(output_dir) / exp_name
    data = {"name": exp_name, "exists": False}

    config_path = exp_dir / "config.yaml"
    if not config_path.exists():
        return data

    data["exists"] = True
    with open(config_path) as f:
        data["config"] = yaml.safe_load(f)

    history_path = exp_dir / "training_history.json"
    if history_path.exists():
        with open(history_path) as f:
            data["history"] = json.load(f)

    grad_path = exp_dir / "gradient_norm_results.json"
    if grad_path.exists():
        with open(grad_path) as f:
            data["gradient_norms"] = json.load(f)

    cand_path = exp_dir / "candidate_eval_results.json"
    if cand_path.exists():
        with open(cand_path) as f:
            data["candidate_eval"] = json.load(f)

    div_path = exp_dir / "divergence.json"
    if div_path.exists():
        with open(div_path) as f:
            data["divergence"] = json.load(f)

    return data


def get_fraction(exp_data):
    """Extract reassignment fraction from experiment config."""
    cfg = exp_data.get("config", {})
    continual = cfg.get("continual", {})
    frac = continual.get("fraction")
    if frac is not None:
        return float(frac)
    name = exp_data["name"]
    if "_f" in name:
        try:
            return float(name.split("_f")[-1].split("_")[0])
        except ValueError:
            pass
    return None


def find_continual_transition_window(steps, candidate_loss):
    """
    For continual learning: define transition relative to actual start/end loss.

    transition_start: last step where candidate_loss > 0.9 * initial_loss
    transition_end:   first step where candidate_loss < 0.1 * initial_loss
    """
    if not steps or not candidate_loss:
        return None, None

    initial_loss = candidate_loss[0]
    if initial_loss < 0.1:
        return None, None

    threshold_high = 0.9 * initial_loss
    threshold_low = max(0.1 * initial_loss, 0.05)

    transition_start = None
    transition_end = None

    for i, cl in enumerate(candidate_loss):
        if cl > threshold_high:
            transition_start = steps[i]

    for i, cl in enumerate(candidate_loss):
        if cl < threshold_low:
            transition_end = steps[i]
            break

    return transition_start, transition_end


def reconstruct_lr_schedule(peak_lr, warmup_steps, max_steps, scheduler_type="constant"):
    lrs = np.zeros(max_steps + 1)
    for step in range(max_steps + 1):
        if scheduler_type == "constant":
            lrs[step] = peak_lr
        elif step < warmup_steps:
            lrs[step] = peak_lr * (step / warmup_steps) if warmup_steps > 0 else peak_lr
        else:
            progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
            lrs[step] = peak_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    return lrs


def compute_q_from_gradient_norms(exp_data):
    """Compute cumulative dissipation Q from gradient norm results."""
    gn = exp_data.get("gradient_norms")
    if gn is None:
        return None

    cfg = exp_data["config"]
    training = cfg["training"]
    peak_lr = float(training["learning_rate"])
    warmup_steps = int(training.get("warmup_steps", 0))
    max_steps = int(training.get("max_steps", 30000))
    scheduler_type = training.get("scheduler", "constant")

    lr_schedule = reconstruct_lr_schedule(peak_lr, warmup_steps, max_steps, scheduler_type)

    steps = np.array(gn["steps"])
    norms_sq = np.array(gn["total_grad_norm_sq"])

    max_step = len(lr_schedule) - 1
    safe_steps = np.clip(steps, 0, max_step).astype(int)
    eta = lr_schedule[safe_steps]

    delta_s = np.diff(steps, prepend=0)
    dissipation = eta * norms_sq * delta_s
    Q_cumulative = np.cumsum(dissipation)

    return {
        "steps": steps.tolist(),
        "Q_cumulative": Q_cumulative.tolist(),
        "dissipation": dissipation.tolist(),
        "Q_total": float(Q_cumulative[-1]),
    }


def compute_q_transition(exp_data, q_data):
    """Compute Q during the transition window."""
    if q_data is None:
        return None

    history = exp_data.get("history", {})
    steps_h = history.get("steps", [])
    new_cand = history.get("new_candidate_loss", [])

    if not steps_h or not new_cand:
        cand_eval = exp_data.get("candidate_eval", {})
        steps_h = cand_eval.get("steps", [])
        new_cand = cand_eval.get("candidate_loss", [])

    if not steps_h or not new_cand:
        return None

    t_start, t_end = find_continual_transition_window(steps_h, new_cand)
    if t_start is None or t_end is None:
        return None

    q_steps = np.array(q_data["steps"])
    q_cum = np.array(q_data["Q_cumulative"])

    Q_at_start = float(np.interp(t_start, q_steps, q_cum))
    Q_at_end = float(np.interp(t_end, q_steps, q_cum))

    return {
        "Q_transition": Q_at_end - Q_at_start,
        "transition_start": t_start,
        "transition_end": t_end,
        "Q_at_start": Q_at_start,
        "Q_at_end": Q_at_end,
    }


def plot_panel_a(ax, experiments):
    """Panel A: Candidate loss trajectories for each f."""
    for exp_data in experiments:
        frac = get_fraction(exp_data)
        if frac is None:
            continue

        history = exp_data.get("history", {})
        steps = history.get("steps", [])
        new_cand = history.get("new_candidate_loss", [])

        if not steps or not new_cand:
            continue

        color = FRAC_COLORS.get(frac, "#333333")
        ax.plot(steps, new_cand, color=color, linewidth=1.5,
                label=f"f={frac}", alpha=0.85)

    ax.set_xlabel("Steps since distribution shift")
    ax.set_ylabel("Candidate loss (new data)")
    ax.set_title("A. Candidate Loss Trajectories")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(bottom=-0.1)
    ax.grid(True, alpha=0.3)


def plot_panel_b(ax, experiments):
    """Panel B: Q_transition vs. f — the main result."""
    fracs = []
    q_trans = []
    labels = []

    for exp_data in experiments:
        frac = get_fraction(exp_data)
        if frac is None or frac == 0.0:
            continue

        q_data = compute_q_from_gradient_norms(exp_data)
        qt = compute_q_transition(exp_data, q_data)
        if qt is None:
            continue

        fracs.append(frac)
        q_trans.append(qt["Q_transition"])
        labels.append(exp_data["name"])

    if not fracs:
        ax.text(0.5, 0.5, "No Q_transition data available",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="gray")
        ax.set_title("B. Q_transition vs. f")
        return

    fracs = np.array(fracs)
    q_trans = np.array(q_trans)
    sort_idx = np.argsort(fracs)
    fracs = fracs[sort_idx]
    q_trans = q_trans[sort_idx]

    ax.plot(fracs, q_trans, "o-", color="#1565C0", linewidth=2, markersize=8)

    # Linear fit
    if len(fracs) >= 2:
        coeffs = np.polyfit(fracs, q_trans, 1)
        fit_x = np.linspace(0, 1, 100)
        fit_y = np.polyval(coeffs, fit_x)
        ax.plot(fit_x, fit_y, "--", color="#1565C0", alpha=0.4, linewidth=1)

        ss_res = np.sum((q_trans - np.polyval(coeffs, fracs)) ** 2)
        ss_tot = np.sum((q_trans - np.mean(q_trans)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        ax.text(0.05, 0.92, f"slope={coeffs[0]:.4f}\nR²={r2:.3f}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_xlabel("Reassignment fraction f")
    ax.set_ylabel("Q_transition (dissipation)")
    ax.set_title("B. Q_transition vs. f")
    ax.set_xlim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)


def plot_panel_c(ax, experiments, output_dir):
    """Panel C: Q_continual vs. Q_scratch comparison."""
    # Try to load Q_scratch from base landauer results
    q_scratch = None
    landauer_path = Path(output_dir) / "landauer_results.json"
    if landauer_path.exists():
        with open(landauer_path) as f:
            landauer = json.load(f)
        for exp in landauer.get("experiments", []):
            if exp.get("K") == 20 and exp.get("Q_transition") is not None:
                q_scratch = exp["Q_transition"]
                break

    # Get Q for f=1.0 (full reassignment)
    q_continual = None
    for exp_data in experiments:
        frac = get_fraction(exp_data)
        if frac == 1.0:
            q_data = compute_q_from_gradient_norms(exp_data)
            qt = compute_q_transition(exp_data, q_data)
            if qt is not None:
                q_continual = qt["Q_transition"]
            break

    bar_labels = []
    bar_values = []
    bar_colors = []

    if q_scratch is not None:
        bar_labels.append("From scratch\n(K=20)")
        bar_values.append(q_scratch)
        bar_colors.append("#2196F3")

    if q_continual is not None:
        bar_labels.append("Continual\n(f=1.0)")
        bar_values.append(q_continual)
        bar_colors.append("#F44336")

    if not bar_values:
        ax.text(0.5, 0.5, "Insufficient data for comparison",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="gray")
        ax.set_title("C. Q_continual vs. Q_scratch")
        return

    bars = ax.bar(bar_labels, bar_values, color=bar_colors, alpha=0.8, width=0.5)
    for bar, val in zip(bars, bar_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    if q_scratch is not None and q_continual is not None:
        ratio = q_continual / q_scratch if q_scratch > 0 else float("inf")
        ax.text(0.5, 0.85, f"Ratio: {ratio:.2f}×",
                transform=ax.transAxes, ha="center", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

    ax.set_ylabel("Q_transition")
    ax.set_title("C. Q_continual vs. Q_scratch")
    ax.grid(True, alpha=0.3, axis="y")


def plot_panel_d(ax, experiments):
    """Panel D: Forgetting dynamics — old_data_candidate_loss over time."""
    for exp_data in experiments:
        frac = get_fraction(exp_data)
        if frac is None:
            continue

        history = exp_data.get("history", {})
        steps = history.get("steps", [])
        old_cand = history.get("old_candidate_loss", [])

        if not steps or not old_cand:
            continue

        color = FRAC_COLORS.get(frac, "#333333")
        ax.plot(steps, old_cand, color=color, linewidth=1.5,
                label=f"f={frac}", alpha=0.85)

    ax.set_xlabel("Steps since distribution shift")
    ax.set_ylabel("Candidate loss (old data)")
    ax.set_title("D. Forgetting Dynamics")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)


def make_summary_json(experiments, output_dir, out_path):
    """Save a JSON summary with Q, transition windows, and divergence stats."""
    summary = []

    for exp_data in experiments:
        frac = get_fraction(exp_data)
        entry = {
            "name": exp_data["name"],
            "fraction": frac,
            "exists": exp_data["exists"],
        }

        if exp_data["exists"]:
            cfg = exp_data.get("config", {})
            continual = cfg.get("continual", {})
            entry["initial_new_candidate_loss"] = continual.get(
                "initial_new_candidate_loss"
            )
            entry["initial_old_candidate_loss"] = continual.get(
                "initial_old_candidate_loss"
            )
            entry["divergence"] = continual.get("divergence", {})

            history = exp_data.get("history", {})
            if history.get("new_candidate_loss"):
                entry["final_new_candidate_loss"] = history["new_candidate_loss"][-1]
            if history.get("old_candidate_loss"):
                entry["final_old_candidate_loss"] = history["old_candidate_loss"][-1]

            q_data = compute_q_from_gradient_norms(exp_data)
            if q_data:
                entry["Q_total"] = q_data["Q_total"]
                qt = compute_q_transition(exp_data, q_data)
                if qt:
                    entry["Q_transition"] = qt["Q_transition"]
                    entry["transition_start"] = qt["transition_start"]
                    entry["transition_end"] = qt["transition_end"]

        summary.append(entry)

    with open(out_path, "w") as f:
        json.dump({"experiments": summary}, f, indent=2)
    print(f"Saved summary to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot continual learning experiment results"
    )
    parser.add_argument("--experiments", type=str, nargs="+", required=True)
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Base experiment output directory")
    parser.add_argument("--output", type=str, default="outputs/continual_analysis/",
                        help="Output directory for figures and summary")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading experiments...")
    experiments = []
    for name in args.experiments:
        exp = load_experiment(name, args.output_dir)
        if exp["exists"]:
            experiments.append(exp)
            frac = get_fraction(exp)
            print(f"  {name}: f={frac}")
        else:
            print(f"  {name}: NOT FOUND (skipping)")

    if not experiments:
        print("No experiments found. Exiting.")
        return

    # Sort by fraction
    experiments.sort(key=lambda e: get_fraction(e) or 0.0)

    # Save summary JSON
    make_summary_json(experiments, args.output_dir, out_dir / "continual_summary.json")

    if not HAS_MPL:
        print("Skipping plots (matplotlib not available)")
        return

    # Create 2×2 figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        "Continual Learning: Thermodynamic Cost of Representational Replacement",
        fontsize=14, fontweight="bold", y=0.98,
    )

    plot_panel_a(axes[0, 0], experiments)
    plot_panel_b(axes[0, 1], experiments)
    plot_panel_c(axes[1, 0], experiments, args.output_dir)
    plot_panel_d(axes[1, 1], experiments)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    fig_path = out_dir / "continual_learning_results.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure to: {fig_path}")

    # --- Additional: Q vs f standalone figure ---
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    plot_panel_b(ax2, experiments)
    fig2.tight_layout()
    fig2_path = out_dir / "q_vs_f.png"
    fig2.savefig(fig2_path, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved Q vs f plot to: {fig2_path}")


if __name__ == "__main__":
    main()
