#!/usr/bin/env python
"""
Plotting script for the f-sweep: measuring representational locality.

Figure 1: Forgetting Phase Diagram
  A: old_candidate_loss at convergence vs. f
  B: new_candidate_loss convergence time vs. f
  C: old_candidate_loss trajectory for all f values

Figure 2: Cost Curves
  A: Q_transition vs. f
  B: Q_total vs. f
  C: Q_total - Q_transition vs. f (hidden cost)

Also prints a summary table.

Usage:
    python scripts/plot_f_sweep.py --output-dir outputs
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
    from matplotlib.ticker import MaxNLocator
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not available.")


FRAC_COLORS = {
    0.0:  "#888888",
    0.01: "#03A9F4",
    0.05: "#00BCD4",
    0.1:  "#2196F3",
    0.25: "#4CAF50",
    0.5:  "#FF9800",
    0.75: "#F44336",
    1.0:  "#9C27B0",
}

ALL_FRACS = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]


def load_experiment(exp_name, output_dir):
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

    return data


def get_fraction(exp_data):
    cfg = exp_data.get("config", {})
    frac = cfg.get("continual", {}).get("fraction")
    if frac is not None:
        return float(frac)
    name = exp_data["name"]
    if "_f" in name:
        try:
            return float(name.split("_f")[-1].split("_")[0])
        except ValueError:
            pass
    return None


def find_adaptation_step(steps, new_cand_loss, threshold=0.1):
    """First step where new_candidate_loss drops below threshold."""
    for i, cl in enumerate(new_cand_loss):
        if cl < threshold:
            return steps[i]
    return None


def compute_q_from_history(history, lr=1e-3):
    """
    Compute cumulative dissipation from training-time gradient norms.
    Q(t) = sum eta * ||grad||^2 * delta_s
    gradient_norm in history is already avg(||grad||^2) per eval interval.
    """
    steps = history.get("steps", [])
    grad_norm_sq = history.get("gradient_norm", [])
    if not steps or not grad_norm_sq:
        return None

    steps = np.array(steps)
    norms = np.array(grad_norm_sq)

    delta_s = np.diff(steps, prepend=0)
    dissipation = lr * norms * delta_s
    Q_cumulative = np.cumsum(dissipation)

    return {
        "steps": steps,
        "Q_cumulative": Q_cumulative,
        "Q_total": float(Q_cumulative[-1]),
    }


def find_transition_window(steps, new_cand_loss):
    """Transition: 90% -> 10% of initial new_candidate_loss."""
    if not steps or not new_cand_loss:
        return None, None

    initial = new_cand_loss[0]
    if initial < 0.1:
        return None, None

    hi = 0.9 * initial
    lo = max(0.1 * initial, 0.05)

    t_start = None
    t_end = None
    for i, cl in enumerate(new_cand_loss):
        if cl > hi:
            t_start = steps[i]
    for i, cl in enumerate(new_cand_loss):
        if cl < lo:
            t_end = steps[i]
            break
    return t_start, t_end


def compute_q_transition(q_data, t_start, t_end):
    if q_data is None or t_start is None or t_end is None:
        return None
    Q_at_start = float(np.interp(t_start, q_data["steps"], q_data["Q_cumulative"]))
    Q_at_end = float(np.interp(t_end, q_data["steps"], q_data["Q_cumulative"]))
    return Q_at_end - Q_at_start


def analyze_experiments(experiments):
    """Build summary rows for all experiments."""
    rows = []
    for exp in experiments:
        frac = get_fraction(exp)
        if frac is None:
            continue

        history = exp.get("history", {})
        steps = history.get("steps", [])
        new_cand = history.get("new_candidate_loss", [])
        old_cand = history.get("old_candidate_loss", [])
        old_unchanged = history.get("old_unchanged_candidate_loss", [])
        old_changed = history.get("old_changed_candidate_loss", [])

        adapt_step = find_adaptation_step(steps, new_cand, threshold=0.1) if new_cand else None

        old_at_adapt = None
        if adapt_step is not None and old_cand:
            idx = next((i for i, s in enumerate(steps) if s >= adapt_step), None)
            if idx is not None:
                old_at_adapt = old_cand[idx]

        old_final = old_cand[-1] if old_cand else None
        new_final = new_cand[-1] if new_cand else None

        cfg = exp.get("config", {})
        lr = float(cfg.get("training", {}).get("learning_rate", 1e-3))
        q_data = compute_q_from_history(history, lr=lr)
        t_start, t_end = find_transition_window(steps, new_cand)
        q_trans = compute_q_transition(q_data, t_start, t_end)
        q_total = q_data["Q_total"] if q_data is not None else None

        rows.append({
            "fraction": frac,
            "adapt_step": adapt_step,
            "old_at_adapt": old_at_adapt,
            "old_final": old_final,
            "new_final": new_final,
            "q_transition": q_trans,
            "q_total": q_total,
            "steps": steps,
            "new_cand": new_cand,
            "old_cand": old_cand,
            "old_unchanged": old_unchanged,
            "old_changed": old_changed,
            "q_data": q_data,
        })

    rows.sort(key=lambda r: r["fraction"])
    return rows


def plot_figure1(rows, out_dir):
    """Forgetting Phase Diagram: 3 panels."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Figure 1: Forgetting Phase Diagram", fontsize=14, fontweight="bold", y=1.02)

    # Panel A: old_candidate_loss at convergence vs. f
    ax = axes[0]
    fracs_a, old_at_adapt_a = [], []
    for r in rows:
        if r["fraction"] == 0.0:
            continue
        if r["old_at_adapt"] is not None:
            fracs_a.append(r["fraction"])
            old_at_adapt_a.append(r["old_at_adapt"])

    if fracs_a:
        colors = [FRAC_COLORS.get(f, "#333") for f in fracs_a]
        ax.scatter(fracs_a, old_at_adapt_a, c=colors, s=100, zorder=5, edgecolors="black", linewidth=0.5)
        ax.plot(fracs_a, old_at_adapt_a, "-", color="#555", alpha=0.5, linewidth=1)
        k_val = 20
        ax.axhline(y=math.log(k_val), color="red", linestyle="--", alpha=0.4, label=f"log(K)={math.log(k_val):.2f}")
        ax.legend(fontsize=9)

    ax.set_xlabel("Reassignment fraction f", fontsize=11)
    ax.set_ylabel("old_candidate_loss at adaptation", fontsize=11)
    ax.set_title("A. Forgetting at Convergence", fontsize=12)
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)

    # Panel B: convergence time vs. f
    ax = axes[1]
    fracs_b, steps_b = [], []
    for r in rows:
        if r["fraction"] == 0.0:
            continue
        if r["adapt_step"] is not None:
            fracs_b.append(r["fraction"])
            steps_b.append(r["adapt_step"])

    if fracs_b:
        colors = [FRAC_COLORS.get(f, "#333") for f in fracs_b]
        ax.scatter(fracs_b, steps_b, c=colors, s=100, zorder=5, edgecolors="black", linewidth=0.5)
        ax.plot(fracs_b, steps_b, "-", color="#555", alpha=0.5, linewidth=1)

    ax.set_xlabel("Reassignment fraction f", fontsize=11)
    ax.set_ylabel("Steps to new_cand_loss < 0.1", fontsize=11)
    ax.set_title("B. Adaptation Speed", fontsize=12)
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Panel C: old_candidate_loss trajectories
    ax = axes[2]
    for r in rows:
        f = r["fraction"]
        if not r["steps"] or not r["old_cand"]:
            continue
        color = FRAC_COLORS.get(f, "#333")
        ax.plot(r["steps"], r["old_cand"], color=color, linewidth=1.5,
                label=f"f={f}", alpha=0.85)

    ax.set_xlabel("Training steps", fontsize=11)
    ax.set_ylabel("old_candidate_loss", fontsize=11)
    ax.set_title("C. Forgetting Dynamics", fontsize=12)
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "figure1_forgetting_phase_diagram.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 1 to: {path}")


def plot_figure2(rows, out_dir):
    """Cost Curves: 3 panels."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Figure 2: Cost Curves", fontsize=14, fontweight="bold", y=1.02)

    active = [r for r in rows if r["fraction"] > 0.0]

    # Panel A: Q_transition vs. f
    ax = axes[0]
    fracs_qt, qt_vals = [], []
    for r in active:
        if r["q_transition"] is not None:
            fracs_qt.append(r["fraction"])
            qt_vals.append(r["q_transition"])

    if fracs_qt:
        colors = [FRAC_COLORS.get(f, "#333") for f in fracs_qt]
        ax.scatter(fracs_qt, qt_vals, c=colors, s=100, zorder=5, edgecolors="black", linewidth=0.5)
        ax.plot(fracs_qt, qt_vals, "-", color="#1565C0", alpha=0.6, linewidth=1.5)

    ax.set_xlabel("Reassignment fraction f", fontsize=11)
    ax.set_ylabel("Q_transition", fontsize=11)
    ax.set_title("A. Transition Dissipation", fontsize=12)
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)

    # Panel B: Q_total vs. f
    ax = axes[1]
    fracs_qtot, qtot_vals = [], []
    for r in active:
        if r["q_total"] is not None:
            fracs_qtot.append(r["fraction"])
            qtot_vals.append(r["q_total"])

    if fracs_qtot:
        colors = [FRAC_COLORS.get(f, "#333") for f in fracs_qtot]
        ax.scatter(fracs_qtot, qtot_vals, c=colors, s=100, zorder=5, edgecolors="black", linewidth=0.5)
        ax.plot(fracs_qtot, qtot_vals, "-", color="#E65100", alpha=0.6, linewidth=1.5)

    ax.set_xlabel("Reassignment fraction f", fontsize=11)
    ax.set_ylabel("Q_total", fontsize=11)
    ax.set_title("B. Total Dissipation", fontsize=12)
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)

    # Panel C: Q_total - Q_transition (hidden cost)
    ax = axes[2]
    fracs_hid, hid_vals = [], []
    for r in active:
        if r["q_transition"] is not None and r["q_total"] is not None:
            fracs_hid.append(r["fraction"])
            hid_vals.append(r["q_total"] - r["q_transition"])

    if fracs_hid:
        colors = [FRAC_COLORS.get(f, "#333") for f in fracs_hid]
        ax.scatter(fracs_hid, hid_vals, c=colors, s=100, zorder=5, edgecolors="black", linewidth=0.5)
        ax.plot(fracs_hid, hid_vals, "-", color="#C62828", alpha=0.6, linewidth=1.5)

    ax.set_xlabel("Reassignment fraction f", fontsize=11)
    ax.set_ylabel("Q_total - Q_transition", fontsize=11)
    ax.set_title("C. Hidden Cost (Post-transition)", fontsize=12)
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "figure2_cost_curves.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 2 to: {path}")


def print_summary_table(rows):
    print()
    print("=" * 110)
    print("f-Sweep Summary: Representational Locality")
    print("=" * 110)
    header = (
        f"{'f':>6}  {'Steps to':>10}  {'old_loss':>10}  {'old_loss':>10}  "
        f"{'new_loss':>10}  {'Q_trans':>12}  {'Q_total':>12}  {'Q_hidden':>12}"
    )
    sub = (
        f"{'':>6}  {'adapt':>10}  {'at adapt':>10}  {'final':>10}  "
        f"{'final':>10}  {'':>12}  {'':>12}  {'':>12}"
    )
    print(header)
    print(sub)
    print("-" * 110)

    for r in rows:
        f_str = f"{r['fraction']:.2f}"
        adapt = f"{r['adapt_step']}" if r["adapt_step"] is not None else "N/A"
        old_a = f"{r['old_at_adapt']:.4f}" if r["old_at_adapt"] is not None else "N/A"
        old_f = f"{r['old_final']:.4f}" if r["old_final"] is not None else "N/A"
        new_f = f"{r['new_final']:.4f}" if r["new_final"] is not None else "N/A"
        qt = f"{r['q_transition']:.4f}" if r["q_transition"] is not None else "N/A"
        qtot = f"{r['q_total']:.4f}" if r["q_total"] is not None else "N/A"
        if r["q_transition"] is not None and r["q_total"] is not None:
            qhid = f"{r['q_total'] - r['q_transition']:.4f}"
        else:
            qhid = "N/A"

        print(f"{f_str:>6}  {adapt:>10}  {old_a:>10}  {old_f:>10}  "
              f"{new_f:>10}  {qt:>12}  {qtot:>12}  {qhid:>12}")

    print("-" * 110)
    print()


def main():
    parser = argparse.ArgumentParser(description="Plot f-sweep results")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--out", type=str, default="outputs/continual_analysis/",
                        help="Directory for output figures")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading experiments...")
    experiments = []
    for frac in ALL_FRACS:
        name = f"continual_reassign_f{frac}"
        exp = load_experiment(name, args.output_dir)
        if exp["exists"]:
            experiments.append(exp)
            print(f"  {name}: loaded")
        else:
            print(f"  {name}: NOT FOUND")

    if not experiments:
        print("No experiments found.")
        return

    rows = analyze_experiments(experiments)
    print_summary_table(rows)

    # Save summary JSON
    summary_for_json = []
    for r in rows:
        entry = {k: v for k, v in r.items()
                 if k not in ("steps", "new_cand", "old_cand", "old_unchanged",
                              "old_changed", "q_data")}
        summary_for_json.append(entry)
    with open(out_dir / "f_sweep_summary.json", "w") as f:
        json.dump(summary_for_json, f, indent=2, default=str)
    print(f"Saved summary to: {out_dir / 'f_sweep_summary.json'}")

    if not HAS_MPL:
        print("Skipping plots (no matplotlib).")
        return

    plot_figure1(rows, out_dir)
    plot_figure2(rows, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
