#!/usr/bin/env python3
"""
Tautology Control: Is ‖∇L‖² just a function of L?

Tests whether the gradient norm is a deterministic function of the current
loss value — ‖∇L‖² ≈ g(L) — which would mean gradient-based scaling laws
contain no independent information beyond the loss trajectory.

Uses existing data from the 10 K-value runs (no new training required).

Output: 5-panel figure + summary statistics table.
"""

import json
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
SAVE_DIR = OUTPUTS / "paper_figures"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

K_VALUES = [3, 5, 7, 10, 13, 17, 20, 25, 30, 36]

COLORS_BY_K = {
    3: "#E74C3C",
    5: "#E67E22",
    7: "#F1C40F",
    10: "#2ECC71",
    13: "#1ABC9C",
    17: "#3498DB",
    20: "#2980B9",
    25: "#9B59B6",
    30: "#8E44AD",
    36: "#34495E",
}

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 7,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────

def load_all_experiments():
    """Load (L, ‖∇L‖², step) for each K."""
    data = {}
    for k in K_VALUES:
        path = OUTPUTS / f"landauer_dense_k{k}" / "training_history.json"
        if not path.exists():
            print(f"  [SKIP] K={k}: {path} not found")
            continue
        with open(path) as f:
            h = json.load(f)
        data[k] = {
            "steps": np.array(h["steps"], dtype=float),
            "L": np.array(h["candidate_loss"], dtype=float),
            "g2": np.array(h["grad_norm_sq"], dtype=float),
            "log_k": math.log(k),
        }
    return data


# ──────────────────────────────────────────────────────────────────────
# Phase detection
# ──────────────────────────────────────────────────────────────────────

def detect_phases(steps, L, log_k):
    """
    Split trajectory into plateau / transition / post-convergence.
    Returns dict with boolean masks for each phase.
    """
    hi = 0.90 * log_k
    lo = 0.10 * log_k

    # Find transition end: first step where L < lo
    t_end_idx = None
    for i, l in enumerate(L):
        if l < lo:
            t_end_idx = i
            break

    # Find transition start: last step before t_end where L > hi
    t_start_idx = None
    if t_end_idx is not None:
        for i in range(t_end_idx):
            if L[i] > hi:
                t_start_idx = i

    # Default fallbacks
    if t_start_idx is None:
        t_start_idx = len(L) // 2
    if t_end_idx is None:
        t_end_idx = len(L) - 1

    t_start = steps[t_start_idx]
    t_end = steps[t_end_idx]

    # Plateau onset: first step where L drops below 1.1 * log_k (past initial transient)
    plateau_onset = 500.0
    for i, l in enumerate(L):
        if l < 1.1 * log_k and steps[i] >= 200:
            plateau_onset = steps[i]
            break

    # Plateau: from onset to t_start (must have at least some extent)
    plateau_start = min(plateau_onset, t_start)
    plateau_mask = (steps >= plateau_start) & (steps <= t_start)

    # Transition: t_start to t_end
    transition_mask = (steps >= t_start) & (steps <= t_end)

    # Post-convergence: t_end to t_end + 5000
    post_mask = (steps >= t_end) & (steps <= t_end + 5000)

    return {
        "plateau": plateau_mask,
        "transition": transition_mask,
        "post": post_mask,
        "t_start": t_start,
        "t_end": t_end,
        "t_start_idx": t_start_idx,
        "t_end_idx": t_end_idx,
        "plateau_onset": plateau_start,
    }


# ──────────────────────────────────────────────────────────────────────
# Panel A: Collapse plot — ‖∇L‖² vs L for all K
# ──────────────────────────────────────────────────────────────────────

def panel_a(ax, data):
    """‖∇L‖² vs L, all K values, color by K."""
    max_log_k = max(d["log_k"] for d in data.values())
    x_max = max_log_k + 0.5  # focus on the range [0, max(log K) + buffer]

    for k in K_VALUES:
        if k not in data:
            continue
        d = data[k]
        valid = (d["L"] > 1e-6) & (d["L"] <= x_max)
        ax.scatter(d["L"][valid], d["g2"][valid],
                   c=COLORS_BY_K[k], s=6, alpha=0.5, label=f"K={k}",
                   edgecolors="none", rasterized=True)

    ax.set_xlim(-0.1, x_max)
    ax.set_yscale("log")
    ax.set_xlabel("L  (candidate loss)")
    ax.set_ylabel("‖∇L‖²")
    ax.set_title("A: ‖∇L‖² vs L  (all K overlaid)")
    ax.legend(ncol=2, loc="upper right", markerscale=3, fontsize=6)


# ──────────────────────────────────────────────────────────────────────
# Panel B: Mean ‖∇L‖² vs K at fixed loss bins
# ──────────────────────────────────────────────────────────────────────

def panel_b(ax, data):
    """For each loss bin, plot mean ‖∇L‖² vs K."""
    bin_edges = np.arange(0.25, 4.01, 0.25)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    cmap = plt.cm.viridis
    n_bins = len(bin_centers)

    plotted_any = False
    for bi, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        center = bin_centers[bi]
        ks_in_bin = []
        means_in_bin = []
        for k in K_VALUES:
            if k not in data:
                continue
            d = data[k]
            mask = (d["L"] >= lo) & (d["L"] < hi)
            if mask.sum() >= 3:
                ks_in_bin.append(k)
                means_in_bin.append(np.mean(d["g2"][mask]))
        if len(ks_in_bin) >= 3:
            color = cmap(bi / n_bins)
            ax.plot(ks_in_bin, means_in_bin, "o-", color=color,
                    markersize=4, linewidth=1.2, alpha=0.8,
                    label=f"L∈[{lo:.2f},{hi:.2f})")
            plotted_any = True

    ax.set_yscale("log")
    ax.set_xlabel("K")
    ax.set_ylabel("Mean ‖∇L‖²")
    ax.set_title("B: Mean ‖∇L‖² vs K  (conditioned on L)")
    if plotted_any:
        ax.legend(ncol=2, loc="best", fontsize=5)
    else:
        ax.text(0.5, 0.5, "Insufficient overlapping data",
                transform=ax.transAxes, ha="center")


# ──────────────────────────────────────────────────────────────────────
# Panel C: Phase-specific ‖∇L‖² vs L
# ──────────────────────────────────────────────────────────────────────

PHASE_COLORS = {
    "plateau": "#3498DB",
    "transition": "#E74C3C",
    "post": "#2ECC71",
}
PHASE_MARKERS = {
    "plateau": "o",
    "transition": "s",
    "post": "^",
}


def panel_c(ax, data, phases):
    """‖∇L‖² vs L, color by phase, marker size encodes K."""
    for phase_name in ["plateau", "transition", "post"]:
        for k in K_VALUES:
            if k not in data or k not in phases:
                continue
            d = data[k]
            mask = phases[k][phase_name]
            if mask.sum() == 0:
                continue
            valid = mask & (d["L"] > 1e-6) & (d["g2"] > 0)
            size = 3 + (k / 36) * 15
            label = f"{phase_name} K={k}" if k in [3, 17, 36] else None
            ax.scatter(d["L"][valid], d["g2"][valid],
                       c=PHASE_COLORS[phase_name],
                       marker=PHASE_MARKERS[phase_name],
                       s=size, alpha=0.35, label=label,
                       edgecolors="none", rasterized=True)

    ax.set_yscale("log")
    ax.set_xlabel("L  (candidate loss)")
    ax.set_ylabel("‖∇L‖²")
    ax.set_title("C: ‖∇L‖² vs L  (by phase)")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles[:9], labels[:9], ncol=3, loc="upper right",
                  markerscale=2, fontsize=5)


# ──────────────────────────────────────────────────────────────────────
# Panel D: Q_actual vs Q_tautological
# ──────────────────────────────────────────────────────────────────────

def compute_tautological_g(data):
    """
    Fit g(L) from pooled data across all K.
    Uses nonparametric binned means + interpolation.
    """
    all_L = np.concatenate([data[k]["L"] for k in K_VALUES if k in data])
    all_g2 = np.concatenate([data[k]["g2"] for k in K_VALUES if k in data])

    valid = (all_L > 1e-6) & (all_g2 > 0)
    all_L = all_L[valid]
    all_g2 = all_g2[valid]

    # Bin into fine bins for smooth interpolation
    bin_edges = np.linspace(0, max(all_L) + 0.1, 200)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_means = np.full(len(bin_centers), np.nan)

    for i in range(len(bin_centers)):
        mask = (all_L >= bin_edges[i]) & (all_L < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_means[i] = np.mean(all_g2[mask])

    # Fill NaN with interpolation
    valid_bins = ~np.isnan(bin_means)
    if valid_bins.sum() < 2:
        return lambda L: np.ones_like(L) * np.nanmean(all_g2)

    def g_taut(L):
        return np.interp(L, bin_centers[valid_bins], bin_means[valid_bins],
                          left=bin_means[valid_bins][0],
                          right=bin_means[valid_bins][-1])

    return g_taut


def panel_d(ax, data, phases, g_taut):
    """Q_actual vs Q_tautological for all K."""
    lr = 0.001  # constant LR across all experiments
    step_spacing = 50.0  # eval_every

    q_actual = []
    q_taut = []
    k_vals = []

    for k in K_VALUES:
        if k not in data or k not in phases:
            continue
        d = data[k]
        ph = phases[k]
        mask = ph["transition"]
        if mask.sum() < 2:
            continue

        L_trans = d["L"][mask]
        g2_trans = d["g2"][mask]

        # Q_actual = sum(eta * g2 * dt) over transition
        q_a = lr * np.sum(g2_trans) * step_spacing

        # Q_tautological = sum(eta * g_taut(L) * dt) over transition
        g2_predicted = g_taut(L_trans)
        q_t = lr * np.sum(g2_predicted) * step_spacing

        q_actual.append(q_a)
        q_taut.append(q_t)
        k_vals.append(k)

    q_actual = np.array(q_actual)
    q_taut = np.array(q_taut)

    if len(q_actual) < 2:
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center")
        return {}, q_actual, q_taut, k_vals

    # Plot
    for i, kv in enumerate(k_vals):
        ax.scatter(q_taut[i], q_actual[i], c=COLORS_BY_K.get(kv, "#333"),
                   s=80, zorder=5, edgecolors="white", linewidth=0.5)
        ax.annotate(f"K={kv}", (q_taut[i], q_actual[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)

    # Diagonal reference
    all_q = np.concatenate([q_actual, q_taut])
    lo, hi = all_q.min() * 0.8, all_q.max() * 1.2
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, linewidth=1, label="y = x (tautology)")

    # R² and MAE
    ss_res = np.sum((q_actual - q_taut) ** 2)
    ss_tot = np.sum((q_actual - np.mean(q_actual)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    mae = np.mean(np.abs(q_actual - q_taut))
    mape = np.mean(np.abs(q_actual - q_taut) / np.maximum(q_actual, 1e-6)) * 100

    ax.set_xlabel("Q_tautological")
    ax.set_ylabel("Q_actual")
    ax.set_title(f"D: Q_actual vs Q_taut  (R²={r2:.3f}, MAPE={mape:.1f}%)")
    ax.legend(loc="upper left")

    return {"R2": r2, "MAE": mae, "MAPE": mape}, q_actual, q_taut, k_vals


# ──────────────────────────────────────────────────────────────────────
# Panel E: ‖∇L‖² / L² vs step
# ──────────────────────────────────────────────────────────────────────

def panel_e(ax, data):
    """‖∇L‖² / L² vs training step for all K."""
    for k in K_VALUES:
        if k not in data:
            continue
        d = data[k]
        valid = d["L"] > 0.05  # avoid division by tiny L near convergence
        ratio = d["g2"][valid] / (d["L"][valid] ** 2)
        ax.plot(d["steps"][valid], ratio, color=COLORS_BY_K[k],
                linewidth=0.9, alpha=0.7, label=f"K={k}")

    ax.set_yscale("log")
    ax.set_xlabel("Training step")
    ax.set_ylabel("‖∇L‖² / L²")
    ax.set_title("E: Normalized gradient vs step")
    ax.legend(ncol=2, loc="upper right", fontsize=6)


# ──────────────────────────────────────────────────────────────────────
# Summary statistics
# ──────────────────────────────────────────────────────────────────────

def compute_correlation_at_loss(data, target_L, window=0.25):
    """
    At a given loss value, compute Pearson r between K and ‖∇L‖².
    """
    ks = []
    mean_g2s = []
    for k in K_VALUES:
        if k not in data:
            continue
        d = data[k]
        mask = (d["L"] >= target_L - window / 2) & (d["L"] < target_L + window / 2)
        if mask.sum() >= 3:
            ks.append(k)
            mean_g2s.append(np.mean(d["g2"][mask]))

    if len(ks) < 3:
        return None, None, len(ks)

    r, p = pearsonr(ks, mean_g2s)
    return r, p, len(ks)


def compute_ratio_cv(data, phases, phase_name):
    """CV of ‖∇L‖²/L² across K values in a given phase."""
    ratios = []
    for k in K_VALUES:
        if k not in data or k not in phases:
            continue
        d = data[k]
        mask = phases[k][phase_name] & (d["L"] > 0.05)
        if mask.sum() < 3:
            continue
        r = np.mean(d["g2"][mask] / (d["L"][mask] ** 2))
        ratios.append(r)
    if len(ratios) < 2:
        return None
    return np.std(ratios) / np.mean(ratios)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    data = load_all_experiments()
    print(f"Loaded {len(data)} / {len(K_VALUES)} experiments")

    if len(data) < 5:
        print("ERROR: Too few experiments found.")
        return

    # Detect phases for each K
    print("Detecting phases...")
    phases = {}
    for k, d in data.items():
        phases[k] = detect_phases(d["steps"], d["L"], d["log_k"])
        ph = phases[k]
        print(f"  K={k:>2}: plateau=[500, {ph['t_start']:.0f}]  "
              f"transition=[{ph['t_start']:.0f}, {ph['t_end']:.0f}]  "
              f"(Δτ = {ph['t_end'] - ph['t_start']:.0f} steps)")

    # Fit tautological g(L)
    print("Fitting g(L) from pooled data...")
    g_taut = compute_tautological_g(data)

    # Build figure
    print("Generating 5-panel figure...")
    fig = plt.figure(figsize=(18, 11))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35,
                           left=0.05, right=0.97, top=0.94, bottom=0.06)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1:])

    panel_a(ax_a, data)
    panel_b(ax_b, data)
    panel_c(ax_c, data, phases)
    d_stats, q_actual, q_taut, k_vals = panel_d(ax_d, data, phases, g_taut)
    panel_e(ax_e, data)

    fig.suptitle("Tautology Control: Is ‖∇L‖² just a function of L?",
                 fontsize=14, fontweight="bold")

    save_path = SAVE_DIR / "fig_tautology_control.png"
    fig.savefig(save_path, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
    print(f"Saved: {save_path.with_suffix('.pdf')}")

    # ── Summary statistics ──
    print()
    print("=" * 60)
    print("TAUTOLOGY CONTROL RESULTS")
    print("=" * 60)

    # Panel A: correlation at fixed L
    r_15, p_15, n_15 = compute_correlation_at_loss(data, 1.5, window=0.3)
    r_25, p_25, n_25 = compute_correlation_at_loss(data, 2.5, window=0.3)

    print()
    print("Panel A: Do curves collapse?")
    if r_15 is not None:
        print(f"  Pearson r(K, ‖∇L‖²) at L≈1.5: r={r_15:+.4f}  (p={p_15:.4f}, n={n_15})")
    else:
        print(f"  Pearson r at L≈1.5: insufficient data (n={n_15})")
    if r_25 is not None:
        print(f"  Pearson r(K, ‖∇L‖²) at L≈2.5: r={r_25:+.4f}  (p={p_25:.4f}, n={n_25})")
    else:
        print(f"  Pearson r at L≈2.5: insufficient data (n={n_25})")
    print("  (r ≈ 0 → collapse → tautology holds)")
    print("  (|r| >> 0 → no collapse → tautology fails)")

    # Panel D: Q comparison
    print()
    print("Panel D: Q_actual vs Q_tautological")
    if d_stats:
        print(f"  R²: {d_stats['R2']:.4f}")
        print(f"  Mean absolute error: {d_stats['MAE']:.6f}")
        print(f"  Mean absolute % error: {d_stats['MAPE']:.1f}%")
        print("  (R² ≈ 1.0 AND MAPE < 10% → tautology holds)")
        print("  (R² << 1.0 OR MAPE >> 10% → tautology fails)")
        print("  NOTE: R² can be high due to shared K-monotonic trend.")
        print("        MAPE is more diagnostic for tautology testing.")
    else:
        print("  Insufficient data")

    # Panel E: ratio constancy
    cv_plateau = compute_ratio_cv(data, phases, "plateau")
    cv_transition = compute_ratio_cv(data, phases, "transition")

    print()
    print("Panel E: ‖∇L‖²/L² constancy")
    if cv_plateau is not None:
        print(f"  CV across K at plateau:    {cv_plateau:.4f}")
    else:
        print(f"  CV at plateau: insufficient data")
    if cv_transition is not None:
        print(f"  CV across K at transition: {cv_transition:.4f}")
    else:
        print(f"  CV at transition: insufficient data")
    print("  (CV ≈ 0 → tautology holds)")
    print("  (CV >> 0 → tautology fails)")

    # ── Verdict ──
    print()
    tautology_score = 0
    n_tests = 0

    if r_15 is not None:
        n_tests += 1
        if abs(r_15) < 0.3:
            tautology_score += 1
    if r_25 is not None:
        n_tests += 1
        if abs(r_25) < 0.3:
            tautology_score += 1
    if d_stats:
        n_tests += 1
        if d_stats["R2"] > 0.9 and d_stats["MAPE"] < 15:
            tautology_score += 1
    if cv_plateau is not None:
        n_tests += 1
        if cv_plateau < 0.3:
            tautology_score += 1
    if cv_transition is not None:
        n_tests += 1
        if cv_transition < 0.3:
            tautology_score += 1

    if n_tests == 0:
        print("VERDICT: Insufficient data for a verdict")
    elif tautology_score >= n_tests * 0.7:
        print(f"VERDICT: Tautology HOLDS  ({tautology_score}/{n_tests} tests pass)")
        print("  → ‖∇L‖² is mostly explained by L alone.")
        print("  → Gradient suppression (Law 3) is a derived quantity, not an independent finding.")
        print("  → Consider demoting Law 3 or reframing it as a consistency check.")
    elif tautology_score <= n_tests * 0.3:
        print(f"VERDICT: Tautology FAILS  ({tautology_score}/{n_tests} tests pass)")
        print("  → ‖∇L‖² depends on K beyond its effect on L.")
        print("  → The gradient landscape is geometrically different at higher K.")
        print("  → Law 3 (gradient suppression) stands as an independent finding.")
    else:
        print(f"VERDICT: MIXED  ({tautology_score}/{n_tests} tests pass)")
        print("  → Partial tautology: some but not all gradient info is predicted by L alone.")
        print("  → Further investigation needed.")

    print()
    print("=" * 60)

    # Save numerical results
    results = {
        "panel_a": {
            "r_at_L1.5": {"r": float(r_15) if r_15 is not None else None,
                           "p": float(p_15) if p_15 is not None else None,
                           "n": n_15},
            "r_at_L2.5": {"r": float(r_25) if r_25 is not None else None,
                           "p": float(p_25) if p_25 is not None else None,
                           "n": n_25},
        },
        "panel_d": {
            "R2": float(d_stats["R2"]) if d_stats else None,
            "MAE": float(d_stats["MAE"]) if d_stats else None,
            "MAPE": float(d_stats["MAPE"]) if d_stats else None,
            "q_actual": q_actual.tolist() if len(q_actual) else [],
            "q_tautological": q_taut.tolist() if len(q_taut) else [],
            "k_values": k_vals,
        },
        "panel_e": {
            "cv_plateau": float(cv_plateau) if cv_plateau is not None else None,
            "cv_transition": float(cv_transition) if cv_transition is not None else None,
        },
        "verdict": {
            "tautology_score": tautology_score,
            "n_tests": n_tests,
        },
    }

    results_path = OUTPUTS / "tautology_control_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to: {results_path}")


if __name__ == "__main__":
    main()
