#!/usr/bin/env python3
"""
Diagnostic: Is Q redundant with τ?

Experiment 1 — Q-vs-τ decomposition (5 panels)
Experiment 4 — Per-K gradient norm anatomy (2 panels)

Uses existing 10 K-value runs. No new training required.

Usage:
    python scripts/diagnostic_q_vs_tau.py
"""

import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
SAVE_DIR = OUTPUTS / "paper_figures"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

K_VALUES = [3, 5, 7, 10, 13, 17, 20, 25, 30, 36]

COLORS_BY_K = {
    3: "#E74C3C", 5: "#E67E22", 7: "#F1C40F", 10: "#2ECC71",
    13: "#1ABC9C", 17: "#3498DB", 20: "#2980B9", 25: "#9B59B6",
    30: "#8E44AD", 36: "#34495E",
}

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
    "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.3,
})


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────

def load_experiment(k: int) -> dict:
    """Load training history for a single K value."""
    exp_dir = OUTPUTS / f"landauer_dense_k{k}"
    with open(exp_dir / "training_history.json") as f:
        history = json.load(f)
    return {
        "k": k,
        "log_k": math.log(k),
        "steps": np.array(history["steps"]),
        "grad_norm_sq": np.array(history["grad_norm_sq"]),
        "candidate_loss": np.array(history["candidate_loss"]),
    }


def find_transition_window(steps, candidate_loss, log_k,
                           thresh_hi=0.95, thresh_lo=0.05):
    """t_start/t_end matching the analysis in analyze_landauer_dense.py."""
    hi = thresh_hi * log_k
    lo = thresh_lo * log_k

    t_end = t_end_idx = None
    for i, cl in enumerate(candidate_loss):
        if cl < lo:
            t_end = steps[i]
            t_end_idx = i
            break
    if t_end is None:
        return None, None

    t_start = None
    for i in range(t_end_idx):
        if candidate_loss[i] > hi:
            t_start = steps[i]

    return t_start, t_end


# ──────────────────────────────────────────────────────────────────────
# Experiment 1: Q-vs-τ decomposition
# ──────────────────────────────────────────────────────────────────────

def compute_decomposition(exp: dict) -> dict:
    """Compute τ, Q, ḡ² metrics for a single experiment."""
    k = exp["k"]
    log_k = exp["log_k"]
    steps = exp["steps"]
    gns = exp["grad_norm_sq"]
    cand = exp["candidate_loss"]
    lr = 0.001  # constant across all dense K experiments

    t_start, t_end = find_transition_window(steps, cand, log_k)
    if t_start is None or t_end is None:
        return None

    tau = t_end - t_start

    # Q_transition: integral of lr * grad_norm_sq over [t_start, t_end]
    mask_trans = (steps >= t_start) & (steps <= t_end)
    s_trans = steps[mask_trans]
    g_trans = gns[mask_trans]
    delta_trans = np.diff(s_trans, prepend=s_trans[0])
    delta_trans[0] = s_trans[0] - t_start if s_trans[0] > t_start else (
        s_trans[1] - s_trans[0] if len(s_trans) > 1 else 1)
    Q_transition = float(np.sum(lr * g_trans * delta_trans))

    # ḡ²_transition: mean gradient norm squared during transition
    g_bar_sq_transition = float(np.mean(g_trans))

    # ḡ²_plateau: mean gradient norm from step 500 to t_start
    plateau_start = 500
    mask_plateau = (steps >= plateau_start) & (steps < t_start)
    if np.any(mask_plateau):
        g_bar_sq_plateau = float(np.mean(gns[mask_plateau]))
    else:
        mask_plateau = steps < t_start
        g_bar_sq_plateau = float(np.mean(gns[mask_plateau])) if np.any(mask_plateau) else g_bar_sq_transition

    # ḡ²_post: mean gradient norm after convergence [t_end, t_end + 2000]
    mask_post = (steps >= t_end) & (steps <= t_end + 2000)
    if np.any(mask_post):
        g_bar_sq_post = float(np.mean(gns[mask_post]))
    else:
        mask_post = steps >= t_end
        g_bar_sq_post = float(np.mean(gns[mask_post])) if np.any(mask_post) else 0.0

    return {
        "k": k, "log_k": log_k,
        "t_start": t_start, "t_end": t_end, "tau": tau,
        "Q_transition": Q_transition,
        "g_bar_sq_transition": g_bar_sq_transition,
        "g_bar_sq_plateau": g_bar_sq_plateau,
        "g_bar_sq_post": g_bar_sq_post,
        "ratio_trans_plateau": g_bar_sq_transition / g_bar_sq_plateau if g_bar_sq_plateau > 0 else float("inf"),
    }


def fit_power_law(x, y):
    """Fit y = a * x^beta in log-log space. Returns (a, beta, R², predicted)."""
    def model(x, a, beta):
        return a * np.power(x, beta)
    try:
        popt, _ = curve_fit(model, x, y, p0=[1.0, 1.0], maxfev=10000)
        pred = model(x, *popt)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return popt[0], popt[1], r2, pred
    except Exception:
        return None, None, None, None


def fit_linear(x, y):
    """Fit y = a*x + b. Returns (a, b, R², predicted)."""
    coeffs = np.polyfit(x, y, 1)
    pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return coeffs[0], coeffs[1], r2, pred


def plot_experiment1(decomps: list):
    """5-panel figure for Experiment 1."""
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35)

    ks = np.array([d["k"] for d in decomps], dtype=float)
    taus = np.array([d["tau"] for d in decomps], dtype=float)
    Qs = np.array([d["Q_transition"] for d in decomps], dtype=float)
    g_trans = np.array([d["g_bar_sq_transition"] for d in decomps])
    g_plat = np.array([d["g_bar_sq_plateau"] for d in decomps])
    ratios = np.array([d["ratio_trans_plateau"] for d in decomps])

    colors = [COLORS_BY_K[d["k"]] for d in decomps]

    # ── Panel A: τ vs K (log-log) ──
    ax_a = fig.add_subplot(gs[0, 0])
    for i, d in enumerate(decomps):
        ax_a.scatter(d["k"], d["tau"], color=colors[i], s=70, zorder=5,
                     edgecolors="white", linewidth=0.5)
        ax_a.annotate(f"K={d['k']}", (d["k"], d["tau"]),
                      textcoords="offset points", xytext=(5, 5), fontsize=7)

    a_tau, beta_tau, r2_tau, pred_tau = fit_power_law(ks, taus)
    k_smooth = np.linspace(ks.min() * 0.8, ks.max() * 1.1, 200)
    ax_a.plot(k_smooth, a_tau * np.power(k_smooth, beta_tau), "r-", linewidth=1.5,
              label=f"τ = {a_tau:.2f}·K^{{{beta_tau:.2f}}}  R²={r2_tau:.4f}")
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlabel("K")
    ax_a.set_ylabel("τ (transition duration, steps)")
    ax_a.set_title("A: τ vs K (log-log)")
    ax_a.legend(loc="upper left")

    # ── Panel B: Q vs K (log-log) ──
    ax_b = fig.add_subplot(gs[0, 1])
    for i, d in enumerate(decomps):
        ax_b.scatter(d["k"], d["Q_transition"], color=colors[i], s=70, zorder=5,
                     edgecolors="white", linewidth=0.5)
        ax_b.annotate(f"K={d['k']}", (d["k"], d["Q_transition"]),
                      textcoords="offset points", xytext=(5, 5), fontsize=7)

    a_Q, alpha_Q, r2_Q, pred_Q = fit_power_law(ks, Qs)
    ax_b.plot(k_smooth, a_Q * np.power(k_smooth, alpha_Q), "r-", linewidth=1.5,
              label=f"Q = {a_Q:.3f}·K^{{{alpha_Q:.2f}}}  R²={r2_Q:.4f}")
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlabel("K")
    ax_b.set_ylabel("Q_transition")
    ax_b.set_title(f"B: Q vs K (log-log), α={alpha_Q:.2f}")
    ax_b.legend(loc="upper left")

    # ── Panel C: ḡ²_transition vs K (THE KEY PANEL) ──
    ax_c = fig.add_subplot(gs[0, 2])
    for i, d in enumerate(decomps):
        ax_c.scatter(d["k"], d["g_bar_sq_transition"], color=colors[i], s=70,
                     zorder=5, edgecolors="white", linewidth=0.5)
        ax_c.annotate(f"K={d['k']}", (d["k"], d["g_bar_sq_transition"]),
                      textcoords="offset points", xytext=(5, 5), fontsize=7)

    # Fit power law to check if flat or increasing
    a_g, beta_g, r2_g, _ = fit_power_law(ks, g_trans)
    ax_c.plot(k_smooth, a_g * np.power(k_smooth, beta_g), "r--", linewidth=1.2,
              label=f"ḡ² ∝ K^{{{beta_g:.2f}}}  R²={r2_g:.4f}")

    # Reference: flat line at mean
    mean_g = np.mean(g_trans)
    ax_c.axhline(mean_g, color="gray", linestyle=":", linewidth=1, alpha=0.7,
                 label=f"flat (mean={mean_g:.4f})")

    ax_c.set_xscale("log")
    ax_c.set_xlabel("K")
    ax_c.set_ylabel("ḡ²_transition (mean ||∇L||² during transition)")
    ax_c.set_title("C: Per-step dissipation rate vs K (KEY)")
    ax_c.legend(loc="best")

    # Annotate verdict
    if abs(beta_g) < 0.15:
        verdict_c = "ḡ² ≈ FLAT → Hyp B (discovery)"
        verdict_color = "#E74C3C"
    elif beta_g > 0.15:
        verdict_c = f"ḡ² INCREASES (β={beta_g:.2f}) → Hyp A (thermo)"
        verdict_color = "#2ECC71"
    else:
        verdict_c = f"ḡ² DECREASES (β={beta_g:.2f})"
        verdict_color = "#F39C12"
    ax_c.text(0.05, 0.95, verdict_c, transform=ax_c.transAxes,
              fontsize=9, fontweight="bold", color=verdict_color,
              verticalalignment="top",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # ── Panel D: ḡ²_transition / ḡ²_plateau vs K ──
    ax_d = fig.add_subplot(gs[1, 0])
    for i, d in enumerate(decomps):
        ax_d.scatter(d["k"], d["ratio_trans_plateau"], color=colors[i], s=70,
                     zorder=5, edgecolors="white", linewidth=0.5)
        ax_d.annotate(f"K={d['k']}", (d["k"], d["ratio_trans_plateau"]),
                      textcoords="offset points", xytext=(5, 5), fontsize=7)

    a_r, beta_r, r2_r, _ = fit_power_law(ks, ratios)
    ax_d.plot(k_smooth, a_r * np.power(k_smooth, beta_r), "r--", linewidth=1.2,
              label=f"ratio ∝ K^{{{beta_r:.2f}}}  R²={r2_r:.4f}")
    ax_d.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.7,
                 label="ratio = 1 (no elevation)")
    ax_d.set_xscale("log")
    ax_d.set_xlabel("K")
    ax_d.set_ylabel("ḡ²_transition / ḡ²_plateau")
    ax_d.set_title("D: Gradient elevation during transition")
    ax_d.legend(loc="best")

    if abs(beta_r) < 0.15 and np.mean(ratios) < 1.3:
        verdict_d = "Ratio ≈ 1, flat → Hyp B"
        vcolor_d = "#E74C3C"
    elif beta_r > 0.15:
        verdict_d = f"Ratio GROWS with K → Hyp A"
        vcolor_d = "#2ECC71"
    else:
        verdict_d = f"Ratio elevated but flat"
        vcolor_d = "#F39C12"
    ax_d.text(0.05, 0.95, verdict_d, transform=ax_d.transAxes,
              fontsize=9, fontweight="bold", color=vcolor_d,
              verticalalignment="top",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # ── Panel E: Q vs τ (residual analysis) ──
    ax_e = fig.add_subplot(gs[1, 1])
    for i, d in enumerate(decomps):
        ax_e.scatter(d["tau"], d["Q_transition"], color=colors[i], s=70,
                     zorder=5, edgecolors="white", linewidth=0.5)
        ax_e.annotate(f"K={d['k']}", (d["tau"], d["Q_transition"]),
                      textcoords="offset points", xytext=(5, 5), fontsize=7)

    # Linear fit Q = m*τ + c
    slope_qt, intercept_qt, r2_qt, pred_qt = fit_linear(taus, Qs)
    tau_smooth = np.linspace(taus.min() * 0.8, taus.max() * 1.1, 200)
    ax_e.plot(tau_smooth, slope_qt * tau_smooth + intercept_qt, "b-", linewidth=1.5,
              label=f"Q = {slope_qt:.5f}·τ + {intercept_qt:.2f}  R²={r2_qt:.4f}")

    # Power law fit Q = a * τ^gamma
    a_qt, gamma_qt, r2_qt_pow, _ = fit_power_law(taus, Qs)
    ax_e.plot(tau_smooth, a_qt * np.power(tau_smooth, gamma_qt), "r--", linewidth=1.2,
              label=f"Q = {a_qt:.4f}·τ^{{{gamma_qt:.2f}}}  R²={r2_qt_pow:.4f}")

    # Pearson correlation
    r_pearson, p_pearson = pearsonr(taus, Qs)

    ax_e.set_xlabel("τ (transition duration)")
    ax_e.set_ylabel("Q_transition")
    ax_e.set_title(f"E: Q vs τ directly  (r={r_pearson:.4f}, p={p_pearson:.2e})")
    ax_e.legend(loc="upper left")

    if r2_qt > 0.99:
        verdict_e = f"R²={r2_qt:.4f} → Q REDUNDANT with τ → Hyp B"
        vcolor_e = "#E74C3C"
    elif r2_qt > 0.95:
        verdict_e = f"R²={r2_qt:.4f} → Highly correlated but not perfect"
        vcolor_e = "#F39C12"
    else:
        verdict_e = f"R²={r2_qt:.4f} → Q NOT redundant with τ → Hyp A"
        vcolor_e = "#2ECC71"
    ax_e.text(0.05, 0.95, verdict_e, transform=ax_e.transAxes,
              fontsize=9, fontweight="bold", color=vcolor_e,
              verticalalignment="top",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # ── Panel F: Summary table ──
    ax_f = fig.add_subplot(gs[1, 2])
    ax_f.axis("off")

    summary_lines = [
        "EXPERIMENT 1 SUMMARY",
        "=" * 40,
        f"τ scaling:  τ ∝ K^{beta_tau:.2f}  (R²={r2_tau:.4f})",
        f"Q scaling:  Q ∝ K^{alpha_Q:.2f}  (R²={r2_Q:.4f})",
        f"",
        f"If Q = ḡ²·τ with ḡ² constant:",
        f"  expect α = β = {beta_tau:.2f}",
        f"  actual α = {alpha_Q:.2f}",
        f"  Δ(α - β) = {alpha_Q - beta_tau:.2f}",
        f"",
        f"ḡ²_trans scaling:  ḡ² ∝ K^{beta_g:.2f}  (R²={r2_g:.4f})",
        f"Elevation ratio ∝ K^{beta_r:.2f}  (R²={r2_r:.4f})",
        f"Q vs τ linear R² = {r2_qt:.4f}",
        f"Q vs τ power R² = {r2_qt_pow:.4f}  (γ={gamma_qt:.2f})",
        f"",
    ]

    # Final verdict
    signals_thermo = 0
    signals_discovery = 0

    if beta_g > 0.15:
        signals_thermo += 1
        summary_lines.append("✓ ḡ² increases with K → Thermo")
    else:
        signals_discovery += 1
        summary_lines.append("✗ ḡ² flat/decreasing → Discovery")

    if r2_qt > 0.99:
        signals_discovery += 1
        summary_lines.append("✗ Q perfectly linear in τ → Discovery")
    else:
        signals_thermo += 1
        summary_lines.append("✓ Q not perfectly linear in τ → Thermo")

    if beta_r > 0.15:
        signals_thermo += 1
        summary_lines.append("✓ Trans/plateau ratio grows → Thermo")
    else:
        signals_discovery += 1
        summary_lines.append("✗ Trans/plateau ratio flat → Discovery")

    summary_lines.append("")
    if signals_thermo > signals_discovery:
        summary_lines.append(f"VERDICT: Hypothesis A (Thermodynamic)")
        summary_lines.append(f"  Score: Thermo {signals_thermo}, Discovery {signals_discovery}")
    elif signals_discovery > signals_thermo:
        summary_lines.append(f"VERDICT: Hypothesis B (Discovery latency)")
        summary_lines.append(f"  Score: Thermo {signals_thermo}, Discovery {signals_discovery}")
    else:
        summary_lines.append(f"VERDICT: AMBIGUOUS")
        summary_lines.append(f"  Score: Thermo {signals_thermo}, Discovery {signals_discovery}")

    ax_f.text(0.05, 0.95, "\n".join(summary_lines), transform=ax_f.transAxes,
              fontsize=8, fontfamily="monospace", verticalalignment="top",
              bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f8f8", alpha=0.9))

    fig.suptitle("Experiment 1: Is Q Redundant with τ?", fontsize=14, fontweight="bold", y=1.02)
    return fig


# ──────────────────────────────────────────────────────────────────────
# Experiment 4: Per-K gradient norm anatomy
# ──────────────────────────────────────────────────────────────────────

def compute_normalized_profiles(exps: list, decomps: list) -> dict:
    """
    Compute gradient norm in normalized time for each K.
    t_norm = (t - t_start) / tau, so 0 = transition start, 1 = transition end.
    """
    profiles = {}
    n_bins = 40
    bin_edges = np.linspace(-1.0, 2.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    for exp, dec in zip(exps, decomps):
        if dec is None:
            continue
        k = exp["k"]
        t_start = dec["t_start"]
        tau = dec["tau"]
        steps = exp["steps"]
        gns = exp["grad_norm_sq"]

        t_norm = (steps - t_start) / tau
        mask = (t_norm >= -1.0) & (t_norm <= 2.0)
        t_norm_valid = t_norm[mask]
        gns_valid = gns[mask]

        binned_means = np.full(n_bins, np.nan)
        for b in range(n_bins):
            in_bin = (t_norm_valid >= bin_edges[b]) & (t_norm_valid < bin_edges[b + 1])
            if np.any(in_bin):
                binned_means[b] = np.mean(gns_valid[in_bin])

        profiles[k] = {
            "bin_centers": bin_centers,
            "gns_binned": binned_means,
            "g_plateau": dec["g_bar_sq_plateau"],
        }

    return profiles


def plot_experiment4(profiles: dict):
    """2-panel figure for Experiment 4."""
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(16, 6))

    # ── Panel A: Raw gradient norm vs normalized time ──
    for k in K_VALUES:
        if k not in profiles:
            continue
        p = profiles[k]
        valid = ~np.isnan(p["gns_binned"])
        ax_a.plot(p["bin_centers"][valid], p["gns_binned"][valid],
                  color=COLORS_BY_K[k], linewidth=1.5, label=f"K={k}", alpha=0.85)

    ax_a.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5, label="t_start")
    ax_a.axvline(1, color="black", linestyle="--", linewidth=0.8, alpha=0.5, label="t_end")
    ax_a.axvspan(0, 1, alpha=0.08, color="#F39C12")
    ax_a.set_xlabel("Normalized time (t - t_start) / τ")
    ax_a.set_ylabel("||∇L||²")
    ax_a.set_title("A: Gradient norm vs normalized time (all K)")
    ax_a.legend(ncol=2, loc="upper right", fontsize=7)

    # ── Panel B: Gradient norm normalized by plateau baseline ──
    for k in K_VALUES:
        if k not in profiles:
            continue
        p = profiles[k]
        valid = ~np.isnan(p["gns_binned"])
        normalized = p["gns_binned"][valid] / p["g_plateau"] if p["g_plateau"] > 0 else p["gns_binned"][valid]
        ax_b.plot(p["bin_centers"][valid], normalized,
                  color=COLORS_BY_K[k], linewidth=1.5, label=f"K={k}", alpha=0.85)

    ax_b.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_b.axvline(1, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_b.axvspan(0, 1, alpha=0.08, color="#F39C12")
    ax_b.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.7, label="plateau baseline")
    ax_b.set_xlabel("Normalized time (t - t_start) / τ")
    ax_b.set_ylabel("||∇L||² / ḡ²_plateau")
    ax_b.set_title("B: Gradient norm relative to plateau (shape comparison)")
    ax_b.legend(ncol=2, loc="upper right", fontsize=7)

    fig.suptitle("Experiment 4: Per-K Gradient Norm Anatomy", fontsize=14, fontweight="bold", y=1.02)
    return fig


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("DIAGNOSTIC: Is Q Redundant with τ?")
    print("=" * 70)

    # Load all experiments
    print("\nLoading experiments...")
    experiments = []
    for k in K_VALUES:
        exp = load_experiment(k)
        experiments.append(exp)
        print(f"  K={k:>2}: {len(exp['steps'])} steps, "
              f"grad_norm range [{exp['grad_norm_sq'].min():.4f}, {exp['grad_norm_sq'].max():.4f}]")

    # ── Experiment 1: Compute decomposition ──
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Q-vs-τ Decomposition")
    print("=" * 70)

    decomps = []
    for exp in experiments:
        d = compute_decomposition(exp)
        decomps.append(d)
        if d is not None:
            print(f"  K={d['k']:>2}  τ={d['tau']:>6}  Q={d['Q_transition']:>10.4f}  "
                  f"ḡ²_trans={d['g_bar_sq_transition']:.6f}  "
                  f"ḡ²_plat={d['g_bar_sq_plateau']:.6f}  "
                  f"ḡ²_post={d['g_bar_sq_post']:.6f}  "
                  f"ratio={d['ratio_trans_plateau']:.3f}")
        else:
            print(f"  K={exp['k']:>2}  FAILED (no transition window)")

    valid_decomps = [d for d in decomps if d is not None]

    # Compute key statistics
    ks = np.array([d["k"] for d in valid_decomps], dtype=float)
    taus = np.array([d["tau"] for d in valid_decomps], dtype=float)
    Qs = np.array([d["Q_transition"] for d in valid_decomps], dtype=float)
    g_trans = np.array([d["g_bar_sq_transition"] for d in valid_decomps])
    ratios = np.array([d["ratio_trans_plateau"] for d in valid_decomps])

    _, beta_tau, r2_tau, _ = fit_power_law(ks, taus)
    _, alpha_Q, r2_Q, _ = fit_power_law(ks, Qs)
    _, beta_g, r2_g, _ = fit_power_law(ks, g_trans)
    _, beta_r, r2_r, _ = fit_power_law(ks, ratios)

    slope_qt, _, r2_qt, _ = fit_linear(taus, Qs)
    _, gamma_qt, r2_qt_pow, _ = fit_power_law(taus, Qs)

    print(f"\n  KEY RESULTS:")
    print(f"  τ ∝ K^{beta_tau:.3f}  (R²={r2_tau:.4f})")
    print(f"  Q ∝ K^{alpha_Q:.3f}  (R²={r2_Q:.4f})")
    print(f"  ḡ²_transition ∝ K^{beta_g:.3f}  (R²={r2_g:.4f})")
    print(f"  ratio ∝ K^{beta_r:.3f}  (R²={r2_r:.4f})")
    print(f"  Q vs τ linear R²={r2_qt:.4f}")
    print(f"  Q vs τ power-law γ={gamma_qt:.3f}  R²={r2_qt_pow:.4f}")
    print(f"  If Q = ḡ²·τ with ḡ² constant: expect α ≈ β = {beta_tau:.3f}")
    print(f"  Actual α = {alpha_Q:.3f}, Δ(α - β) = {alpha_Q - beta_tau:.3f}")

    print(f"\n  INTERPRETATION:")
    if abs(beta_g) < 0.15:
        print(f"  → ḡ² is FLAT (β_g={beta_g:.3f}): per-step dissipation independent of K")
        print(f"    This means Q ≈ const × τ. Hypothesis B (discovery) is favored.")
    elif beta_g > 0.15:
        print(f"  → ḡ² INCREASES with K (β_g={beta_g:.3f}): per-step cost grows")
        print(f"    Q ≠ const × τ. Hypothesis A (thermodynamic) survives.")
    else:
        print(f"  → ḡ² DECREASES with K (β_g={beta_g:.3f}): unexpected.")

    if r2_qt > 0.99:
        print(f"  → Q is perfectly linear in τ (R²={r2_qt:.4f}): Q IS redundant")
    elif r2_qt > 0.95:
        print(f"  → Q highly correlated with τ (R²={r2_qt:.4f}): mostly redundant")
    else:
        print(f"  → Q NOT perfectly linear in τ (R²={r2_qt:.4f}): Q has info beyond τ")

    # ── Generate Experiment 1 figure ──
    fig1 = plot_experiment1(valid_decomps)
    fig1_path = SAVE_DIR / "fig_diagnostic_q_vs_tau.pdf"
    fig1.savefig(fig1_path, bbox_inches="tight")
    fig1.savefig(fig1_path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig1)
    print(f"\n  Saved: {fig1_path}")
    print(f"  Saved: {fig1_path.with_suffix('.png')}")

    # ── Experiment 4: Gradient norm anatomy ──
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Per-K Gradient Norm Anatomy")
    print("=" * 70)

    profiles = compute_normalized_profiles(experiments, decomps)

    # Check self-similarity: compute peak height in normalized profiles
    print("\n  Normalized profile peak heights (relative to plateau):")
    for k in K_VALUES:
        if k not in profiles:
            continue
        p = profiles[k]
        valid = ~np.isnan(p["gns_binned"])
        trans_mask = valid & (p["bin_centers"] >= 0) & (p["bin_centers"] <= 1)
        if np.any(trans_mask):
            peak_norm = np.max(p["gns_binned"][trans_mask]) / p["g_plateau"] if p["g_plateau"] > 0 else 0
            mean_norm = np.mean(p["gns_binned"][trans_mask]) / p["g_plateau"] if p["g_plateau"] > 0 else 0
            print(f"    K={k:>2}:  peak/plateau = {peak_norm:.3f},  mean/plateau = {mean_norm:.3f}")

    fig4 = plot_experiment4(profiles)
    fig4_path = SAVE_DIR / "fig_diagnostic_grad_anatomy.pdf"
    fig4.savefig(fig4_path, bbox_inches="tight")
    fig4.savefig(fig4_path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig4)
    print(f"\n  Saved: {fig4_path}")
    print(f"  Saved: {fig4_path.with_suffix('.png')}")

    # ── Save numerical results ──
    results = {
        "experiment_1": {
            "decompositions": valid_decomps,
            "scaling": {
                "tau_exponent": float(beta_tau), "tau_R2": float(r2_tau),
                "Q_exponent": float(alpha_Q), "Q_R2": float(r2_Q),
                "g_bar_sq_exponent": float(beta_g), "g_bar_sq_R2": float(r2_g),
                "ratio_exponent": float(beta_r), "ratio_R2": float(r2_r),
                "Q_vs_tau_linear_R2": float(r2_qt),
                "Q_vs_tau_power_gamma": float(gamma_qt),
                "Q_vs_tau_power_R2": float(r2_qt_pow),
                "alpha_minus_beta": float(alpha_Q - beta_tau),
            },
        },
    }

    def jsonify(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results_path = OUTPUTS / "diagnostic_q_vs_tau_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=jsonify)
    print(f"\n  Saved results: {results_path}")

    # ── Final verdict ──
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    thermo_score = 0
    discovery_score = 0

    if beta_g > 0.15:
        print("  [THERMO]    ḡ² increases with K")
        thermo_score += 1
    elif abs(beta_g) < 0.15:
        print("  [DISCOVERY] ḡ² flat across K")
        discovery_score += 1
    else:
        print(f"  [???]       ḡ² decreases with K (β={beta_g:.3f})")

    if r2_qt > 0.99:
        print("  [DISCOVERY] Q perfectly linear in τ")
        discovery_score += 1
    elif r2_qt > 0.95:
        print("  [MIXED]     Q highly correlated with τ but not perfect")
        thermo_score += 0.5
        discovery_score += 0.5
    else:
        print("  [THERMO]    Q not redundant with τ")
        thermo_score += 1

    if beta_r > 0.15:
        print("  [THERMO]    Transition/plateau ratio grows with K")
        thermo_score += 1
    elif np.mean(ratios) > 1.3:
        print("  [MIXED]     Ratio elevated but not growing")
        thermo_score += 0.5
        discovery_score += 0.5
    else:
        print("  [DISCOVERY] Transition not distinct from plateau")
        discovery_score += 1

    delta_exp = alpha_Q - beta_tau
    if abs(delta_exp) > 0.15:
        print(f"  [THERMO]    α ≠ β: exponent gap Δ = {delta_exp:.3f}")
        thermo_score += 1
    else:
        print(f"  [DISCOVERY] α ≈ β: exponent gap Δ = {delta_exp:.3f}")
        discovery_score += 1

    print(f"\n  Score:  Thermodynamic = {thermo_score},  Discovery = {discovery_score}")
    if thermo_score > discovery_score + 1:
        print("  → HYPOTHESIS A (Thermodynamic) SURVIVES")
        print("  → Proceed to Experiments 2-3 (LR sweep, batch size sweep)")
    elif discovery_score > thermo_score + 1:
        print("  → HYPOTHESIS B (Discovery latency) CONFIRMED")
        print("  → Q is redundant with τ. Reframe paper around circuit discovery time.")
    else:
        print("  → AMBIGUOUS — data partially consistent with both")
        print("  → Recommend running Experiment 2 (LR sweep) for a definitive test")

    print("\nDone.")


if __name__ == "__main__":
    main()
