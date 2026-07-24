#!/usr/bin/env python3
"""
Exponent Robustness Check: Leave-One-Out Jackknife + Bootstrap

Tests sensitivity of the three scaling law exponents:
  Law 1:  τ ∝ K^α      (10 K-values)
  Law 2:  τ ∝ η^γ      (4 η-values, K=20 fixed)
  Law 3:  ḡ²_plateau ∝ K^β  (10 K-values)

Methods: LOO jackknife, bootstrap (Law 1), and alternative model comparison.

No new training required — pure reanalysis of existing data.
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

K_VALUES = np.array([3, 5, 7, 10, 13, 17, 20, 25, 30, 36], dtype=float)
LR_VALUES_ALL = [3e-4, 5e-4, 1e-3, 2e-3]  # exclude 5e-3 (never transitioned)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
    "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.3,
})


# ──────────────────────────────────────────────────────────────────────
# Data extraction
# ──────────────────────────────────────────────────────────────────────

def find_transition_window(steps, L, log_k, thresh_hi=0.95, thresh_lo=0.05):
    """Return (t_start, t_end) using candidate loss thresholds."""
    hi = thresh_hi * log_k
    lo = thresh_lo * log_k
    t_end_idx = None
    for i, l in enumerate(L):
        if l < lo:
            t_end_idx = i
            break
    if t_end_idx is None:
        return None, None
    t_start = None
    for i in range(t_end_idx):
        if L[i] > hi:
            t_start = steps[i]
    t_end = steps[t_end_idx]
    return t_start, t_end


def load_k_sweep_data():
    """Load τ and ḡ²_plateau for all 10 K values."""
    tau_vals = []
    g2_plateau_vals = []
    valid_k = []

    for k in K_VALUES:
        ki = int(k)
        path = OUTPUTS / f"landauer_dense_k{ki}" / "training_history.json"
        if not path.exists():
            continue
        with open(path) as f:
            h = json.load(f)

        steps = np.array(h["steps"], dtype=float)
        L = np.array(h["candidate_loss"], dtype=float)
        g2 = np.array(h["grad_norm_sq"], dtype=float)
        log_k = math.log(k)

        t_start, t_end = find_transition_window(steps, L, log_k)
        if t_start is None or t_end is None:
            continue

        tau = t_end - t_start

        # Plateau ḡ²: mean g2 from plateau onset to t_start
        plat_mask = (steps >= 500) & (steps <= t_start)
        if plat_mask.sum() < 2:
            plat_mask = steps <= t_start
        g2_plat = float(np.mean(g2[plat_mask])) if plat_mask.sum() > 0 else None

        valid_k.append(k)
        tau_vals.append(tau)
        g2_plateau_vals.append(g2_plat)

    return np.array(valid_k), np.array(tau_vals), np.array(g2_plateau_vals)


def load_lr_sweep_data():
    """Load τ for LR sweep (K=20 fixed)."""
    with open(OUTPUTS / "lr_sweep_results.json") as f:
        results = json.load(f)

    eta_vals = []
    tau_vals = []
    for m in results["metrics"]:
        if m.get("no_transition") or m.get("diverged"):
            continue
        eta_vals.append(m["lr"])
        tau_vals.append(m["tau"])

    return np.array(eta_vals), np.array(tau_vals)


# ──────────────────────────────────────────────────────────────────────
# Fitting utilities
# ──────────────────────────────────────────────────────────────────────

def power_law(x, a, alpha):
    return a * np.power(x, alpha)


def fit_power_law(x, y, p0=None):
    """Fit y = a * x^alpha. Returns (a, alpha, R²)."""
    if p0 is None:
        p0 = [1.0, 1.0]
    try:
        popt, _ = curve_fit(power_law, x, y, p0=p0, maxfev=10000)
        y_pred = power_law(x, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return popt[0], popt[1], r2
    except Exception:
        return None, None, None


def fit_log_linear(x, y):
    """Fit y = c * log(x) + b. Returns (c, b, R²)."""
    log_x = np.log(x)
    coeffs = np.polyfit(log_x, y, 1)
    y_pred = np.polyval(coeffs, log_x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return coeffs[0], coeffs[1], r2


def fit_linear(x, y):
    """Fit y = c * x + b. Returns (c, b, R²)."""
    coeffs = np.polyfit(x, y, 1)
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return coeffs[0], coeffs[1], r2


def compute_aic(y, y_pred, n_params):
    n = len(y)
    rss = np.sum((y - y_pred) ** 2)
    if rss <= 0 or n <= n_params:
        return float("inf")
    return n * np.log(rss / n) + 2 * n_params


# ──────────────────────────────────────────────────────────────────────
# Analysis 1 & 3: LOO Jackknife on K-sweep
# ──────────────────────────────────────────────────────────────────────

def loo_jackknife(x, y, p0, label):
    """Run LOO jackknife for power law fit. Returns dict of results."""
    n = len(x)

    # Full fit
    a_full, alpha_full, r2_full = fit_power_law(x, y, p0=p0)
    print(f"\n{label}")
    print(f"  Full fit (n={n}): exponent = {alpha_full:.4f}, R² = {r2_full:.4f}")

    # LOO
    alphas_loo = []
    for i in range(n):
        x_drop = np.delete(x, i)
        y_drop = np.delete(y, i)
        _, alpha_i, _ = fit_power_law(x_drop, y_drop, p0=p0)
        if alpha_i is not None:
            alphas_loo.append(alpha_i)
            delta = abs(alpha_i - alpha_full)
            print(f"    Drop x={x[i]:>8.4g}:  exponent = {alpha_i:.4f}  (Δ = {delta:+.4f})")
        else:
            alphas_loo.append(np.nan)
            print(f"    Drop x={x[i]:>8.4g}:  fit failed")

    alphas_loo = np.array(alphas_loo)
    valid = ~np.isnan(alphas_loo)

    loo_mean = np.mean(alphas_loo[valid])
    loo_std = np.std(alphas_loo[valid])
    loo_min = np.min(alphas_loo[valid])
    loo_max = np.max(alphas_loo[valid])

    # Most / least influential
    deltas = np.abs(alphas_loo - alpha_full)
    deltas[~valid] = -1
    most_idx = np.argmax(deltas)
    least_idx = np.argmin(deltas[valid])
    # Re-index least to the valid subset
    valid_indices = np.where(valid)[0]

    print(f"\n  LOO mean ± std:    {loo_mean:.4f} ± {loo_std:.4f}")
    print(f"  LOO range:         [{loo_min:.4f}, {loo_max:.4f}]")
    print(f"  Most influential:  x={x[most_idx]:.4g}  (Δ = {deltas[most_idx]:.4f})")
    print(f"  Least influential: x={x[valid_indices[least_idx]]:.4g}  (Δ = {deltas[valid_indices[least_idx]]:.4f})")

    return {
        "alpha_full": alpha_full,
        "r2_full": r2_full,
        "loo_alphas": alphas_loo,
        "loo_mean": loo_mean,
        "loo_std": loo_std,
        "loo_min": loo_min,
        "loo_max": loo_max,
        "most_influential_x": float(x[most_idx]),
        "most_influential_delta": float(deltas[most_idx]),
        "least_influential_x": float(x[valid_indices[least_idx]]),
        "least_influential_delta": float(deltas[valid_indices[least_idx]]),
    }


# ──────────────────────────────────────────────────────────────────────
# Analysis 4: Bootstrap (Law 1)
# ──────────────────────────────────────────────────────────────────────

def bootstrap_power_law(x, y, p0, n_bootstrap=2000, seed=42):
    """Bootstrap resampling for power law exponent."""
    rng = np.random.default_rng(seed)
    n = len(x)
    alphas = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        x_boot = x[idx]
        y_boot = y[idx]
        if len(np.unique(x_boot)) < 4:
            continue
        _, alpha_i, _ = fit_power_law(x_boot, y_boot, p0=p0)
        if alpha_i is not None and np.isfinite(alpha_i):
            alphas.append(alpha_i)

    alphas = np.array(alphas)
    return {
        "alphas": alphas,
        "median": float(np.median(alphas)),
        "mean": float(np.mean(alphas)),
        "std": float(np.std(alphas)),
        "ci_2_5": float(np.percentile(alphas, 2.5)),
        "ci_97_5": float(np.percentile(alphas, 97.5)),
        "n_valid": len(alphas),
        "n_total": n_bootstrap,
    }


# ──────────────────────────────────────────────────────────────────────
# Analysis 5: Alternative model comparison on LOO subsets
# ──────────────────────────────────────────────────────────────────────

def loo_model_comparison(K, tau):
    """Compare power law vs log-linear vs linear across LOO subsets."""
    n = len(K)
    wins = {"power": 0, "log": 0, "linear": 0}
    results = []

    for i in range(n):
        K_drop = np.delete(K, i)
        tau_drop = np.delete(tau, i)

        # Power law
        a_p, alpha_p, r2_p = fit_power_law(K_drop, tau_drop, p0=[1, 1.3])

        # Log-linear
        _, _, r2_log = fit_log_linear(K_drop, tau_drop)

        # Linear
        _, _, r2_lin = fit_linear(K_drop, tau_drop)

        best = max([("power", r2_p), ("log", r2_log), ("linear", r2_lin)],
                    key=lambda x: x[1] if x[1] is not None else -1)
        wins[best[0]] += 1

        results.append({
            "dropped_K": float(K[i]),
            "r2_power": r2_p,
            "r2_log": r2_log,
            "r2_linear": r2_lin,
            "winner": best[0],
        })

    print(f"\n  Model comparison across {n} LOO subsets:")
    print(f"    Power law wins: {wins['power']}/{n}")
    print(f"    Log(K) wins:    {wins['log']}/{n}")
    print(f"    Linear wins:    {wins['linear']}/{n}")

    for r in results:
        print(f"    Drop K={r['dropped_K']:>5.0f}:  "
              f"Power R²={r['r2_power']:.4f}  "
              f"Log R²={r['r2_log']:.4f}  "
              f"Lin R²={r['r2_linear']:.4f}  "
              f"→ {r['winner']}")

    return wins, results


# ──────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────

def make_figure(K, tau, law1_loo, law1_boot, law2_loo, law3_loo, save_path):
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.38, wspace=0.35,
                           left=0.06, right=0.97, top=0.93, bottom=0.07)

    # ── Panel A: Bootstrap histogram of α (Law 1) ──
    ax_a = fig.add_subplot(gs[0, 0])
    alphas = law1_boot["alphas"]
    ax_a.hist(alphas, bins=50, color="#3498DB", alpha=0.7, edgecolor="white",
              linewidth=0.3, density=True)
    ax_a.axvline(law1_loo["alpha_full"], color="#E74C3C", linewidth=2,
                 label=f'Full fit α = {law1_loo["alpha_full"]:.3f}')
    ax_a.axvline(law1_boot["ci_2_5"], color="#E74C3C", linewidth=1,
                 linestyle="--", label=f'95% CI [{law1_boot["ci_2_5"]:.3f}, {law1_boot["ci_97_5"]:.3f}]')
    ax_a.axvline(law1_boot["ci_97_5"], color="#E74C3C", linewidth=1,
                 linestyle="--")
    # LOO range as shaded region
    ax_a.axvspan(law1_loo["loo_min"], law1_loo["loo_max"],
                 alpha=0.15, color="#F39C12",
                 label=f'LOO range [{law1_loo["loo_min"]:.3f}, {law1_loo["loo_max"]:.3f}]')
    ax_a.set_xlabel("Exponent α  (τ ∝ K^α)")
    ax_a.set_ylabel("Density")
    ax_a.set_title("A: Bootstrap distribution of α  (Law 1)")
    ax_a.legend(fontsize=7, loc="upper left")

    # ── Panel B: LOO α vs dropped K (Law 1) ──
    ax_b = fig.add_subplot(gs[0, 1])
    loo_alphas = law1_loo["loo_alphas"]
    valid = ~np.isnan(loo_alphas)
    ax_b.scatter(K[valid], loo_alphas[valid], c="#3498DB", s=80,
                 zorder=5, edgecolors="white", linewidth=0.5)
    for i, ki in enumerate(K):
        if valid[i]:
            ax_b.annotate(f"{loo_alphas[i]:.3f}",
                          (ki, loo_alphas[i]),
                          textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax_b.axhline(law1_loo["alpha_full"], color="#E74C3C", linewidth=1.5,
                 linestyle="-", label=f'Full α = {law1_loo["alpha_full"]:.3f}')
    ax_b.fill_between([K.min() - 1, K.max() + 1],
                       law1_loo["alpha_full"] - law1_loo["loo_std"],
                       law1_loo["alpha_full"] + law1_loo["loo_std"],
                       alpha=0.15, color="#E74C3C")
    ax_b.set_xlim(K.min() - 2, K.max() + 2)
    ax_b.set_xlabel("Dropped K value")
    ax_b.set_ylabel("Fitted α  (remaining 9 points)")
    ax_b.set_title("B: LOO sensitivity  (Law 1: τ ∝ K^α)")
    ax_b.legend(fontsize=8)

    # ── Panel C: LOO β vs dropped K (Law 3) ──
    ax_c = fig.add_subplot(gs[0, 2])
    loo_betas = law3_loo["loo_alphas"]
    valid3 = ~np.isnan(loo_betas)
    ax_c.scatter(K[valid3], loo_betas[valid3], c="#9B59B6", s=80,
                 zorder=5, edgecolors="white", linewidth=0.5)
    for i, ki in enumerate(K):
        if valid3[i]:
            ax_c.annotate(f"{loo_betas[i]:.3f}",
                          (ki, loo_betas[i]),
                          textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax_c.axhline(law3_loo["alpha_full"], color="#8E44AD", linewidth=1.5,
                 linestyle="-", label=f'Full β = {law3_loo["alpha_full"]:.3f}')
    ax_c.fill_between([K.min() - 1, K.max() + 1],
                       law3_loo["alpha_full"] - law3_loo["loo_std"],
                       law3_loo["alpha_full"] + law3_loo["loo_std"],
                       alpha=0.15, color="#8E44AD")
    ax_c.set_xlim(K.min() - 2, K.max() + 2)
    ax_c.set_xlabel("Dropped K value")
    ax_c.set_ylabel("Fitted β  (remaining 9 points)")
    ax_c.set_title("C: LOO sensitivity  (Law 3: ḡ² ∝ K^β)")
    ax_c.legend(fontsize=8)

    # ── Panel D: LOO γ vs dropped η (Law 2) ──
    ax_d = fig.add_subplot(gs[1, 0])
    eta_arr = np.array(LR_VALUES_ALL)
    loo_gammas = law2_loo["loo_alphas"]
    valid2 = ~np.isnan(loo_gammas)
    if valid2.sum() > 0:
        ax_d.scatter(eta_arr[valid2], loo_gammas[valid2], c="#E67E22", s=80,
                     zorder=5, edgecolors="white", linewidth=0.5)
        for i, ei in enumerate(eta_arr):
            if valid2[i]:
                ax_d.annotate(f"{loo_gammas[i]:.3f}",
                              (ei, loo_gammas[i]),
                              textcoords="offset points", xytext=(6, 4), fontsize=7)
        ax_d.axhline(law2_loo["alpha_full"], color="#E74C3C", linewidth=1.5,
                     linestyle="-", label=f'Full γ = {law2_loo["alpha_full"]:.3f}')
        ax_d.fill_between([eta_arr.min() * 0.8, eta_arr.max() * 1.2],
                           law2_loo["alpha_full"] - law2_loo["loo_std"],
                           law2_loo["alpha_full"] + law2_loo["loo_std"],
                           alpha=0.15, color="#E74C3C")
    ax_d.set_xscale("log")
    ax_d.set_xlabel("Dropped η value")
    ax_d.set_ylabel("Fitted γ  (remaining 3 points)")
    ax_d.set_title("D: LOO sensitivity  (Law 2: τ ∝ η^γ)")
    ax_d.legend(fontsize=8)

    # ── Panel E: Summary comparison bar chart ──
    ax_e = fig.add_subplot(gs[1, 1:])

    laws = ["Law 1\nτ ∝ K^α", "Law 2\nτ ∝ η^γ", "Law 3\nḡ² ∝ K^β"]
    full_vals = [law1_loo["alpha_full"], law2_loo["alpha_full"], law3_loo["alpha_full"]]
    loo_stds = [law1_loo["loo_std"], law2_loo["loo_std"], law3_loo["loo_std"]]
    loo_ranges_lo = [law1_loo["loo_min"], law2_loo["loo_min"], law3_loo["loo_min"]]
    loo_ranges_hi = [law1_loo["loo_max"], law2_loo["loo_max"], law3_loo["loo_max"]]

    x_pos = np.arange(len(laws))
    colors = ["#3498DB", "#E67E22", "#9B59B6"]

    bars = ax_e.bar(x_pos, full_vals, width=0.4, color=colors, alpha=0.8,
                    edgecolor="white", linewidth=0.5)
    # LOO error bars (range)
    for i in range(len(laws)):
        lo_err = full_vals[i] - loo_ranges_lo[i]
        hi_err = loo_ranges_hi[i] - full_vals[i]
        ax_e.errorbar(x_pos[i], full_vals[i],
                       yerr=[[lo_err], [hi_err]],
                       fmt="none", ecolor="black", capsize=8, capthick=2,
                       linewidth=2)
        ax_e.errorbar(x_pos[i] + 0.05, full_vals[i],
                       yerr=loo_stds[i],
                       fmt="none", ecolor="#E74C3C", capsize=6, capthick=1.5,
                       linewidth=1.5)
        ax_e.text(x_pos[i], loo_ranges_hi[i] + 0.05,
                  f"σ_LOO = {loo_stds[i]:.3f}",
                  ha="center", fontsize=8, color="#E74C3C")

    # Bootstrap CI for Law 1
    ci_lo = law1_boot["ci_2_5"]
    ci_hi = law1_boot["ci_97_5"]
    ax_e.plot([x_pos[0] - 0.25, x_pos[0] + 0.25], [ci_lo, ci_lo],
              color="#2ECC71", linewidth=2, linestyle="--")
    ax_e.plot([x_pos[0] - 0.25, x_pos[0] + 0.25], [ci_hi, ci_hi],
              color="#2ECC71", linewidth=2, linestyle="--",
              label=f"Bootstrap 95% CI")

    ax_e.set_xticks(x_pos)
    ax_e.set_xticklabels(laws)
    ax_e.set_ylabel("Exponent value")
    ax_e.set_title("E: Exponent stability summary")
    ax_e.legend(fontsize=8)

    # Annotate bars
    for i, (bar, val) in enumerate(zip(bars, full_vals)):
        ax_e.text(bar.get_x() + bar.get_width() / 2, val / 2,
                  f"{val:.3f}", ha="center", va="center",
                  fontsize=12, fontweight="bold", color="white")

    fig.suptitle("Exponent Robustness: Leave-One-Out Jackknife + Bootstrap",
                 fontsize=14, fontweight="bold")
    fig.savefig(save_path, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {save_path}")
    print(f"Saved: {save_path.with_suffix('.pdf')}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("EXPONENT ROBUSTNESS CHECK")
    print("=" * 70)

    # Load data
    print("\nLoading K-sweep data...")
    K, tau_k, g2_plat = load_k_sweep_data()
    print(f"  K values: {K.astype(int).tolist()}")
    print(f"  τ values: {tau_k.tolist()}")
    print(f"  ḡ² values: {[f'{v:.4f}' for v in g2_plat]}")

    print("\nLoading LR-sweep data...")
    eta, tau_eta = load_lr_sweep_data()
    print(f"  η values: {eta.tolist()}")
    print(f"  τ values: {tau_eta.tolist()}")

    # ── Law 1: τ ∝ K^α ──
    print("\n" + "─" * 70)
    law1_loo = loo_jackknife(K, tau_k, p0=[1, 1.3],
                              label="LAW 1: τ ∝ K^α  (10 K-values)")

    print("\n  Bootstrap resampling (n=2000)...")
    law1_boot = bootstrap_power_law(K, tau_k, p0=[1, 1.3], n_bootstrap=2000)
    print(f"    Median α: {law1_boot['median']:.4f}")
    print(f"    Mean ± std: {law1_boot['mean']:.4f} ± {law1_boot['std']:.4f}")
    print(f"    95% CI: [{law1_boot['ci_2_5']:.4f}, {law1_boot['ci_97_5']:.4f}]")
    print(f"    Valid resamples: {law1_boot['n_valid']}/{law1_boot['n_total']}")

    print("\n  Alternative model comparison (LOO):")
    wins1, model_results1 = loo_model_comparison(K, tau_k)

    # ── Law 2: τ ∝ η^γ ──
    print("\n" + "─" * 70)
    law2_loo = loo_jackknife(eta, tau_eta, p0=[1e6, 0.8],
                              label="LAW 2: τ ∝ η^γ  (4 η-values, K=20)")

    # ── Law 3: ḡ² ∝ K^β ──
    print("\n" + "─" * 70)
    # Filter out any NaN g2 values (K=3 has no plateau before transition)
    valid_g2 = ~np.isnan(g2_plat) & (g2_plat > 0)
    law3_loo = loo_jackknife(K[valid_g2], g2_plat[valid_g2], p0=[10, -0.6],
                              label="LAW 3: ḡ²_plateau ∝ K^β  (K-sweep)")

    # ── Summary table ──
    print("\n" + "=" * 70)
    print("ROBUSTNESS RESULTS")
    print("=" * 70)

    print(f"""
LAW 1: τ ∝ K^α
  Full fit:            α = {law1_loo['alpha_full']:.4f}  (R² = {law1_loo['r2_full']:.4f})
  LOO:                 α = {law1_loo['loo_mean']:.4f} ± {law1_loo['loo_std']:.4f}  [range: {law1_loo['loo_min']:.4f}, {law1_loo['loo_max']:.4f}]
  Bootstrap 95% CI:    [{law1_boot['ci_2_5']:.4f}, {law1_boot['ci_97_5']:.4f}]
  Most influential K:  {law1_loo['most_influential_x']:.0f}  (Δα = {law1_loo['most_influential_delta']:.4f})
  Power law wins LOO:  {wins1['power']}/10 subsets

LAW 2: τ ∝ η^γ
  Full fit:            γ = {law2_loo['alpha_full']:.4f}  (R² = {law2_loo['r2_full']:.4f})
  LOO:                 γ = {law2_loo['loo_mean']:.4f} ± {law2_loo['loo_std']:.4f}  [range: {law2_loo['loo_min']:.4f}, {law2_loo['loo_max']:.4f}]
  Most influential η:  {law2_loo['most_influential_x']:.0e}  (Δγ = {law2_loo['most_influential_delta']:.4f})

LAW 3: ḡ² ∝ K^β
  Full fit:            β = {law3_loo['alpha_full']:.4f}  (R² = {law3_loo['r2_full']:.4f})
  LOO:                 β = {law3_loo['loo_mean']:.4f} ± {law3_loo['loo_std']:.4f}  [range: {law3_loo['loo_min']:.4f}, {law3_loo['loo_max']:.4f}]
  Most influential K:  {law3_loo['most_influential_x']:.0f}  (Δβ = {law3_loo['most_influential_delta']:.4f})""")

    print("\nSTABILITY VERDICT:")
    stable_1 = law1_loo["loo_std"] < 0.1
    stable_2 = law2_loo["loo_std"] < 0.2
    stable_3 = law3_loo["loo_std"] < 0.1
    print(f"  α stable (LOO std < 0.1)?  {'YES' if stable_1 else 'NO'}  (σ = {law1_loo['loo_std']:.4f})")
    print(f"  γ stable (LOO std < 0.2)?  {'YES' if stable_2 else 'NO'}  (σ = {law2_loo['loo_std']:.4f})")
    print(f"  β stable (LOO std < 0.1)?  {'YES' if stable_3 else 'NO'}  (σ = {law3_loo['loo_std']:.4f})")
    print()

    # ── Figure ──
    print("Generating figure...")
    # Pad law3 LOO alphas back to full K array length for plotting
    if valid_g2.sum() < len(K):
        loo_full = np.full(len(K), np.nan)
        loo_full[valid_g2] = law3_loo["loo_alphas"]
        law3_loo_plot = dict(law3_loo)
        law3_loo_plot["loo_alphas"] = loo_full
    else:
        law3_loo_plot = law3_loo

    save_path = SAVE_DIR / "fig_exponent_robustness.png"
    make_figure(K, tau_k, law1_loo, law1_boot, law2_loo, law3_loo_plot, save_path)

    # ── Save JSON ──
    results = {
        "law1": {
            "exponent_name": "alpha",
            "full_fit": law1_loo["alpha_full"],
            "r2_full": law1_loo["r2_full"],
            "loo_mean": law1_loo["loo_mean"],
            "loo_std": law1_loo["loo_std"],
            "loo_range": [law1_loo["loo_min"], law1_loo["loo_max"]],
            "bootstrap_95ci": [law1_boot["ci_2_5"], law1_boot["ci_97_5"]],
            "bootstrap_median": law1_boot["median"],
            "most_influential": {"x": law1_loo["most_influential_x"],
                                  "delta": law1_loo["most_influential_delta"]},
            "power_law_wins_loo": wins1["power"],
            "stable": bool(stable_1),
        },
        "law2": {
            "exponent_name": "gamma",
            "full_fit": law2_loo["alpha_full"],
            "r2_full": law2_loo["r2_full"],
            "loo_mean": law2_loo["loo_mean"],
            "loo_std": law2_loo["loo_std"],
            "loo_range": [law2_loo["loo_min"], law2_loo["loo_max"]],
            "most_influential": {"x": law2_loo["most_influential_x"],
                                  "delta": law2_loo["most_influential_delta"]},
            "stable": bool(stable_2),
        },
        "law3": {
            "exponent_name": "beta",
            "full_fit": law3_loo["alpha_full"],
            "r2_full": law3_loo["r2_full"],
            "loo_mean": law3_loo["loo_mean"],
            "loo_std": law3_loo["loo_std"],
            "loo_range": [law3_loo["loo_min"], law3_loo["loo_max"]],
            "most_influential": {"x": law3_loo["most_influential_x"],
                                  "delta": law3_loo["most_influential_delta"]},
            "stable": bool(stable_3),
        },
    }

    results_path = OUTPUTS / "exponent_robustness_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {results_path}")


if __name__ == "__main__":
    main()
