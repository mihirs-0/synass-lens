#!/usr/bin/env python3
"""
Analyze the dense K-sweep Landauer scaling experiment.

Loads results from all 10 K values, computes Q_transition under multiple
definitions, fits Q vs log(K) against linear / power-law / quadratic
alternatives, and generates a publication-quality 4-panel figure.

Usage:
    python scripts/analyze_landauer_dense.py
    python scripts/analyze_landauer_dense.py --output-dir outputs --save-dir outputs/paper_figures
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

try:
    from scipy.optimize import curve_fit
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"

K_VALUES = [3, 5, 7, 10, 13, 17, 20, 25, 30, 36]
EXPERIMENT_NAMES = [f"landauer_dense_k{k}" for k in K_VALUES]

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

def load_json(path: Path) -> Optional[Dict]:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_experiment(name: str, output_dir: Path) -> Optional[Dict]:
    """Load all results for a single experiment."""
    import yaml

    exp_dir = output_dir / name
    history = load_json(exp_dir / "training_history.json")
    grad_norms = load_json(exp_dir / "gradient_norm_results.json")
    candidate_eval = load_json(exp_dir / "candidate_eval_results.json")

    if history is None:
        print(f"  [SKIP] {name}: no training_history.json")
        return None

    cfg_path = exp_dir / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
    else:
        print(f"  [SKIP] {name}: no config.yaml")
        return None

    k = int(cfg["data"]["k"])
    return {
        "name": name,
        "k": k,
        "log_k": math.log(k),
        "config": cfg,
        "history": history,
        "grad_norms": grad_norms,
        "candidate_eval": candidate_eval,
    }


# ──────────────────────────────────────────────────────────────────────
# Transition window detection
# ──────────────────────────────────────────────────────────────────────

def find_transition_window(steps, candidate_loss, log_k,
                           thresh_hi=0.95, thresh_lo=0.05):
    """
    t_end:   first step where candidate_loss < thresh_lo * log(K)
    t_start: last step BEFORE t_end where candidate_loss > thresh_hi * log(K)

    Finding t_end first and constraining t_start to precede it prevents
    late instability spikes (common with constant LR / no clipping) from
    corrupting the transition window.
    """
    hi = thresh_hi * log_k
    lo = thresh_lo * log_k

    # Step 1: find t_end (first convergence)
    t_end = None
    t_end_idx = None
    for i, cl in enumerate(candidate_loss):
        if cl < lo:
            t_end = steps[i]
            t_end_idx = i
            break

    if t_end is None:
        return None, None

    # Step 2: find t_start — last step before t_end still on the plateau
    t_start = None
    for i in range(t_end_idx):
        if candidate_loss[i] > hi:
            t_start = steps[i]

    return t_start, t_end


def find_grad_spike_window(steps, grad_norm_sq, baseline_mult=2.0):
    """Window where grad_norm > baseline_mult * baseline."""
    baseline = np.median(grad_norm_sq[:max(5, len(grad_norm_sq) // 10)])
    threshold = baseline_mult * baseline
    in_spike = np.array(grad_norm_sq) > threshold
    if not np.any(in_spike):
        return None, None
    indices = np.where(in_spike)[0]
    return steps[indices[0]], steps[indices[-1]]


def find_zgap_window(steps, z_gap, log_k, lo_thresh=0.1, hi_thresh=0.95):
    """Window from first z_gap > lo_thresh to z_gap > hi_thresh * log(K)."""
    t_start = None
    t_end = None
    for i, g in enumerate(z_gap):
        if g > lo_thresh and t_start is None:
            t_start = steps[i]
        if g > hi_thresh * log_k:
            t_end = steps[i]
            break
    return t_start, t_end


# ──────────────────────────────────────────────────────────────────────
# Dissipation integrals
# ──────────────────────────────────────────────────────────────────────

def compute_q_integral(steps, grad_norm_sq, lr, t_start, t_end):
    """
    Q = sum(lr * grad_norm_sq[t] * delta_t) for t in [t_start, t_end].
    Uses trapezoidal-style: each checkpoint's contribution spans the
    interval from the previous checkpoint to the current one.
    """
    if t_start is None or t_end is None:
        return None

    steps = np.array(steps, dtype=float)
    gns = np.array(grad_norm_sq, dtype=float)

    mask = (steps >= t_start) & (steps <= t_end)
    if not np.any(mask):
        return None

    s = steps[mask]
    g = gns[mask]
    delta = np.diff(s, prepend=s[0])
    delta[0] = s[0] - t_start if s[0] > t_start else (s[1] - s[0] if len(s) > 1 else 1)

    return float(np.sum(lr * g * delta))


def compute_all_q_definitions(exp: Dict) -> Dict:
    """Compute Q under four definitions for a single experiment."""
    k = exp["k"]
    log_k = exp["log_k"]
    lr = float(exp["config"]["training"]["learning_rate"])

    # Prefer candidate_eval_results.json (post-hoc, dense); fall back to training history
    if exp["candidate_eval"] is not None:
        cand_steps = exp["candidate_eval"]["steps"]
        cand_loss = exp["candidate_eval"]["candidate_loss"]
        z_gap = exp["candidate_eval"].get("z_gap")
    elif "candidate_loss" in exp["history"]:
        cand_steps = exp["history"]["steps"]
        cand_loss = exp["history"]["candidate_loss"]
        z_gap = None
    else:
        return {"k": k, "log_k": log_k, "error": "no candidate loss data"}

    # Gradient norms: prefer dedicated computation, fall back to training history
    if exp["grad_norms"] is not None:
        grad_steps = exp["grad_norms"]["steps"]
        grad_norm_sq = exp["grad_norms"]["total_grad_norm_sq"]
    elif "grad_norm_sq" in exp["history"]:
        grad_steps = exp["history"]["steps"]
        grad_norm_sq = exp["history"]["grad_norm_sq"]
    else:
        return {"k": k, "log_k": log_k, "error": "no gradient norm data"}

    # Q_transition: standard definition (95%/5% of log(K))
    t_start, t_end = find_transition_window(cand_steps, cand_loss, log_k)
    q_transition = compute_q_integral(grad_steps, grad_norm_sq, lr, t_start, t_end)

    # Q_total_conv: from step 0 to convergence (t_end)
    q_total_conv = compute_q_integral(grad_steps, grad_norm_sq, lr, 0, t_end)

    # Q_gradspike: integral over grad_norm > 2x baseline
    gs_start, gs_end = find_grad_spike_window(
        np.array(grad_steps), np.array(grad_norm_sq)
    )
    q_gradspike = compute_q_integral(grad_steps, grad_norm_sq, lr, gs_start, gs_end)

    # Q_zgap: from first z_gap > 0.1 to z_gap > 0.95 * log(K)
    q_zgap = None
    if z_gap is not None:
        zg_start, zg_end = find_zgap_window(cand_steps, z_gap, log_k)
        if zg_start is not None and zg_end is not None:
            q_zgap = compute_q_integral(grad_steps, grad_norm_sq, lr, zg_start, zg_end)

    return {
        "k": k,
        "log_k": log_k,
        "transition_start": t_start,
        "transition_end": t_end,
        "Q_transition": q_transition,
        "Q_total_conv": q_total_conv,
        "Q_gradspike": q_gradspike,
        "Q_zgap": q_zgap,
    }


# ──────────────────────────────────────────────────────────────────────
# Model fitting
# ──────────────────────────────────────────────────────────────────────

def fit_linear(log_k, q):
    """Q = c * log(K) + b"""
    coeffs = np.polyfit(log_k, q, 1)
    predicted = np.polyval(coeffs, log_k)
    return coeffs, predicted


def fit_power_law(k_arr, q):
    """Q = a * K^alpha  ⟹  log(Q) = log(a) + alpha*log(K)"""
    if HAS_SCIPY:
        try:
            def model(x, a, alpha):
                return a * np.power(x, alpha)
            popt, _ = curve_fit(model, k_arr, q, p0=[1.0, 0.5], maxfev=10000)
            predicted = model(k_arr, *popt)
            return popt, predicted
        except Exception:
            pass
    # Fallback: linearise in log-log space
    try:
        mask = (k_arr > 0) & (q > 0)
        if mask.sum() < 2:
            return None, None
        log_k = np.log(k_arr[mask])
        log_q = np.log(q[mask])
        coeffs = np.polyfit(log_k, log_q, 1)
        alpha = coeffs[0]
        a = np.exp(coeffs[1])
        predicted = a * np.power(k_arr, alpha)
        return np.array([a, alpha]), predicted
    except Exception:
        return None, None


def fit_quadratic_log(log_k, q):
    """Q = a*(log K)^2 + b*log(K) + c"""
    coeffs = np.polyfit(log_k, q, 2)
    predicted = np.polyval(coeffs, log_k)
    return coeffs, predicted


def compute_r2(observed, predicted):
    ss_res = np.sum((observed - predicted) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


def compute_aic_bic(observed, predicted, n_params):
    n = len(observed)
    residuals = observed - predicted
    rss = np.sum(residuals ** 2)
    if rss <= 0:
        return float("-inf"), float("-inf")
    log_likelihood = -n / 2 * (np.log(2 * np.pi * rss / n) + 1)
    aic = 2 * n_params - 2 * log_likelihood
    bic = n_params * np.log(n) - 2 * log_likelihood
    return aic, bic


def run_model_comparison(k_arr, log_k_arr, q_arr):
    """Fit all three models and return comparison dict."""
    results = {}

    # Linear: Q = c * log(K) + b  (2 params)
    coeffs_lin, pred_lin = fit_linear(log_k_arr, q_arr)
    r2_lin = compute_r2(q_arr, pred_lin)
    aic_lin, bic_lin = compute_aic_bic(q_arr, pred_lin, 2)
    results["linear"] = {
        "slope": float(coeffs_lin[0]),
        "intercept": float(coeffs_lin[1]),
        "R2": float(r2_lin),
        "AIC": float(aic_lin),
        "BIC": float(bic_lin),
        "predicted": pred_lin.tolist(),
    }

    # Power law: Q = a * K^alpha  (2 params)
    popt_pow, pred_pow = fit_power_law(k_arr, q_arr)
    if popt_pow is not None:
        r2_pow = compute_r2(q_arr, pred_pow)
        aic_pow, bic_pow = compute_aic_bic(q_arr, pred_pow, 2)
        results["power_law"] = {
            "a": float(popt_pow[0]),
            "alpha": float(popt_pow[1]),
            "R2": float(r2_pow),
            "AIC": float(aic_pow),
            "BIC": float(bic_pow),
            "predicted": pred_pow.tolist(),
        }

    # Quadratic in log: Q = a*(logK)^2 + b*logK + c  (3 params)
    coeffs_quad, pred_quad = fit_quadratic_log(log_k_arr, q_arr)
    r2_quad = compute_r2(q_arr, pred_quad)
    aic_quad, bic_quad = compute_aic_bic(q_arr, pred_quad, 3)
    results["quadratic_log"] = {
        "a": float(coeffs_quad[0]),
        "b": float(coeffs_quad[1]),
        "c": float(coeffs_quad[2]),
        "R2": float(r2_quad),
        "AIC": float(aic_quad),
        "BIC": float(bic_quad),
        "predicted": pred_quad.tolist(),
    }

    return results


# ──────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────

def make_4panel_figure(experiments: List[Dict], q_results: List[Dict],
                       model_fits: Dict, save_path: Path):
    """
    Panel A: Q_transition vs log(K) with linear + power-law fits
    Panel B: log(Q) vs log(K) — power-law test (should be a line if Q ∝ K^α)
    Panel C: Candidate loss curves for all K values (zoomed to transition)
    Panel D: Gradient norm profiles for all K values (zoomed to transition)
    """
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # ── Panel A: Q_transition vs log(K) ──
    ax_a = fig.add_subplot(gs[0, 0])

    valid = [(r["log_k"], r["Q_transition"], r["k"]) for r in q_results
             if r.get("Q_transition") is not None]
    if not valid:
        print("  [WARN] No valid Q_transition values for Panel A")
        ax_a.text(0.5, 0.5, "No data", transform=ax_a.transAxes, ha="center")
    else:
        log_ks = np.array([v[0] for v in valid])
        qs = np.array([v[1] for v in valid])
        ks_plot = [v[2] for v in valid]

        for lk, q, kv in zip(log_ks, qs, ks_plot):
            ax_a.scatter(lk, q, color=COLORS_BY_K.get(kv, "#333"),
                         s=60, zorder=5, edgecolors="white", linewidth=0.5)
            ax_a.annotate(f"K={kv}", (lk, q), textcoords="offset points",
                          xytext=(5, 5), fontsize=7)

        # Linear fit line
        lin = model_fits.get("Q_transition", {}).get("linear")
        if lin:
            x_fit = np.linspace(log_ks.min() - 0.1, log_ks.max() + 0.1, 100)
            y_fit = lin["slope"] * x_fit + lin["intercept"]
            ax_a.plot(x_fit, y_fit, "k--", alpha=0.7, linewidth=1.5,
                      label=f"Linear: R²={lin['R2']:.4f}")

        pow_fit = model_fits.get("Q_transition", {}).get("power_law")
        if pow_fit:
            k_fit = np.exp(x_fit)
            y_pow = pow_fit["a"] * np.power(k_fit, pow_fit["alpha"])
            ax_a.plot(x_fit, y_pow, "r:", alpha=0.5, linewidth=1,
                      label=f"Power: R²={pow_fit['R2']:.4f}")

        ax_a.set_xlabel("log(K)")
        ax_a.set_ylabel("Q_transition")
        ax_a.set_title("A: Dissipation vs log(K)")
        ax_a.legend(loc="upper left")

    # ── Panel B: log(Q) vs log(K) — power-law test ──
    ax_b = fig.add_subplot(gs[0, 1])

    if valid:
        # Filter to positive Q values for log transform
        pos = [(lk, q, kv) for lk, q, kv in zip(log_ks, qs, ks_plot) if q > 0]
        if pos:
            lk_pos = np.array([v[0] for v in pos])
            lq_pos = np.log(np.array([v[1] for v in pos]))
            kv_pos = [v[2] for v in pos]

            for lk, lq, kv in zip(lk_pos, lq_pos, kv_pos):
                ax_b.scatter(lk, lq, color=COLORS_BY_K.get(kv, "#333"),
                             s=60, zorder=5, edgecolors="white", linewidth=0.5)
                ax_b.annotate(f"K={kv}", (lk, lq), textcoords="offset points",
                              xytext=(5, 5), fontsize=7)

            # Linear fit in log-log space: log(Q) = alpha * log(K) + log(a)
            coeffs_ll = np.polyfit(lk_pos, lq_pos, 1)
            alpha_ll = coeffs_ll[0]
            x_ll = np.linspace(lk_pos.min() - 0.1, lk_pos.max() + 0.1, 100)
            y_ll = np.polyval(coeffs_ll, x_ll)
            pred_ll = np.polyval(coeffs_ll, lk_pos)
            ss_res = np.sum((lq_pos - pred_ll) ** 2)
            ss_tot = np.sum((lq_pos - np.mean(lq_pos)) ** 2)
            r2_ll = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            ax_b.plot(x_ll, y_ll, "r-", alpha=0.7, linewidth=1.5,
                      label=f"α={alpha_ll:.2f}, R²={r2_ll:.4f}")

            # Reference line for Landauer: slope = 1 (Q ∝ K^1 ≈ log-linear)
            # Actually log(Q) vs log(K) with slope 1 means Q ∝ K,
            # but the Landauer prediction is Q ∝ log(K), which is concave
            # in log-log space. Show it as a reference curve.
            k_ref = np.exp(x_ll)
            landauer_ref = np.log(np.maximum(
                lin["slope"] * x_ll + lin["intercept"], 1e-6)) if lin else None
            if landauer_ref is not None:
                ax_b.plot(x_ll, landauer_ref, "k--", alpha=0.5, linewidth=1,
                          label="Landauer (∝ log K)")

            ax_b.set_xlabel("log(K)")
            ax_b.set_ylabel("log(Q_transition)")
            ax_b.set_title(f"B: Log-log power-law test (slope α={alpha_ll:.2f})")
            ax_b.legend(loc="upper left")
        else:
            ax_b.text(0.5, 0.5, "No positive Q", transform=ax_b.transAxes, ha="center")
    else:
        ax_b.text(0.5, 0.5, "No data", transform=ax_b.transAxes, ha="center")

    # ── Panel C: Candidate loss curves (zoomed to transition region) ──
    ax_c = fig.add_subplot(gs[1, 0])

    # Determine x-axis limit: 2x the latest transition end, or 15K default
    t_ends = [r["transition_end"] for r in q_results
              if r.get("transition_end") is not None]
    x_max = int(max(t_ends) * 2) if t_ends else 15000
    x_max = max(x_max, 5000)  # at least 5K

    for exp in experiments:
        k = exp["k"]
        log_k = exp["log_k"]
        color = COLORS_BY_K.get(k, "#333")

        if exp["candidate_eval"] is not None:
            steps = exp["candidate_eval"]["steps"]
            cand_loss = exp["candidate_eval"]["candidate_loss"]
        elif "candidate_loss" in exp["history"]:
            steps = exp["history"]["steps"]
            cand_loss = exp["history"]["candidate_loss"]
        else:
            continue

        # Clip to transition region to avoid late instability spikes
        s_arr = np.array(steps)
        c_arr = np.array(cand_loss)
        mask = s_arr <= x_max
        ax_c.plot(s_arr[mask], c_arr[mask], color=color, linewidth=1.2,
                  label=f"K={k}", alpha=0.8)
        ax_c.axhline(log_k, color=color, linestyle=":", linewidth=0.5, alpha=0.4)

    ax_c.set_xlim(0, x_max)
    ax_c.set_xlabel("Training step")
    ax_c.set_ylabel("Candidate loss")
    ax_c.set_title("C: Phase transitions across K values")
    ax_c.legend(ncol=2, loc="upper right", fontsize=6)

    # ── Panel D: Gradient norm profiles (log y-axis, zoomed x) ──
    ax_d = fig.add_subplot(gs[1, 1])

    for exp in experiments:
        k = exp["k"]
        color = COLORS_BY_K.get(k, "#333")

        # Prefer training_history (eval_every=50) for 2x resolution vs checkpoints
        if "grad_norm_sq" in exp["history"]:
            steps = exp["history"]["steps"]
            gns = exp["history"]["grad_norm_sq"]
        elif exp["grad_norms"] is not None:
            steps = exp["grad_norms"]["steps"]
            gns = exp["grad_norms"]["total_grad_norm_sq"]
        else:
            continue

        s_arr = np.array(steps)
        g_arr = np.array(gns, dtype=float)
        mask = s_arr <= x_max
        # Clamp zeros for log scale
        g_plot = np.clip(g_arr[mask], 1e-4, None)
        ax_d.plot(s_arr[mask], g_plot, color=color, linewidth=0.9,
                  label=f"K={k}", alpha=0.8)

    ax_d.set_yscale("log")
    ax_d.set_xlim(0, x_max)
    ax_d.set_xlabel("Training step")
    ax_d.set_ylabel("||∇L||²")
    ax_d.set_title("D: Gradient norm profiles (log scale)")
    ax_d.legend(ncol=2, loc="upper right", fontsize=6)

    fig.savefig(save_path, bbox_inches="tight")
    png_path = save_path.with_suffix(".png")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")
    print(f"  Saved: {png_path}")


def make_small_multiples_figure(experiments: List[Dict], q_results: List[Dict],
                                save_path: Path):
    """
    2×5 small-multiples grid: each panel shows one K value with gradient
    norm (left y-axis, log scale) and candidate loss (right y-axis)
    overlaid. This makes the temporal correspondence between loss drop
    and gradient spike visually obvious for each K.
    """
    n = len(experiments)
    ncols = 5
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3.5 * nrows),
                             sharex=False)
    if nrows == 1:
        axes = axes.reshape(1, -1)

    # x-axis limit per experiment: 2x transition end
    t_end_map = {r["k"]: r.get("transition_end") for r in q_results}

    for idx, exp in enumerate(experiments):
        row, col = divmod(idx, ncols)
        ax_gn = axes[row, col]
        k = exp["k"]
        log_k = exp["log_k"]
        color = COLORS_BY_K.get(k, "#333")

        t_end = t_end_map.get(k)
        x_lim = int(t_end * 2.5) if t_end else 15000
        x_lim = max(x_lim, 3000)

        # Gradient norms from training history (highest resolution)
        if "grad_norm_sq" in exp["history"]:
            gn_steps = np.array(exp["history"]["steps"])
            gn_vals = np.array(exp["history"]["grad_norm_sq"], dtype=float)
        elif exp["grad_norms"] is not None:
            gn_steps = np.array(exp["grad_norms"]["steps"])
            gn_vals = np.array(exp["grad_norms"]["total_grad_norm_sq"], dtype=float)
        else:
            continue

        mask = gn_steps <= x_lim
        gn_plot = np.clip(gn_vals[mask], 1e-4, None)
        ax_gn.plot(gn_steps[mask], gn_plot, color="#E74C3C", linewidth=0.8,
                   alpha=0.85, label="||∇L||²")
        ax_gn.set_yscale("log")
        ax_gn.set_ylabel("||∇L||²", color="#E74C3C", fontsize=8)
        ax_gn.tick_params(axis="y", labelcolor="#E74C3C", labelsize=7)

        # Candidate loss on secondary y-axis
        ax_cl = ax_gn.twinx()

        if exp["candidate_eval"] is not None:
            cl_steps = np.array(exp["candidate_eval"]["steps"])
            cl_vals = np.array(exp["candidate_eval"]["candidate_loss"])
        elif "candidate_loss" in exp["history"]:
            cl_steps = np.array(exp["history"]["steps"])
            cl_vals = np.array(exp["history"]["candidate_loss"])
        else:
            cl_steps, cl_vals = np.array([]), np.array([])

        if len(cl_steps) > 0:
            cl_mask = cl_steps <= x_lim
            ax_cl.plot(cl_steps[cl_mask], cl_vals[cl_mask], color="#3498DB",
                       linewidth=1.0, alpha=0.85, label="Cand. loss")
            ax_cl.axhline(log_k, color="#3498DB", linestyle=":", linewidth=0.5,
                          alpha=0.4)
            ax_cl.set_ylabel("Cand. loss", color="#3498DB", fontsize=8)
            ax_cl.tick_params(axis="y", labelcolor="#3498DB", labelsize=7)

        # Mark transition window
        if t_end is not None:
            t_start = None
            for r in q_results:
                if r["k"] == k:
                    t_start = r.get("transition_start")
                    break
            if t_start is not None:
                ax_gn.axvspan(t_start, t_end, alpha=0.12, color="#F39C12")

        ax_gn.set_xlim(0, x_lim)
        ax_gn.set_title(f"K={k}  (log K={log_k:.2f})", fontsize=9)
        ax_gn.set_xlabel("Step", fontsize=8)
        ax_gn.tick_params(axis="x", labelsize=7)

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle("Gradient Norm & Candidate Loss per K (transition region)",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    png_path = save_path.with_suffix(".png")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")
    print(f"  Saved: {png_path}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze dense K-sweep Landauer scaling experiment"
    )
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--save-dir", type=str, default="outputs/paper_figures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load all experiments
    print("Loading experiments...")
    experiments = []
    for name in EXPERIMENT_NAMES:
        exp = load_experiment(name, output_dir)
        if exp is not None:
            experiments.append(exp)
    experiments.sort(key=lambda x: x["k"])

    if not experiments:
        print("ERROR: No experiments found. Check that training has completed.")
        return

    print(f"Loaded {len(experiments)} / {len(K_VALUES)} experiments")
    print(f"  K values: {[e['k'] for e in experiments]}")

    # Compute Q under all definitions
    print("\nComputing Q definitions...")
    q_results = []
    for exp in experiments:
        qr = compute_all_q_definitions(exp)
        q_results.append(qr)
        qt = qr.get("Q_transition")
        qtc = qr.get("Q_total_conv")
        qgs = qr.get("Q_gradspike")
        qzg = qr.get("Q_zgap")
        print(f"  K={qr['k']:>2}  Q_trans={qt if qt is not None else 'N/A':>12}"
              f"  Q_total_conv={qtc if qtc is not None else 'N/A':>12}"
              f"  Q_spike={qgs if qgs is not None else 'N/A':>12}"
              f"  Q_zgap={qzg if qzg is not None else 'N/A':>12}")

    # Model fitting for each Q definition
    print("\nFitting models...")
    all_fits = {}
    q_defs = ["Q_transition", "Q_total_conv", "Q_gradspike", "Q_zgap"]

    for q_name in q_defs:
        valid = [(r["k"], r["log_k"], r[q_name]) for r in q_results
                 if r.get(q_name) is not None]
        if len(valid) < 3:
            print(f"  {q_name}: insufficient data ({len(valid)} points)")
            continue

        k_arr = np.array([v[0] for v in valid], dtype=float)
        log_k_arr = np.array([v[1] for v in valid])
        q_arr = np.array([v[2] for v in valid])

        fits = run_model_comparison(k_arr, log_k_arr, q_arr)
        all_fits[q_name] = fits

        lin = fits["linear"]
        print(f"\n  {q_name} ({len(valid)} points):")
        print(f"    Linear:    slope={lin['slope']:.6f}  "
              f"intercept={lin['intercept']:.6f}  R²={lin['R2']:.4f}  "
              f"AIC={lin['AIC']:.2f}  BIC={lin['BIC']:.2f}")
        if "power_law" in fits:
            pw = fits["power_law"]
            print(f"    Power law: a={pw['a']:.6f}  alpha={pw['alpha']:.6f}  "
                  f"R²={pw['R2']:.4f}  AIC={pw['AIC']:.2f}  BIC={pw['BIC']:.2f}")
        ql = fits["quadratic_log"]
        print(f"    Quad log:  a={ql['a']:.6f}  b={ql['b']:.6f}  c={ql['c']:.6f}  "
              f"R²={ql['R2']:.4f}  AIC={ql['AIC']:.2f}  BIC={ql['BIC']:.2f}")

    # Evaluate success/failure criteria
    print("\n" + "=" * 70)
    print("ASSESSMENT")
    print("=" * 70)

    lin_fit = all_fits.get("Q_transition", {}).get("linear")
    if lin_fit:
        r2 = lin_fit["R2"]
        slope = lin_fit["slope"]
        if r2 > 0.95:
            print(f"  R² = {r2:.4f} > 0.95  ✓  Linear fit is strong")
        elif r2 > 0.90:
            print(f"  R² = {r2:.4f} ∈ [0.90, 0.95)  ~  Marginal")
        else:
            print(f"  R² = {r2:.4f} < 0.90  ✗  Linear fit is weak")

        # Compare AIC/BIC
        for alt in ["power_law", "quadratic_log"]:
            alt_fit = all_fits.get("Q_transition", {}).get(alt)
            if alt_fit:
                if lin_fit["AIC"] < alt_fit["AIC"]:
                    print(f"  AIC: Linear ({lin_fit['AIC']:.2f}) < {alt} ({alt_fit['AIC']:.2f})  ✓")
                else:
                    print(f"  AIC: Linear ({lin_fit['AIC']:.2f}) >= {alt} ({alt_fit['AIC']:.2f})  ✗")

        # Slope consistency across Q definitions
        slopes = []
        for q_name in q_defs:
            f = all_fits.get(q_name, {}).get("linear")
            if f:
                slopes.append((q_name, f["slope"]))
        if len(slopes) > 1:
            s_vals = [s for _, s in slopes]
            mean_s = np.mean(s_vals)
            spread = (max(s_vals) - min(s_vals)) / abs(mean_s) * 100
            print(f"\n  Slope consistency across Q definitions:")
            for name, s in slopes:
                print(f"    {name}: c = {s:.6f}")
            print(f"  Spread: {spread:.1f}% (target: <20%)")
    else:
        print("  No linear fit available for Q_transition")

    # Save full results
    results_output = {
        "experiments": [
            {"name": e["name"], "k": e["k"], "log_k": e["log_k"]}
            for e in experiments
        ],
        "q_results": q_results,
        "model_fits": {},
    }
    # Convert numpy arrays in model_fits to lists for JSON serialization
    for q_name, fits in all_fits.items():
        results_output["model_fits"][q_name] = {}
        for model_name, fit in fits.items():
            clean_fit = {}
            for fk, fv in fit.items():
                if isinstance(fv, np.ndarray):
                    clean_fit[fk] = fv.tolist()
                elif isinstance(fv, (np.floating, np.integer)):
                    clean_fit[fk] = float(fv)
                else:
                    clean_fit[fk] = fv
            results_output["model_fits"][q_name][model_name] = clean_fit

    results_path = output_dir / "landauer_dense_results.json"
    with open(results_path, "w") as f:
        json.dump(results_output, f, indent=2)
    print(f"\nSaved results to: {results_path}")

    # Generate figures
    print("\nGenerating figures...")
    fig_path = save_dir / "fig_landauer_dense.pdf"
    make_4panel_figure(experiments, q_results, all_fits, fig_path)

    fig_sm_path = save_dir / "fig_landauer_dense_multiples.pdf"
    make_small_multiples_figure(experiments, q_results, fig_sm_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
