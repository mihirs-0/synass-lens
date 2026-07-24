#!/usr/bin/env python
"""
Model comparison: statistically distinguish log(K) from alternative scaling
functions for Q_transition vs K.

For each architecture, fits four models:
  1. Log:       Q = a * log(K) + b        (2 params — the Landauer hypothesis)
  2. Linear:    Q = a * K + b             (2 params)
  3. Power law: Q = a * K^alpha + b       (3 params)
  4. Sqrt:      Q = a * sqrt(K) + b       (2 params)

Reports R², AIC, BIC, and leave-one-out cross-validation error for each.
Produces a 2×3 model comparison figure (top: fits, bottom: residuals).

Usage:
    python scripts/compare_scaling_models.py [--output-dir outputs]
"""

import sys
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.compute_landauer_cost import process_experiment


# ------------------------------------------------------------------
# Architecture groups (mirrors plot_architecture_comparison.py)
# ------------------------------------------------------------------
ARCH_GROUPS = {
    "Transformer": {
        "experiments": [
            "landauer_k10", "landauer_k15", "landauer_k20",
            "landauer_k25", "landauer_k36",
            "landauer_k50", "landauer_k75", "landauer_k100", "landauer_k150",
        ],
        "color": "#2196F3",
        "marker": "o",
    },
    "Gated MLP": {
        "experiments": [
            "gatedmlp_k10", "gatedmlp_k15", "gatedmlp_k20",
            "gatedmlp_k25", "gatedmlp_k36",
            "gatedmlp_k50", "gatedmlp_k75", "gatedmlp_k100", "gatedmlp_k150",
        ],
        "color": "#FF5722",
        "marker": "s",
    },
    "RNN (LSTM)": {
        "experiments": [
            "rnn_k10", "rnn_k15", "rnn_k20",
            "rnn_k25", "rnn_k36",
            "rnn_k50", "rnn_k75", "rnn_k100", "rnn_k150",
        ],
        "color": "#4CAF50",
        "marker": "D",
    },
}

MODEL_STYLES = {
    "log":    {"color": "#1565C0", "ls": "-",  "label": "log(K)"},
    "linear": {"color": "#C62828", "ls": "--", "label": "linear"},
    "power":  {"color": "#6A1B9A", "ls": "-.", "label": "power law"},
    "sqrt":   {"color": "#E65100", "ls": ":",  "label": "√K"},
}


# ------------------------------------------------------------------
# Model comparison core
# ------------------------------------------------------------------
def compare_models(K_values, Q_values):
    """Compare scaling models for Q vs K data.

    Returns dict keyed by model name, each containing R², AIC, BIC,
    LOOCV, RSS, predictions.
    """
    K = np.array(K_values, dtype=float)
    Q = np.array(Q_values, dtype=float)
    n = len(K)

    models = {}

    # Model 1: Q = a * log(K) + b
    X_log = np.column_stack([np.log(K), np.ones(n)])
    coeffs_log = np.linalg.lstsq(X_log, Q, rcond=None)[0]
    Q_pred_log = X_log @ coeffs_log
    rss_log = np.sum((Q - Q_pred_log) ** 2)
    models["log"] = {"params": 2, "rss": rss_log, "pred": Q_pred_log,
                      "coeffs": coeffs_log, "func": lambda K, c=coeffs_log: c[0] * np.log(K) + c[1]}

    # Model 2: Q = a * K + b
    X_lin = np.column_stack([K, np.ones(n)])
    coeffs_lin = np.linalg.lstsq(X_lin, Q, rcond=None)[0]
    Q_pred_lin = X_lin @ coeffs_lin
    rss_lin = np.sum((Q - Q_pred_lin) ** 2)
    models["linear"] = {"params": 2, "rss": rss_lin, "pred": Q_pred_lin,
                         "coeffs": coeffs_lin, "func": lambda K, c=coeffs_lin: c[0] * K + c[1]}

    # Model 3: Q = a * K^alpha + b (power law)
    def power_law(K, a, alpha, b):
        return a * np.power(K, alpha) + b

    try:
        popt, _ = curve_fit(power_law, K, Q, p0=[1.0, 0.5, 0.0], maxfev=10000)
        Q_pred_pow = power_law(K, *popt)
        rss_pow = np.sum((Q - Q_pred_pow) ** 2)
        models["power"] = {"params": 3, "rss": rss_pow, "pred": Q_pred_pow,
                            "coeffs": popt, "func": lambda K, p=popt: power_law(K, *p)}
    except RuntimeError:
        models["power"] = {"params": 3, "rss": np.inf, "pred": np.full(n, np.nan),
                            "coeffs": None, "func": lambda K: np.full_like(K, np.nan)}

    # Model 4: Q = a * sqrt(K) + b
    X_sqrt = np.column_stack([np.sqrt(K), np.ones(n)])
    coeffs_sqrt = np.linalg.lstsq(X_sqrt, Q, rcond=None)[0]
    Q_pred_sqrt = X_sqrt @ coeffs_sqrt
    rss_sqrt = np.sum((Q - Q_pred_sqrt) ** 2)
    models["sqrt"] = {"params": 2, "rss": rss_sqrt, "pred": Q_pred_sqrt,
                       "coeffs": coeffs_sqrt, "func": lambda K, c=coeffs_sqrt: c[0] * np.sqrt(K) + c[1]}

    # Compute AIC, BIC, R², LOOCV for each
    ss_tot = np.sum((Q - np.mean(Q)) ** 2)
    results = {}
    for name, m in models.items():
        p = m["params"]
        rss = m["rss"]
        r2 = 1 - rss / ss_tot if ss_tot > 0 else 0
        aic = n * np.log(rss / n + 1e-30) + 2 * p
        bic = n * np.log(rss / n + 1e-30) + p * np.log(n)

        # Leave-one-out cross validation
        loocv = _loocv(K, Q, name)

        results[name] = {
            "R2": round(r2, 6),
            "AIC": round(aic, 4),
            "BIC": round(bic, 4),
            "LOOCV": round(loocv, 6),
            "RSS": round(rss, 8),
            "params": p,
            "predictions": m["pred"].tolist(),
            "func": m["func"],
        }

    return results


def _loocv(K, Q, model_name):
    """Leave-one-out cross-validation mean squared error."""
    n = len(K)
    errors = []
    for i in range(n):
        K_train = np.delete(K, i)
        Q_train = np.delete(Q, i)
        K_test = K[i]
        Q_test = Q[i]

        try:
            if model_name == "log":
                X = np.column_stack([np.log(K_train), np.ones(n - 1)])
                c = np.linalg.lstsq(X, Q_train, rcond=None)[0]
                pred = c[0] * np.log(K_test) + c[1]
            elif model_name == "linear":
                X = np.column_stack([K_train, np.ones(n - 1)])
                c = np.linalg.lstsq(X, Q_train, rcond=None)[0]
                pred = c[0] * K_test + c[1]
            elif model_name == "sqrt":
                X = np.column_stack([np.sqrt(K_train), np.ones(n - 1)])
                c = np.linalg.lstsq(X, Q_train, rcond=None)[0]
                pred = c[0] * np.sqrt(K_test) + c[1]
            elif model_name == "power":
                def plaw(k, a, alpha, b):
                    return a * np.power(k, alpha) + b
                popt, _ = curve_fit(plaw, K_train, Q_train, p0=[1.0, 0.5, 0.0],
                                    maxfev=10000)
                pred = plaw(K_test, *popt)
            else:
                pred = Q_test
            errors.append((Q_test - pred) ** 2)
        except (RuntimeError, np.linalg.LinAlgError):
            errors.append(np.inf)

    return float(np.mean(errors))


# ------------------------------------------------------------------
# Data gathering
# ------------------------------------------------------------------
def gather_results(output_dir):
    """Process all experiments and group by architecture."""
    all_results = {}
    for arch_name, info in ARCH_GROUPS.items():
        results = []
        for exp_name in info["experiments"]:
            exp_dir = Path(output_dir) / exp_name
            if not (exp_dir / "config.yaml").exists():
                continue
            result = process_experiment(exp_name, output_dir)
            if result is not None and result["Q_transition"] is not None:
                results.append(result)
        results.sort(key=lambda r: r["K"])
        all_results[arch_name] = results
        print(f"  {arch_name}: {len(results)} experiments with valid Q_transition")
    return all_results


# ------------------------------------------------------------------
# Printing
# ------------------------------------------------------------------
def print_comparison_table(arch_name, comparison):
    """Print a formatted comparison table for one architecture."""
    print(f"\n  {arch_name}")
    print(f"  {'Model':<12} {'R²':>8} {'AIC':>10} {'BIC':>10} {'LOOCV':>12} {'params':>6}")
    print(f"  {'-'*60}")
    best_aic = min(v["AIC"] for v in comparison.values())
    for name in ["log", "linear", "sqrt", "power"]:
        v = comparison[name]
        marker = " <-- best" if v["AIC"] == best_aic else ""
        print(f"  {name:<12} {v['R2']:>8.4f} {v['AIC']:>10.2f} {v['BIC']:>10.2f} "
              f"{v['LOOCV']:>12.6f} {v['params']:>6}{marker}")


# ------------------------------------------------------------------
# Figure: Model Comparison Panel (2×3 grid)
# ------------------------------------------------------------------
def plot_model_comparison(all_results, all_comparisons, save_path):
    """
    2×3 grid figure:
      Top row:    Q vs K with all four model fits overlaid (one per architecture)
      Bottom row: Residuals from log fit vs residuals from linear fit
    """
    arch_names = [a for a in ARCH_GROUPS if a in all_comparisons]
    n_arch = len(arch_names)
    if n_arch == 0:
        print("  No data for model comparison figure.")
        return

    fig, axes = plt.subplots(2, n_arch, figsize=(6 * n_arch, 9),
                             gridspec_kw={"height_ratios": [2, 1]})
    if n_arch == 1:
        axes = axes.reshape(2, 1)

    for col, arch_name in enumerate(arch_names):
        results = all_results[arch_name]
        comparison = all_comparisons[arch_name]
        K_arr = np.array([r["K"] for r in results], dtype=float)
        Q_arr = np.array([r["Q_transition"] for r in results])
        arch_color = ARCH_GROUPS[arch_name]["color"]
        arch_marker = ARCH_GROUPS[arch_name]["marker"]

        # --- Top panel: fits ---
        ax_top = axes[0, col]
        ax_top.scatter(K_arr, Q_arr, color=arch_color, marker=arch_marker,
                       s=80, zorder=10, edgecolors="black", linewidths=0.7,
                       label="data")

        K_smooth = np.linspace(K_arr.min() * 0.9, K_arr.max() * 1.05, 200)
        for model_name, style in MODEL_STYLES.items():
            if model_name in comparison and comparison[model_name]["R2"] > -10:
                func = comparison[model_name]["func"]
                try:
                    Q_fit = func(K_smooth)
                    aic_val = comparison[model_name]["AIC"]
                    ax_top.plot(K_smooth, Q_fit,
                                color=style["color"], ls=style["ls"], lw=1.8,
                                label=f"{style['label']} (AIC={aic_val:.1f})",
                                alpha=0.85)
                except Exception:
                    pass

        ax_top.set_title(arch_name, fontsize=12, fontweight="bold")
        ax_top.set_xlabel("K")
        if col == 0:
            ax_top.set_ylabel(r"$Q_{\mathrm{transition}}$")
        ax_top.legend(fontsize=7, loc="upper left")
        ax_top.grid(True, alpha=0.3)

        # --- Bottom panel: residuals (log vs linear) ---
        ax_bot = axes[1, col]
        for model_name, style in [("log", MODEL_STYLES["log"]),
                                   ("linear", MODEL_STYLES["linear"])]:
            if model_name in comparison:
                preds = np.array(comparison[model_name]["predictions"])
                residuals = Q_arr - preds
                ax_bot.scatter(K_arr, residuals, color=style["color"],
                               marker=arch_marker, s=50, label=style["label"],
                               edgecolors="black", linewidths=0.5, zorder=5)
                ax_bot.plot(K_arr, residuals, color=style["color"],
                            ls=style["ls"], lw=1, alpha=0.5)

        ax_bot.axhline(0, color="gray", ls="-", lw=0.8)
        ax_bot.set_xlabel("K")
        if col == 0:
            ax_bot.set_ylabel("Residual")
        ax_bot.set_title("Residuals: log vs linear", fontsize=9)
        ax_bot.legend(fontsize=7)
        ax_bot.grid(True, alpha=0.3)

    fig.suptitle("Model Comparison: Why log(K) and Not Something Else?",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved model comparison figure to: {save_path}")
    plt.close()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Model comparison: log(K) vs alternative scaling functions"
    )
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--save-path", type=str,
                        default="outputs/figures/model_comparison.png")
    args = parser.parse_args()

    print("Gathering Landauer results for model comparison...")
    all_results = gather_results(args.output_dir)

    print("\n" + "=" * 70)
    print("Scaling Model Comparison — AIC / BIC / LOOCV")
    print("=" * 70)

    all_comparisons = {}
    summary_for_json = {}

    for arch_name, results in all_results.items():
        if len(results) < 3:
            print(f"\n  {arch_name}: only {len(results)} points, need >= 3 for comparison")
            continue

        K_vals = [r["K"] for r in results]
        Q_vals = [r["Q_transition"] for r in results]
        comparison = compare_models(K_vals, Q_vals)

        all_comparisons[arch_name] = comparison
        print_comparison_table(arch_name, comparison)

        # JSON-safe version (drop the lambda)
        summary_for_json[arch_name] = {
            name: {k: v for k, v in m.items() if k != "func"}
            for name, m in comparison.items()
        }
        summary_for_json[arch_name]["K_values"] = K_vals
        summary_for_json[arch_name]["Q_values"] = Q_vals

    print("\n" + "=" * 70)

    # Determine overall winner per architecture
    for arch_name, comparison in all_comparisons.items():
        best = min(comparison, key=lambda m: comparison[m]["AIC"])
        delta_aic = comparison["linear"]["AIC"] - comparison["log"]["AIC"]
        print(f"\n  {arch_name}: best model by AIC = {best}")
        print(f"    ΔAIC(linear − log) = {delta_aic:+.2f}  "
              f"({'log preferred' if delta_aic > 0 else 'linear preferred' if delta_aic < 0 else 'tied'})")

    # Save JSON
    json_path = Path(args.output_dir) / "scaling_model_comparison.json"
    with open(json_path, "w") as f:
        json.dump(summary_for_json, f, indent=2)
    print(f"\nSaved JSON results to: {json_path}")

    # Plot
    if all_comparisons:
        plot_model_comparison(all_results, all_comparisons, args.save_path)


if __name__ == "__main__":
    main()
