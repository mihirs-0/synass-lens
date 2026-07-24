#!/usr/bin/env python
"""
Label noise analysis: testing Ziyin's entropic force prediction.

Ziyin predicts: if the Landauer scaling Q ∝ log(K) is driven by entropic
forces (the η·||∇L||² term), then adding label noise should AMPLIFY the
effect. More gradient noise → stronger entropic contribution → larger
dissipation → steeper Landauer slope.

This script:
  1. Loads gradient_norm_results.json and candidate_eval_results.json for
     each (K, p_noise) experiment.
  2. Computes Q_zgap for each experiment using z-gap window thresholds.
  3. For each noise level, fits Q_zgap vs log(K).
  4. Tests whether the Landauer constant c (slope) increases with noise.
  5. Generates diagnostic plots and a summary table.

Experiment naming convention:
  p_noise = 0.0:  landauer_k{K}       (existing experiments)
  p_noise > 0:    landauer_k{K}_noise{p_noise}
  (K=36 uses landauer_k36_60k as baseline)

Usage:
    python scripts/analyze_label_noise.py
    python scripts/analyze_label_noise.py --noise-levels 0.0 0.05 0.10 0.20
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════

BASELINE_NAMES = {
    10: "landauer_k10",
    15: "landauer_k15",
    20: "landauer_k20",
    25: "landauer_k25",
    36: "landauer_k36_60k",
}


def noise_experiment_name(K, p_noise):
    """Get the experiment directory name for a (K, p_noise) pair."""
    if p_noise == 0.0:
        return BASELINE_NAMES.get(K, f"landauer_k{K}")
    # Format noise level consistently: 0.05, 0.10, 0.20 (always 2 decimal places)
    noise_str = f"{p_noise:.2f}"
    return f"landauer_k{K}_noise{noise_str}"


def load_experiment(exp_name, output_dir):
    """Load gradient norms and candidate eval for one experiment."""
    exp_dir = Path(output_dir) / exp_name

    grad_path = exp_dir / "gradient_norm_results.json"
    cand_path = exp_dir / "candidate_eval_results.json"
    config_path = exp_dir / "config.yaml"

    if not grad_path.exists() or not cand_path.exists():
        return None

    with open(grad_path) as f:
        grad_data = json.load(f)
    with open(cand_path) as f:
        cand_data = json.load(f)

    # Try to load config for metadata
    config = None
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)

    return {
        "grad": grad_data,
        "cand": cand_data,
        "config": config,
    }


# ═══════════════════════════════════════════════════════════════════════
# Q_zgap computation (same as compute_landauer_comprehensive.py)
# ═══════════════════════════════════════════════════════════════════════

def find_zgap_window(cand_steps, z_gap):
    """
    zgap_start: first step where z_gap > 0.5
    zgap_end:   first step where z_gap > 0.9 × max(z_gap)
    """
    cand_steps = np.array(cand_steps, dtype=float)
    z_gap = np.array(z_gap, dtype=float)
    max_zgap = float(np.max(z_gap))

    zgap_start = None
    mask_start = z_gap > 0.5
    if np.any(mask_start):
        zgap_start = float(cand_steps[np.argmax(mask_start)])

    zgap_end = None
    threshold_end = 0.9 * max_zgap
    mask_end = z_gap > threshold_end
    if np.any(mask_end):
        zgap_end = float(cand_steps[np.argmax(mask_end)])

    if zgap_start is not None and zgap_end is not None:
        if zgap_start >= zgap_end:
            after = cand_steps >= zgap_start
            if np.any(after):
                idx_after = np.where(after)[0]
                best = idx_after[np.argmax(z_gap[idx_after])]
                zgap_end = float(cand_steps[best])
                if zgap_end <= zgap_start and best + 1 < len(cand_steps):
                    zgap_end = float(cand_steps[best + 1])

    return zgap_start, zgap_end, max_zgap


def compute_Q_zgap(grad_data, cand_data, lr):
    """
    Compute Q_zgap for a single experiment.

    Returns dict with Q_zgap, delta_L_zgap, Q_over_deltaL, window info,
    convergence status, and plateau duration.
    """
    grad_steps = np.array(grad_data["steps"], dtype=float)
    grad_norm_sq = np.array(grad_data["total_grad_norm_sq"], dtype=float)

    cand_steps = np.array(cand_data["steps"], dtype=float)
    candidate_loss = np.array(cand_data["candidate_loss"], dtype=float)
    z_gap = np.array(cand_data["z_gap"], dtype=float)

    K = int(cand_data["k"])
    log_K = math.log(K)

    # ── Convergence check ──
    converged = bool(candidate_loss[-1] < 0.5 * log_K)

    # ── Plateau duration ──
    low, high = 0.8 * log_K, 1.2 * log_K
    in_plateau = (candidate_loss >= low) & (candidate_loss <= high)
    max_plateau_duration = 0
    current_start = None
    for i in range(len(in_plateau)):
        if in_plateau[i]:
            if current_start is None:
                current_start = cand_steps[i]
        else:
            if current_start is not None:
                dur = cand_steps[i] - current_start
                max_plateau_duration = max(max_plateau_duration, dur)
                current_start = None
    if current_start is not None:
        dur = cand_steps[-1] - current_start
        max_plateau_duration = max(max_plateau_duration, dur)

    # ── Z-gap window ──
    zgap_start, zgap_end, max_zgap = find_zgap_window(cand_steps, z_gap)

    Q_zgap = None
    delta_L_zgap = None
    Q_over_deltaL = None

    if zgap_start is not None and zgap_end is not None and max_zgap >= 0.5:
        # Cumulative dissipation
        delta_s = np.diff(grad_steps, prepend=0)
        Q_cum = np.cumsum(lr * grad_norm_sq * delta_s)

        Q_start = float(np.interp(zgap_start, grad_steps, Q_cum))
        Q_end = float(np.interp(zgap_end, grad_steps, Q_cum))
        Q_zgap = Q_end - Q_start

        cl_start = float(np.interp(zgap_start, cand_steps, candidate_loss))
        cl_end = float(np.interp(zgap_end, cand_steps, candidate_loss))
        delta_L_zgap = cl_start - cl_end

        if delta_L_zgap > 0:
            Q_over_deltaL = Q_zgap / delta_L_zgap

    # Convergence step
    conv_step = None
    below = candidate_loss < 0.5
    count = 0
    for i, cl in enumerate(candidate_loss):
        if cl < 0.5:
            count += 1
            if count >= 10:
                conv_step = float(cand_steps[i - 9])
                break
        else:
            count = 0
    if conv_step is None and converged:
        conv_step = float(cand_steps[-1])

    return {
        "K": K,
        "log_K": round(log_K, 4),
        "converged": converged,
        "convergence_step": conv_step,
        "plateau_duration": float(max_plateau_duration),
        "Q_zgap": round(Q_zgap, 4) if Q_zgap is not None else None,
        "delta_L_zgap": round(delta_L_zgap, 4) if delta_L_zgap is not None else None,
        "Q_over_deltaL": round(Q_over_deltaL, 4) if Q_over_deltaL is not None else None,
        "zgap_start": zgap_start,
        "zgap_end": zgap_end,
        "max_zgap": round(max_zgap, 4),
    }


# ═══════════════════════════════════════════════════════════════════════
# Linear fit
# ═══════════════════════════════════════════════════════════════════════

def linear_fit(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    if len(x) < 2:
        return {"slope": None, "intercept": None, "r_squared": None}
    coeffs = np.polyfit(x, y, 1)
    slope, intercept = coeffs
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r2),
    }


# ═══════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════

NOISE_COLORS = {0.0: "#1f77b4", 0.05: "#ff7f0e", 0.10: "#2ca02c", 0.20: "#d62728"}


def plot_results(all_results, noise_levels, fits_by_noise, c_vs_noise,
                 raw_data, output_path):
    """Generate 5-panel label noise analysis figure."""
    if not HAS_MPL:
        print("matplotlib not available — skipping plots")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Label Noise Experiment: Ziyin Entropic Force Test",
                 fontsize=14, y=0.98)

    # ── Panel 1: candidate_loss vs step for K=20, all noise levels ──
    ax = axes[0, 0]
    target_K = 20
    # Fallback to K=10 if K=20 not available
    if not any(K == target_K for (_, K) in raw_data):
        target_K = 10
    for (p_noise, K), data in sorted(raw_data.items()):
        if K != target_K:
            continue
        c = NOISE_COLORS.get(p_noise, "gray")
        cand = data.get("cand", {})
        if cand:
            ax.plot(cand["steps"], cand["candidate_loss"],
                    label=f"p={p_noise}", color=c, alpha=0.8)
    ax.axhline(math.log(target_K), color="gray", linestyle=":", alpha=0.4,
               label=f"log({target_K})")
    ax.set_xlabel("Step")
    ax.set_ylabel("Candidate Loss")
    ax.set_title(f"K={target_K}: Candidate Loss vs Noise Level")
    ax.legend(fontsize=8)

    # ── Panel 2: Q_zgap vs log(K) per noise level ──
    ax = axes[0, 1]
    for p_noise in noise_levels:
        p_str = str(p_noise)
        if p_str not in all_results:
            continue
        results = all_results[p_str]
        valid = [r for r in results if r["Q_zgap"] is not None]
        if not valid:
            continue
        c = NOISE_COLORS.get(p_noise, "gray")
        lk = [r["log_K"] for r in valid]
        qq = [r["Q_zgap"] for r in valid]
        ax.scatter(lk, qq, color=c, marker="o", s=60, zorder=5,
                   label=f"p={p_noise}")
        # Regression line
        fit = fits_by_noise.get(p_str, {})
        if fit.get("slope") is not None and len(valid) >= 2:
            x_fit = np.linspace(min(lk) - 0.1, max(lk) + 0.1, 50)
            y_fit = fit["slope"] * x_fit + fit["intercept"]
            ax.plot(x_fit, y_fit, "--", color=c, alpha=0.5)
    ax.set_xlabel("log(K)")
    ax.set_ylabel("Q_zgap")
    ax.set_title("Landauer Scaling per Noise Level")
    ax.legend(fontsize=8)

    # ── Panel 3: Landauer constant c vs p_noise (THE MONEY PLOT) ──
    ax = axes[0, 2]
    if c_vs_noise["p_noise"] and c_vs_noise["c_values"]:
        valid = [(p, c, r) for p, c, r in
                 zip(c_vs_noise["p_noise"],
                     c_vs_noise["c_values"],
                     c_vs_noise["r_squared_values"])
                 if c is not None]
        if valid:
            pn = [v[0] for v in valid]
            cv = [v[1] for v in valid]
            r2 = [v[2] for v in valid]
            ax.plot(pn, cv, "o-", color="#1f77b4", markersize=8, linewidth=2)
            for i, (p, c, r) in enumerate(valid):
                ax.annotate(f"R²={r:.2f}", (p, c), fontsize=7,
                            textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("Label Noise Probability (p_noise)")
    ax.set_ylabel("Landauer Constant c (slope)")
    ax.set_title("Landauer Constant vs Noise")
    ax.axhline(120.3, color="gray", linestyle=":", alpha=0.4,
               label="Transformer baseline")
    ax.legend(fontsize=8)

    # ── Panel 4: Plateau duration vs K per noise level ──
    ax = axes[1, 0]
    for p_noise in noise_levels:
        p_str = str(p_noise)
        if p_str not in all_results:
            continue
        results = all_results[p_str]
        c = NOISE_COLORS.get(p_noise, "gray")
        K_vals = [r["K"] for r in results if r["converged"]]
        durations = [r["plateau_duration"] for r in results if r["converged"]]
        if K_vals:
            ax.plot(K_vals, durations, "o-", color=c, label=f"p={p_noise}",
                    alpha=0.8)
    ax.set_xlabel("K")
    ax.set_ylabel("Plateau Duration (steps)")
    ax.set_title("Plateau Duration vs K")
    ax.legend(fontsize=8)

    # ── Panel 5: Q/ΔL ratio vs K per noise level ──
    ax = axes[1, 1]
    for p_noise in noise_levels:
        p_str = str(p_noise)
        if p_str not in all_results:
            continue
        results = all_results[p_str]
        c = NOISE_COLORS.get(p_noise, "gray")
        valid = [r for r in results if r["Q_over_deltaL"] is not None]
        if valid:
            ax.plot([r["K"] for r in valid],
                    [r["Q_over_deltaL"] for r in valid],
                    "o-", color=c, label=f"p={p_noise}", alpha=0.8)
    ax.set_xlabel("K")
    ax.set_ylabel("Q/ΔL")
    ax.set_title("Dissipation Ratio vs K")
    ax.legend(fontsize=8)

    # ── Panel 6: z_gap vs step for K=20, all noise levels ──
    ax = axes[1, 2]
    for (p_noise, K), data in sorted(raw_data.items()):
        if K != target_K:
            continue
        c = NOISE_COLORS.get(p_noise, "gray")
        cand = data.get("cand", {})
        if cand:
            ax.plot(cand["steps"], cand["z_gap"],
                    label=f"p={p_noise}", color=c, alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("z-gap")
    ax.set_title(f"K={target_K}: z-gap vs Noise Level")
    ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════

def print_summary(all_results, noise_levels, fits_by_noise, c_vs_noise, verdict):
    print()
    print("Label Noise Experiment: Testing Ziyin's Entropic Force Prediction")
    print("══════════════════════════════════════════════════════════════")

    header = (f"{'p_noise':>7}  {'K=10 Q':>10}  {'K=20 Q':>10}  "
              f"{'K=36 Q':>10}  {'slope(c)':>10}  {'R²':>7}")
    print(header)
    print("-" * len(header))

    for p_noise in noise_levels:
        p_str = str(p_noise)
        results = all_results.get(p_str, [])
        Q_by_K = {r["K"]: r["Q_zgap"] for r in results}
        fit = fits_by_noise.get(p_str, {})

        def fmt_q(K):
            q = Q_by_K.get(K)
            return f"{q:>10.2f}" if q is not None else f"{'—':>10}"

        slope_str = f"{fit['slope']:>10.2f}" if fit.get("slope") else f"{'—':>10}"
        r2_str = f"{fit['r_squared']:>7.3f}" if fit.get("r_squared") else f"{'—':>7}"

        print(f"{p_noise:>7.2f}  {fmt_q(10)}  {fmt_q(20)}  "
              f"{fmt_q(36)}  {slope_str}  {r2_str}")

    print("══════════════════════════════════════════════════════════════")

    # Trend analysis
    valid_c = [(p, c) for p, c in
               zip(c_vs_noise.get("p_noise", []),
                   c_vs_noise.get("c_values", []))
               if c is not None]
    if len(valid_c) >= 2:
        c_vals = [c for _, c in valid_c]
        if c_vals[-1] > c_vals[0] * 1.2:
            trend = "INCREASING"
        elif c_vals[-1] < c_vals[0] * 0.8:
            trend = "DECREASING"
        else:
            trend = "APPROXIMATELY CONSTANT"
        print(f"Landauer constant c vs noise: {trend}")
        print(f"  c range: [{min(c_vals):.1f}, {max(c_vals):.1f}]")
    else:
        print("Landauer constant c vs noise: insufficient data")

    print(f"\nVERDICT: {verdict}")
    print("══════════════════════════════════════════════════════════════")
    print()


# ═══════════════════════════════════════════════════════════════════════
# Verdict determination
# ═══════════════════════════════════════════════════════════════════════

def determine_verdict(c_vs_noise):
    """Determine the scientific verdict from the Landauer constant vs noise."""
    valid_c = [(p, c) for p, c in
               zip(c_vs_noise.get("p_noise", []),
                   c_vs_noise.get("c_values", []))
               if c is not None]

    if len(valid_c) < 2:
        return ("INCONCLUSIVE: insufficient noise levels with valid scaling "
                "fits to determine trend.")

    c_vals = [c for _, c in valid_c]
    c_baseline = c_vals[0]
    c_max = max(c_vals[1:])  # highest among non-baseline

    if c_max > c_baseline * 1.2:
        return ("ENTROPIC: Landauer constant c increases with label noise, "
                "consistent with Ziyin's prediction that entropic forces "
                "(η·||∇L||²) drive the dissipation scaling. More gradient "
                "noise → stronger entropic contribution → larger c.")
    elif c_max < c_baseline * 0.8:
        return ("NOISE DISRUPTS: Landauer constant c decreases with label "
                "noise, suggesting the phenomenon is disrupted by noise "
                "rather than amplified. This is inconsistent with the "
                "entropic force mechanism.")
    else:
        return ("GEOMETRIC: Landauer constant c is approximately constant "
                "across noise levels. The dissipation scaling is driven by "
                "loss landscape geometry, not gradient noise. The entropic "
                "force mechanism is NOT the primary driver.")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Analyze label noise experiment results"
    )
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--K-values", type=int, nargs="+",
                        default=[10, 20, 36])
    parser.add_argument("--noise-levels", type=float, nargs="+",
                        default=[0.0, 0.05, 0.10, 0.20])
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate used in experiments")
    parser.add_argument("--output", type=str,
                        default="outputs/label_noise_analysis.json")
    parser.add_argument("--figure", type=str,
                        default="outputs/figures/label_noise_analysis.png")
    args = parser.parse_args()

    print("Label Noise Analysis")
    print(f"  K values: {args.K_values}")
    print(f"  Noise levels: {args.noise_levels}")
    print()

    # ── Load all experiments ──
    all_results = {}   # p_noise_str -> list of per-K results
    raw_data = {}      # (p_noise, K) -> {grad, cand} for plotting
    n_loaded = 0
    n_missing = 0

    for p_noise in args.noise_levels:
        p_str = str(p_noise)
        all_results[p_str] = []

        for K in args.K_values:
            exp_name = noise_experiment_name(K, p_noise)
            data = load_experiment(exp_name, args.output_dir)

            if data is None:
                print(f"  MISSING: {exp_name}")
                n_missing += 1
                continue

            n_loaded += 1
            print(f"  Loaded: {exp_name} (K={K}, p_noise={p_noise})")

            # Compute Q_zgap
            result = compute_Q_zgap(data["grad"], data["cand"], args.lr)
            result["p_noise"] = p_noise
            result["experiment"] = exp_name
            all_results[p_str].append(result)

            raw_data[(p_noise, K)] = data

    print(f"\nLoaded {n_loaded} experiments, {n_missing} missing")

    if n_loaded == 0:
        print("No experiments found. Run training first.")
        return

    # ── Fit Q_zgap vs log(K) for each noise level ──
    fits_by_noise = {}
    for p_noise in args.noise_levels:
        p_str = str(p_noise)
        results = all_results.get(p_str, [])
        valid = [r for r in results if r["Q_zgap"] is not None]
        if len(valid) >= 2:
            lk = [r["log_K"] for r in valid]
            qq = [r["Q_zgap"] for r in valid]
            fits_by_noise[p_str] = linear_fit(lk, qq)
        else:
            fits_by_noise[p_str] = {"slope": None, "intercept": None,
                                    "r_squared": None}

    # ── Landauer constant c vs p_noise ──
    c_vs_noise = {
        "p_noise": [],
        "c_values": [],
        "r_squared_values": [],
    }
    for p_noise in args.noise_levels:
        p_str = str(p_noise)
        fit = fits_by_noise.get(p_str, {})
        c_vs_noise["p_noise"].append(p_noise)
        c_vs_noise["c_values"].append(fit.get("slope"))
        c_vs_noise["r_squared_values"].append(fit.get("r_squared"))

    # ── Verdict ──
    verdict = determine_verdict(c_vs_noise)

    # ── Print summary ──
    print_summary(all_results, args.noise_levels, fits_by_noise,
                  c_vs_noise, verdict)

    # ── Plot ──
    fig_path = Path(args.figure)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plot_results(all_results, args.noise_levels, fits_by_noise,
                 c_vs_noise, raw_data, str(fig_path))

    # ── Save JSON ──
    output = {
        "noise_levels": args.noise_levels,
        "results_by_noise": {},
        "landauer_constant_vs_noise": c_vs_noise,
        "fits_by_noise": fits_by_noise,
        "verdict": verdict,
    }

    for p_noise in args.noise_levels:
        p_str = str(p_noise)
        results = all_results.get(p_str, [])
        output["results_by_noise"][p_str] = {
            "K_values": [r["K"] for r in results],
            "Q_zgap": [r["Q_zgap"] for r in results],
            "plateau_duration": [r["plateau_duration"] for r in results],
            "converged": [r["converged"] for r in results],
            "fit": fits_by_noise.get(p_str, {}),
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved results: {output_path}")


if __name__ == "__main__":
    main()
