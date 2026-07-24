#!/usr/bin/env python3
"""
Phase Boundary Analysis: Map η*(K) from training results.

Reads training histories from the phase boundary grid, determines success/fail
for each (K, η, seed) run, locates the critical boundary η*(K), and fits
the boundary scaling.

Produces 4 figures:
  1. Phase diagram — (K, η) grid colored by outcome
  2. Candidate loss trajectories near the boundary
  3. τ heatmap for successful runs
  4. η*(K) boundary with power-law fit

Usage:
    python scripts/analyze_phase_boundary.py
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
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
SAVE_DIR = OUTPUTS / "paper_figures"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

GRID = {
    5:  [5e-3, 7e-3, 1e-2, 1.5e-2, 2e-2],
    10: [3e-3, 5e-3, 7e-3, 1e-2],
    20: [2e-3, 3e-3, 5e-3, 7e-3],
    36: [1e-3, 2e-3, 3e-3, 5e-3],
}

LR_STR_MAP = {
    5e-3: "5e-3", 7e-3: "7e-3", 1e-2: "1e-2", 1.5e-2: "1.5e-2", 2e-2: "2e-2",
    3e-3: "3e-3", 1e-3: "1e-3", 2e-3: "2e-3",
}

SEEDS = [42, 123, 7]

# Prior-experiment name overrides (seed=42 runs that lived under different names).
# Only include runs that SUCCEEDED — the 200K-step re-runs of failures are separate.
PRIOR_OVERRIDES = {
    (20, 2e-3, 42): "lr_sweep_eta2e-3",     # τ≈7550, succeeded at 50K steps
    (36, 1e-3, 42): "landauer_dense_k36",    # τ≈6350, succeeded at 50K steps
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

def lr_to_str(lr):
    return LR_STR_MAP.get(lr, f"{lr:.0e}")


def run_name(k, lr, seed):
    return f"pb_K{k}_lr{lr_to_str(lr)}_s{seed}"


def find_history(k, lr, seed):
    """Find training_history.json, checking both pb_ names and prior overrides."""
    candidates = [
        OUTPUTS / run_name(k, lr, seed) / "training_history.json",
    ]
    override = PRIOR_OVERRIDES.get((k, lr, seed))
    if override:
        candidates.append(OUTPUTS / override / "training_history.json")

    for path in candidates:
        if path.exists():
            return path
    return None


def classify_run(steps, cand_loss, log_k, max_steps=200000):
    """
    Determine SUCCESS or FAIL for a single run.
    SUCCESS if EITHER:
      (a) final candidate_loss < 5% of log(K)  — stable convergence, OR
      (b) min candidate_loss < 1% of log(K)    — deep convergence was reached
          but later reverted due to unclipped gradient spikes.
    Condition (b) ensures runs that genuinely solved the task aren't penalised
    for post-convergence instability caused by the no-clipping regime.
    Also computes τ = t_end - t_start using 95%/5% thresholds.
    """
    hi = 0.95 * log_k
    lo = 0.05 * log_k
    deep = 0.01 * log_k

    min_loss = float(min(cand_loss))
    solved = cand_loss[-1] < lo or min_loss < deep

    # Find t_end: first step where loss < lo
    t_end_idx = None
    for i, l in enumerate(cand_loss):
        if l < lo:
            t_end_idx = i
            break

    if t_end_idx is None or not solved:
        return {
            "success": False,
            "tau": None,
            "t_start": None,
            "t_end": None,
            "final_loss": float(cand_loss[-1]),
            "max_step": float(steps[-1]),
        }

    t_end = float(steps[t_end_idx])

    # Find t_start: last step before t_end where loss > hi
    t_start = 0.0
    for i in range(t_end_idx):
        if cand_loss[i] > hi:
            t_start = float(steps[i])

    tau = t_end - t_start

    return {
        "success": True,
        "tau": tau,
        "t_start": t_start,
        "t_end": t_end,
        "final_loss": float(cand_loss[-1]),
        "max_step": float(steps[-1]),
    }


def load_all_results():
    """Load and classify every run in the grid."""
    results = {}
    n_found = 0
    n_missing = 0

    for k, lrs in sorted(GRID.items()):
        for lr in lrs:
            for seed in SEEDS:
                path = find_history(k, lr, seed)
                if path is None:
                    n_missing += 1
                    continue

                with open(path) as f:
                    h = json.load(f)

                steps = np.array(h["steps"], dtype=float)
                cand_loss = np.array(h["candidate_loss"], dtype=float)
                log_k = math.log(k)

                result = classify_run(steps, cand_loss, log_k)
                result["K"] = k
                result["lr"] = lr
                result["seed"] = seed
                result["steps"] = steps
                result["cand_loss"] = cand_loss
                result["log_k"] = log_k

                key = (k, lr, seed)
                results[key] = result
                n_found += 1

    print(f"Loaded {n_found} runs ({n_missing} missing)")
    return results


def aggregate(results):
    """Aggregate per-(K, η) across seeds."""
    agg = defaultdict(lambda: {
        "seeds": [], "successes": 0, "total": 0, "taus": [],
        "all_results": [],
    })

    for (k, lr, seed), r in results.items():
        key = (k, lr)
        agg[key]["seeds"].append(seed)
        agg[key]["total"] += 1
        agg[key]["all_results"].append(r)
        if r["success"]:
            agg[key]["successes"] += 1
            agg[key]["taus"].append(r["tau"])

    for key in agg:
        a = agg[key]
        a["K"] = key[0]
        a["lr"] = key[1]
        a["success_rate"] = a["successes"] / a["total"] if a["total"] > 0 else 0
        a["mean_tau"] = float(np.mean(a["taus"])) if a["taus"] else None
        a["std_tau"] = float(np.std(a["taus"])) if len(a["taus"]) >= 2 else None

    return dict(agg)


def find_boundary(agg):
    """
    For each K, determine η*(K):
      η_max_success = max η where all seeds succeed
      η_min_fail    = min η where at least one seed fails
      η*            = geometric mean of those two
    """
    K_values = sorted(set(k for k, _ in agg.keys()))
    boundary = {}

    for k in K_values:
        entries = [(lr, agg[(k, lr)]) for _, lr in agg if _ == k]
        if not entries:
            continue
        entries.sort(key=lambda x: x[0])

        max_all_succeed = None
        min_any_fail = None

        for lr, a in entries:
            if a["success_rate"] == 1.0:
                if max_all_succeed is None or lr > max_all_succeed:
                    max_all_succeed = lr
            if a["success_rate"] < 1.0:
                if min_any_fail is None or lr < min_any_fail:
                    min_any_fail = lr

        eta_star = None
        if max_all_succeed is not None and min_any_fail is not None:
            eta_star = math.sqrt(max_all_succeed * min_any_fail)
        elif max_all_succeed is not None:
            eta_star = max_all_succeed  # lower bound
        elif min_any_fail is not None:
            eta_star = min_any_fail  # upper bound

        boundary[k] = {
            "eta_star": eta_star,
            "max_all_succeed": max_all_succeed,
            "min_any_fail": min_any_fail,
        }

    return boundary


# ──────────────────────────────────────────────────────────────────────
# Figure 1: Phase diagram
# ──────────────────────────────────────────────────────────────────────

def plot_phase_diagram(ax, agg, boundary):
    """(K, η) grid colored by success rate."""
    for (k, lr), a in agg.items():
        rate = a["success_rate"]
        if rate == 1.0:
            color = "#2ECC71"
            marker = "o"
        elif rate == 0.0:
            color = "#E74C3C"
            marker = "X"
        else:
            color = "#F1C40F"
            marker = "D"

        # Size proportional to 1/τ for successful runs
        if a["mean_tau"] is not None and a["mean_tau"] > 0:
            size = max(30, min(200, 3000 / a["mean_tau"]))
        else:
            size = 60

        ax.scatter(k, lr, c=color, marker=marker, s=size, zorder=5,
                   edgecolors="white", linewidth=0.5)

        label = f"{a['successes']}/{a['total']}"
        ax.annotate(label, (k, lr), textcoords="offset points",
                    xytext=(8, 0), fontsize=6, va="center")

    # Plot boundary
    K_bnd = sorted(boundary.keys())
    eta_bnd = [boundary[k]["eta_star"] for k in K_bnd if boundary[k]["eta_star"] is not None]
    K_bnd_valid = [k for k in K_bnd if boundary[k]["eta_star"] is not None]

    if len(K_bnd_valid) >= 2:
        ax.plot(K_bnd_valid, eta_bnd, "k--", linewidth=2, alpha=0.7,
                label="η*(K) boundary", zorder=4)

    # Shade regions
    all_K = sorted(set(k for k, _ in agg.keys()))
    all_lr = sorted(set(lr for _, lr in agg.keys()))
    if all_K and all_lr:
        ax.fill_between([min(all_K) - 2, max(all_K) + 2],
                        min(all_lr) * 0.5, min(all_lr) * 0.5,
                        alpha=0)  # placeholder

    ax.set_yscale("log")
    ax.set_xlabel("K  (number of candidates)")
    ax.set_ylabel("η  (learning rate)")
    ax.set_title("Phase Diagram: Discoverable vs Undiscoverable")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ECC71',
               markersize=10, label='All succeed'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#F1C40F',
               markersize=10, label='Mixed'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='#E74C3C',
               markersize=10, label='All fail'),
    ]
    if len(K_bnd_valid) >= 2:
        legend_elements.append(Line2D([0], [0], color='k', linestyle='--',
                                       linewidth=2, label='η*(K)'))
    ax.legend(handles=legend_elements, loc="upper right", fontsize=7)


# ──────────────────────────────────────────────────────────────────────
# Figure 2: Candidate loss near boundary
# ──────────────────────────────────────────────────────────────────────

def plot_boundary_trajectories(axes, results, agg, boundary):
    """For each K, show candidate loss at η values straddling η*(K)."""
    K_values = sorted(set(r["K"] for r in results.values()))

    COLORS_BY_SEED = {42: "#E74C3C", 123: "#3498DB", 7: "#2ECC71"}

    for idx, k in enumerate(K_values):
        if idx >= len(axes):
            break
        ax = axes[idx]
        log_k = math.log(k)

        # Find η values near the boundary
        lrs = sorted(set(lr for (ki, lr) in agg if ki == k))

        for lr in lrs:
            a = agg.get((k, lr))
            if a is None:
                continue

            for r in a["all_results"]:
                seed = r["seed"]
                steps = r["steps"]
                cl = r["cand_loss"]
                color = COLORS_BY_SEED.get(seed, "#999")
                linestyle = "-" if r["success"] else "--"
                alpha = 0.7

                # Truncate display to 2× the transition or 50K for readability
                x_max = min(50000, float(steps[-1]))
                if r["t_end"] is not None:
                    x_max = min(x_max, r["t_end"] * 2)
                mask = steps <= x_max

                ax.plot(steps[mask], cl[mask], color=color,
                        linestyle=linestyle, linewidth=0.8, alpha=alpha,
                        label=f"η={lr:.0e} s={seed}" if seed == 42 else None)

        ax.axhline(log_k, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)
        ax.axhline(0.05 * log_k, color="green", linestyle=":", linewidth=0.5, alpha=0.3)
        ax.set_xlabel("Step")
        ax.set_ylabel("Candidate loss")
        ax.set_title(f"K={k}  (log K={log_k:.2f})")

        eta_star = boundary.get(k, {}).get("eta_star")
        if eta_star is not None:
            ax.set_title(f"K={k}  (η*≈{eta_star:.1e})")


# ──────────────────────────────────────────────────────────────────────
# Figure 3: τ heatmap
# ──────────────────────────────────────────────────────────────────────

def plot_tau_heatmap(ax, agg):
    """Heatmap of mean τ(K, η)."""
    K_all = sorted(set(k for k, _ in agg.keys()))
    lr_all = sorted(set(lr for _, lr in agg.keys()))

    tau_grid = np.full((len(lr_all), len(K_all)), np.nan)

    for i, lr in enumerate(lr_all):
        for j, k in enumerate(K_all):
            a = agg.get((k, lr))
            if a is not None and a["mean_tau"] is not None:
                tau_grid[i, j] = a["mean_tau"]

    # Use log scale for tau
    tau_log = np.log10(np.where(np.isnan(tau_grid), 1, tau_grid))
    tau_log[np.isnan(tau_grid)] = np.nan

    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad("lightgray")

    im = ax.imshow(tau_log, aspect="auto", cmap=cmap, origin="lower",
                    interpolation="nearest")

    ax.set_xticks(range(len(K_all)))
    ax.set_xticklabels([str(int(k)) for k in K_all])
    ax.set_yticks(range(len(lr_all)))
    ax.set_yticklabels([f"{lr:.0e}" for lr in lr_all])
    ax.set_xlabel("K")
    ax.set_ylabel("η")
    ax.set_title("Mean τ (log₁₀ scale; gray = fail)")

    # Annotate cells
    for i, lr in enumerate(lr_all):
        for j, k in enumerate(K_all):
            a = agg.get((k, lr))
            if a is not None:
                if a["mean_tau"] is not None:
                    txt = f"{a['mean_tau']:.0f}"
                else:
                    txt = "✗"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=6, color="white" if a["mean_tau"] and a["mean_tau"] > 5000 else "black")

    plt.colorbar(im, ax=ax, label="log₁₀(τ)", shrink=0.8)


# ──────────────────────────────────────────────────────────────────────
# Figure 4: η*(K) with fit
# ──────────────────────────────────────────────────────────────────────

def plot_boundary_fit(ax, boundary):
    """Log-log plot of η*(K) with power-law fit."""
    K_vals = []
    eta_vals = []
    for k, b in sorted(boundary.items()):
        if b["eta_star"] is not None:
            K_vals.append(k)
            eta_vals.append(b["eta_star"])

    if len(K_vals) < 2:
        ax.text(0.5, 0.5, "Insufficient boundary data",
                transform=ax.transAxes, ha="center")
        return {}

    K_arr = np.array(K_vals, dtype=float)
    eta_arr = np.array(eta_vals)

    ax.scatter(K_arr, eta_arr, c="#2980B9", s=80, zorder=5,
               edgecolors="white", linewidth=0.5)
    for ki, ei in zip(K_arr, eta_arr):
        ax.annotate(f"K={int(ki)}", (ki, ei), textcoords="offset points",
                    xytext=(8, 4), fontsize=8)

    fit_result = {}

    # Power-law fit: η* = a · K^(-δ)
    if len(K_arr) >= 3:
        try:
            def power_model(K, a, delta):
                return a * np.power(K, -delta)

            popt, pcov = curve_fit(power_model, K_arr, eta_arr, p0=[0.1, 0.5])
            a_fit, delta_fit = popt
            perr = np.sqrt(np.diag(pcov))

            K_smooth = np.linspace(K_arr.min() * 0.8, K_arr.max() * 1.2, 100)
            eta_smooth = power_model(K_smooth, *popt)

            y_pred = power_model(K_arr, *popt)
            ss_res = np.sum((eta_arr - y_pred) ** 2)
            ss_tot = np.sum((eta_arr - np.mean(eta_arr)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            ax.plot(K_smooth, eta_smooth, "r-", linewidth=2, alpha=0.7,
                    label=f"η* = {a_fit:.4f} · K^(-{delta_fit:.3f})\nR² = {r2:.4f}")

            fit_result = {
                "a": float(a_fit),
                "delta": float(delta_fit),
                "a_err": float(perr[0]),
                "delta_err": float(perr[1]),
                "R2": float(r2),
            }

            # Extrapolate K* for typical LRs
            for typical_eta in [1e-3, 3e-4]:
                if a_fit > 0 and delta_fit > 0:
                    K_star = (typical_eta / a_fit) ** (-1 / delta_fit)
                    ax.axhline(typical_eta, color="gray", linestyle=":",
                               linewidth=0.5, alpha=0.5)
                    ax.text(K_arr.max() * 0.9, typical_eta * 1.1,
                            f"η={typical_eta:.0e} → K*≈{K_star:.0f}",
                            fontsize=7, color="gray")
                    fit_result[f"K_star_at_{typical_eta:.0e}"] = float(K_star)

        except Exception as e:
            print(f"  Power-law fit failed: {e}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("K")
    ax.set_ylabel("η*(K)")
    ax.set_title("Critical Boundary η*(K)")
    ax.legend(loc="upper right", fontsize=8)

    return fit_result


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("PHASE BOUNDARY ANALYSIS")
    print("=" * 70)

    # Load all results
    results = load_all_results()

    if not results:
        print("\nNo results found. Have the training runs completed?")
        print("Expected directories: outputs/pb_K*_lr*_s*/training_history.json")
        return

    # Aggregate
    agg = aggregate(results)

    # Print outcome matrix
    print("\nOutcome matrix:")
    print(f"{'K':>4} {'η':>10} {'Rate':>8} {'Mean τ':>10} {'Seeds':>12}")
    print("-" * 50)
    for (k, lr) in sorted(agg.keys()):
        a = agg[(k, lr)]
        rate_str = f"{a['successes']}/{a['total']}"
        tau_str = f"{a['mean_tau']:.0f}" if a["mean_tau"] is not None else "—"
        seeds_str = ",".join(str(s) for s in sorted(a["seeds"]))
        print(f"{k:>4} {lr:>10.1e} {rate_str:>8} {tau_str:>10} {seeds_str:>12}")

    # Find boundary
    boundary = find_boundary(agg)
    print("\nBoundary η*(K):")
    for k, b in sorted(boundary.items()):
        if b["eta_star"] is not None:
            lo = f"all-succeed ≤ {b['max_all_succeed']:.1e}" if b['max_all_succeed'] else "?"
            hi = f"some-fail ≥ {b['min_any_fail']:.1e}" if b['min_any_fail'] else "?"
            print(f"  K={k:>3}: η* ≈ {b['eta_star']:.4e}  ({lo}, {hi})")
        else:
            print(f"  K={k:>3}: insufficient data")

    # ── Generate figures ──
    print("\nGenerating figures...")

    # Figure 1: Phase diagram (single panel, large)
    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    plot_phase_diagram(ax1, agg, boundary)
    fig1.savefig(SAVE_DIR / "fig_phase_diagram.png", bbox_inches="tight")
    fig1.savefig(SAVE_DIR / "fig_phase_diagram.pdf", bbox_inches="tight")
    plt.close(fig1)
    print("  Saved: fig_phase_diagram.png")

    # Figure 2: Boundary trajectories (one panel per K)
    K_values = sorted(set(r["K"] for r in results.values()))
    n_k = len(K_values)
    fig2, axes2 = plt.subplots(1, n_k, figsize=(5 * n_k, 4), squeeze=False)
    plot_boundary_trajectories(axes2[0], results, agg, boundary)
    fig2.suptitle("Candidate Loss Near Phase Boundary", fontsize=13, fontweight="bold")
    fig2.tight_layout()
    fig2.savefig(SAVE_DIR / "fig_phase_boundary_trajectories.png", bbox_inches="tight")
    fig2.savefig(SAVE_DIR / "fig_phase_boundary_trajectories.pdf", bbox_inches="tight")
    plt.close(fig2)
    print("  Saved: fig_phase_boundary_trajectories.png")

    # Figure 3: τ heatmap
    fig3, ax3 = plt.subplots(1, 1, figsize=(8, 5))
    plot_tau_heatmap(ax3, agg)
    fig3.savefig(SAVE_DIR / "fig_tau_heatmap.png", bbox_inches="tight")
    fig3.savefig(SAVE_DIR / "fig_tau_heatmap.pdf", bbox_inches="tight")
    plt.close(fig3)
    print("  Saved: fig_tau_heatmap.png")

    # Figure 4: Boundary fit
    fig4, ax4 = plt.subplots(1, 1, figsize=(7, 5))
    fit_result = plot_boundary_fit(ax4, boundary)
    fig4.savefig(SAVE_DIR / "fig_eta_star_boundary.png", bbox_inches="tight")
    fig4.savefig(SAVE_DIR / "fig_eta_star_boundary.pdf", bbox_inches="tight")
    plt.close(fig4)
    print("  Saved: fig_eta_star_boundary.png")

    # ── Summary table ──
    print("\n" + "=" * 70)
    print("PHASE BOUNDARY RESULTS")
    print("=" * 70)

    for k in sorted(set(ki for ki, _ in agg.keys())):
        print(f"\nK={k}:")
        for lr in sorted(set(lr for ki, lr in agg if ki == k)):
            a = agg[(k, lr)]
            tau_str = f"mean τ = {a['mean_tau']:.0f}" if a["mean_tau"] is not None else "never transitioned"
            print(f"  η={lr:.1e}:   {a['successes']}/{a['total']} succeed, {tau_str}")

        b = boundary.get(k, {})
        if b.get("eta_star"):
            print(f"  → η*({k}) ≈ {b['eta_star']:.4e}")

    if fit_result:
        print(f"\nBoundary fit: η*(K) = {fit_result['a']:.4f} · K^(-{fit_result['delta']:.3f})")
        print(f"  R² = {fit_result['R2']:.4f}")
        print(f"  δ = {fit_result['delta']:.3f} ± {fit_result.get('delta_err', 0):.3f}")
        if "K_star_at_1e-03" in fit_result:
            print(f"  Implied K* at η=1e-3: {fit_result['K_star_at_1e-03']:.0f}")
        if "K_star_at_3e-04" in fit_result:
            print(f"  Implied K* at η=3e-4: {fit_result['K_star_at_3e-04']:.0f}")

    # Sharpness assessment
    n_sharp = 0
    n_mushy = 0
    for k in sorted(set(ki for ki, _ in agg.keys())):
        lrs = sorted(set(lr for ki, lr in agg if ki == k))
        rates = [agg[(k, lr)]["success_rate"] for lr in lrs]
        has_mixed = any(0 < r < 1 for r in rates)
        if has_mixed:
            n_mushy += 1
        else:
            if len(set(r > 0 for r in rates)) > 1:
                n_sharp += 1

    print(f"\nBOUNDARY SHARPNESS:")
    print(f"  Sharp (all-or-none transitions): {n_sharp} K values")
    print(f"  Mushy (mixed outcomes):           {n_mushy} K values")

    print("\n" + "=" * 70)

    # ── Save JSON ──
    summary = {
        "grid_results": {},
        "boundary": {},
        "fit": fit_result,
    }

    for (k, lr), a in sorted(agg.items()):
        key = f"K={k},lr={lr:.1e}"
        summary["grid_results"][key] = {
            "K": k, "lr": lr,
            "successes": a["successes"], "total": a["total"],
            "success_rate": a["success_rate"],
            "mean_tau": a["mean_tau"], "std_tau": a["std_tau"],
            "seeds": sorted(a["seeds"]),
        }

    for k, b in sorted(boundary.items()):
        summary["boundary"][f"K={k}"] = {
            "eta_star": b["eta_star"],
            "max_all_succeed": b["max_all_succeed"],
            "min_any_fail": b["min_any_fail"],
        }

    summary_path = OUTPUTS / "phase_boundary_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
