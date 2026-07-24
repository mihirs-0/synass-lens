#!/usr/bin/env python3
"""
Experiment 2: Learning Rate Sweep Analysis

How does plateau duration τ depend on learning rate η?

Loads training histories for η ∈ {3e-4, 5e-4, 1e-3, 2e-3, 5e-3} at K=20,
computes τ, Q_transition, ḡ², and generates a 5-panel figure:

  A: τ vs η (log-log) with power-law fit τ = a·η^γ
  B: Q_transition vs η
  C: η·τ vs η (total parameter displacement)
  D: ḡ²_transition vs η (sanity check)
  E: Candidate loss curves overlaid for all η

Usage:
    python scripts/analyze_lr_sweep.py
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

K = 20
LOG_K = math.log(K)

LR_VALUES = [3e-4, 5e-4, 1e-3, 2e-3, 5e-3]
LR_LABELS = ["3e-4", "5e-4", "1e-3", "2e-3", "5e-3"]
EXPERIMENT_NAMES = [f"lr_sweep_eta{label}" for label in LR_LABELS]

COLORS_BY_LR = {
    3e-4: "#2980B9",
    5e-4: "#3498DB",
    1e-3: "#2ECC71",
    2e-3: "#E67E22",
    5e-3: "#E74C3C",
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

def load_experiment(name: str, lr: float) -> dict | None:
    """Load training history for a single LR run."""
    exp_dir = OUTPUTS / name
    history_path = exp_dir / "training_history.json"
    if not history_path.exists():
        print(f"  [SKIP] {name}: no training_history.json")
        return None
    with open(history_path) as f:
        history = json.load(f)
    return {
        "name": name,
        "lr": lr,
        "steps": np.array(history["steps"]),
        "grad_norm_sq": np.array(history["grad_norm_sq"]),
        "candidate_loss": np.array(history["candidate_loss"]),
        "loss_z_shuffled": np.array(history["loss_z_shuffled"]),
        "train_loss": np.array(history["train_loss"]),
    }


# ──────────────────────────────────────────────────────────────────────
# Transition detection (same thresholds as main analysis)
# ──────────────────────────────────────────────────────────────────────

def find_transition_window(steps, candidate_loss, log_k=LOG_K,
                           thresh_hi=0.95, thresh_lo=0.05):
    """
    t_end:   first step where candidate_loss < thresh_lo * log(K)
    t_start: last step BEFORE t_end where candidate_loss > thresh_hi * log(K)
    """
    hi = thresh_hi * log_k
    lo = thresh_lo * log_k

    t_end = t_end_idx = None
    for i, cl in enumerate(candidate_loss):
        if cl < lo:
            t_end = steps[i]
            t_end_idx = i
            break
    if t_end is None:
        return None, None, None, None

    t_start = t_start_idx = None
    for i in range(t_end_idx):
        if candidate_loss[i] > hi:
            t_start = steps[i]
            t_start_idx = i

    return t_start, t_end, t_start_idx, t_end_idx


def detect_divergence(steps, train_loss):
    """Check if the run diverged (NaN or loss explosion)."""
    for i, l in enumerate(train_loss):
        if math.isnan(l) or math.isinf(l):
            return True, int(steps[i])
        if i > 10 and l > 5 * train_loss[5]:
            return True, int(steps[i])
    return False, None


# ──────────────────────────────────────────────────────────────────────
# Metrics computation
# ──────────────────────────────────────────────────────────────────────

def compute_metrics(exp: dict) -> dict | None:
    """Compute τ, Q, ḡ² for a single LR run."""
    lr = exp["lr"]
    steps = exp["steps"]
    gns = exp["grad_norm_sq"]
    cand = exp["candidate_loss"]

    diverged, div_step = detect_divergence(steps, exp["train_loss"])
    if diverged:
        return {
            "lr": lr, "name": exp["name"],
            "diverged": True, "diverged_at_step": div_step,
        }

    t_start, t_end, t_start_idx, t_end_idx = find_transition_window(steps, cand)
    if t_start is None or t_end is None:
        return {
            "lr": lr, "name": exp["name"],
            "diverged": False, "no_transition": True,
        }

    tau = t_end - t_start

    # Q_transition: integral of η·‖∇L‖² over [t_start, t_end]
    mask_trans = (steps >= t_start) & (steps <= t_end)
    s_trans = steps[mask_trans]
    g_trans = gns[mask_trans]
    delta_trans = np.diff(s_trans, prepend=s_trans[0])
    delta_trans[0] = (s_trans[0] - t_start if s_trans[0] > t_start
                      else (s_trans[1] - s_trans[0] if len(s_trans) > 1 else 1))
    Q_transition = float(np.sum(lr * g_trans * delta_trans))

    # ḡ²_transition
    g_bar_sq_transition = float(np.mean(g_trans))

    # ḡ²_plateau: mean gradient norm from step 500 to t_start
    plateau_start = 500
    mask_plateau = (steps >= plateau_start) & (steps < t_start)
    if np.any(mask_plateau):
        g_bar_sq_plateau = float(np.mean(gns[mask_plateau]))
    else:
        mask_plateau = steps < t_start
        g_bar_sq_plateau = float(np.mean(gns[mask_plateau])) if np.any(mask_plateau) else g_bar_sq_transition

    eta_tau = lr * tau

    return {
        "lr": lr, "name": exp["name"],
        "diverged": False, "no_transition": False,
        "t_start": int(t_start), "t_end": int(t_end), "tau": int(tau),
        "Q_transition": Q_transition,
        "g_bar_sq_transition": g_bar_sq_transition,
        "g_bar_sq_plateau": g_bar_sq_plateau,
        "eta_tau": eta_tau,
    }


# ──────────────────────────────────────────────────────────────────────
# Fitting
# ──────────────────────────────────────────────────────────────────────

def fit_power_law(x, y):
    """Fit y = a * x^gamma. Returns (a, gamma, R², predicted)."""
    def model(x, a, gamma):
        return a * np.power(x, gamma)
    try:
        popt, _ = curve_fit(model, x, y, p0=[1.0, -1.0], maxfev=10000)
        pred = model(x, *popt)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return popt[0], popt[1], r2, pred
    except Exception:
        return None, None, None, None


def fit_linear_loglog(x, y):
    """Fit log(y) = gamma*log(x) + log(a). Returns (a, gamma, R²)."""
    lx, ly = np.log(x), np.log(y)
    coeffs = np.polyfit(lx, ly, 1)
    gamma, log_a = coeffs
    a = np.exp(log_a)
    pred = np.polyval(coeffs, lx)
    ss_res = np.sum((ly - pred) ** 2)
    ss_tot = np.sum((ly - np.mean(ly)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return a, gamma, r2


# ──────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────

def make_figure(experiments: list, metrics: list):
    """5-panel figure for the LR sweep."""
    valid = [m for m in metrics if not m.get("diverged") and not m.get("no_transition")]
    all_m = metrics

    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35)

    # ── Panel A: τ vs η (log-log) ──
    ax_a = fig.add_subplot(gs[0, 0])
    if len(valid) >= 2:
        lrs = np.array([m["lr"] for m in valid])
        taus = np.array([m["tau"] for m in valid], dtype=float)
        sort_idx = np.argsort(lrs)
        lrs, taus = lrs[sort_idx], taus[sort_idx]

        for m in valid:
            ax_a.scatter(m["lr"], m["tau"], color=COLORS_BY_LR[m["lr"]],
                         s=80, zorder=5, edgecolors="black", linewidths=0.5)

        a_fit, gamma, r2, pred = fit_power_law(lrs, taus)
        if a_fit is not None:
            lr_fine = np.logspace(np.log10(lrs.min() * 0.7), np.log10(lrs.max() * 1.3), 100)
            tau_fine = a_fit * np.power(lr_fine, gamma)
            ax_a.plot(lr_fine, tau_fine, "k--", linewidth=1.5, alpha=0.7,
                      label=f"$\\tau = {a_fit:.1f} \\cdot \\eta^{{{gamma:.2f}}}$\n$R^2 = {r2:.3f}$")

            # Reference lines
            if abs(gamma + 1) > 0.1:
                a_ref1 = taus[len(taus)//2] * lrs[len(lrs)//2]
                ax_a.plot(lr_fine, a_ref1 / lr_fine, ":", color="gray", alpha=0.4,
                          label="$\\tau \\propto 1/\\eta$ (γ=−1)")
            ax_a.plot(lr_fine, np.full_like(lr_fine, np.mean(taus)), ":",
                      color="lightblue", alpha=0.4,
                      label="$\\tau = $ const (γ=0)")

        # Mark diverged runs
        for m in all_m:
            if m.get("diverged"):
                ax_a.axvline(m["lr"], color=COLORS_BY_LR[m["lr"]], linestyle=":",
                             alpha=0.5, label=f"η={m['lr']:.0e} diverged")

        ax_a.set_xscale("log")
        ax_a.set_yscale("log")
        ax_a.set_xlabel("Learning rate η")
        ax_a.set_ylabel("Plateau duration τ (steps)")
        ax_a.set_title("A: τ vs η")
        ax_a.legend(fontsize=7)

    # ── Panel B: Q_transition vs η ──
    ax_b = fig.add_subplot(gs[0, 1])
    if len(valid) >= 2:
        lrs_v = np.array([m["lr"] for m in valid])
        Qs = np.array([m["Q_transition"] for m in valid])
        sort_idx = np.argsort(lrs_v)
        lrs_v, Qs = lrs_v[sort_idx], Qs[sort_idx]

        for m in valid:
            ax_b.scatter(m["lr"], m["Q_transition"], color=COLORS_BY_LR[m["lr"]],
                         s=80, zorder=5, edgecolors="black", linewidths=0.5)

        a_q, gamma_q, r2_q, _ = fit_power_law(lrs_v, Qs)
        if a_q is not None:
            lr_fine = np.logspace(np.log10(lrs_v.min() * 0.7), np.log10(lrs_v.max() * 1.3), 100)
            Q_fine = a_q * np.power(lr_fine, gamma_q)
            ax_b.plot(lr_fine, Q_fine, "k--", linewidth=1.5, alpha=0.7,
                      label=f"$Q \\propto \\eta^{{{gamma_q:.2f}}}$, $R^2={r2_q:.3f}$")

        ax_b.set_xscale("log")
        ax_b.set_yscale("log")
        ax_b.set_xlabel("Learning rate η")
        ax_b.set_ylabel("$Q_{\\mathrm{transition}}$ (η·‖∇L‖²·Δt)")
        ax_b.set_title("B: $Q_{\\mathrm{transition}}$ vs η")
        ax_b.legend(fontsize=7)

    # ── Panel C: η·τ vs η ──
    ax_c = fig.add_subplot(gs[0, 2])
    if len(valid) >= 2:
        lrs_v = np.array([m["lr"] for m in valid])
        eta_taus = np.array([m["eta_tau"] for m in valid])
        sort_idx = np.argsort(lrs_v)
        lrs_v, eta_taus = lrs_v[sort_idx], eta_taus[sort_idx]

        for m in valid:
            ax_c.scatter(m["lr"], m["eta_tau"], color=COLORS_BY_LR[m["lr"]],
                         s=80, zorder=5, edgecolors="black", linewidths=0.5)

        mean_eta_tau = np.mean(eta_taus)
        cv = np.std(eta_taus) / mean_eta_tau if mean_eta_tau > 0 else float("inf")
        ax_c.axhline(mean_eta_tau, color="gray", linestyle="--", alpha=0.5,
                      label=f"mean={mean_eta_tau:.1f}, CV={cv:.2f}")

        a_et, gamma_et, r2_et, _ = fit_power_law(lrs_v, eta_taus)
        if a_et is not None and abs(gamma_et) > 0.05:
            lr_fine = np.logspace(np.log10(lrs_v.min() * 0.7), np.log10(lrs_v.max() * 1.3), 100)
            et_fine = a_et * np.power(lr_fine, gamma_et)
            ax_c.plot(lr_fine, et_fine, "k--", linewidth=1.5, alpha=0.7,
                      label=f"$\\eta\\tau \\propto \\eta^{{{gamma_et:.2f}}}$, $R^2={r2_et:.3f}$")

        ax_c.set_xscale("log")
        ax_c.set_yscale("log")
        ax_c.set_xlabel("Learning rate η")
        ax_c.set_ylabel("η · τ (parameter displacement)")
        ax_c.set_title("C: η·τ vs η")
        ax_c.legend(fontsize=7)

    # ── Panel D: ḡ² vs η ──
    ax_d = fig.add_subplot(gs[1, 0])
    if len(valid) >= 2:
        lrs_v = np.array([m["lr"] for m in valid])
        g_trans = np.array([m["g_bar_sq_transition"] for m in valid])
        g_plat = np.array([m["g_bar_sq_plateau"] for m in valid])
        sort_idx = np.argsort(lrs_v)
        lrs_v_s = lrs_v[sort_idx]
        g_trans_s = g_trans[sort_idx]
        g_plat_s = g_plat[sort_idx]

        ax_d.scatter(lrs_v_s, g_trans_s, color="#E74C3C", s=70, zorder=5,
                     edgecolors="black", linewidths=0.5, label="ḡ² transition")
        ax_d.scatter(lrs_v_s, g_plat_s, color="#3498DB", s=70, zorder=5,
                     edgecolors="black", linewidths=0.5, marker="s", label="ḡ² plateau")

        ax_d.set_xscale("log")
        ax_d.set_yscale("log")
        ax_d.set_xlabel("Learning rate η")
        ax_d.set_ylabel("Mean ‖∇L‖²")
        ax_d.set_title("D: ḡ² vs η")
        ax_d.legend(fontsize=7)

    # ── Panel E: Candidate loss curves ──
    ax_e = fig.add_subplot(gs[1, 1:])
    for exp in experiments:
        if exp is None:
            continue
        lr = exp["lr"]
        label = f"η={lr:.0e}"
        ax_e.plot(exp["steps"], exp["candidate_loss"],
                  color=COLORS_BY_LR[lr], linewidth=1.2, alpha=0.85, label=label)

    ax_e.axhline(LOG_K, color="gray", linestyle=":", alpha=0.4, label=f"log(K)={LOG_K:.2f}")
    ax_e.axhline(0.05 * LOG_K, color="gray", linestyle="--", alpha=0.3, label="5% threshold")
    ax_e.set_xlabel("Training step")
    ax_e.set_ylabel("Candidate loss")
    ax_e.set_title(f"E: Candidate loss curves (K={K})")
    ax_e.legend(fontsize=7, loc="upper right")

    fig.suptitle(f"Experiment 2: Learning Rate Sweep at K={K}", fontsize=13, fontweight="bold", y=1.01)

    for fmt in ("pdf", "png"):
        path = SAVE_DIR / f"fig_lr_sweep.{fmt}"
        fig.savefig(path)
        print(f"  Saved: {path}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# Interpretation
# ──────────────────────────────────────────────────────────────────────

def interpret_gamma(gamma, r2):
    """Return a text interpretation of the exponent."""
    if r2 < 0.7:
        return "Poor fit — relationship may be non-monotonic or noisy."
    if gamma < -0.8:
        return ("γ ≈ −1 → DIFFUSIVE. τ ∝ 1/η. Fixed parameter displacement η·τ = const. "
                "The disambiguation circuit lies at a fixed distance from init; "
                "η just sets the step size for the random walk.")
    if gamma > -0.3:
        return ("γ ≈ 0 → STEP-COUNT. τ independent of η. "
                "Discovery requires a fixed number of gradient observations, "
                "not a fixed distance. Sample complexity dominates.")
    return ("γ between −1 and 0 → SUBLINEAR SPEEDUP. "
            "Partial benefit from larger steps; noise starts to hurt at high η.")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Experiment 2: Learning Rate Sweep Analysis")
    print("=" * 70)

    # Load all experiments
    experiments = []
    for name, lr in zip(EXPERIMENT_NAMES, LR_VALUES):
        exp = load_experiment(name, lr)
        experiments.append(exp)

    loaded = [e for e in experiments if e is not None]
    print(f"\nLoaded {len(loaded)}/{len(LR_VALUES)} experiments.")

    if len(loaded) < 2:
        print("[ERROR] Need at least 2 completed runs for analysis. Exiting.")
        return

    # Compute metrics
    print("\nComputing metrics...")
    metrics = []
    for exp in experiments:
        if exp is None:
            continue
        m = compute_metrics(exp)
        metrics.append(m)
        if m.get("diverged"):
            print(f"  η={m['lr']:.0e}: DIVERGED at step {m['diverged_at_step']}")
        elif m.get("no_transition"):
            print(f"  η={m['lr']:.0e}: No transition detected (may need more steps)")
        else:
            print(f"  η={m['lr']:.0e}: τ={m['tau']:,}  Q={m['Q_transition']:.2f}  "
                  f"η·τ={m['eta_tau']:.1f}  ḡ²_trans={m['g_bar_sq_transition']:.4f}  "
                  f"ḡ²_plat={m['g_bar_sq_plateau']:.4f}")

    valid = [m for m in metrics if not m.get("diverged") and not m.get("no_transition")]

    # Primary analysis: τ vs η
    if len(valid) >= 3:
        lrs = np.array([m["lr"] for m in valid])
        taus = np.array([m["tau"] for m in valid], dtype=float)
        sort_idx = np.argsort(lrs)
        lrs, taus = lrs[sort_idx], taus[sort_idx]

        a_fit, gamma, r2, _ = fit_power_law(lrs, taus)
        a_ll, gamma_ll, r2_ll = fit_linear_loglog(lrs, taus)

        print(f"\n{'='*70}")
        print("PRIMARY RESULT: τ vs η")
        print(f"{'='*70}")
        print(f"  Power-law fit (nonlinear):  τ = {a_fit:.1f} · η^{gamma:.3f}  R²={r2:.4f}")
        print(f"  Log-log linear fit:          γ = {gamma_ll:.3f}  R²={r2_ll:.4f}")
        print(f"\n  Interpretation: {interpret_gamma(gamma, r2)}")

        # η·τ analysis
        eta_taus = np.array([m["eta_tau"] for m in valid])
        mean_et = np.mean(eta_taus)
        cv_et = np.std(eta_taus) / mean_et if mean_et > 0 else float("inf")
        print(f"\n  η·τ values: {[f'{et:.1f}' for et in eta_taus[np.argsort(lrs)]]}")
        print(f"  η·τ mean = {mean_et:.1f}, CV = {cv_et:.3f}")
        if cv_et < 0.15:
            print("  → η·τ ≈ CONSTANT. Parameter displacement is conserved.")
        elif cv_et < 0.3:
            print("  → η·τ roughly constant (moderate variation).")
        else:
            print("  → η·τ NOT constant. Displacement grows or shrinks with η.")

    # Check for non-monotonicity (Kramers resonance)
    if len(valid) >= 4:
        lrs_sorted = np.array(sorted([m["lr"] for m in valid]))
        tau_sorted = np.array([next(m["tau"] for m in valid if m["lr"] == lr) for lr in lrs_sorted])
        diffs = np.diff(tau_sorted)
        if np.any(diffs > 0) and np.any(diffs < 0):
            print("\n  ⚠ NON-MONOTONIC τ(η) detected — possible Kramers resonance!")
            min_idx = np.argmin(tau_sorted)
            print(f"    Minimum τ at η={lrs_sorted[min_idx]:.0e}")

    # Save results
    results = {
        "experiment": "lr_sweep",
        "K": K,
        "log_K": LOG_K,
        "lr_values": LR_VALUES,
        "metrics": metrics,
    }
    if len(valid) >= 3:
        results["fit"] = {
            "gamma": float(gamma),
            "a": float(a_fit),
            "r2": float(r2),
            "gamma_loglog": float(gamma_ll),
            "r2_loglog": float(r2_ll),
            "interpretation": interpret_gamma(gamma, r2),
        }

    results_path = OUTPUTS / "lr_sweep_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved results to: {results_path}")

    # Generate figure
    print("\nGenerating figure...")
    make_figure(experiments, metrics)

    print("\nDone.")


if __name__ == "__main__":
    main()
