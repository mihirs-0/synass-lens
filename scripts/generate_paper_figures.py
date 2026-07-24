#!/usr/bin/env python3
"""
Generate all figures for the results section of the paper.

Output directory: outputs/paper_figures/
Figure names match LaTeX \ref labels exactly:
    fig_loss_curves.pdf
    fig_dashboard.pdf
    fig_tautology.pdf
    fig_linear.pdf
    fig_arch_comparison.pdf
    fig_noise.pdf
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
SAVE_DIR = OUTPUTS / "paper_figures"
SAVE_DIR.mkdir(exist_ok=True)

COLORS = {
    2: "#E74C3C",
    3: "#E67E22",
    5: "#95A5A6",
    10: "#3498DB",
    15: "#9B59B6",
    20: "#E67E22",
    25: "#2ECC71",
    36: "#27AE60",
    50: "#1ABC9C",
    75: "#8E44AD",
    100: "#C0392B",
}

ARCH_COLORS = {
    "Transformer": "#3498DB",
    "Gated MLP": "#E74C3C",
    "RNN (LSTM)": "#27AE60",
}

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def load_history(name: str) -> Optional[Dict]:
    p = OUTPUTS / name / "training_history.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def load_json(path: Path) -> Optional[Dict]:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_candidate_eval(name: str) -> Optional[Dict]:
    """Load candidate_eval_results.json for *name*."""
    p = OUTPUTS / name / "candidate_eval_results.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: fig_loss_curves — K-sweep candidate loss overlay
# ─────────────────────────────────────────────────────────────────────────────
def make_fig_loss_curves():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2),
                             gridspec_kw={"width_ratios": [2, 1]})

    # Panel A: candidate loss curves
    ax = axes[0]
    experiments = [
        ("landauer_k5", 5),
        ("landauer_k10", 10),
        ("landauer_k20", 20),
        ("landauer_k36_60k", 36),
    ]
    for name, k in experiments:
        ce = load_candidate_eval(name)
        if ce is None:
            continue
        steps = np.array(ce["steps"])
        cl = np.array(ce["candidate_loss"])
        ax.plot(steps, cl, color=COLORS[k], label=f"$K={k}$", linewidth=1.5)
        ax.axhline(math.log(k), color=COLORS[k], linestyle=":", alpha=0.4,
                   linewidth=0.8)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Candidate Loss (nats)")
    ax.set_title("A.  Two-Phase Learning: Loss Plateaus at $\\log K$")
    ax.legend(loc="upper right")
    ax.set_ylim(-0.2, 4.5)

    # Panel B: z-gap evolution for K=10
    ax2 = axes[1]
    ce = load_candidate_eval("landauer_k10")
    if ce and "z_gap" in ce:
        steps = np.array(ce["steps"])
        zgap = np.array(ce["z_gap"])
        ax2.plot(steps, zgap, color=COLORS[10], linewidth=1.5)
        ax2.axhline(0, color="gray", linestyle="-", alpha=0.3)
        ax2.fill_between(steps, 0, zgap, alpha=0.15, color=COLORS[10])
        ax2.set_xlabel("Training Step")
        ax2.set_ylabel("$z$-gap (nats)")
        ax2.set_title("B.  $z$-Dependence Emerges ($K{=}10$)")

    fig.tight_layout(w_pad=3)
    fig.savefig(SAVE_DIR / "fig_loss_curves.pdf")
    fig.savefig(SAVE_DIR / "fig_loss_curves.png")
    plt.close(fig)
    print("  -> fig_loss_curves")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: fig_dashboard — mechanistic probes for K=20
# ─────────────────────────────────────────────────────────────────────────────
def make_fig_dashboard():
    exp = "temp_lr1e3_bs128_k20"
    h = load_history(exp)
    ce = load_candidate_eval(exp)
    probe_dir = OUTPUTS / exp
    K = 20

    fig = plt.figure(figsize=(13, 8.5))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.35)

    # --- Panel A: Candidate loss + train loss ---
    ax = fig.add_subplot(gs[0, 0])
    steps = np.array(h["steps"])
    tl = np.array(h["train_loss"])
    ax.plot(steps, tl, label="Train loss", color="#3498DB", linewidth=1.2)
    if ce:
        cand_steps = np.array(ce["steps"])
        cand_loss = np.array(ce["candidate_loss"])
        ax.plot(cand_steps, cand_loss, label="Candidate loss", color="#9B59B6",
                linewidth=1.2)
    ax.axhline(math.log(K), color="gray", linestyle=":", alpha=0.5,
               label=f"$\\log {K} = {math.log(K):.2f}$")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss (nats)")
    ax.set_title("A)  Training & Candidate Loss")
    ax.legend(fontsize=7)

    # --- Load probe data ---
    all_probes = load_json(probe_dir / "probe_results" / "all_probes.json")
    probe_steps_list = all_probes["steps"] if all_probes else []

    # --- Panel B: Attention to z ---
    ax = fig.add_subplot(gs[0, 1])
    if all_probes and "attention_to_z" in all_probes.get("probe_results", {}):
        att_raw = all_probes["probe_results"]["attention_to_z"]
        n_layers = None
        layer_means = {}
        for step_str in sorted(att_raw.keys(), key=int):
            entry = att_raw[step_str]
            att_by_layer = entry["attention_to_z"]
            if n_layers is None:
                n_layers = len(att_by_layer)
                for li in range(n_layers):
                    layer_means[li] = []
            for li in range(n_layers):
                heads = att_by_layer[li]
                layer_means[li].append(np.mean(heads) if isinstance(heads, list)
                                       else heads)
        probe_steps_arr = np.array(sorted(int(s) for s in att_raw.keys()))
        for li in range(n_layers):
            ax.plot(probe_steps_arr, layer_means[li], label=f"L{li}",
                    linewidth=1.2)
        ax.set_xlabel("Step")
        ax.set_ylabel("Attention to $z$")
        ax.set_title("B)  Attention to Selector ($z$)")
        ax.legend(fontsize=7, ncol=2)
    else:
        ax.text(0.5, 0.5, "Probe data not found", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="gray")
        ax.set_title("B)  Attention to Selector ($z$)")

    # --- Panel C: Logit lens ---
    ax = fig.add_subplot(gs[0, 2])
    if all_probes and "logit_lens" in all_probes.get("probe_results", {}):
        ll_raw = all_probes["probe_results"]["logit_lens"]
        n_layers = None
        layer_probs = {}
        for step_str in sorted(ll_raw.keys(), key=int):
            entry = ll_raw[step_str]
            probs = entry["correct_prob_by_layer"]
            if n_layers is None:
                n_layers = len(probs)
                for li in range(n_layers):
                    layer_probs[li] = []
            for li in range(n_layers):
                layer_probs[li].append(probs[li])
        probe_steps_arr = np.array(sorted(int(s) for s in ll_raw.keys()))
        layer_names = ["Embed"] + [f"Layer {i}" for i in range(1, n_layers)]
        for li in range(n_layers):
            style = "--" if li == 0 else "-"
            ax.plot(probe_steps_arr, layer_probs[li], label=layer_names[li],
                    linewidth=1.2, linestyle=style)
        ax.set_xlabel("Step")
        ax.set_ylabel("$P(\\mathrm{correct})$")
        ax.set_title("C)  Logit Lens: $P(\\mathrm{correct})$")
        ax.legend(fontsize=6, ncol=2)
    else:
        ax.text(0.5, 0.5, "Probe data not found", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="gray")
        ax.set_title("C)  Logit Lens: $P(\\mathrm{correct})$")

    # --- Panel D: z-gap (from candidate eval) ---
    ax = fig.add_subplot(gs[1, 0])
    if ce and "z_gap" in ce:
        cand_steps = np.array(ce["steps"])
        zgap = np.array(ce["z_gap"])
        ax.plot(cand_steps, zgap, color="#E74C3C", linewidth=1.2)
        ax.fill_between(cand_steps, 0, zgap, alpha=0.12, color="#E74C3C")
        ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
    ax.set_xlabel("Step")
    ax.set_ylabel("$z$-gap (nats)")
    ax.set_title("D)  $z$-Shuffle Diagnostic")

    # --- Panel E: Combined / overlay ---
    ax = fig.add_subplot(gs[1, 1:])
    acc = np.array(h.get("train_accuracy", [0]*len(steps)))
    ax.plot(steps, acc, label="Train accuracy", color="#3498DB",
            linewidth=1.5)

    if ce:
        cand_steps = np.array(ce["steps"])
        cand_loss = np.array(ce["candidate_loss"])
        cl_norm = 1.0 - (cand_loss - cand_loss.min()) / (cand_loss.max() - cand_loss.min() + 1e-12)
        ax.plot(cand_steps, cl_norm, label="Candidate loss (inv, norm)",
                color="#E67E22", linewidth=1.5)
        if "z_gap" in ce:
            zgap = np.array(ce["z_gap"])
            zgap_norm = zgap / (zgap.max() + 1e-12)
            ax.plot(cand_steps, zgap_norm, label="$z$-gap (norm)", color="#E74C3C",
                    linewidth=1.5)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Normalised Metric")
    ax.set_title("E)  Combined View: The Phase Transition ($K{=}20$)")
    ax.legend(fontsize=7, loc="center right")
    ax.set_ylim(-0.05, 1.1)

    fig.suptitle(
        f"Mechanistic Analysis Dashboard  —  $K={K}$,  $|\\mathcal{{B}}|=1000$",
        fontsize=13, y=1.01,
    )
    fig.savefig(SAVE_DIR / "fig_dashboard.pdf")
    fig.savefig(SAVE_DIR / "fig_dashboard.png")
    plt.close(fig)
    print("  -> fig_dashboard")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: fig_tautology — dissipation scaling (4 panels)
# ─────────────────────────────────────────────────────────────────────────────
def make_fig_tautology():
    taut = load_json(OUTPUTS / "tautology_check_results.json")
    decomp = load_json(OUTPUTS / "plateau_decomposition_results.json")
    regime = load_json(OUTPUTS / "regime_fit_results.json")

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # Panel A: Tautology test — Q_zgap vs ΔL_zgap
    ax = axes[0, 0]
    if taut and "experiments" in taut:
        data = taut["experiments"]
        q_vals = [d["Q_zgap"] for d in data]
        dl_vals = [d["delta_L_zgap"] for d in data]
        k_ints = [d["K"] for d in data]
        for q, dl, k in zip(q_vals, dl_vals, k_ints):
            ax.scatter(dl, q, s=120, color=COLORS.get(k, "gray"), zorder=5,
                       edgecolors="k", linewidth=0.5)
            ax.annotate(f"$K={k}$", (dl, q), fontsize=7,
                        xytext=(8, 5), textcoords="offset points")
        mx = max(max(q_vals), max(dl_vals)) * 1.1
        ax.plot([0, mx], [0, mx], "k--", alpha=0.3,
                label="$Q = \\Delta L$ (tautology)")
        ax.fill_between([0, mx], [0, mx], [0, 0], alpha=0.06, color="blue")
        ax.set_xlabel("$\\Delta \\mathcal{L}_{\\Delta_z}$ (loss decrease)")
        ax.set_ylabel("$Q_{\\Delta_z}$ (gradient dissipation)")
        ax.set_title("A.  Tautology Test: $Q$ vs $\\Delta\\mathcal{L}$")
        ax.legend(fontsize=7)

    # Panel B: Q_excess vs log(K)
    ax = axes[0, 1]
    if taut and "experiments" in taut:
        data = taut["experiments"]
        logk = [math.log(d["K"]) for d in data]
        q_excess = [d["Q_excess"] for d in data]
        k_ints = [d["K"] for d in data]

        for lk, qe, k in zip(logk, q_excess, k_ints):
            ax.scatter(lk, qe, s=120, color=COLORS.get(k, "gray"), zorder=5,
                       edgecolors="k", linewidth=0.5)
            ax.annotate(f"$K={k}$", (lk, qe), fontsize=7,
                        xytext=(8, 5), textcoords="offset points")

        if len(logk) >= 2:
            logk_arr = np.array(logk)
            qe_arr = np.array(q_excess)
            slope, intercept = np.polyfit(logk_arr, qe_arr, 1)
            ss_res = np.sum((qe_arr - (slope * logk_arr + intercept))**2)
            ss_tot = np.sum((qe_arr - qe_arr.mean())**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            x_fit = np.linspace(min(logk) - 0.2, max(logk) + 0.2, 100)
            ax.plot(x_fit, slope * x_fit + intercept, "k-", linewidth=1.5,
                    label=f"$Q_{{\\mathrm{{excess}}}} = {slope:.1f}\\,\\log K "
                          f"{intercept:+.1f}$\n$R^2 = {r2:.3f}$")
            ax.legend(fontsize=7)

        ax.set_xlabel("$\\log K$")
        ax.set_ylabel("$Q_{\\mathrm{excess}}$")
        ax.set_title("B.  Excess Dissipation vs $\\log K$")

    # Panel C: Plateau decomposition
    ax = axes[1, 0]
    if decomp and "experiments" in decomp:
        data = decomp["experiments"]
        k_labels = [f"$K={d['K']}$" for d in data]
        q_prior = [d.get("Q_prior", 0) for d in data]
        q_plat = [d.get("Q_plateau", 0) for d in data]
        q_trans = [d.get("Q_transition", 0) for d in data]

        x = np.arange(len(data))
        width = 0.55
        ax.bar(x, q_prior, width, label="$Q_{\\mathrm{prior}}$",
               color="#BDD7EE")
        ax.bar(x, q_plat, width, bottom=q_prior,
               label="$Q_{\\mathrm{plateau}}$", color="#6BAED6")
        ax.bar(x, q_trans, width,
               bottom=[p + pl for p, pl in zip(q_prior, q_plat)],
               label="$Q_{\\mathrm{transition}}$", color="#2171B5")
        ax.set_xticks(x)
        ax.set_xticklabels(k_labels)
        ax.set_ylabel("Dissipation $Q$")
        ax.set_title("C.  Plateau Decomposition")
        ax.legend(fontsize=7)

    # Panel D: Regime-aware fit
    ax = axes[1, 1]
    if regime:
        best = regime.get("Q_zgap", regime)
        k_star = best.get("best_K_star", best.get("best_k_star", 7.1))
        c = best.get("best_c", 120.3)
        r2 = best.get("best_r_squared", best.get("best_r2", 0.92))

        # Use the Q_zgap K_values and Q_values from regime fit for all K
        # including low-K points (K=2,3,5 etc.)
        regime_ks = best.get("K_values", [])
        regime_qs = best.get("Q_values", [])

        if regime_ks and regime_qs:
            logk_all = [math.log(k) for k in regime_ks]
            q_all = regime_qs
            k_ints_all = regime_ks
        elif taut and "experiments" in taut:
            data = taut["experiments"]
            logk_all = [math.log(d["K"]) for d in data]
            q_all = [d["Q_zgap"] for d in data]
            k_ints_all = [d["K"] for d in data]
        else:
            logk_all, q_all, k_ints_all = [], [], []

        if logk_all:
            for lk, q, k in zip(logk_all, q_all, k_ints_all):
                ax.scatter(lk, q, s=120, color=COLORS.get(k, "gray"),
                           zorder=5, edgecolors="k", linewidth=0.5)
                ax.annotate(f"$K={k}$", (lk, q), fontsize=7,
                            xytext=(8, 5), textcoords="offset points")

        ax.axvline(math.log(k_star), color="red", linestyle=":",
                   alpha=0.5, label=f"$K^* = {k_star:.1f}$")
        x_fit = np.linspace(math.log(k_star), 4.2, 100)
        y_fit = c * (x_fit - math.log(k_star))
        ax.plot(x_fit, y_fit, "k-", linewidth=1.5,
                label=f"$Q = {c:.1f}\\,\\log(K/K^*)$\n$R^2 = {r2:.3f}$")
        ax.set_xlabel("$\\log K$")
        ax.set_ylabel("$Q_{\\Delta_z}$")
        ax.set_title("D.  Regime-Aware Fit: $Q = c \\cdot \\log(K/K^*)$")
        ax.legend(fontsize=7)

    fig.suptitle("Gradient Dissipation During the Phase Transition",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(SAVE_DIR / "fig_tautology.pdf")
    fig.savefig(SAVE_DIR / "fig_tautology.png")
    plt.close(fig)
    print("  -> fig_tautology")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: fig_linear — linear capacity failure
# ─────────────────────────────────────────────────────────────────────────────
def make_fig_linear():
    import shutil

    src = OUTPUTS / "figures" / "twolayer_linear_analysis.png"
    if src.exists():
        shutil.copy2(src, SAVE_DIR / "fig_linear.png")

    lin_analysis = load_json(OUTPUTS / "twolayer_linear_analysis.json")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    if lin_analysis and "forward_results" in lin_analysis:
        fwd = lin_analysis["forward_results"]

        # Panel A: Summary bar chart — final candidate loss vs log K
        ax = axes[0]
        ks = [d["K"] for d in fwd]
        final_losses = [d["final_candidate_loss"] for d in fwd]
        logks = [math.log(k) for k in ks]
        x = np.arange(len(ks))
        bars = ax.bar(x, final_losses, 0.5,
                      color=[COLORS.get(k, "gray") for k in ks],
                      edgecolor="k", linewidth=0.5, alpha=0.8)
        for xi, lk, k in zip(x, logks, ks):
            ax.plot([xi - 0.3, xi + 0.3], [lk, lk], "k--", linewidth=1.0,
                    alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([f"$K={k}$" for k in ks])
        ax.set_ylabel("Final Candidate Loss (nats)")
        ax.set_title("A.  Loss Converges to $\\log K$ (Stuck)")
        ax.legend(["$\\log K$"], fontsize=7, loc="upper left")

        # Panel B: Final z-gap — essentially zero
        ax = axes[1]
        final_zgaps = [d["final_zgap"] for d in fwd]
        max_zgaps = [d["max_zgap"] for d in fwd]
        bars = ax.bar(x - 0.15, max_zgaps, 0.3,
                      color=[COLORS.get(k, "gray") for k in ks],
                      edgecolor="k", linewidth=0.5, alpha=0.5,
                      label="Max $z$-gap")
        bars2 = ax.bar(x + 0.15, final_zgaps, 0.3,
                       color=[COLORS.get(k, "gray") for k in ks],
                       edgecolor="k", linewidth=0.5, alpha=0.9,
                       label="Final $z$-gap")
        ax.set_xticks(x)
        ax.set_xticklabels([f"$K={k}$" for k in ks])
        ax.set_ylabel("$z$-gap (nats)")
        ax.set_title("B.  $z$-gap $\\approx 0$: Model Ignores $z$")
        ax.legend(fontsize=7)
        for xi, val in zip(x, max_zgaps):
            ax.text(xi - 0.15, val + 0.0003, f"{val:.4f}", ha="center",
                    fontsize=6)

        # Panel C: Final accuracy — stuck at 1/K (chance)
        ax = axes[2]
        accs = [d["final_accuracy"] for d in fwd]
        chance = [1.0 / k for k in ks]
        bars = ax.bar(x, accs, 0.5,
                      color=[COLORS.get(k, "gray") for k in ks],
                      edgecolor="k", linewidth=0.5, alpha=0.8)
        for xi, ch, k in zip(x, chance, ks):
            ax.plot([xi - 0.3, xi + 0.3], [ch, ch], "k--", linewidth=1.0,
                    alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([f"$K={k}$" for k in ks])
        ax.set_ylabel("Final Accuracy")
        ax.set_title("C.  Accuracy at Chance ($1/K$)")
        ax.legend(["$1/K$"], fontsize=7, loc="upper right")

    fig.suptitle("Two-Layer Linear Network: Permanent Capacity Failure",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(SAVE_DIR / "fig_linear.pdf")
    fig.savefig(SAVE_DIR / "fig_linear.png")
    plt.close(fig)
    print("  -> fig_linear")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: fig_arch_comparison — Transformer vs Gated MLP vs LSTM
# ─────────────────────────────────────────────────────────────────────────────
def make_fig_arch_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    scaling = load_json(OUTPUTS / "scaling_model_comparison.json")

    # Collect transition steps for each architecture
    arch_configs = {
        "Transformer": [
            ("landauer_k10", 10), ("landauer_k15", 15),
            ("landauer_k20", 20), ("landauer_k25", 25),
            ("landauer_k36_60k", 36),
        ],
        "Gated MLP": [
            ("gatedmlp_k10", 10), ("gatedmlp_k15", 15),
            ("gatedmlp_k20", 20), ("gatedmlp_k25", 25),
            ("gatedmlp_k36", 36), ("gatedmlp_k50", 50),
            ("gatedmlp_k75", 75), ("gatedmlp_k100", 100),
        ],
        "RNN (LSTM)": [
            ("rnn_k10", 10), ("rnn_k15", 15),
            ("rnn_k20", 20), ("rnn_k25", 25),
            ("rnn_k36", 36), ("rnn_k50", 50),
            ("rnn_k75", 75), ("rnn_k100", 100),
        ],
    }

    def find_transition(name, k):
        """Transition step: first step where candidate_loss < log(k)/2."""
        ce = load_candidate_eval(name)
        if ce is None:
            return None
        cl = np.array(ce["candidate_loss"])
        steps = np.array(ce["steps"])
        threshold = math.log(k) / 2
        for s, c in zip(steps, cl):
            if c < threshold:
                return int(s)
        return None

    # Panel A: Q_transition vs log(K) for all three architectures
    ax = axes[0]
    if scaling:
        for arch_name in ["Transformer", "Gated MLP", "RNN (LSTM)"]:
            arch_data = scaling.get(arch_name)
            if arch_data is None:
                continue
            color = ARCH_COLORS.get(arch_name, "gray")
            ks = arch_data.get("K_values", [])
            qs = arch_data.get("Q_values", [])
            if ks and qs:
                logks = [math.log(k) for k in ks]
                ax.scatter(logks, qs, s=80, color=color, zorder=5,
                           edgecolors="k", linewidth=0.5, label=arch_name)

                log_fit = arch_data.get("log", {})
                r2 = log_fit.get("R2", 0)
                preds = log_fit.get("predictions", [])
                if preds:
                    ax.plot(logks, preds, color=color, linestyle="--",
                            alpha=0.6, linewidth=1.2)
                elif len(logks) >= 2:
                    lk_a = np.array(logks)
                    q_a = np.array(qs)
                    slope, intercept = np.polyfit(lk_a, q_a, 1)
                    x_f = np.linspace(min(logks) - 0.1, max(logks) + 0.1, 100)
                    ax.plot(x_f, slope * x_f + intercept, color=color,
                            linestyle="--", alpha=0.6, linewidth=1.2)

    ax.set_xlabel("$\\log K$")
    ax.set_ylabel("$Q_{\\mathrm{transition}}$")
    ax.set_title("A.  Transition Dissipation vs $\\log K$")
    ax.legend(fontsize=7)

    # Panel B: Plateau duration vs K
    ax = axes[1]
    for arch_name, experiments in arch_configs.items():
        color = ARCH_COLORS[arch_name]
        ks_plot, durations = [], []
        for exp_name, k in experiments:
            ts = find_transition(exp_name, k)
            if ts is not None:
                ks_plot.append(k)
                durations.append(ts)
        if ks_plot:
            ax.plot(ks_plot, durations, "o-", color=color, linewidth=1.5,
                    markersize=6, label=arch_name)

    ax.set_xlabel("$K$")
    ax.set_ylabel("Transition Step")
    ax.set_title("B.  Plateau Duration vs $K$")
    ax.legend(fontsize=7)

    # Panel C: Candidate loss curves overlay for K=20 across architectures
    ax = axes[2]
    arch_k20 = {
        "Transformer": "landauer_k20",
        "Gated MLP": "gatedmlp_k20",
        "RNN (LSTM)": "rnn_k20",
    }
    for arch_name, exp_name in arch_k20.items():
        ce = load_candidate_eval(exp_name)
        if ce:
            steps = np.array(ce["steps"])
            cl = np.array(ce["candidate_loss"])
            ax.plot(steps, cl, color=ARCH_COLORS[arch_name], linewidth=1.5,
                    label=arch_name)
    ax.axhline(math.log(20), color="gray", linestyle=":", alpha=0.5,
               label="$\\log 20$")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Candidate Loss (nats)")
    ax.set_title("C.  Candidate Loss at $K{=}20$")
    ax.legend(fontsize=7)

    fig.suptitle(
        "Architecture Ablation: Is the Disambiguation Lag Universal?",
        fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(SAVE_DIR / "fig_arch_comparison.pdf")
    fig.savefig(SAVE_DIR / "fig_arch_comparison.png")
    plt.close(fig)
    print("  -> fig_arch_comparison")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: fig_noise — label noise experiment
# ─────────────────────────────────────────────────────────────────────────────
def make_fig_noise():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    experiments = [
        ("landauer_k20", 0.0, "#3498DB"),
        ("landauer_k20_noise0.05", 0.05, "#E67E22"),
        ("landauer_k20_noise0.10", 0.10, "#27AE60"),
        ("landauer_k20_noise0.20", 0.20, "#E74C3C"),
    ]

    K = 20

    # Panel A: Candidate loss curves
    ax = axes[0]
    for name, p, color in experiments:
        ce = load_candidate_eval(name)
        if ce is None:
            continue
        steps = np.array(ce["steps"])
        cl = np.array(ce["candidate_loss"])
        ax.plot(steps, cl, color=color, linewidth=1.5,
                label=f"$p={p:.2f}$" + (" (baseline)" if p == 0 else ""))
    ax.axhline(math.log(K), color="gray", linestyle=":", alpha=0.5,
               label=f"$\\log {K}$")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Candidate Loss (nats)")
    ax.set_title("A.  Loss Curves: Noise Accelerates Escape")
    ax.legend(fontsize=7)

    # Panel B: z-gap evolution (from candidate eval)
    ax = axes[1]
    for name, p, color in experiments:
        ce = load_candidate_eval(name)
        if ce is None or "z_gap" not in ce:
            continue
        steps = np.array(ce["steps"])
        zgap = np.array(ce["z_gap"])
        ax.plot(steps, zgap, color=color, linewidth=1.5,
                label=f"$p={p:.2f}$")
    ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("$z$-gap (nats)")
    ax.set_title("B.  $z$-Dependence Onset")
    ax.legend(fontsize=7)

    # Panel C: Summary bar chart — transition step and final z-gap
    ax = axes[2]
    p_vals = []
    trans_steps = []
    final_zgaps = []
    for name, p, color in experiments:
        ce = load_candidate_eval(name)
        if ce is None:
            continue
        steps = np.array(ce["steps"])
        cl = np.array(ce["candidate_loss"])
        threshold = math.log(K) / 2
        ts = None
        for s, c in zip(steps, cl):
            if c < threshold:
                ts = int(s)
                break
        p_vals.append(p)
        trans_steps.append(ts if ts else int(steps[-1]))
        if "z_gap" in ce:
            final_zgaps.append(float(ce["z_gap"][-1]))
        else:
            final_zgaps.append(0)

    colors_bar = [e[2] for e in experiments[:len(p_vals)]]
    x = np.arange(len(p_vals))
    width = 0.35
    bars1 = ax.bar(x - width/2, trans_steps, width, color=colors_bar,
                   alpha=0.7, edgecolor="k", linewidth=0.5,
                   label="Transition step")
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, final_zgaps, width, color=colors_bar,
                    alpha=0.35, edgecolor="k", linewidth=0.5, hatch="//",
                    label="Final $z$-gap")

    ax.set_xticks(x)
    ax.set_xticklabels([f"$p={p:.2f}$" for p in p_vals])
    ax.set_ylabel("Transition Step")
    ax2.set_ylabel("Final $z$-gap (nats)")
    ax.set_title("C.  Non-Monotonic Effect of Noise")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")

    for bar, val in zip(bars1, trans_steps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f"{val:,}", ha="center", va="bottom", fontsize=7)

    fig.suptitle(
        "Label Noise as a Probe of Saddle Geometry  —  $K{=}20$",
        fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(SAVE_DIR / "fig_noise.pdf")
    fig.savefig(SAVE_DIR / "fig_noise.png")
    plt.close(fig)
    print("  -> fig_noise")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Generating paper figures in {SAVE_DIR}/")
    print()
    make_fig_loss_curves()
    make_fig_dashboard()
    make_fig_tautology()
    make_fig_linear()
    make_fig_arch_comparison()
    make_fig_noise()
    print()
    print(f"Done. All figures saved to {SAVE_DIR}/")
