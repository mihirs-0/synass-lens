#!/usr/bin/env python
"""
Paper-ready dataset-geometry figure for Q_transition.

Layout:
  A. eta_sweep n_b=1000 fixed: Q_transition vs log K, with apparent log-K
     law overlaid.  Shows the local pattern at fixed n_b.
  B. confound v2 fixed-D: Q_transition vs log K, two D values.  Shows
     sign reversal of the K-dependence under fixed D.
  C. confound v2 joint identifiable: Q_transition vs n_b · K^α with the
     fitted α=0.78 from M7.  Shows collapse with R² and AIC annotation.
  D. Model comparison: AIC bar plot for the five candidate forms in v2.

Reads:
  results/joint_nb_k_q_model.json
Writes:
  results/figures/dataset_geometry_q_transition.png
  results/dataset_geometry_q_transition_caption.md
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ETA_SWEEP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ETA_SWEEP_ROOT.parent))
from eta_sweep.config import RESULTS_DIR  # noqa: E402

IN_JSON = RESULTS_DIR / "joint_nb_k_q_model.json"
OUT_FIG = RESULTS_DIR / "figures" / "dataset_geometry_q_transition.png"
OUT_CAPTION = RESULTS_DIR / "dataset_geometry_q_transition_caption.md"

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
})


def main() -> None:
    raw = json.load(open(IN_JSON))
    eta_fits = raw["fits"].get("eta_sweep_eta1e-3")
    v2_fits = raw["fits"].get("confound_v2")
    if eta_fits is None or v2_fits is None:
        raise RuntimeError("missing required fits in joint_nb_k_q_model.json")

    df_eta = pd.DataFrame(eta_fits["data"])
    df_v2 = pd.DataFrame(v2_fits["data"])
    eta_M1 = eta_fits["models"]["M1_logK"]
    v2_M7 = v2_fits["models"]["M7_nb_Kpow"]
    alpha = float(v2_M7["alpha"])
    a_M7 = float(v2_M7["a"])
    b_M7 = float(v2_M7["b"])

    # --- Figure ----------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))

    # (A) eta_sweep n_b=1000 fixed
    ax = axes[0, 0]
    K = df_eta["K"].to_numpy(dtype=float)
    Q = df_eta["Q_transition"].to_numpy(dtype=float)
    ax.scatter(np.log(K), Q, s=80, color="#2E86AB",
               edgecolors="black", zorder=3, label="data (η=1e-3, K∈{10,15,20,25,36})")
    if eta_M1.get("c") is not None:
        K_grid = np.geomspace(K.min(), K.max(), 60)
        ax.plot(np.log(K_grid),
                eta_M1["c"] * np.log(K_grid) + eta_M1["b"],
                "--", color="#E76F51", linewidth=1.6,
                label=(f"Q = {eta_M1['c']:.2f}·log K + {eta_M1['b']:.2f}\n"
                       f"R² = {eta_M1['R2']:.2f}"))
    ax.set_xlabel("log K")
    ax.set_ylabel("Q_transition")
    ax.set_title("(A) eta_sweep, n_b = 1000 fixed\nApparent log-K law (held-out grad)",
                 fontsize=10)
    ax.legend(fontsize=8, loc="best")
    # Annotate K values
    for ki, qi in zip(K, Q):
        ax.annotate(f"K={int(ki)}", (math.log(ki), qi),
                    fontsize=7, xytext=(5, 4), textcoords="offset points")

    # (B) confound v2 fixed-D, both D values
    ax = axes[0, 1]
    cmap = plt.get_cmap("plasma")
    Ds = sorted(df_v2["D"].unique())
    for i, D in enumerate(Ds):
        sub = df_v2[df_v2["D"] == D].sort_values("K")
        K_b = sub["K"].to_numpy(dtype=float)
        Q_b = sub["Q_transition"].to_numpy(dtype=float)
        col = cmap(i / max(len(Ds) - 1, 1))
        D_label = "10k" if abs(D - 10000) < 50 else ("20k" if abs(D - 20000) < 50 else f"{D//1000}k")
        ax.scatter(np.log(K_b), Q_b, s=80, color=col,
                   edgecolors="black", zorder=3, label=f"D ≈ {D_label}")
        # Affine fit per fixed D
        if len(K_b) >= 2:
            logK_b = np.log(K_b)
            A = np.vstack([logK_b, np.ones_like(logK_b)]).T
            (c, b), *_ = np.linalg.lstsq(A, Q_b, rcond=None)
            yhat = c * logK_b + b
            ss_res = float(np.sum((Q_b - yhat) ** 2))
            ss_tot = float(np.sum((Q_b - Q_b.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            K_grid = np.geomspace(K_b.min(), K_b.max(), 50)
            ax.plot(np.log(K_grid), c * np.log(K_grid) + b,
                    "--", color=col, linewidth=1.4,
                    label=f"  c = {c:+.2f}, R² = {r2:.2f}")
        for ki, qi in zip(K_b, Q_b):
            ax.annotate(f"K={int(ki)}", (math.log(ki), qi),
                        fontsize=7, xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("log K")
    ax.set_ylabel("Q_transition")
    ax.set_title("(B) confound v2, fixed D (n_b ∝ 1/K)\nLog-K slope reverses sign at fixed D",
                 fontsize=10)
    ax.legend(fontsize=7, loc="best")

    # (C) v2 collapse onto n_b · K^α with α = 0.78
    ax = axes[1, 0]
    x_v2 = df_v2["n_b"].to_numpy(dtype=float) * np.power(df_v2["K"].to_numpy(dtype=float), alpha)
    Q_v2 = df_v2["Q_transition"].to_numpy(dtype=float)
    # Re-fit Q = a · x + b on this transformed axis to overlay best line.
    A = np.vstack([x_v2, np.ones_like(x_v2)]).T
    (a_lin, b_lin), *_ = np.linalg.lstsq(A, Q_v2, rcond=None)
    yhat = a_lin * x_v2 + b_lin
    ss_res = float(np.sum((Q_v2 - yhat) ** 2))
    ss_tot = float(np.sum((Q_v2 - Q_v2.mean()) ** 2))
    r2_lin = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    # Color points by D for visual reference.
    for i, D in enumerate(Ds):
        sub_idx = (df_v2["D"] == D).to_numpy()
        col = cmap(i / max(len(Ds) - 1, 1))
        D_label = "10k" if abs(D - 10000) < 50 else ("20k" if abs(D - 20000) < 50 else f"{D//1000}k")
        ax.scatter(x_v2[sub_idx], Q_v2[sub_idx], s=90, color=col,
                   edgecolors="black", zorder=3, label=f"D ≈ {D_label}")
        for j, idx in enumerate(np.where(sub_idx)[0]):
            ki = int(df_v2["K"].iloc[idx])
            ax.annotate(f"K={ki}", (x_v2[idx], Q_v2[idx]),
                        fontsize=7, xytext=(6, -8), textcoords="offset points")
    x_grid = np.linspace(x_v2.min(), x_v2.max(), 80)
    ax.plot(x_grid, a_lin * x_grid + b_lin, "--", color="#7B2CBF",
            linewidth=1.7,
            label=(f"Q = a · n_b · K^α + b\n"
                   f"α = {alpha:.3f} ± {v2_M7.get('alpha_se', 0):.3f}\n"
                   f"R² (M7) = {v2_M7['R2']:.3f},  AIC = {v2_M7['AIC']:.1f}"))
    ax.set_xlabel(f"n_b · K^α   (α = {alpha:.3f})")
    ax.set_ylabel("Q_transition")
    ax.set_title("(C) confound v2: joint collapse onto n_b · K^α\nfitted α from M7 = 0.78 ± 0.07",
                 fontsize=10)
    ax.legend(fontsize=7, loc="best")

    # (D) Model comparison bar plot — AIC for the 5 candidates in v2
    ax = axes[1, 1]
    model_keys = ["M1_logK", "M2_D", "M3_nb", "M5_nb_logK", "M7_nb_Kpow"]
    model_labels = [
        "Q ~ log K",
        "Q ~ D",
        "Q ~ n_b",
        "Q ~ n_b · log K",
        "Q ~ n_b · K^α",
    ]
    aics, r2s = [], []
    for mk in model_keys:
        m = v2_fits["models"].get(mk, {})
        aics.append(m.get("AIC"))
        r2s.append(m.get("R2"))
    # Plot AIC as bars; lower is better.  Highlight the winner.
    aics_clean = [a if a is not None else max(x for x in aics if x is not None) + 5 for a in aics]
    min_aic_idx = int(np.argmin(aics_clean))
    colors = ["#cccccc"] * len(aics_clean)
    colors[min_aic_idx] = "#7B2CBF"
    bars = ax.bar(range(len(aics_clean)), aics_clean, color=colors,
                  edgecolor="black", linewidth=0.8)
    ax.set_xticks(range(len(model_labels)))
    ax.set_xticklabels(model_labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("AIC (lower = better)")
    ax.set_title("(D) Model comparison on confound v2  (n=8 cells)",
                 fontsize=10)
    # Annotate R² and AIC above each bar.
    y_top = max(aics_clean) + 1
    for i, (a, r) in enumerate(zip(aics_clean, r2s)):
        r_s = f"{r:.2f}" if r is not None else "—"
        a_s = f"{a:.1f}" if a is not None else "—"
        ax.text(i, a + 0.5, f"AIC={a_s}\nR²={r_s}",
                ha="center", va="bottom", fontsize=7)
    ax.set_ylim(min(aics_clean) - 4, y_top + 5)
    ax.text(min_aic_idx, aics_clean[min_aic_idx] - 1.5,
            "★ winner", ha="center", color="#7B2CBF",
            fontsize=9, fontweight="bold")

    # Suptitle
    fig.suptitle(
        "Q_transition is not a function of log K alone.  "
        "The apparent log-K law at fixed n_b (A) reverses sign at fixed D (B).  "
        "Both regimes are explained by Q_transition ≈ a · n_b · K^α  (C, D).",
        fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG)
    plt.close(fig)
    print(f"wrote {OUT_FIG}")

    # --- Caption file ----------------------------------------------------
    cap = []
    cap.append("# Caption — `dataset_geometry_q_transition.png`")
    cap.append("")
    cap.append("**Figure X.  Dataset geometry, not ambiguity log K, governs the transition-phase work Q_transition.**")
    cap.append("")
    cap.append("**(A) Fixed-n_b view (eta_sweep, η = 10⁻³).**  Q_transition vs log K, K ∈ {10, 15, 20, 25, 36}, with n_b = 1000 held constant so D = 1000 · K varies linearly with K.  The five points are seed-averaged transition-phase Q values from the held-out-batch gradient norm; the dashed line is the affine fit Q_transition = "
               f"{eta_M1['c']:.2f} · log K + {eta_M1['b']:.2f}, R² = {eta_M1['R2']:.2f}.  In this regime Q_transition appears to follow a clean log-K law.")
    cap.append("")
    cap.append("**(B) Fixed-D view (confound v2, parent-repo controls with matching scheduler/task config).**  Q_transition vs log K at two fixed dataset sizes D ∈ {10 000, 20 000}, with n_b varied as D/K so K ∈ {5, 10, 20, 36} sweeps n_b ∈ {2 000, 1 000, 500, 278} at D = 10k and {4 000, 2 000, 1 000, 556} at D = 20k.  Per-D affine fits show **negative log-K slopes** (c = "
               "−5.8, R² = 0.28 at D = 10k; c = −41.6, R² = 0.74 at D = 20k).  The sign of the K-dependence flips when D rather than n_b is held constant — Q_transition is not a function of K alone.")
    cap.append("")
    cap.append(f"**(C) Joint collapse with n_b · K^α.**  Confound v2 cells plotted against the joint geometry variable n_b · K^α with α = {alpha:.3f} fitted by nonlinear regression (model M7 in panel D).  The eight cells from the two D values collapse onto a single ascending line, demonstrating that the fixed-n_b log-K law in (A) and the fixed-D sign reversal in (B) are two slices through the same n_b · K^α surface.  α = 0.776 ± {v2_M7.get('alpha_se', 0):.3f} from M7; corroborated by α = 0.876 ± 0.060 in confound v1 despite v1 using a different scheduler.")
    cap.append("")
    cap.append("**(D) Model comparison on confound v2 (n = 8).**  AIC bars for five candidate functional forms.  Lower AIC is better; the winner (★) is M7, Q_transition ≈ a · n_b · K^α + b.  Models that omit n_b (M1 \"Q ~ log K\", M2 \"Q ~ D\") fail decisively (AIC ≥ 56, R² ≤ 0.73), while joint forms with n_b succeed: M5 (Q ~ n_b · log K) achieves R² = 0.75 and M7 (Q ~ n_b · K^α) achieves R² = 0.92, ΔAIC = 7 over M5.  Adding a bare n_b term to M5 (giving M6) does not substantially improve the fit, confirming that the joint coupling — n_b multiplied by a K-dependent factor — is the operative geometry.")
    cap.append("")
    cap.append("**Take-away.**  The entropy gap log K sets the height of the plateau (candidate_loss saturates near log K), but the work done during the symmetry-breaking snap is not a function of log K alone.  The apparent log-K law observed at fixed n_b is a slice through a joint scaling Q_transition ≈ a · n_b · K^α with α ≈ 0.78–0.88 (sub-linear in K).  The plateau-decomposition headline 'Q_transition carries the log-K scaling' was therefore a fixed-n_b artefact; the underlying scaling involves dataset geometry (n_b, K) jointly, not ambiguity alone.")
    cap.append("")
    cap.append("**Caveats.**  Panels A and B/C/D use different gradient definitions (held-out unclipped vs training-batch with grad_clip = 1.0); absolute Q magnitudes therefore differ between A and B–D and the two should not be compared in absolute terms.  Confound v2 has n = 8 cells (single seed); α point estimate has SE = 0.073 (about 9% relative).  The fitted α value at η = 3 × 10⁻⁴ in eta_sweep with the K = 50/75/100 extension is α ≈ 2.4, much larger than confound v2's α ≈ 0.78 — this is driven by the K ≥ 75 break in Q vs log K and reflects a regime change at large K, not the underlying small-K scaling.")
    cap.append("")
    cap.append(f"**Files.** Figure: `{OUT_FIG.relative_to(RESULTS_DIR.parent)}`.  Underlying data: `results/joint_nb_k_q_model.json` and `results/d_vs_k_reconciliation.json`.")
    OUT_CAPTION.write_text("\n".join(cap))
    print(f"wrote {OUT_CAPTION}")


if __name__ == "__main__":
    main()
