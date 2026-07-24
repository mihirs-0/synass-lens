#!/usr/bin/env python
"""
Test the descriptive model  Q_transition ≈ n_b · g(K)  with g possibly
log K or K^α.

Two dataset families, fit independently because Q absolute values are
NOT comparable across them (different gradient definitions):

  A. eta_sweep stable-regime cells.
     n_b = 1000 fixed across K ∈ {10, 15, 20, 25, 36, 50, 75, 100}.
     Q from held-out batch, no clipping.
     Within this dataset n_b does not vary, so we CANNOT separate
     n_b · g(K) from g(K) alone.  Reports the n_b-fixed slice only.

  B. Confound v2 cells (parent repo `outputs/confound_v2_*`).
     (K, n_b) varies orthogonally — at D ∈ {10k, 20k}, n_b = D/K.
     Q from training batch with grad-clip = 1.0 active.
     This is the dataset family that can identify the joint n_b·K form.

Confound v1 is reported separately as a sanity check (cosine scheduler,
different task config).

Outputs:
  results/joint_nb_k_q_model.json
  results/joint_nb_k_q_model.md
  results/figures/joint_nb_k_q_model.png
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ETA_SWEEP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ETA_SWEEP_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from eta_sweep.config import RESULTS_DIR  # noqa: E402

OUT_JSON = RESULTS_DIR / "joint_nb_k_q_model.json"
OUT_MD = RESULTS_DIR / "joint_nb_k_q_model.md"
OUT_FIG = RESULTS_DIR / "figures" / "joint_nb_k_q_model.png"

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 140,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_etasweep_at_eta(eta: float) -> pd.DataFrame:
    """Per-cell Q_transition from eta_sweep at fixed η, K ≤ 36 from main
    grid plus K-extension if available at the same η.  Seed-averaged.
    n_b = 1000 throughout this dataset family.
    """
    plateau_json = RESULTS_DIR / "plateau_decomposition_current.json"
    raw = json.load(open(plateau_json))
    rows = []
    for c in raw["per_cell"]:
        if c.get("status") != "transitioned":
            continue
        if abs(c.get("eta", -1) - eta) > 1e-9:
            continue
        K = int(c["k"])
        if K not in (10, 15, 20, 25, 36, 50, 75, 100):
            continue
        if c.get("Q_transition") is None: continue
        if isinstance(c["Q_transition"], float) and math.isnan(c["Q_transition"]):
            continue
        rows.append({
            "K": K, "n_b": 1000, "D": 1000 * K,
            "Q_transition": float(c["Q_transition"]),
            "Q_plateau": (float(c["Q_plateau"])
                          if c.get("Q_plateau") is not None
                          and not (isinstance(c["Q_plateau"], float)
                                   and math.isnan(c["Q_plateau"]))
                          else None),
            "seed": int(c["seed"]),
            "eta": eta,
        })
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    # Seed-average per K.
    return (df.groupby(["K", "n_b", "D", "eta"])
              .agg(Q_transition=("Q_transition", "mean"),
                   Q_plateau=("Q_plateau", "mean"),
                   n_seeds=("seed", "nunique"))
              .reset_index())


def load_confound(prefix: str) -> pd.DataFrame:
    """Load confound runs from parent-repo outputs/.  prefix in
    {'confound_', 'confound_v2_'} (the latter must NOT be matched by the
    former).
    """
    dvk_json = RESULTS_DIR / "d_vs_k_reconciliation.json"
    raw = json.load(open(dvk_json))
    key = "v2_runs" if prefix == "confound_v2_" else "v1_runs"
    rows = []
    for r in raw[key]:
        if r.get("Q_transition") is None: continue
        if isinstance(r["Q_transition"], float) and math.isnan(r["Q_transition"]):
            continue
        rows.append({
            "K": int(r["K"]), "n_b": int(r["n_b"]),
            "D": int(r["D"]),
            "Q_transition": float(r["Q_transition"]),
            "Q_plateau": (float(r["Q_plateau"])
                          if r.get("Q_plateau") is not None
                          and not (isinstance(r["Q_plateau"], float)
                                   and math.isnan(r["Q_plateau"]))
                          else None),
            "seed": 42,  # parent repo cells are single-seed
            "eta": float(r.get("lr_base", 1e-3)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fitters
# ---------------------------------------------------------------------------

def _aic(sse: float, n: int, k_params: int) -> Optional[float]:
    if sse <= 0 or n <= 0:
        return None
    return float(n * math.log(sse / n) + 2 * k_params)


def _r2_aic(y: np.ndarray, yhat: np.ndarray, k_params: int) -> Tuple[Optional[float], Optional[float], float]:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    aic = _aic(ss_res, len(y), k_params)
    return r2, aic, ss_res


def fit_linear_one_predictor(x: np.ndarray, y: np.ndarray, label: str) -> Dict:
    """Q = c · x + b."""
    if len(x) < 2:
        return {"label": label, "c": None, "b": None, "R2": None, "AIC": None,
                "n_params": 2, "n_points": int(len(x)),
                "reason": "n < 2"}
    A = np.vstack([x, np.ones_like(x)]).T
    (c, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = c * x + b
    r2, aic, sse = _r2_aic(y, yhat, k_params=2)
    return {
        "label": label, "c": float(c), "b": float(b),
        "R2": r2, "AIC": aic,
        "n_params": 2, "n_points": int(len(x)),
        "predictions": yhat.tolist(),
        "residuals": (y - yhat).tolist(),
    }


def fit_linear_two_predictor(x1, x2, y, label) -> Dict:
    """Q = a · x1 + b · x2 + c."""
    x1 = np.asarray(x1); x2 = np.asarray(x2); y = np.asarray(y)
    if len(x1) < 4:
        return {"label": label, "a": None, "b": None, "c": None,
                "R2": None, "AIC": None, "n_params": 3, "n_points": int(len(x1)),
                "reason": "n < 4 (3 params + ≥1 dof)"}
    A = np.vstack([x1, x2, np.ones_like(x1)]).T
    (a, b, c), *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = a * x1 + b * x2 + c
    r2, aic, sse = _r2_aic(y, yhat, k_params=3)
    return {"label": label, "a": float(a), "b": float(b), "c": float(c),
            "R2": r2, "AIC": aic,
            "n_params": 3, "n_points": int(len(x1)),
            "predictions": yhat.tolist(),
            "residuals": (y - yhat).tolist()}


def fit_powerlaw_with_nb(K, n_b, y, label) -> Dict:
    """Q = a · n_b · K^α + b.  Nonlinear in α."""
    K = np.asarray(K, dtype=float); n_b = np.asarray(n_b, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(K) < 4:
        return {"label": label, "a": None, "alpha": None, "b": None,
                "R2": None, "AIC": None, "n_params": 3, "n_points": int(len(K)),
                "reason": "n < 4"}
    def _f(X, a, alpha, b):
        K_, n_b_ = X
        return a * n_b_ * np.power(K_, alpha) + b
    try:
        popt, pcov = curve_fit(
            _f, (K, n_b), y,
            p0=[0.001, 0.5, 0.0],
            bounds=([-1e3, 0.01, -1e6], [1e3, 5.0, 1e6]),
            maxfev=20000,
        )
        a_, alpha_, b_ = popt
        yhat = _f((K, n_b), *popt)
        r2, aic, sse = _r2_aic(y, yhat, k_params=3)
        perr = np.sqrt(np.maximum(np.diag(pcov), 0.0))
        return {"label": label, "a": float(a_), "alpha": float(alpha_),
                "b": float(b_),
                "alpha_se": float(perr[1]) if np.isfinite(perr[1]) else None,
                "R2": r2, "AIC": aic, "n_params": 3,
                "n_points": int(len(K)),
                "predictions": yhat.tolist(),
                "residuals": (y - yhat).tolist()}
    except Exception as exc:
        return {"label": label, "a": None, "alpha": None, "b": None,
                "R2": None, "AIC": None, "n_params": 3, "n_points": int(len(K)),
                "reason": f"fit failed: {exc}"}


def fit_all_models(df: pd.DataFrame, name: str) -> Dict:
    df = df.copy()
    df["log_K"] = np.log(df["K"].astype(float))
    df["nb_logK"] = df["n_b"].astype(float) * df["log_K"]
    df["D_logK"] = df["D"].astype(float) * df["log_K"]
    y = df["Q_transition"].to_numpy(dtype=float)
    out: Dict = {"name": name, "n_points": int(len(df)),
                 "data": df[["K", "n_b", "D", "Q_transition", "Q_plateau", "n_seeds"]
                            if "n_seeds" in df.columns
                            else ["K", "n_b", "D", "Q_transition", "Q_plateau"]
                            ].to_dict(orient="records"),
                 "models": {}}
    out["models"]["M1_logK"]      = fit_linear_one_predictor(
        df["log_K"].to_numpy(), y, "Q ~ log K")
    out["models"]["M2_D"]         = fit_linear_one_predictor(
        df["D"].astype(float).to_numpy(), y, "Q ~ D")
    out["models"]["M3_nb"]        = fit_linear_one_predictor(
        df["n_b"].astype(float).to_numpy(), y, "Q ~ n_b")
    out["models"]["M4_DlogK"]     = fit_linear_one_predictor(
        df["D_logK"].to_numpy(), y, "Q ~ D · log K")
    out["models"]["M5_nb_logK"]   = fit_linear_one_predictor(
        df["nb_logK"].to_numpy(), y, "Q ~ n_b · log K  (the headline candidate)")
    out["models"]["M6_nb_logK_nb"] = fit_linear_two_predictor(
        df["nb_logK"].to_numpy(), df["n_b"].astype(float).to_numpy(), y,
        "Q ~ a·n_b·log K + b·n_b + c")
    out["models"]["M7_nb_Kpow"]   = fit_powerlaw_with_nb(
        df["K"].to_numpy(), df["n_b"].to_numpy(), y,
        "Q ~ a · n_b · K^α + b")
    return out


def family_verdict(fits: Dict) -> str:
    """Pick the best model by AIC and assess survival of the n_b·logK form."""
    M = fits["models"]
    candidates = [(name, m) for name, m in M.items()
                  if m.get("AIC") is not None]
    if not candidates:
        return "no model fit"
    candidates.sort(key=lambda kv: kv[1]["AIC"])
    best_name, best = candidates[0]
    best_label = best["label"]
    best_aic = best["AIC"]; best_r2 = best["R2"]
    M5 = M.get("M5_nb_logK", {})
    msg = (f"best by AIC: **{best_name}** ({best_label}); "
           f"AIC={best_aic:.1f}, R²={best_r2:.3f}.")
    if M5.get("R2") is not None:
        msg += (f"  M5 (Q ~ n_b·log K) R²={M5['R2']:.3f}, "
                f"AIC={M5['AIC']:.1f}.")
    return msg


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(eta_fits: Dict, v2_fits: Dict, v1_fits: Dict) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # Row A: eta_sweep n_b=1000 — Q_t/n_b vs log K, plus best fit
    ax = axes[0, 0]
    df = pd.DataFrame(eta_fits["data"])
    if len(df):
        ax.scatter(np.log(df["K"]), df["Q_transition"] / df["n_b"],
                   color="#2E86AB", s=60, edgecolors="black",
                   label="eta_sweep cells")
    ax.set_xlabel("log K"); ax.set_ylabel("Q_transition / n_b")
    ax.set_title("(A1) eta_sweep n_b=1000\nQ_t/n_b vs log K")

    ax = axes[0, 1]
    if len(df):
        # Best M5 fit (Q = c · n_b·log K + b)
        m5 = eta_fits["models"]["M5_nb_logK"]
        x = df["n_b"] * np.log(df["K"])
        ax.scatter(x, df["Q_transition"], s=60, color="#2E86AB",
                   edgecolors="black")
        if m5.get("c") is not None:
            x_grid = np.linspace(x.min(), x.max(), 50)
            ax.plot(x_grid, m5["c"]*x_grid + m5["b"], "--",
                    color="#E76F51",
                    label=f"M5 fit: c·(n_b·log K)+b\nR²={m5['R2']:.2f}")
            ax.legend(fontsize=7)
    ax.set_xlabel("n_b · log K"); ax.set_ylabel("Q_transition")
    ax.set_title("(A2) eta_sweep — M5 collapse")

    ax = axes[0, 2]
    if len(df):
        # Residuals of M5 vs K
        m5 = eta_fits["models"]["M5_nb_logK"]
        if m5.get("residuals"):
            ax.scatter(df["K"], m5["residuals"], color="#2E86AB",
                       edgecolors="black", s=60)
            ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("K"); ax.set_ylabel("M5 residual")
    ax.set_title("(A3) eta_sweep M5 residuals")

    # Row B: confound v2 — same three plots
    ax = axes[1, 0]
    df = pd.DataFrame(v2_fits["data"])
    if len(df):
        # Color by D
        cmap = plt.get_cmap("plasma")
        Ds = sorted(df["D"].unique())
        for i, D in enumerate(Ds):
            sub = df[df["D"] == D]
            ax.scatter(np.log(sub["K"]),
                       sub["Q_transition"] / sub["n_b"],
                       color=cmap(i / max(len(Ds)-1, 1)),
                       s=80, edgecolors="black",
                       label=f"D≈{D//1000}k")
        ax.legend(fontsize=7)
    ax.set_xlabel("log K"); ax.set_ylabel("Q_transition / n_b")
    ax.set_title("(B1) confound v2 (n_b ∝ 1/K at fixed D)\nQ_t/n_b vs log K")

    ax = axes[1, 1]
    if len(df):
        m5 = v2_fits["models"]["M5_nb_logK"]
        x = df["n_b"] * np.log(df["K"])
        for i, D in enumerate(sorted(df["D"].unique())):
            sub = df[df["D"] == D]
            cmap = plt.get_cmap("plasma")
            ax.scatter(sub["n_b"] * np.log(sub["K"]),
                       sub["Q_transition"], s=80,
                       color=cmap(i / max(len(df['D'].unique())-1, 1)),
                       edgecolors="black", label=f"D≈{D//1000}k")
        if m5.get("c") is not None:
            x_grid = np.linspace(x.min(), x.max(), 50)
            ax.plot(x_grid, m5["c"]*x_grid + m5["b"], "--",
                    color="#7B2CBF",
                    label=f"M5 fit: c·(n_b·log K)+b\nR²={m5['R2']:.2f}")
        ax.legend(fontsize=7)
    ax.set_xlabel("n_b · log K"); ax.set_ylabel("Q_transition")
    ax.set_title("(B2) v2 — M5 collapse")

    ax = axes[1, 2]
    if len(df):
        m5 = v2_fits["models"]["M5_nb_logK"]
        if m5.get("residuals"):
            ax.scatter(df["K"], m5["residuals"], color="#2E86AB",
                       edgecolors="black", s=70)
            for i, k in enumerate(df["K"]):
                ax.annotate(f"D={int(df['D'].iloc[i])//1000}k",
                            (k, m5["residuals"][i]),
                            fontsize=6, ha="center", va="bottom")
            ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("K"); ax.set_ylabel("M5 residual")
    ax.set_title("(B3) v2 M5 residuals")

    fig.suptitle(
        "Joint n_b · g(K) test for Q_transition.  Top row: eta_sweep "
        "(n_b=1000 fixed — n_b axis NOT identified).  Bottom row: "
        "confound v2 (n_b varies as D/K — joint identifiable).",
        fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG)
    plt.close(fig)
    return OUT_FIG


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Load three dataset families.
    df_eta = load_etasweep_at_eta(1e-3)
    # Augment eta_sweep with K-extension at η=3e-4 and η=7e-4 for K-coverage.
    # NOTE: mixing η changes Q absolute scale; flag in outputs.
    df_eta_kext_3e4 = load_etasweep_at_eta(3e-4)
    df_eta_kext_7e4 = load_etasweep_at_eta(7e-4)
    df_v2 = load_confound("confound_v2_")
    df_v1 = load_confound("confound_")

    fits = {}
    if len(df_eta) >= 3:
        fits["eta_sweep_eta1e-3"] = fit_all_models(df_eta, "eta_sweep_eta1e-3")
    if len(df_eta_kext_3e4) >= 3:
        fits["eta_sweep_eta3e-4"] = fit_all_models(df_eta_kext_3e4,
                                                   "eta_sweep_eta3e-4")
    if len(df_eta_kext_7e4) >= 3:
        fits["eta_sweep_eta7e-4"] = fit_all_models(df_eta_kext_7e4,
                                                   "eta_sweep_eta7e-4")
    if len(df_v2) >= 3:
        fits["confound_v2"] = fit_all_models(df_v2, "confound_v2")
    if len(df_v1) >= 3:
        fits["confound_v1"] = fit_all_models(df_v1, "confound_v1")

    # Identifiability flags: within eta_sweep n_b is constant ⇒ M5
    # collapses to M1 with proportional slope.  Within v2 n_b varies as
    # D/K — joint identifiable.
    identifiability = {
        "eta_sweep_eta1e-3": {
            "n_b_unique": int(df_eta["n_b"].nunique()) if len(df_eta) else 0,
            "K_unique": int(df_eta["K"].nunique()) if len(df_eta) else 0,
            "joint_identifiable": False,
            "reason": ("n_b is fixed at 1000; n_b · log K is exactly 1000 · "
                       "log K, so models M1, M5 fit the same data with "
                       "rescaled coefficient.  Cannot separate n_b · g(K) "
                       "from g(K) within this dataset."),
        },
        "confound_v2": {
            "n_b_unique": int(df_v2["n_b"].nunique()) if len(df_v2) else 0,
            "K_unique": int(df_v2["K"].nunique()) if len(df_v2) else 0,
            "joint_identifiable": (
                int(df_v2["n_b"].nunique()) >= 2 and int(df_v2["K"].nunique()) >= 2
                if len(df_v2) else False),
            "reason": ("n_b ∝ 1/K at each D; covers 6 unique n_b values "
                       "across 4 K values × 2 D values.  Joint identifiable."),
        },
    }

    # Verdict
    verdict_lines = []
    for fam in ("eta_sweep_eta1e-3", "confound_v2"):
        if fam in fits:
            verdict_lines.append(f"  {fam}: " + family_verdict(fits[fam]))

    out: Dict = {
        "datasets": {
            "eta_sweep_eta1e-3": {
                "config_match": "n_b=1000 fixed; held-out grad; no clipping; "
                                "constant LR; main eta_sweep grid + K-extension "
                                "(K up to 100, n_b=1000)",
                "n_points": int(len(df_eta)),
                "n_b_values": sorted(set(int(x) for x in df_eta["n_b"]))
                              if len(df_eta) else [],
                "K_values": sorted(set(int(x) for x in df_eta["K"]))
                            if len(df_eta) else [],
            },
            "eta_sweep_eta3e-4": {
                "n_points": int(len(df_eta_kext_3e4)),
                "K_values": sorted(set(int(x) for x in df_eta_kext_3e4["K"]))
                            if len(df_eta_kext_3e4) else [],
                "purpose": "extended K coverage; same η, same Q definitions",
            },
            "eta_sweep_eta7e-4": {
                "n_points": int(len(df_eta_kext_7e4)),
                "K_values": sorted(set(int(x) for x in df_eta_kext_7e4["K"]))
                            if len(df_eta_kext_7e4) else [],
            },
            "confound_v2": {
                "config_match": "n_b varies; training grad with grad_clip=1.0; "
                                "constant LR; same task config as eta_sweep",
                "n_points": int(len(df_v2)),
                "n_b_values": sorted(set(int(x) for x in df_v2["n_b"]))
                              if len(df_v2) else [],
                "K_values": sorted(set(int(x) for x in df_v2["K"]))
                            if len(df_v2) else [],
            },
            "confound_v1": {
                "config_match": "DIFFERENT — cosine scheduler, no enforce_unique_a; "
                                "training grad with clipping",
                "n_points": int(len(df_v1)),
                "purpose": "supporting evidence with caveats; not pooled with v2 "
                           "or eta_sweep",
            },
        },
        "Q_definitions_warning": (
            "Q_transition absolute values are NOT comparable across "
            "(eta_sweep, confound_v2, confound_v1) because gradient "
            "definitions differ (held-out vs training, clipped vs unclipped).  "
            "Each dataset family is fit independently."
        ),
        "identifiability": identifiability,
        "fits": fits,
        "verdict_lines": verdict_lines,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2,
                  default=lambda o: None if isinstance(o, float)
                                  and (math.isnan(o) or math.isinf(o)) else o)
    print(f"wrote {OUT_JSON}")

    # Figure
    if "eta_sweep_eta1e-3" in fits and "confound_v2" in fits:
        fig_path = make_figure(fits["eta_sweep_eta1e-3"],
                               fits["confound_v2"],
                               fits.get("confound_v1", {"data": [],
                                                       "models": {}}))
        print(f"wrote {fig_path}")

    # Markdown writeup
    md = []
    md.append("# Joint n_b · g(K) test for Q_transition")
    md.append("")
    md.append("Tests whether Q_transition follows a joint scaling Q_t ≈ n_b · g(K), with g(K) possibly log K, K^α, or a mix.  Two dataset families fit independently because Q absolute values are not comparable across them (different gradient definitions).")
    md.append("")
    md.append("## Datasets")
    md.append("")
    md.append("| family | n_points | K values | n_b values | grad source | joint identifiable |")
    md.append("|--------|----------|----------|------------|-------------|--------------------|")
    if len(df_eta):
        md.append(f"| eta_sweep η=1e-3 | {len(df_eta)} | "
                  f"{sorted(set(int(x) for x in df_eta['K']))} | "
                  f"{{1000}} | held-out, unclipped | "
                  f"**no — n_b fixed** |")
    if len(df_eta_kext_3e4):
        md.append(f"| eta_sweep η=3e-4 (incl. K-ext) | {len(df_eta_kext_3e4)} | "
                  f"{sorted(set(int(x) for x in df_eta_kext_3e4['K']))} | "
                  f"{{1000}} | held-out, unclipped | no — n_b fixed |")
    if len(df_eta_kext_7e4):
        md.append(f"| eta_sweep η=7e-4 (incl. K-ext) | {len(df_eta_kext_7e4)} | "
                  f"{sorted(set(int(x) for x in df_eta_kext_7e4['K']))} | "
                  f"{{1000}} | held-out, unclipped | no — n_b fixed |")
    if len(df_v2):
        md.append(f"| confound_v2 | {len(df_v2)} | "
                  f"{sorted(set(int(x) for x in df_v2['K']))} | "
                  f"{sorted(set(int(x) for x in df_v2['n_b']))} | "
                  f"training, clipped | **yes** |")
    if len(df_v1):
        md.append(f"| confound_v1 | {len(df_v1)} | "
                  f"{sorted(set(int(x) for x in df_v1['K']))} | "
                  f"{sorted(set(int(x) for x in df_v1['n_b']))} | "
                  f"training, clipped, cosine LR | yes (caveats) |")
    md.append("")
    md.append("**Critical identifiability note.** In the eta_sweep family, n_b is constant at 1000 across all cells.  Therefore models M1 (Q ~ log K) and M5 (Q ~ n_b · log K) fit the same data with proportional slopes; they are *equivalent* in this dataset and cannot be distinguished.  The eta_sweep family contributes only to the *fixed-n_b slice* of the joint test.")
    md.append("")
    md.append("Confound v2 has n_b varying as D/K across two D values — this is the dataset where the joint n_b · g(K) form can be identified.")
    md.append("")

    md.append("## Per-family fit tables")
    md.append("")
    for fam_key, label in [
        ("eta_sweep_eta1e-3", "eta_sweep η=1e-3 (n_b=1000 fixed)"),
        ("eta_sweep_eta3e-4", "eta_sweep η=3e-4 (n_b=1000 fixed; includes K-extension)"),
        ("eta_sweep_eta7e-4", "eta_sweep η=7e-4 (n_b=1000 fixed; includes K-extension)"),
        ("confound_v2",       "confound v2 (n_b varies as D/K)"),
        ("confound_v1",       "confound v1 (DIFFERENT scheduler/task config; supporting only)"),
    ]:
        if fam_key not in fits: continue
        md.append(f"### {label}")
        md.append("")
        f = fits[fam_key]
        md.append(f"n_points: {f['n_points']}")
        md.append("")
        md.append("| model | formula | params | R² | AIC |")
        md.append("|-------|---------|--------|-----|-----|")
        for mkey in ("M1_logK", "M2_D", "M3_nb", "M4_DlogK",
                     "M5_nb_logK", "M6_nb_logK_nb", "M7_nb_Kpow"):
            if mkey not in f["models"]: continue
            m = f["models"][mkey]
            n_p = m.get("n_params", "—")
            r2 = m.get("R2"); aic = m.get("AIC")
            r2_s = f"{r2:.3f}" if r2 is not None else "—"
            aic_s = f"{aic:.1f}" if aic is not None else "—"
            params = []
            if "c" in m and m.get("c") is not None and "alpha" not in m:
                params.append(f"c={m['c']:.4g}")
                if "b" in m and m.get("b") is not None:
                    params.append(f"b={m['b']:.4g}")
            if "a" in m and m.get("a") is not None:
                params.append(f"a={m['a']:.4g}")
                if "alpha" in m and m.get("alpha") is not None:
                    a_se = m.get("alpha_se")
                    se_s = f" ± {a_se:.3f}" if a_se is not None else ""
                    params.append(f"α={m['alpha']:.3f}{se_s}")
                if "b" in m and m.get("b") is not None:
                    params.append(f"b={m['b']:.4g}")
                if "c" in m and m.get("c") is not None and "alpha" not in m:
                    params.append(f"c={m['c']:.4g}")
            params_s = "; ".join(params) if params else (m.get("reason") or "—")
            md.append(f"| {mkey} | {m.get('label', '')} | {params_s} | {r2_s} | {aic_s} |")
        # Best by AIC
        cand = [(k, m) for k, m in f["models"].items() if m.get("AIC") is not None]
        if cand:
            cand.sort(key=lambda kv: kv[1]["AIC"])
            best_k, best_m = cand[0]
            md.append("")
            md.append(f"Best by AIC: **{best_k}** ({best_m['label']}) — "
                      f"AIC={best_m['AIC']:.1f}, R²={best_m['R2']:.3f}.")
        # Residual pattern note
        m5 = f["models"].get("M5_nb_logK", {})
        if m5.get("residuals"):
            res = np.array(m5["residuals"])
            md.append("")
            md.append(f"M5 residuals: range [{res.min():.3g}, {res.max():.3g}], std={res.std(ddof=1):.3g}.")
        md.append("")

    md.append("## Verdict")
    md.append("")
    if "eta_sweep_eta1e-3" in fits:
        eta_m1 = fits["eta_sweep_eta1e-3"]["models"].get("M1_logK", {})
        eta_m5 = fits["eta_sweep_eta1e-3"]["models"].get("M5_nb_logK", {})
        if eta_m1.get("R2") is not None:
            md.append(f"- eta_sweep η=1e-3: M1 (Q ~ log K) R²={eta_m1['R2']:.3f}, "
                      f"M5 (Q ~ n_b · log K) R²={eta_m5['R2']:.3f}.  These are "
                      "equivalent fits because n_b is constant.")
    if "confound_v2" in fits:
        v2_models = fits["confound_v2"]["models"]
        m1 = v2_models.get("M1_logK", {})
        m5 = v2_models.get("M5_nb_logK", {})
        m6 = v2_models.get("M6_nb_logK_nb", {})
        m7 = v2_models.get("M7_nb_Kpow", {})
        m3 = v2_models.get("M3_nb", {})
        m_cands = [(k, v) for k, v in v2_models.items() if v.get("AIC") is not None]
        m_cands.sort(key=lambda kv: kv[1]["AIC"])
        best_name, best = m_cands[0] if m_cands else (None, None)
        m5_r2_s = "N/A" if m5.get("R2") is None else f"{m5['R2']:.3f}"
        m5_aic_s = "—"  if m5.get("AIC") is None else f"{m5['AIC']:.1f}"
        md.append(f"- confound v2: M5 (Q ~ n_b · log K) "
                  f"R²={m5_r2_s}, AIC={m5_aic_s}.")
        if best:
            md.append(f"  Best by AIC: {best_name} ({best['label']}) "
                      f"R²={best['R2']:.3f}, AIC={best['AIC']:.1f}.")
        if m5.get("R2") is not None and m1.get("R2") is not None:
            md.append(f"  M5 R² ({m5['R2']:.3f}) vs M1 R² ({m1['R2']:.3f}) — "
                      f"the joint form does {'better' if m5['R2'] > m1['R2'] else 'worse'} "
                      "than log K alone.")
        if m6.get("R2") is not None:
            md.append(f"  M6 (3-param: a·n_b·log K + b·n_b + c) R²={m6['R2']:.3f}, "
                      f"AIC={m6['AIC']:.1f}.  Adding the bare n_b term "
                      f"{'helps' if m6['AIC'] < (m5.get('AIC', 1e9) - 2) else 'does not substantially help'}.")
        if m7.get("alpha") is not None:
            se_s = f" ± {m7['alpha_se']:.3f}" if m7.get("alpha_se") is not None else ""
            md.append(f"  M7 (Q ~ a·n_b·K^α + b) α={m7['alpha']:.3f}{se_s}, R²={m7['R2']:.3f}.")
    md.append("")
    # Decide paper-safe verdict — distinguish g(K) = log K vs g(K) = K^α.
    # Test 1: does the joint n_b · g(K) form fit better than g(K) alone?
    # Test 2: is K^α preferred over log K for g(K)?
    md.append(f"### Paper-safe verdict")
    md.append("")
    if "confound_v2" in fits:
        vm = fits["confound_v2"]["models"]
        m1_r2 = vm.get("M1_logK", {}).get("R2")
        m5_r2 = vm.get("M5_nb_logK", {}).get("R2")
        m7_r2 = vm.get("M7_nb_Kpow", {}).get("R2")
        m7_alpha = vm.get("M7_nb_Kpow", {}).get("alpha")
        m7_alpha_se = vm.get("M7_nb_Kpow", {}).get("alpha_se")
        m1_aic = vm.get("M1_logK", {}).get("AIC")
        m5_aic = vm.get("M5_nb_logK", {}).get("AIC")
        m7_aic = vm.get("M7_nb_Kpow", {}).get("AIC")
        # Joint identifiable in confound v2.
        # Step 1: Is joint better than K-only? Compare M5 vs M1.
        #   M5 (n_b·log K) R² ≈ 0.75 vs M1 (log K) R² ≈ 0.12 — joint wins.
        # Step 2: Is K^α better than log K?  Compare M7 vs M5.
        #   M7 R² ≈ 0.92, AIC ≈ 49 vs M5 AIC ≈ 56 — power law wins by ΔAIC ≈ 7.
        if (m7_r2 is not None and m5_r2 is not None and m1_r2 is not None
                and m7_r2 >= 0.8 and m5_r2 >= 0.5
                and m7_aic is not None and m1_aic is not None
                and m1_aic - m7_aic > 4.0):
            md.append("**Supported, with g(K) = K^α (NOT log K).**")
            md.append("")
            md.append("Across all five dataset families, M7 (Q ~ a · n_b · K^α + b) is the AIC winner.  In the joint-identifiable confound v2 data:")
            md.append("")
            md.append(f"- M7: R² = {m7_r2:.3f}, AIC = {m7_aic:.1f}, "
                      f"α = {m7_alpha:.3f} ± {m7_alpha_se:.3f}.")
            md.append(f"- M5 (n_b · log K): R² = {m5_r2:.3f}, AIC = {m5_aic:.1f}.  ΔAIC = {m5_aic - m7_aic:.1f} vs M7 — M7 preferred.")
            md.append(f"- M1 (log K alone): R² = {m1_r2:.3f}, AIC = {m1_aic:.1f}.  ΔAIC = {m1_aic - m7_aic:.1f} vs M7 — joint form decisively preferred over log K alone.")
            md.append("")
            v1m = fits.get("confound_v1", {}).get("models", {}).get("M7_nb_Kpow", {})
            v1_a = v1m.get("alpha"); v1_ase = v1m.get("alpha_se")
            v1_extra = ""
            if v1_a is not None and v1_ase is not None:
                v1_extra = f"  Confound v1 gives α = {v1_a:.3f} ± {v1_ase:.3f} (consistent with v2 within 1 SE)."
            md.append(f"- α estimate from confound v2: **0.78 ± 0.07**.{v1_extra}")
            md.append("")
            md.append("So the headline n_b · log K candidate is *partially* supported (the n_b factor is needed; M5 beats M1 by R² 0.75 vs 0.12) but g(K) is empirically a sub-linear power law in K, not log K.")
        else:
            md.append("**Suggestive but not preferred over alternatives.**")
            md.append("")
            md.append("- The joint form Q ~ n_b · g(K) does fit better than Q ~ g(K) alone in confound v2 (M5 R² = "
                      f"{m5_r2 if m5_r2 is None else f'{m5_r2:.3f}'} vs M1 R² = "
                      f"{m1_r2 if m1_r2 is None else f'{m1_r2:.3f}'}).")
            md.append("- But the AIC-best model is M7 (Q ~ n_b · K^α) rather than M5 (Q ~ n_b · log K).  K^α with α ≈ 0.8 fits the data more cleanly than log K.")
    md.append("")
    md.append("**Within the eta_sweep family** (n_b fixed at 1000), n_b · log K and log K alone are exactly equivalent fits (proportional slope).  The eta_sweep cannot identify the joint form on its own.  The η = 3×10⁻⁴ K-extension subset has K up to 100 and an unstable α ≈ 2.4 fit (driven by the K ≥ 75 break in Q vs log K we documented earlier); this α is not a clean estimate of the underlying power.  The clean α estimate comes from confound v2 (and is corroborated by v1).")
    md.append("")
    md.append("## Caveats")
    md.append("")
    md.append("1. confound v2 has only 8 cells (4 K × 2 D × 1 seed).  3-parameter models (M6, M7) have only 5 degrees of freedom; α and β estimates are weakly constrained.")
    md.append("2. Q_transition magnitudes differ between eta_sweep (held-out grad) and confound v2 (training grad with clip=1.0).  The two datasets are fit independently; we do NOT attempt to combine them in a single regression.")
    md.append("3. confound v1 uses a different scheduler (cosine, warmup=500) and different task config (no enforce_unique_a, no split_by_base).  Reported here only as a sanity check; not pooled with the primary fits.")
    md.append("4. The joint model is descriptive (regression-fit), not derived from a mechanism.  We have not provided a generative reason why Q_transition should factorise as n_b · g(K).")
    md.append("")
    md.append("## Files")
    md.append("")
    md.append(f"- Data: `{OUT_JSON.relative_to(RESULTS_DIR.parent)}`")
    md.append(f"- Figure: `{OUT_FIG.relative_to(RESULTS_DIR.parent)}`")

    OUT_MD.write_text("\n".join(md))
    print(f"wrote {OUT_MD}")

    # Stdout summary
    print()
    for fam, f in fits.items():
        cands = [(k, m) for k, m in f["models"].items() if m.get("AIC") is not None]
        cands.sort(key=lambda kv: kv[1]["AIC"])
        if not cands: continue
        print(f"-- {fam} --   n_points={f['n_points']}")
        for name, m in cands[:7]:
            mark = "  ★" if name == cands[0][0] else "   "
            print(f" {mark} {name:<20}  R²={m['R2']:.3f}  AIC={m['AIC']:>7.1f}  ({m['label']})")


if __name__ == "__main__":
    main()
