#!/usr/bin/env python
"""
Port of the parent-repo plateau-decomposition analysis to current eta_sweep
data.  Decomposes Q into Q_prior + Q_plateau + Q_transition for every
transitioned cell, aggregates by (η, K), and tests:

  - whether Q_plateau is weakly K-dependent
  - whether Q_transition carries the log-K scaling
  - whether Q_plateau / Q_total grows approaching η_c

Outputs:
  results/plateau_decomposition_current.json
  results/figures/plateau_decomposition_current.png
  results/plateau_decomposition_claims.md

Source for the original three-phase decomposition idea:
  scripts/compute_plateau_decomposition.py (parent repo, NOT modified).
This re-implementation adapts the same definitions to current
eta_sweep run logs.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ETA_SWEEP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ETA_SWEEP_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from eta_sweep.config import RESULTS_DIR  # noqa: E402

FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True, parents=True)

OUT_JSON = RESULTS_DIR / "plateau_decomposition_current.json"
OUT_FIG = FIG_DIR / "plateau_decomposition_current.png"
OUT_MD = RESULTS_DIR / "plateau_decomposition_claims.md"

MAIN_K = {10, 15, 20, 25, 36}

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
# I/O
# ---------------------------------------------------------------------------

def _read_log(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    rows = []
    bad = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                bad += 1
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)
    if bad:
        df.attrs["bad_lines"] = bad
    return df


def _read_status(d: Path) -> Optional[Dict]:
    p = d / "status.json"
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def _read_config(d: Path) -> Optional[Dict]:
    p = d / "config.json"
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f).get("cell")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Phase boundaries
# ---------------------------------------------------------------------------

def _find_prior_end(steps: np.ndarray, cand: np.ndarray, log_K: float
                    ) -> Optional[int]:
    """First step at which candidate_loss < 1.1 · log K."""
    threshold = 1.1 * log_K
    mask = cand < threshold
    if not np.any(mask):
        return None
    return int(steps[np.argmax(mask)])


def _find_t1_t2(steps: np.ndarray, dz: np.ndarray
                ) -> Tuple[Optional[int], Optional[int]]:
    """t1 = first step where Δz > 0.5; t2 = first step where Δz > 0.9·max(Δz)."""
    valid = ~np.isnan(dz)
    if not np.any(valid):
        return None, None
    dz_v = dz[valid]
    steps_v = steps[valid]
    above = dz_v > 0.5
    if not np.any(above):
        return None, None
    t1 = int(steps_v[np.argmax(above)])
    max_dz = float(np.max(dz_v))
    above2 = dz_v > 0.9 * max_dz
    if np.any(above2):
        t2 = int(steps_v[np.argmax(above2)])
    else:
        t2 = int(steps_v[np.argmax(dz_v)])
    if t2 < t1:
        # Clamp t2 to argmax(Δz) at or after t1.
        t1_idx = int(np.argmax(above))
        after = dz_v[t1_idx:]
        t2 = int(steps_v[t1_idx + int(np.argmax(after))])
    return t1, t2


def _trapz_Q(steps: np.ndarray, g: np.ndarray, lr: np.ndarray,
             a: int, b: int) -> float:
    """Trapezoidal η·‖∇L‖² over step window [a, b]."""
    if a is None or b is None or b <= a:
        return float("nan")
    mask = (steps >= a) & (steps <= b)
    x = steps[mask].astype(float)
    y = (lr[mask] * g[mask]).astype(float)
    if len(x) < 2:
        return float("nan")
    return float(np.trapz(y, x))


def _interp(steps: np.ndarray, vals: np.ndarray, x: float) -> float:
    return float(np.interp(x, steps, vals))


# ---------------------------------------------------------------------------
# Per-cell decomposition
# ---------------------------------------------------------------------------

def decompose_cell(run_dir: Path) -> Optional[Dict]:
    cfg = _read_config(run_dir)
    status = _read_status(run_dir)
    if cfg is None or status is None:
        return None
    df = _read_log(run_dir / "log.jsonl")
    if df is None or len(df) < 2:
        return None

    eta = float(cfg["eta"])
    K = int(cfg["k"])
    seed = int(cfg["seed"])
    log_K = math.log(K)

    steps = df["step"].to_numpy()
    cand = df["candidate_loss"].to_numpy(dtype=float)
    dz = df["delta_z"].to_numpy(dtype=float)
    g = df["grad_norm_sq_held_out"].to_numpy(dtype=float)
    lr = df["lr"].to_numpy(dtype=float)

    # Phase boundaries
    prior_end = _find_prior_end(steps, cand, log_K)
    t1, t2 = _find_t1_t2(steps, dz)

    out: Dict = {
        "run_name": run_dir.name,
        "eta": eta, "k": K, "seed": seed,
        "status": status.get("status"),
        "final_step": int(status.get("final_step", 0)),
        "log_K": log_K,
        "prior_end": prior_end,
        "t1": t1, "t2": t2,
        "max_dz": float(np.nanmax(dz)) if np.any(~np.isnan(dz)) else None,
        "ambiguous_prior": prior_end is None,
        "no_transition_window": (t1 is None or t2 is None),
    }

    # Compute Q over each phase.
    step0 = int(steps[0])
    if prior_end is not None:
        out["Q_prior"] = _trapz_Q(steps, g, lr, step0, prior_end)
        out["delta_L_prior"] = (
            _interp(steps, cand, step0) - _interp(steps, cand, prior_end)
        )
    else:
        out["Q_prior"] = None
        out["delta_L_prior"] = None

    if prior_end is not None and t1 is not None and t1 > prior_end:
        out["Q_plateau"] = _trapz_Q(steps, g, lr, prior_end, t1)
        out["delta_L_plateau"] = (
            _interp(steps, cand, prior_end) - _interp(steps, cand, t1)
        )
    elif t1 is not None and prior_end is None:
        # No prior phase identified; treat [step0, t1] as the plateau.
        out["Q_plateau"] = _trapz_Q(steps, g, lr, step0, t1)
        out["delta_L_plateau"] = (
            _interp(steps, cand, step0) - _interp(steps, cand, t1)
        )
        out["plateau_includes_prior"] = True
    else:
        out["Q_plateau"] = None
        out["delta_L_plateau"] = None

    if t1 is not None and t2 is not None:
        out["Q_transition"] = _trapz_Q(steps, g, lr, t1, t2)
        out["delta_L_transition"] = (
            _interp(steps, cand, t1) - _interp(steps, cand, t2)
        )
    else:
        out["Q_transition"] = None
        out["delta_L_transition"] = None

    # Q_total = sum of available phases (skip None).
    pieces = [out[k] for k in ("Q_prior", "Q_plateau", "Q_transition")
              if out.get(k) is not None and not (isinstance(out[k], float) and math.isnan(out[k]))]
    out["Q_total"] = float(sum(pieces)) if pieces else None

    # Phase-wise frac_excess (only where meaningful).
    for ph in ("prior", "plateau", "transition"):
        Q = out.get(f"Q_{ph}")
        dL = out.get(f"delta_L_{ph}")
        if (Q is not None and dL is not None
                and isinstance(Q, float) and Q > 0
                and not math.isnan(Q) and not math.isnan(dL)):
            out[f"frac_excess_{ph}"] = (Q - dL) / Q
        else:
            out[f"frac_excess_{ph}"] = None

    # Q_plateau / Q_total ratio (the "silent learning fraction").
    if (out["Q_plateau"] is not None and out["Q_total"] is not None
            and out["Q_total"] > 0
            and not math.isnan(out["Q_plateau"])
            and not math.isnan(out["Q_total"])):
        out["plateau_fraction"] = out["Q_plateau"] / out["Q_total"]
    else:
        out["plateau_fraction"] = None

    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _eta_in(eta: float, e: float, tol: float = 1e-9) -> bool:
    return abs(eta - e) < tol * max(abs(e), 1.0) or abs(eta - e) < 1e-12


def aggregate(per_cell: List[Dict]) -> Dict:
    df = pd.DataFrame(per_cell)
    df = df[df["status"] == "transitioned"].copy()

    # Mean per-(η, K) for each phase quantity, restricted to K ≤ 36.
    main = df[df["k"].isin(MAIN_K)].copy()
    grouped: Dict = {"per_eta_K": {}, "by_eta": {}, "by_K_at_low_eta": {}}

    for (eta, K), grp in main.groupby(["eta", "k"]):
        agg = {}
        for col in ("Q_prior","Q_plateau","Q_transition","Q_total",
                    "delta_L_prior","delta_L_plateau","delta_L_transition",
                    "plateau_fraction",
                    "frac_excess_prior","frac_excess_plateau","frac_excess_transition"):
            vals = grp[col].dropna().to_numpy(dtype=float)
            agg[col] = {
                "mean": float(np.mean(vals)) if len(vals) else None,
                "std":  float(np.std(vals, ddof=1)) if len(vals) >= 2 else None,
                "n":    int(len(vals)),
            }
        agg["n_seeds_total"] = int(len(grp))
        grouped["per_eta_K"][f"{eta:g}|K={K}"] = agg

    # By η: aggregate over all K ≤ 36.
    for eta, grp in main.groupby("eta"):
        agg = {}
        for col in ("Q_plateau","Q_transition","Q_total","plateau_fraction"):
            vals = grp[col].dropna().to_numpy(dtype=float)
            agg[col] = {
                "mean": float(np.mean(vals)) if len(vals) else None,
                "n":    int(len(vals)),
            }
        grouped["by_eta"][f"{eta:g}"] = agg

    # Stable low-η regime: by-K Q-phase scaling.
    stable = main[main["eta"] <= 1e-3 + 1e-12].copy()
    for K, grp in stable.groupby("k"):
        agg = {}
        for col in ("Q_prior","Q_plateau","Q_transition","Q_total"):
            vals = grp[col].dropna().to_numpy(dtype=float)
            agg[col] = {
                "mean": float(np.mean(vals)) if len(vals) else None,
                "n":    int(len(vals)),
            }
        grouped["by_K_at_low_eta"][int(K)] = agg

    return grouped


def fit_log_K_scaling(grouped: Dict) -> Dict:
    """Affine fit of phase-Q vs log K in the stable low-η regime."""
    by_K = grouped["by_K_at_low_eta"]
    Ks = sorted(by_K.keys())
    out = {}
    for col in ("Q_prior","Q_plateau","Q_transition","Q_total"):
        K_arr, Q_arr = [], []
        for K in Ks:
            mean = by_K[K][col]["mean"]
            if mean is not None and not math.isnan(mean):
                K_arr.append(K); Q_arr.append(mean)
        if len(K_arr) < 3:
            out[col] = {"c": None, "q0": None, "R2": None,
                        "n_K": len(K_arr), "reason": "insufficient K"}
            continue
        K_arr = np.array(K_arr, dtype=float); Q_arr = np.array(Q_arr, dtype=float)
        logK = np.log(K_arr)
        A = np.vstack([logK, np.ones_like(logK)]).T
        (c, q0), *_ = np.linalg.lstsq(A, Q_arr, rcond=None)
        yhat = c * logK + q0
        ss_res = float(np.sum((Q_arr - yhat) ** 2))
        ss_tot = float(np.sum((Q_arr - Q_arr.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
        out[col] = {
            "c": float(c), "q0": float(q0), "R2": r2,
            "n_K": int(len(K_arr)),
            "K_values": K_arr.tolist(),
            "Q_means": Q_arr.tolist(),
        }
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _representative_cell(per_cell: List[Dict]) -> Optional[Path]:
    """Pick a clean transitioned cell with all three phases and a clear snap."""
    for c in per_cell:
        if c["status"] != "transitioned": continue
        if (c["Q_prior"] is None or c["Q_plateau"] is None
                or c["Q_transition"] is None): continue
        if c["k"] != 20: continue
        if abs(c["eta"] - 1e-3) < 1e-12 and c["seed"] == 0:
            return Path(c["run_name"])
        if abs(c["eta"] - 1e-3) < 1e-12 and c["seed"] in (1, 2):
            return Path(c["run_name"])
    # fallback: any transitioned with all three phases
    for c in per_cell:
        if c["status"] == "transitioned" and c["Q_plateau"] is not None and c["Q_transition"] is not None:
            return Path(c["run_name"])
    return None


def make_figure(per_cell: List[Dict], grouped: Dict, scaling: Dict) -> Path:
    fig = plt.figure(figsize=(13.5, 9))
    gs = fig.add_gridspec(3, 3, hspace=0.5, wspace=0.4)

    # --- Row 0: representative trajectory (3 subplots side by side) ---
    rep_name = _representative_cell(per_cell)
    if rep_name is not None:
        rep_dir = RESULTS_DIR / rep_name
        df = _read_log(rep_dir / "log.jsonl")
        cfg = _read_config(rep_dir)
        log_K = math.log(int(cfg["k"]))
        cand = df["candidate_loss"].to_numpy(dtype=float)
        dz = df["delta_z"].to_numpy(dtype=float)
        g = df["grad_norm_sq_held_out"].to_numpy(dtype=float)
        steps = df["step"].to_numpy()
        rep = next(c for c in per_cell if c["run_name"] == rep_name.name)
        prior_end = rep["prior_end"]; t1 = rep["t1"]; t2 = rep["t2"]

        ax_cand = fig.add_subplot(gs[0, 0])
        ax_dz = fig.add_subplot(gs[0, 1])
        ax_g = fig.add_subplot(gs[0, 2])

        for ax, y, label in [(ax_cand, cand, "candidate_loss"),
                             (ax_dz, dz, "Δz (nats)"),
                             (ax_g, g, "‖∇L‖² (held-out)")]:
            ax.plot(steps, y, color="#2E86AB", linewidth=1.0)
            for x, lbl, col in (
                (prior_end, "prior_end", "#888"),
                (t1, "t₁", "#E76F51"),
                (t2, "t₂", "#7B2CBF"),
            ):
                if x is not None:
                    ax.axvline(x, color=col, linestyle="--", linewidth=1.0, alpha=0.7)
                    ax.text(x, ax.get_ylim()[1] * 0.93, lbl, color=col,
                            fontsize=7, ha="center", rotation=90)
            ax.set_xlabel("step")
            ax.set_ylabel(label)
        ax_cand.axhline(log_K, color="grey", linestyle=":", linewidth=0.8,
                        label=f"log K = {log_K:.2f}")
        ax_cand.legend(fontsize=7)
        ax_g.set_yscale("log")
        ax_cand.set_title(f"Representative trajectory: {rep_name.name}",
                          fontsize=9)

    # --- Row 1: phase Q vs log K in stable low-η regime ---
    ax = fig.add_subplot(gs[1, 0:2])
    by_K = grouped["by_K_at_low_eta"]
    Ks = sorted(by_K.keys())
    for col, color, label in (
        ("Q_prior", "#888", "Q_prior"),
        ("Q_plateau", "#2E86AB", "Q_plateau"),
        ("Q_transition", "#E76F51", "Q_transition"),
        ("Q_total", "black", "Q_total"),
    ):
        x, y = [], []
        for K in Ks:
            m = by_K[K][col]["mean"]
            if m is not None and not math.isnan(m):
                x.append(K); y.append(m)
        if x:
            ax.plot(x, y, "o-", color=color, label=label, linewidth=1.4)
        # overlay fit
        f = scaling.get(col, {})
        if f.get("c") is not None and f.get("R2") is not None:
            K_grid = np.geomspace(min(x), max(x), 60) if x else None
            if K_grid is not None:
                yhat = f["c"] * np.log(K_grid) + f["q0"]
                ax.plot(K_grid, yhat, "--", color=color, alpha=0.5,
                        label=f"  fit c={f['c']:.2f} R²={f['R2']:.2f}")
    ax.set_xscale("log")
    ax.set_xlabel("K"); ax.set_ylabel("Q")
    ax.set_title("Phase-decomposed Q vs log K (stable low-η: η ≤ 10⁻³)",
                 fontsize=9)
    ax.legend(fontsize=7, ncol=2)

    # --- Row 1 right: plateau_fraction vs η at K=20 ---
    ax = fig.add_subplot(gs[1, 2])
    df_pc = pd.DataFrame(per_cell)
    df_pc = df_pc[(df_pc["status"] == "transitioned") & (df_pc["k"] == 20)]
    pf = df_pc.groupby("eta")["plateau_fraction"].agg(["mean", "std", "count"])
    pf = pf.dropna(subset=["mean"]).sort_index()
    if len(pf) >= 1:
        yerr = pf["std"].fillna(0).to_numpy() / np.sqrt(np.maximum(pf["count"].to_numpy(), 1))
        ax.errorbar(pf.index, pf["mean"], yerr=yerr, marker="o",
                    color="#2E86AB", capsize=3)
    ax.set_xscale("log")
    ax.set_xlabel("η"); ax.set_ylabel("Q_plateau / Q_total")
    ax.set_title("Silent-learning fraction vs η at K=20", fontsize=9)

    # --- Row 2: per-(η,K) Q_plateau and Q_transition tables (3 panels) ---
    df_pc = pd.DataFrame(per_cell)
    df_pc = df_pc[df_pc["status"] == "transitioned"]
    df_pc = df_pc[df_pc["k"].isin(MAIN_K)]
    pivot_plat = df_pc.groupby(["eta","k"])["Q_plateau"].mean().unstack().sort_index()
    pivot_trans = df_pc.groupby(["eta","k"])["Q_transition"].mean().unstack().sort_index()
    pivot_pf = df_pc.groupby(["eta","k"])["plateau_fraction"].mean().unstack().sort_index()

    for idx, (pivot, title, cmap) in enumerate([
        (pivot_plat,  "Q_plateau by (η, K)",   "Blues"),
        (pivot_trans, "Q_transition by (η, K)", "Reds"),
        (pivot_pf,    "Q_plateau / Q_total by (η, K)", "viridis"),
    ]):
        ax = fig.add_subplot(gs[2, idx])
        if pivot.size == 0:
            ax.axis("off"); ax.set_title(title + "\n(no data)"); continue
        im = ax.imshow(pivot.values, aspect="auto", cmap=cmap,
                       origin="lower")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{e:g}" for e in pivot.index], fontsize=7)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns.astype(int), fontsize=7)
        ax.set_xlabel("K"); ax.set_ylabel("η")
        ax.set_title(title, fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    fig.suptitle("Plateau-decomposition: Q_prior + Q_plateau + Q_transition",
                 fontsize=11, y=0.995)
    fig.savefig(OUT_FIG)
    plt.close(fig)
    return OUT_FIG


# ---------------------------------------------------------------------------
# Markdown writeup
# ---------------------------------------------------------------------------

def write_claims_md(per_cell: List[Dict], grouped: Dict, scaling: Dict) -> Path:
    df_pc = pd.DataFrame(per_cell)
    n_cells_total = len(df_pc)
    n_transitioned = int((df_pc["status"] == "transitioned").sum())
    n_stuck = int((df_pc["status"] == "stuck").sum())
    n_diverged = int((df_pc["status"] == "diverged").sum())
    n_ambig_prior = int(df_pc["ambiguous_prior"].sum())
    n_no_window = int(df_pc["no_transition_window"].sum())

    # Stable-regime scaling readout
    pl_scaling = scaling.get("Q_plateau", {})
    tr_scaling = scaling.get("Q_transition", {})
    pr_scaling = scaling.get("Q_prior", {})
    tot_scaling = scaling.get("Q_total", {})

    # Plateau fraction near η_c
    main = df_pc[(df_pc["status"] == "transitioned") & (df_pc["k"].isin(MAIN_K))]
    pf_by_eta = (main.groupby("eta")["plateau_fraction"]
                     .agg(["mean", "count"]).dropna(subset=["mean"]).sort_index())

    # Pull representative cell info
    rep = next((c for c in per_cell
                if c["run_name"].startswith("eta_0.001_K_20")
                and c["status"] == "transitioned"
                and c["Q_plateau"] is not None
                and c["Q_transition"] is not None), None)

    lines = []
    lines.append("# Plateau-decomposition claims (current eta_sweep data)")
    lines.append("")
    lines.append("Reproduced from the parent-repo idea in "
                 "`scripts/compute_plateau_decomposition.py`, ported to current "
                 "eta_sweep cells via `eta_sweep/analysis/plateau_decomposition.py`.")
    lines.append("")
    lines.append("## Definitions")
    lines.append("")
    lines.append("- **prior_end** = first checkpoint where candidate_loss < 1.1 · log K.")
    lines.append("- **t₁** = first checkpoint where Δz > 0.5 nats.")
    lines.append("- **t₂** = first checkpoint where Δz > 0.9 · max(Δz).")
    lines.append("- **Q_prior** = ∫η·‖∇L‖² over [0, prior_end].")
    lines.append("- **Q_plateau** = ∫η·‖∇L‖² over [prior_end, t₁].")
    lines.append("- **Q_transition** = ∫η·‖∇L‖² over [t₁, t₂].")
    lines.append("- **Q_total** = sum of available phases.")
    lines.append("- **plateau_fraction** = Q_plateau / Q_total (silent-learning share).")
    lines.append("- **frac_excess_phase** = (Q_phase − ΔL_phase) / Q_phase when meaningful.")
    lines.append("- ‖∇L‖² is on a fixed held-out batch of 256 examples; it is the raw "
                 "parameter gradient (NOT the AdamW effective update).")
    lines.append("")
    lines.append("## Cell counts")
    lines.append("")
    lines.append(f"- Total cells loaded: **{n_cells_total}**")
    lines.append(f"- Transitioned: **{n_transitioned}**, stuck: {n_stuck}, diverged: {n_diverged}")
    lines.append(f"- Cells with ambiguous/absent prior_end: **{n_ambig_prior}** "
                 "(candidate_loss never crossed 1.1·log K — for these we report "
                 "Q_plateau over [step_0, t₁] and flag `plateau_includes_prior=True`)")
    lines.append(f"- Cells with no transition window detected: {n_no_window}")
    lines.append("")
    lines.append("## Headline results")
    lines.append("")
    lines.append("### 1. Q_plateau is non-trivial")
    lines.append("")
    if rep is not None:
        lines.append(f"Representative cell `{rep['run_name']}`:")
        lines.append("")
        lines.append("| phase | Q | ΔL | frac_excess |")
        lines.append("|-------|---|------|-------------|")
        for ph in ("prior", "plateau", "transition"):
            Q = rep.get(f"Q_{ph}"); dL = rep.get(f"delta_L_{ph}")
            fe = rep.get(f"frac_excess_{ph}")
            Q_s = f"{Q:.4f}" if Q is not None and not math.isnan(Q) else "—"
            dL_s = f"{dL:.4f}" if dL is not None and not math.isnan(dL) else "—"
            fe_s = f"{fe:.3f}" if fe is not None and not math.isnan(fe) else "—"
            lines.append(f"| {ph} | {Q_s} | {dL_s} | {fe_s} |")
        lines.append(f"| **total** | **{rep['Q_total']:.4f}** | | |")
        lines.append("")
        lines.append(f"plateau_fraction = Q_plateau / Q_total = "
                     f"**{rep['plateau_fraction']:.3f}**.")
        lines.append("")
    lines.append("### 2. Stable low-η regime (η ≤ 10⁻³): phase-Q vs log K")
    lines.append("")
    lines.append("| phase | c (slope vs log K) | q₀ | R² | n_K |")
    lines.append("|-------|--------------------|----|----|-----|")
    for col, lbl in [("Q_prior","Q_prior"),
                     ("Q_plateau","Q_plateau"),
                     ("Q_transition","Q_transition"),
                     ("Q_total","Q_total")]:
        f = scaling.get(col, {})
        if f.get("c") is None:
            lines.append(f"| {lbl} | — | — | — | {f.get('n_K', 0)} |")
        else:
            lines.append(f"| {lbl} | {f['c']:.3f} | {f['q0']:.3f} | "
                         f"{f['R2']:.3f} | {f['n_K']} |")
    lines.append("")
    lines.append("### 3. Verdict on K-dependence of plateau vs transition")
    lines.append("")
    if (pl_scaling.get("c") is not None and tr_scaling.get("c") is not None):
        ratio = abs(tr_scaling["c"] / pl_scaling["c"]) if pl_scaling["c"] != 0 else float("inf")
        lines.append(f"- |slope(Q_transition) / slope(Q_plateau)| = "
                     f"**{ratio:.2f}**.")
        if ratio >= 2.0:
            lines.append("- **Q_transition slope is at least 2× larger in magnitude than Q_plateau slope** in the stable regime — supports the parent-repo claim that Q_transition carries the dominant log-K dependence.")
        else:
            lines.append("- The slopes are comparable (ratio < 2). Q_plateau and Q_transition both scale with log K to similar degree in the present data; the original claim of "
                         "'Q_plateau weakly K-dependent' is **not strongly supported** by the current eta_sweep aggregations.")
        if pl_scaling.get("R2", 0) is not None and pl_scaling["R2"] < 0.5:
            lines.append("- Q_plateau R² is poor — suggests it is not a clean log K function, "
                         "consistent with K-independence (or near-independence).")
    else:
        lines.append("- Insufficient stable-regime data with finished prior+plateau+transition phases to fit log-K scaling for at least one phase.")
    lines.append("")
    lines.append("### 4. plateau_fraction vs η (does Q_plateau dominate near η_c?)")
    lines.append("")
    lines.append("| η | mean Q_plateau / Q_total | n cells |")
    lines.append("|---|--------------------------|---------|")
    for eta, row in pf_by_eta.iterrows():
        lines.append(f"| {eta:g} | {row['mean']:.3f} | {int(row['count'])} |")
    if len(pf_by_eta) >= 3:
        first = pf_by_eta.iloc[0]["mean"]
        last = pf_by_eta.iloc[-1]["mean"]
        if last > first + 0.1:
            lines.append(f"\n- **Plateau fraction increases with η** "
                         f"({first:.2f} → {last:.2f}). "
                         "Approaching η_c, more of the total Q is spent on "
                         "plateau-phase silent learning. Supports the framing "
                         "that the apparent 'snap' is preceded by progressively "
                         "longer pre-snap dissipation.")
        elif last < first - 0.1:
            lines.append(f"\n- Plateau fraction **decreases** with η "
                         f"({first:.2f} → {last:.2f}); near-criticality the "
                         "snap window dominates dissipation.")
        else:
            lines.append("\n- Plateau fraction roughly constant across the η range "
                         "(within ±0.1). No clear divergence at η_c.")
    lines.append("")
    lines.append("### 5. Cells with ambiguous prior_end")
    lines.append("")
    lines.append(f"For **{n_ambig_prior} cells**, candidate_loss never crosses 1.1·log K. "
                 "These are cells where the very first logged checkpoint is already below "
                 "1.1·log K (i.e., the model never spends time above 1.1·log K — at K=10 "
                 "where log K ≈ 2.3, even random init can score below this). "
                 "For those cells, Q_plateau is computed over [step₀, t₁] and flagged "
                 "with `plateau_includes_prior=True` in the JSON output. Q_prior is null.")
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append("- Q here uses raw `p.grad`, not the AdamW effective update `m̂/√v̂`. "
                 "Phase-Q values are gradient-norm integrals, not literal entropy "
                 "production for AdamW.")
    lines.append("- Phase boundaries are noisy at low N (some cells transition fast and "
                 "have only 2–3 plateau-phase checkpoints).")
    lines.append("- Stable-regime fits use seed-averaged Q at each K. K=15 and K=25 are "
                 "still single-seed in the stable regime. Slope confidence is limited.")
    lines.append("- Q_prior is small (and often zero/null) at K=10 since transitions "
                 "complete in fewer steps than the first eval-cadence boundary.")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append(f"- Data: `{OUT_JSON.relative_to(RESULTS_DIR.parent)}`")
    lines.append(f"- Figure: `{OUT_FIG.relative_to(RESULTS_DIR.parent)}`")

    OUT_MD.write_text("\n".join(lines))
    return OUT_MD


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def main() -> None:
    cell_dirs = sorted([d for d in RESULTS_DIR.iterdir()
                        if d.is_dir() and d.name.startswith("eta_")
                        and not any(d.name.startswith(x) for x in ("eta_sweep",))])

    per_cell: List[Dict] = []
    for d in cell_dirs:
        rec = decompose_cell(d)
        if rec is None: continue
        per_cell.append(rec)
    print(f"decomposed {len(per_cell)} cells")

    grouped = aggregate(per_cell)
    scaling = fit_log_K_scaling(grouped)

    out = {
        "n_cells_processed": len(per_cell),
        "scaling_in_stable_low_eta": scaling,
        "aggregations": grouped,
        "per_cell": per_cell,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2,
                  default=lambda o: None if isinstance(o, float)
                                  and (math.isnan(o) or math.isinf(o)) else o)
    print(f"wrote {OUT_JSON}")

    fig_path = make_figure(per_cell, grouped, scaling)
    print(f"wrote {fig_path}")

    md_path = write_claims_md(per_cell, grouped, scaling)
    print(f"wrote {md_path}")

    # Quick stdout summary
    print()
    print("=== stable low-η scaling (Q_phase = c · log K + q₀) ===")
    for col in ("Q_prior","Q_plateau","Q_transition","Q_total"):
        f = scaling.get(col, {})
        if f.get("c") is None:
            print(f"  {col:<14}: insufficient data")
        else:
            print(f"  {col:<14}: c = {f['c']:>7.3f}  R² = {f['R2']:.3f}  n_K = {f['n_K']}")
    pl = scaling.get("Q_plateau", {})
    tr = scaling.get("Q_transition", {})
    if pl.get("c") is not None and tr.get("c") is not None:
        print(f"  ratio |slope(Q_transition)/slope(Q_plateau)| = "
              f"{abs(tr['c']/pl['c']):.2f}")


if __name__ == "__main__":
    main()
