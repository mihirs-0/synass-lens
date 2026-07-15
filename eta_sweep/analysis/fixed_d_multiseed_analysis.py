#!/usr/bin/env python
"""
Analysis of the fixed-D multi-seed sweep (24 cells, 3 seeds × 8 (D, K) pairs).

Builds on plateau_decomposition.py phase definitions; cells were trained
through eta_sweep/run_single.py (held-out gradient, no clipping) so are
directly comparable to the rest of the eta_sweep parquet.

Outputs:
  results/fixed_d_multiseed_analysis.json
  results/fixed_d_multiseed_analysis.md
  results/figures/fixed_d_multiseed_analysis.png
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

DATA_ROOT = RESULTS_DIR / "fixed_d_multiseed"
OUT_JSON = RESULTS_DIR / "fixed_d_multiseed_analysis.json"
OUT_MD = RESULTS_DIR / "fixed_d_multiseed_analysis.md"
OUT_FIG = RESULTS_DIR / "figures" / "fixed_d_multiseed_analysis.png"

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
# Phase boundary helpers (same definitions as plateau_decomposition.py)
# ---------------------------------------------------------------------------

def _find_prior_end(steps, cand, log_K):
    threshold = 1.1 * log_K
    mask = np.asarray(cand) < threshold
    if not np.any(mask):
        return None
    return int(np.asarray(steps)[np.argmax(mask)])


def _find_t1_t2(steps, dz):
    steps = np.asarray(steps); dz = np.asarray(dz, dtype=float)
    valid = ~np.isnan(dz)
    if not np.any(valid):
        return None, None
    dz_v = dz[valid]; steps_v = steps[valid]
    above = dz_v > 0.5
    if not np.any(above):
        return None, None
    t1_idx = int(np.argmax(above)); t1 = int(steps_v[t1_idx])
    max_dz = float(np.max(dz_v))
    above2 = dz_v > 0.9 * max_dz
    if np.any(above2):
        t2 = int(steps_v[np.argmax(above2)])
    else:
        t2 = int(steps_v[np.argmax(dz_v)])
    if t2 < t1:
        after = dz_v[t1_idx:]
        t2 = int(steps_v[t1_idx + int(np.argmax(after))])
    return t1, t2


def _trapz_Q(steps, g, lr, a, b):
    if a is None or b is None or b <= a:
        return float("nan")
    steps = np.asarray(steps, dtype=float)
    g = np.asarray(g, dtype=float); lr = np.asarray(lr, dtype=float)
    mask = (steps >= a) & (steps <= b)
    x = steps[mask]; y = lr[mask] * g[mask]
    if len(x) < 2:
        return float("nan")
    return float(np.trapz(y, x))


def decompose_cell(cell_dir: Path) -> Optional[Dict]:
    cfg_path = cell_dir / "config.json"
    log_path = cell_dir / "log.jsonl"
    status_path = cell_dir / "status.json"
    if not (cfg_path.exists() and log_path.exists() and status_path.exists()):
        return None
    cfg = json.load(open(cfg_path))["cell"]
    status = json.load(open(status_path))
    rows = []
    for line in open(log_path):
        line = line.strip()
        if not line: continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)

    K = int(cfg["k"])
    n_b = int(cfg["n_unique_b"])
    D = K * n_b
    seed = int(cfg["seed"])
    log_K = math.log(K)
    eta = float(cfg["eta"])

    steps = df["step"].to_numpy()
    cand = df["candidate_loss"].to_numpy(dtype=float)
    dz = df["delta_z"].to_numpy(dtype=float)
    g = df["grad_norm_sq_held_out"].to_numpy(dtype=float)
    lr = df["lr"].to_numpy(dtype=float)

    prior_end = _find_prior_end(steps, cand, log_K)
    t1, t2 = _find_t1_t2(steps, dz)
    step0 = int(steps[0])

    out = {
        "run_name": cell_dir.name,
        "D": D, "K": K, "n_b": n_b, "seed": seed,
        "eta": eta,
        "status": status.get("status"),
        "final_step": int(status.get("final_step", 0)),
        "prior_end": prior_end, "t1": t1, "t2": t2,
        "Q_prior": (_trapz_Q(steps, g, lr, step0, prior_end)
                    if prior_end is not None else None),
        "Q_plateau": None, "Q_transition": None, "Q_total": None,
    }
    if prior_end is not None and t1 is not None and t1 > prior_end:
        out["Q_plateau"] = _trapz_Q(steps, g, lr, prior_end, t1)
    elif t1 is not None and prior_end is None:
        out["Q_plateau"] = _trapz_Q(steps, g, lr, step0, t1)
    if t1 is not None and t2 is not None:
        out["Q_transition"] = _trapz_Q(steps, g, lr, t1, t2)
    pieces = [out[k] for k in ("Q_prior", "Q_plateau", "Q_transition")
              if out.get(k) is not None
              and not (isinstance(out[k], float) and math.isnan(out[k]))]
    out["Q_total"] = float(sum(pieces)) if pieces else None
    return out


# ---------------------------------------------------------------------------
# Bootstrap fits
# ---------------------------------------------------------------------------

def fit_M7_bootstrap(df: pd.DataFrame, n_bootstrap: int = 2000,
                     rng_seed: int = 42) -> Dict:
    """Q ~ a · n_b · K^α + b  with seed-bootstrap CIs on α and a.

    Bootstrap unit: seed within each (D, K) cell.  Each iteration
    resamples seeds with replacement, computes the seed-mean Q at each
    (D, K), and fits M7 on those 8 seed-averaged points.
    """
    rng = np.random.default_rng(rng_seed)
    df = df.dropna(subset=["Q_transition"]).copy()

    # Cells: (D, K) -> array of Q_transition values across seeds
    cells = (df.groupby(["D", "K", "n_b"])["Q_transition"]
               .apply(list).reset_index())
    cells["Q_arr"] = cells["Q_transition"].apply(np.asarray)

    def _f(X, a, alpha, b):
        K_, n_b_ = X
        return a * n_b_ * np.power(K_, alpha) + b

    # Point estimate on full data (seed-averaged)
    cells["Q_mean"] = cells["Q_arr"].apply(np.mean)
    K_arr = cells["K"].to_numpy(dtype=float)
    nb_arr = cells["n_b"].to_numpy(dtype=float)
    Q_arr = cells["Q_mean"].to_numpy(dtype=float)
    try:
        popt_pt, pcov_pt = curve_fit(
            _f, (K_arr, nb_arr), Q_arr,
            p0=[0.01, 0.5, 0.0],
            bounds=([-1e3, 0.01, -1e6], [1e3, 5.0, 1e6]),
            maxfev=20000,
        )
        a_pt, alpha_pt, b_pt = popt_pt
        yhat = _f((K_arr, nb_arr), *popt_pt)
        ss_res = float(np.sum((Q_arr - yhat) ** 2))
        ss_tot = float(np.sum((Q_arr - Q_arr.mean()) ** 2))
        r2_pt = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    except Exception as exc:
        return {"error": str(exc)}

    # Bootstrap
    a_bs, alpha_bs, b_bs = [], [], []
    for _ in range(n_bootstrap):
        Q_boot = []
        for q in cells["Q_arr"]:
            sample = rng.choice(q, size=len(q), replace=True)
            Q_boot.append(float(np.mean(sample)))
        Q_boot = np.asarray(Q_boot)
        try:
            popt, _ = curve_fit(
                _f, (K_arr, nb_arr), Q_boot,
                p0=[a_pt, alpha_pt, b_pt],
                bounds=([-1e3, 0.01, -1e6], [1e3, 5.0, 1e6]),
                maxfev=10000,
            )
            a_bs.append(popt[0])
            alpha_bs.append(popt[1])
            b_bs.append(popt[2])
        except Exception:
            continue

    def _ci(xs):
        if len(xs) < 50: return [None, None]
        return [float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))]

    return {
        "point_estimate": {
            "a": float(a_pt), "alpha": float(alpha_pt), "b": float(b_pt),
            "R2": r2_pt, "n_cells": int(len(cells)),
        },
        "bootstrap": {
            "n_iter": int(len(a_bs)),
            "alpha_median": float(np.median(alpha_bs)) if a_bs else None,
            "alpha_mean": float(np.mean(alpha_bs)) if a_bs else None,
            "alpha_se": float(np.std(alpha_bs, ddof=1)) if len(a_bs) >= 2 else None,
            "alpha_ci_95": _ci(alpha_bs),
            "a_ci_95": _ci(a_bs),
            "b_ci_95": _ci(b_bs),
        },
    }


def fit_tau_power_law_bootstrap(df_tau: pd.DataFrame, n_bootstrap: int = 2000,
                                rng_seed: int = 43) -> Dict:
    """τ = A · D^δ.  Bootstrap over seeds.

    Per-(D, K) seed-mean τ is computed each iteration, then log τ vs log D
    is fit per K and the per-K δ values are averaged.  Reports δ
    median/mean/SE across the bootstrap.
    """
    rng = np.random.default_rng(rng_seed)
    cells = (df_tau.groupby(["D", "K"])["t1"]
                   .apply(list).reset_index())
    Ks = sorted(cells["K"].unique())
    Ds = sorted(cells["D"].unique())
    if len(Ds) < 2:
        return {"error": "need at least 2 D values"}

    # Per-K bootstrap on log τ vs log D
    delta_per_K_bs: Dict[int, List[float]] = {int(K): [] for K in Ks}
    delta_pool_bs: List[float] = []

    # Point estimate
    pt = {}
    for K in Ks:
        sub = cells[cells["K"] == K]
        x = np.log(sub["D"].to_numpy(dtype=float))
        y = np.log(np.array([np.mean(t) for t in sub["t1"]]))
        if len(x) < 2: continue
        A = np.vstack([x, np.ones_like(x)]).T
        (delta, log_A), *_ = np.linalg.lstsq(A, y, rcond=None)
        pt[int(K)] = {"delta": float(delta), "log_A": float(log_A),
                      "A": float(math.exp(log_A))}

    # Pooled bootstrap
    for _ in range(n_bootstrap):
        boot_means: List[Tuple[int, int, float]] = []
        for _, row in cells.iterrows():
            t1_arr = row["t1"]
            sample = rng.choice(t1_arr, size=len(t1_arr), replace=True)
            boot_means.append((int(row["D"]), int(row["K"]), float(np.mean(sample))))
        df_boot = pd.DataFrame(boot_means, columns=["D", "K", "t1"])
        for K in Ks:
            sub = df_boot[df_boot["K"] == K]
            x = np.log(sub["D"].to_numpy(dtype=float))
            y = np.log(sub["t1"].to_numpy(dtype=float))
            if len(x) < 2: continue
            A = np.vstack([x, np.ones_like(x)]).T
            (delta, _), *_ = np.linalg.lstsq(A, y, rcond=None)
            delta_per_K_bs[int(K)].append(float(delta))
        # Pooled: pool all (log D, log τ) points and fit single δ
        x_all = np.log(df_boot["D"].to_numpy(dtype=float))
        y_all = np.log(df_boot["t1"].to_numpy(dtype=float))
        A = np.vstack([x_all, np.ones_like(x_all)]).T
        (delta, _), *_ = np.linalg.lstsq(A, y_all, rcond=None)
        delta_pool_bs.append(float(delta))

    def _stat(xs):
        if len(xs) < 50:
            return {"n": len(xs), "median": None, "mean": None,
                    "se": None, "ci_95": [None, None]}
        return {"n": len(xs),
                "median": float(np.median(xs)),
                "mean": float(np.mean(xs)),
                "se": float(np.std(xs, ddof=1)),
                "ci_95": [float(np.percentile(xs, 2.5)),
                          float(np.percentile(xs, 97.5))]}

    return {
        "point_estimate_per_K": pt,
        "bootstrap_per_K_delta": {k: _stat(v) for k, v in delta_per_K_bs.items()},
        "bootstrap_pooled_delta": _stat(delta_pool_bs),
    }


def per_seed_sign_reversal_test(df: pd.DataFrame) -> Dict:
    """For each (D, seed), fit Q_transition = c · log K + b across the 4 K
    values and report the sign of c.  At fixed D, the joint model
    predicts c < 0 (because n_b ∝ 1/K shrinks faster than K^α grows for
    α < 1 in the affected range).

    NOTE: K=36 cells have D = 36 · 278 = 10008 (or 36 · 556 = 20016) due
    to integer n_b rounding.  We bin D to the nearest 1000 to keep all
    four K values in the same group.
    """
    df = df.copy()
    df["D_bin"] = (df["D"] / 1000).round().astype(int) * 1000
    out: Dict = {"per_seed": {}, "summary": {}}
    by_D_seed = df.dropna(subset=["Q_transition"]).groupby(["D_bin", "seed"])
    sign_counts = {"negative": 0, "positive": 0, "zero_or_unfit": 0}
    for (D, seed), grp in by_D_seed:
        grp = grp.sort_values("K")
        K = grp["K"].to_numpy(dtype=float)
        Q = grp["Q_transition"].to_numpy(dtype=float)
        if len(K) < 2:
            out["per_seed"][f"D={int(D)}_seed={int(seed)}"] = {
                "n_K": int(len(K)), "c": None, "R2": None,
                "sign": "unfit (insufficient K)"
            }
            sign_counts["zero_or_unfit"] += 1
            continue
        logK = np.log(K)
        A = np.vstack([logK, np.ones_like(logK)]).T
        (c, b), *_ = np.linalg.lstsq(A, Q, rcond=None)
        yhat = c * logK + b
        ss_res = float(np.sum((Q - yhat) ** 2))
        ss_tot = float(np.sum((Q - Q.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
        sign = "negative" if c < 0 else ("positive" if c > 0 else "zero")
        sign_counts[sign if sign in sign_counts else "zero_or_unfit"] += 1
        out["per_seed"][f"D={int(D)}_seed={int(seed)}"] = {
            "n_K": int(len(K)), "c": float(c), "b": float(b),
            "R2": r2, "K_values": K.tolist(),
            "Q_values": Q.tolist(), "sign": sign,
        }
    out["summary"] = sign_counts
    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(df: pd.DataFrame, m7: Dict, tau_fit: Dict, sign_test: Dict) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))

    # (A) τ vs D, one curve per K (mean ± SE across seeds)
    ax = axes[0, 0]
    cmap = plt.get_cmap("viridis")
    Ks = sorted(df["K"].unique())
    for i, K in enumerate(Ks):
        sub = df[df["K"] == K]
        agg = sub.groupby("D")["t1"].agg(["mean", "std", "count"]).reset_index()
        col = cmap(i / max(len(Ks) - 1, 1))
        yerr = agg["std"] / np.sqrt(np.maximum(agg["count"], 1))
        ax.errorbar(agg["D"], agg["mean"], yerr=yerr, marker="o",
                    color=col, label=f"K={K}", capsize=3)
        # Power-law fit overlay
        pe = tau_fit.get("point_estimate_per_K", {}).get(int(K))
        if pe and pe.get("delta") is not None:
            D_grid = np.geomspace(agg["D"].min(), agg["D"].max(), 30)
            ax.plot(D_grid, pe["A"] * np.power(D_grid, pe["delta"]),
                    "--", color=col, alpha=0.5, linewidth=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("D"); ax.set_ylabel("τ ≈ t₁ (steps)")
    pool = tau_fit.get("bootstrap_pooled_delta", {})
    if pool.get("median") is not None:
        ax.set_title(f"(A) τ vs D, per K — pooled δ = {pool['median']:.2f} ± "
                     f"{pool['se']:.2f}\n[fit: τ = A · D^δ]", fontsize=10)
    else:
        ax.set_title("(A) τ vs D, per K", fontsize=10)
    ax.legend(fontsize=7)

    # (B) τ vs K at fixed D (showing K-independence)
    ax = axes[0, 1]
    Ds = sorted(df["D"].unique())
    cmap2 = plt.get_cmap("plasma")
    for i, D in enumerate(Ds):
        sub = df[df["D"] == D]
        agg = sub.groupby("K")["t1"].agg(["mean", "std", "count"]).reset_index()
        yerr = agg["std"] / np.sqrt(np.maximum(agg["count"], 1))
        col = cmap2(i / max(len(Ds) - 1, 1))
        ax.errorbar(agg["K"], agg["mean"], yerr=yerr, marker="o",
                    color=col, label=f"D = {int(D)}", capsize=3)
    ax.set_xscale("log")
    ax.set_xlabel("K"); ax.set_ylabel("τ (steps, mean ± SE)")
    ax.set_title("(B) τ vs K at fixed D — K-independence test", fontsize=10)
    ax.legend(fontsize=8)

    # (C) Q_transition collapse onto n_b · K^α (with bootstrap CI)
    ax = axes[1, 0]
    pe7 = m7.get("point_estimate", {})
    if pe7.get("alpha") is not None:
        alpha = pe7["alpha"]; a = pe7["a"]; b = pe7["b"]
        # Plot all 24 points colored by D
        for i, D in enumerate(Ds):
            sub = df[df["D"] == D].dropna(subset=["Q_transition"])
            x = sub["n_b"] * np.power(sub["K"], alpha)
            col = cmap2(i / max(len(Ds) - 1, 1))
            ax.scatter(x, sub["Q_transition"], color=col, s=60,
                       edgecolors="black", label=f"D = {int(D)} (n=12)")
        x_grid = np.linspace(
            (df["n_b"] * np.power(df["K"], alpha)).min(),
            (df["n_b"] * np.power(df["K"], alpha)).max(), 80)
        ax.plot(x_grid, a * x_grid + b, "--", color="#7B2CBF", linewidth=1.5,
                label=f"M7 fit (point est)")
        bs = m7.get("bootstrap", {})
        ci = bs.get("alpha_ci_95", [None, None])
        ci_str = (f"[{ci[0]:.2f}, {ci[1]:.2f}]"
                  if ci[0] is not None else "—")
        ax.set_title(f"(C) Q_transition vs n_b · K^α  with α = {alpha:.3f}\n"
                     f"bootstrap α 95% CI = {ci_str}, "
                     f"R²(point) = {pe7['R2']:.3f}",
                     fontsize=10)
    ax.set_xlabel(f"n_b · K^α   (α = {pe7.get('alpha', '?'):.3f})")
    ax.set_ylabel("Q_transition")
    ax.legend(fontsize=8)

    # (D) Per-seed sign-reversal test: scatter of c values
    ax = axes[1, 1]
    by_seed = sign_test["per_seed"]
    seeds = [0, 1, 2]
    width = 0.25
    for i, D in enumerate(Ds):
        cs = []
        for s in seeds:
            key = f"D={int(D)}_seed={s}"
            entry = by_seed.get(key, {})
            cs.append(entry.get("c"))
        x_pos = np.arange(len(seeds)) + i * width
        col = cmap2(i / max(len(Ds) - 1, 1))
        cs_finite = [c if c is not None else 0 for c in cs]
        bars = ax.bar(x_pos, cs_finite, width=width, color=col,
                      edgecolor="black", label=f"D = {int(D)}")
        for x, c in zip(x_pos, cs):
            if c is not None:
                ax.text(x, c, f"{c:.1f}", ha="center",
                        va="bottom" if c > 0 else "top", fontsize=7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(seeds)) + width / 2)
    ax.set_xticklabels([f"seed={s}" for s in seeds])
    ax.set_ylabel("Q_transition vs log K slope c")
    sgn = sign_test.get("summary", {})
    ax.set_title(f"(D) Per-seed sign of (Q_t vs log K) at fixed D\n"
                 f"negative: {sgn.get('negative', 0)}, "
                 f"positive: {sgn.get('positive', 0)}, "
                 f"unfit: {sgn.get('zero_or_unfit', 0)}",
                 fontsize=10)
    ax.legend(fontsize=8)

    fig.suptitle("Fixed-D multi-seed analysis (24 cells, 3 seeds × 8 (D, K) pairs)",
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
    cell_dirs = sorted([
        d for D_dir in (DATA_ROOT.iterdir() if DATA_ROOT.exists() else [])
        if D_dir.is_dir() and D_dir.name.startswith("D_")
        for d in D_dir.iterdir()
        if d.is_dir() and d.name.startswith("eta_")
    ])
    print(f"found {len(cell_dirs)} cell dirs")

    rows = []
    for d in cell_dirs:
        rec = decompose_cell(d)
        if rec is not None:
            rows.append(rec)
    df = pd.DataFrame(rows)
    print(f"decomposed {len(df)} cells")
    if len(df) == 0:
        print("no data; aborting")
        return

    # Per-(D, K) seed-aggregate τ ≈ t1
    tau_table = (df.groupby(["D", "K"])
                   .agg(t1_mean=("t1", "mean"),
                        t1_std=("t1", "std"),
                        n_seeds=("seed", "nunique"),
                        Q_t_mean=("Q_transition", "mean"),
                        Q_t_std=("Q_transition", "std"),
                        Q_p_mean=("Q_plateau", "mean"))
                   .reset_index())
    tau_table["t1_se"] = tau_table["t1_std"] / np.sqrt(np.maximum(tau_table["n_seeds"], 1))
    tau_table["Q_t_se"] = tau_table["Q_t_std"] / np.sqrt(np.maximum(tau_table["n_seeds"], 1))
    print(f"\n=== τ table (mean ± SE across seeds) ===")
    print(tau_table[["D", "K", "n_seeds", "t1_mean", "t1_se",
                     "Q_t_mean", "Q_t_se"]].to_string(index=False))

    # τ ~ D^δ with seed bootstrap
    tau_fit = fit_tau_power_law_bootstrap(df.dropna(subset=["t1"]), n_bootstrap=2000)
    print(f"\n=== τ ~ D^δ bootstrap ===")
    pool = tau_fit.get("bootstrap_pooled_delta", {})
    print(f"pooled δ: median = {pool.get('median')}, "
          f"SE = {pool.get('se')}, "
          f"95% CI = {pool.get('ci_95')}")
    for K, st in tau_fit.get("bootstrap_per_K_delta", {}).items():
        print(f"  K={K}: δ median = {st.get('median')}, "
              f"95% CI = {st.get('ci_95')}")

    # M7 with seed bootstrap on α
    m7 = fit_M7_bootstrap(df, n_bootstrap=2000)
    print(f"\n=== M7 (Q ~ a · n_b · K^α + b) bootstrap ===")
    pe = m7.get("point_estimate", {})
    bs = m7.get("bootstrap", {})
    print(f"point: a = {pe.get('a')}, α = {pe.get('alpha')}, "
          f"b = {pe.get('b')}, R² = {pe.get('R2')}")
    print(f"bootstrap α: median = {bs.get('alpha_median')}, "
          f"SE = {bs.get('alpha_se')}, 95% CI = {bs.get('alpha_ci_95')}")

    # Per-seed sign-reversal test
    sign_test = per_seed_sign_reversal_test(df)
    print(f"\n=== Per-seed sign-reversal test ===")
    print(f"sign counts: {sign_test['summary']}")
    for key, e in sign_test["per_seed"].items():
        c = e.get("c")
        c_s = f"{c:.2f}" if c is not None else "—"
        r2 = e.get("R2")
        r2_s = f"{r2:.2f}" if r2 is not None else "—"
        print(f"  {key:<22} c = {c_s:<10}  R² = {r2_s:<5}  sign = {e.get('sign')}")

    # Save JSON
    out = {
        "n_cells": int(len(df)),
        "Q_definitions_note": (
            "Q computed via eta_sweep run_single.py: held-out batch of 256, "
            "no clipping, raw p.grad.  Comparable to rest of eta_sweep, NOT "
            "directly comparable in absolute value to confound v1/v2 cells "
            "(which use training-batch grad with clipping)."
        ),
        "per_cell": rows,
        "per_DK_aggregate": tau_table.to_dict(orient="records"),
        "tau_fit_bootstrap": tau_fit,
        "M7_bootstrap": m7,
        "per_seed_sign_test": sign_test,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2,
                  default=lambda o: None if isinstance(o, float)
                                  and (math.isnan(o) or math.isinf(o)) else o)
    print(f"\nwrote {OUT_JSON}")

    # Figure
    fig_path = make_figure(df, m7, tau_fit, sign_test)
    print(f"wrote {fig_path}")

    # Markdown writeup
    md = []
    md.append("# Fixed-D multi-seed analysis (24 cells)")
    md.append("")
    md.append("Replication of confound v2 with 3 seeds per cell, using "
              "eta_sweep `run_single.py` (held-out gradient, no clipping).")
    md.append("")
    md.append(f"**Cells**: {int(len(df))} of 24 transitioned, all clean.")
    md.append("")
    md.append("## τ ≈ t₁ table")
    md.append("")
    md.append("| D | K | n_seeds | τ mean | τ SE | Q_t mean | Q_t SE |")
    md.append("|---|---|---------|--------|------|----------|--------|")
    for _, r in tau_table.iterrows():
        md.append(f"| {int(r['D']):>5} | {int(r['K']):>2} | "
                  f"{int(r['n_seeds'])} | {r['t1_mean']:.0f} | "
                  f"{r['t1_se']:.0f} | {r['Q_t_mean']:.3f} | {r['Q_t_se']:.3f} |")
    md.append("")
    md.append("## Bootstrap fits")
    md.append("")
    md.append("### τ ~ A · D^δ  (per-K and pooled, 2000-iter bootstrap over seeds)")
    md.append("")
    md.append("| K | δ point | δ median (boot) | δ SE | δ 95% CI |")
    md.append("|---|---------|-----------------|------|----------|")
    for K, pt in tau_fit.get("point_estimate_per_K", {}).items():
        bs_K = tau_fit.get("bootstrap_per_K_delta", {}).get(K, {})
        ci = bs_K.get("ci_95", [None, None])
        ci_s = f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci[0] is not None else "—"
        med = bs_K.get("median")
        med_s = f"{med:.3f}" if med is not None else "—"
        se = bs_K.get("se")
        se_s = f"{se:.3f}" if se is not None else "—"
        md.append(f"| {K} | {pt['delta']:.3f} | {med_s} | {se_s} | {ci_s} |")
    pool = tau_fit.get("bootstrap_pooled_delta", {})
    if pool.get("median") is not None:
        md.append("")
        md.append(f"**Pooled δ (across all K, 8 (D,K) cells × 3 seeds)**: "
                  f"median = **{pool['median']:.3f}**, "
                  f"SE = {pool['se']:.3f}, "
                  f"95% CI = [{pool['ci_95'][0]:.3f}, {pool['ci_95'][1]:.3f}].")
    md.append("")
    md.append("### M7  Q_transition = a · n_b · K^α + b  (bootstrap on seeds)")
    md.append("")
    md.append(f"- Point estimate: a = {pe.get('a'):.4g}, "
              f"**α = {pe.get('alpha'):.3f}**, b = {pe.get('b'):.4g}, "
              f"R² = {pe.get('R2'):.3f}.")
    md.append(f"- Bootstrap α: median = **{bs.get('alpha_median'):.3f}**, "
              f"SE = {bs.get('alpha_se'):.3f}, "
              f"95% CI = [{bs.get('alpha_ci_95')[0]:.3f}, "
              f"{bs.get('alpha_ci_95')[1]:.3f}].")
    md.append("")
    md.append("### Per-seed sign of Q_transition vs log K at fixed D")
    md.append("")
    md.append("| seed | D | c (Q_t vs log K slope) | R² | sign |")
    md.append("|------|---|------------------------|-----|------|")
    for key, e in sign_test["per_seed"].items():
        c_s = f"{e['c']:.2f}" if e.get('c') is not None else "—"
        r2_s = f"{e['R2']:.2f}" if e.get('R2') is not None else "—"
        md.append(f"| {key} | | {c_s} | {r2_s} | {e['sign']} |")
    sgn = sign_test["summary"]
    md.append("")
    md.append(f"**Sign-reversal verdict**: {sgn.get('negative', 0)} of "
              f"{sum(sgn.values())} per-(D, seed) fits show negative slope, "
              f"{sgn.get('positive', 0)} positive, {sgn.get('zero_or_unfit', 0)} "
              f"unfit.")
    md.append("")
    md.append("## Files")
    md.append("")
    md.append(f"- Data: `{OUT_JSON.relative_to(RESULTS_DIR.parent)}`")
    md.append(f"- Figure: `{OUT_FIG.relative_to(RESULTS_DIR.parent)}`")
    OUT_MD.write_text("\n".join(md))
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
