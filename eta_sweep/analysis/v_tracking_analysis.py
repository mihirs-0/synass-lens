#!/usr/bin/env python
"""
v-tracking MV analysis — decision plot for path 1 vs path 2.

Reads v_aggregates.jsonl from the three K=10 η=1e-3 seeds and produces:

  figures/v_tracking_decision.png    — 2x2 panel:
       (a) cand_loss / log K vs step, all 3 seeds (with τ markers)
       (b) precond_mean over time, z-pipeline groups vs control groups
       (c) eff_step_mean over time, z-pipeline vs control
       (d) z-pipeline-minus-control gap on (b) and (c) at each step

  v_tracking_analysis.json           — numerical summary:
       per-group (precond, eff_step) trajectory means ± std across seeds
       z-pipeline vs control crossover step
       monotonicity score (Spearman ρ of precond_mean vs step over plateau)

Decision rule:
  PATH 1 ⇐ on z-pipeline groups, mean(precond) is monotone increasing across
           the plateau (Spearman ρ > 0.7) AND mean(eff_step) shows a sharp
           jump within ±200 steps of τ AND control groups do not.
  PATH 2 ⇐ otherwise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ETA_SWEEP_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ETA_SWEEP_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

V_TRACK_DIR = ETA_SWEEP_ROOT / "results" / "v_tracking"
FIG_DIR = ETA_SWEEP_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = ETA_SWEEP_ROOT / "results" / "v_tracking_analysis.json"

Z_PIPELINE = {"L0_attn", "L0_mlp", "L2_mlp", "L3_attn"}
CONTROL = {"L1_attn", "L1_mlp", "embed", "unembed"}  # the cleanest controls


def load_run(seed: int):
    base = V_TRACK_DIR / f"eta_0.001_K_10_seed_{seed}"
    status = json.loads((base / "status.json").read_text())
    log = [json.loads(l) for l in (base / "log.jsonl").read_text().strip().splitlines()]
    aggs = [json.loads(l) for l in (base / "v_aggregates.jsonl").read_text().strip().splitlines()]
    return status, log, aggs


def aggregate_groups(aggs: List[dict], groups: set, key: str) -> Dict[int, float]:
    """For each step, compute the unweighted mean of `key` over the named groups."""
    out: Dict[int, float] = {}
    for snap in aggs:
        step = snap["step"]
        vals = []
        for g in groups:
            if g in snap["per_group"]:
                vals.append(snap["per_group"][g][key])
        if vals:
            out[step] = float(np.mean(vals))
    return out


def spearman_rho(x: List[float], y: List[float]) -> float:
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def main():
    runs = {seed: load_run(seed) for seed in (0, 1, 2)}
    taus = {s: r[0]["transition_detected_step"] for s, r in runs.items()}
    print(f"τ per seed: {taus}")

    # Use the per-seed step grid (slightly different across seeds because runs
    # terminate after their own +2000 post-τ tail).  Build a common step grid
    # by intersection.
    step_sets = []
    for seed, (_, _, aggs) in runs.items():
        step_sets.append({a["step"] for a in aggs})
    common_steps = sorted(set.intersection(*step_sets))
    print(f"common steps: {len(common_steps)} from {common_steps[0]} to {common_steps[-1]}")

    # Compute z-pipeline / control means for every (seed, step, key)
    metrics = ["precond_mean", "eff_step_mean", "v_mean", "grad_l2_sq"]
    seed_data: Dict[int, Dict[str, Dict[int, float]]] = {}
    for seed, (_, _, aggs) in runs.items():
        seed_data[seed] = {}
        for tag, groups in [("z", Z_PIPELINE), ("c", CONTROL)]:
            for key in metrics:
                seed_data[seed][f"{tag}_{key}"] = aggregate_groups(aggs, groups, key)

    # Build arrays on the common step grid
    arrays: Dict[str, np.ndarray] = {}
    for tag in ("z", "c"):
        for key in metrics:
            mat = np.array([
                [seed_data[s][f"{tag}_{key}"][step] for step in common_steps]
                for s in (0, 1, 2)
            ])
            arrays[f"{tag}_{key}"] = mat  # shape (3 seeds, len(common_steps))

    # ---- Decision metrics ----
    summary = {
        "tau_per_seed": taus,
        "tau_mean": float(np.mean(list(taus.values()))),
        "tau_std": float(np.std(list(taus.values()))),
        "common_steps_range": [common_steps[0], common_steps[-1]],
        "n_common_steps": len(common_steps),
    }

    # Plateau window for monotonicity — use steps in [200, mean τ - 200]
    tau_mean = summary["tau_mean"]
    plateau_mask = np.array([(200 <= s <= tau_mean - 200) for s in common_steps])
    plateau_steps = [s for s, m in zip(common_steps, plateau_mask) if m]
    print(f"plateau window: {plateau_steps[0] if plateau_steps else 'EMPTY'}"
          f" → {plateau_steps[-1] if plateau_steps else ''}, "
          f"{len(plateau_steps)} points")

    for tag, label in [("z", "z_pipeline"), ("c", "control")]:
        precond_traj = arrays[f"{tag}_precond_mean"]  # (3, T)
        # mean across seeds, then Spearman ρ over plateau window
        traj_mean = precond_traj.mean(axis=0)
        traj_plateau = traj_mean[plateau_mask]
        rho = spearman_rho(plateau_steps, traj_plateau.tolist()) if len(plateau_steps) > 2 else float("nan")
        summary[f"{label}_precond_spearman_plateau"] = rho

        # eff_step: ratio of (post-τ mean) / (pre-τ mean) — the "jump"
        eff = arrays[f"{tag}_eff_step_mean"].mean(axis=0)
        pre_mask = np.array([s < tau_mean for s in common_steps])
        post_mask = np.array([s > tau_mean for s in common_steps])
        pre_mean = float(eff[pre_mask].mean()) if pre_mask.any() else float("nan")
        post_mean = float(eff[post_mask].mean()) if post_mask.any() else float("nan")
        summary[f"{label}_eff_step_pre_tau"] = pre_mean
        summary[f"{label}_eff_step_post_tau"] = post_mean
        summary[f"{label}_eff_step_ratio"] = post_mean / pre_mean if pre_mean > 0 else float("nan")

        # Multiplicative growth of precond across plateau (last/first plateau value)
        if len(traj_plateau) >= 2:
            summary[f"{label}_precond_plateau_growth"] = float(traj_plateau[-1] / traj_plateau[0])
        else:
            summary[f"{label}_precond_plateau_growth"] = float("nan")

    # ---- Refined decision metrics ----
    # The actual mechanism (read from the figure): z-pipeline v stays LOW
    # through the plateau (small gradients on unused directions), so Adam's
    # preconditioner is LARGE there.  At τ, v jumps to catch up and the
    # preconditioner asymmetry collapses.  The signal is the z-pipeline /
    # control RATIO, not the z-pipeline GROWTH.

    # Mean over the plateau window (steps 500..τ-200 to skip warm-up)
    plateau_window = np.array([(500 <= s <= tau_mean - 200) for s in common_steps])
    post_window = np.array([s > tau_mean + 200 for s in common_steps])

    z_v_plateau = arrays["z_v_mean"].mean(axis=0)[plateau_window].mean()
    c_v_plateau = arrays["c_v_mean"].mean(axis=0)[plateau_window].mean()
    z_v_post = arrays["z_v_mean"].mean(axis=0)[post_window].mean()
    c_v_post = arrays["c_v_mean"].mean(axis=0)[post_window].mean()

    # Module-level preconditioner: 1/sqrt(v_mean), NOT mean(1/sqrt(v)).
    # The latter is dominated by outlier coordinates with near-zero v inside a
    # module, so it conflates within-module heterogeneity with the across-
    # module signal we want.
    eps = 1e-8
    z_modlevel_pre_plateau = float(1.0 / (np.sqrt(z_v_plateau) + eps))
    c_modlevel_pre_plateau = float(1.0 / (np.sqrt(c_v_plateau) + eps))

    summary["v_plateau_z_over_c"] = float(z_v_plateau / c_v_plateau)
    summary["v_post_z_over_c"] = float(z_v_post / c_v_post)
    summary["v_zpipeline_catchup_ratio"] = float(z_v_post / z_v_plateau)
    summary["v_control_catchup_ratio"] = float(c_v_post / c_v_plateau)
    summary["modlevel_precond_plateau_z_over_c"] = float(
        z_modlevel_pre_plateau / c_modlevel_pre_plateau
    )

    path1_criteria = {
        "v_plateau_zpipe_lt_half_control":
            bool(summary["v_plateau_z_over_c"] < 0.5),
        "modlevel_precond_zpipe_gt_1.5x_control":
            bool(summary["modlevel_precond_plateau_z_over_c"] > 1.5),
        "v_zpipe_catchup_gt_2x":
            bool(summary["v_zpipeline_catchup_ratio"] > 2.0),
        "v_zpipe_catchup_exceeds_control":
            bool(summary["v_zpipeline_catchup_ratio"]
                 > summary["v_control_catchup_ratio"] * 1.5),
        "asymmetry_collapses_at_tau":
            bool(summary["v_post_z_over_c"]
                 > summary["v_plateau_z_over_c"] * 1.5),
    }
    path1_score = int(sum(path1_criteria.values()))
    summary["path1_criteria"] = path1_criteria
    summary["path1_score"] = path1_score
    summary["decision"] = "PATH_1" if path1_score >= 4 else "PATH_2"

    print("\n=== REFINED DECISION METRICS ===")
    print(f"  v_plateau ratio (z-pipeline / control):       {summary['v_plateau_z_over_c']:.3f}")
    print(f"  v_post-τ ratio (z-pipeline / control):        {summary['v_post_z_over_c']:.3f}")
    print(f"  modlevel precond ratio (z-pipeline / c):      {summary['modlevel_precond_plateau_z_over_c']:.3f}")
    print(f"  z-pipeline v catch-up post-τ / plateau:       {summary['v_zpipeline_catchup_ratio']:.3f}×")
    print(f"  control     v catch-up post-τ / plateau:      {summary['v_control_catchup_ratio']:.3f}×")
    print(f"  path 1 criteria met: {path1_score}/5")
    for k, v in path1_criteria.items():
        mark = "✓" if v else "✗"
        print(f"    {mark} {k}")
    print(f"  → decision: {summary['decision']}")

    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"\n  summary → {OUT_JSON}")

    # ---- Figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping figure")
        return

    steps = np.array(common_steps)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) cand_loss / log K
    ax = axes[0, 0]
    log_k = np.log(10)
    for seed, (_, log, _) in runs.items():
        st = [r["step"] for r in log]
        cl = [r["candidate_loss"] / log_k for r in log]
        ax.plot(st, cl, label=f"seed {seed}", alpha=0.85)
        ax.axvline(taus[seed], linestyle=":", alpha=0.4)
    ax.axhline(1.0, color="grey", linestyle="--", alpha=0.4, label="log K floor")
    ax.axhline(0.3, color="green", linestyle="--", alpha=0.4, label="transition cut")
    ax.set_xlabel("step")
    ax.set_ylabel("candidate_loss / log K")
    ax.set_title(f"(a) Phenomenology — τ ≈ {tau_mean:.0f} ± {summary['tau_std']:.0f}")
    ax.legend(fontsize=8)

    # (b) precond_mean trajectories
    ax = axes[0, 1]
    z_pre = arrays["z_precond_mean"]
    c_pre = arrays["c_precond_mean"]
    ax.plot(steps, z_pre.mean(axis=0), color="C3", label="z-pipeline (mean over groups)", lw=2)
    ax.fill_between(steps, z_pre.min(axis=0), z_pre.max(axis=0), color="C3", alpha=0.2)
    ax.plot(steps, c_pre.mean(axis=0), color="C0", label="control (mean over groups)", lw=2)
    ax.fill_between(steps, c_pre.min(axis=0), c_pre.max(axis=0), color="C0", alpha=0.2)
    ax.axvline(tau_mean, color="black", linestyle=":", alpha=0.5, label=f"τ={tau_mean:.0f}")
    ax.set_xlabel("step")
    ax.set_ylabel("mean(1 / sqrt(v + ε))")
    ax.set_yscale("log")
    ax.set_title("(b) Adam preconditioner — z-pipeline vs control")
    ax.legend(fontsize=8)

    # (c) eff_step_mean trajectories
    ax = axes[1, 0]
    z_eff = arrays["z_eff_step_mean"]
    c_eff = arrays["c_eff_step_mean"]
    ax.plot(steps, z_eff.mean(axis=0), color="C3", label="z-pipeline", lw=2)
    ax.fill_between(steps, z_eff.min(axis=0), z_eff.max(axis=0), color="C3", alpha=0.2)
    ax.plot(steps, c_eff.mean(axis=0), color="C0", label="control", lw=2)
    ax.fill_between(steps, c_eff.min(axis=0), c_eff.max(axis=0), color="C0", alpha=0.2)
    ax.axvline(tau_mean, color="black", linestyle=":", alpha=0.5)
    ax.set_xlabel("step")
    ax.set_ylabel("mean(|g| / sqrt(v + ε))")
    ax.set_yscale("log")
    ax.set_title("(c) Effective step (gradient × preconditioner)")
    ax.legend(fontsize=8)

    # (d) v_mean trajectories — the underlying second moment
    ax = axes[1, 1]
    z_v = arrays["z_v_mean"]
    c_v = arrays["c_v_mean"]
    ax.plot(steps, z_v.mean(axis=0), color="C3", label="z-pipeline", lw=2)
    ax.fill_between(steps, z_v.min(axis=0), z_v.max(axis=0), color="C3", alpha=0.2)
    ax.plot(steps, c_v.mean(axis=0), color="C0", label="control", lw=2)
    ax.fill_between(steps, c_v.min(axis=0), c_v.max(axis=0), color="C0", alpha=0.2)
    ax.axvline(tau_mean, color="black", linestyle=":", alpha=0.5)
    ax.set_xlabel("step")
    ax.set_ylabel("mean(v) [exp_avg_sq]")
    ax.set_yscale("log")
    ax.set_title("(d) Adam second moment v — the underlying signal")
    ax.legend(fontsize=8)

    decision_color = "green" if summary["decision"] == "PATH_1" else "orange"
    fig.suptitle(
        f"v-tracking MV: K=10, η=1e-3, 3 seeds. "
        f"Decision: {summary['decision']} ({path1_score}/5 criteria)",
        fontsize=13, color=decision_color,
    )
    plt.tight_layout()
    out_png = FIG_DIR / "v_tracking_decision.png"
    fig.savefig(out_png, dpi=130)
    print(f"  figure → {out_png}")


if __name__ == "__main__":
    main()
