#!/usr/bin/env python
"""
Experiment 1 (v2): Dataset-Size Confound Control

Reviewer concern: D = n_unique_b × K, so larger K means more training data.
The plateau and scaling laws might reflect dataset size, not ambiguity.

Design: Hold total dataset size D constant by varying n_unique_b inversely with K.
- Target D = 10,000 examples: K=5 (n=2000), K=10 (1000), K=20 (500), K=36 (~278)
- Target D = 20,000 examples: K=5 (4000), K=10 (2000), K=20 (1000), K=36 (~556)

CRITICAL: Uses identical hyperparameters to baseline landauer_dense experiments:
- split_by_base=True, enforce_unique_a_first_char_per_b=True
- warmup_steps=0, scheduler=constant
- η=1e-3, seed=42

Expected output: Table of τ(K) at fixed D. If τ still scales with K,
the confound is ruled out.
"""

import sys
import json
import math
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.experiment_helpers import (
    make_config, run_parallel, run_exists, load_history, detect_tau,
)
from omegaconf import OmegaConf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ─────────────────────────────────────────────────────────────────
OUTPUT_DIR = "outputs"
FIG_DIR = Path(OUTPUT_DIR) / "paper_figures"

TARGETS = {
    10000: {5: 2000, 10: 1000, 20: 500, 36: 278},
    20000: {5: 4000, 10: 2000, 20: 1000, 36: 556},
}

SEED = 42
LR = 1e-3
MAX_STEPS = 50000

# Run names use v2 prefix to avoid collision with old (mis-configured) runs
RUN_PREFIX = "confound_v2"


def fit_power_law(k_arr, tau_arr, n_bootstrap=2000):
    """Fit τ = C * K^δ in log-log space with bootstrap CIs."""
    mask = np.isfinite(tau_arr) & (tau_arr > 0)
    k_arr, tau_arr = k_arr[mask], tau_arr[mask]
    if len(k_arr) < 2:
        return None, None, None
    log_k, log_tau = np.log(k_arr), np.log(tau_arr)
    A = np.vstack([log_k, np.ones(len(log_k))]).T
    delta, _ = np.linalg.lstsq(A, log_tau, rcond=None)[0]
    rng = np.random.RandomState(42)
    n = len(log_k)
    deltas = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        try:
            d_b, _ = np.linalg.lstsq(
                np.vstack([log_k[idx], np.ones(len(idx))]).T,
                log_tau[idx], rcond=None)[0]
            deltas.append(d_b)
        except Exception:
            continue
    deltas = np.array(deltas)
    return delta, np.percentile(deltas, 2.5), np.percentile(deltas, 97.5)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT 1 (v2): DATASET-SIZE CONFOUND CONTROL")
    print("  Config: split_by_base=True, enforce_unique_a_first_char=True,")
    print("          warmup=0, scheduler=constant (matches baseline)")
    print("=" * 70)

    # ── Build jobs ─────────────────────────────────────────────────────────
    jobs = []
    all_configs = []
    for D, k_map in TARGETS.items():
        for k, n_b in k_map.items():
            name = f"{RUN_PREFIX}_D{D}_K{k}_n{n_b}"
            all_configs.append((D, k, n_b, name))
            if run_exists(name, min_steps=MAX_STEPS * 0.3, output_dir=OUTPUT_DIR):
                print(f"  SKIP {name} (already exists)")
                continue

            print(f"  QUEUE {name}")
            cfg = make_config(
                experiment_name=name,
                k=k,
                n_unique_b=n_b,
                lr=LR,
                seed=SEED,
                max_steps=MAX_STEPS,
                early_stop_frac=0.01,
            )
            jobs.append({
                "cfg_dict": OmegaConf.to_container(cfg, resolve=True),
                "mapping_path": None,
                "output_dir": OUTPUT_DIR,
                "name": name,
            })

    # ── Run ────────────────────────────────────────────────────────────────
    if jobs:
        run_parallel(jobs, max_workers=6, label="confound-v2")
    else:
        print("  All runs already exist!")

    # ── Convergence diagnostics ───────────────────────────────────────────
    print(f"\n{'='*80}")
    print("CONVERGENCE DIAGNOSTICS")
    print(f"{'='*80}")
    print(f"{'D':<8} {'K':<6} {'n_b':<8} {'Final loss':>12} {'Cand acc':>10} "
          f"{'Z-gap':>8} {'Steps':>8} {'Early?':>8}")
    print("-" * 80)

    results = {}
    convergence_info = {}
    for D, k, n_b, name in all_configs:
        h = load_history(name, OUTPUT_DIR)
        if h is None:
            print(f"  {D:<8} {k:<6} {n_b:<8} {'MISS':>12}")
            continue

        final_loss = h["first_target_loss"][-1] if h.get("first_target_loss") else None
        cand_acc = h["candidate_accuracy"][-1] if h.get("candidate_accuracy") else None
        z_shuf = h["loss_z_shuffled"][-1] if h.get("loss_z_shuffled") else None
        z_gap = (z_shuf - final_loss) if (z_shuf is not None and final_loss is not None) else None
        total_steps = h["steps"][-1] if h.get("steps") else 0
        early = h.get("early_stopped", False)

        fl_str = f"{final_loss:.4f}" if final_loss is not None else "N/A"
        ca_str = f"{cand_acc:.1%}" if cand_acc is not None else "N/A"
        zg_str = f"{z_gap:.2f}" if z_gap is not None else "N/A"
        ea_str = "Yes" if early else "No"

        print(f"  {D:<8} {k:<6} {n_b:<8} {fl_str:>12} {ca_str:>10} "
              f"{zg_str:>8} {total_steps:>8} {ea_str:>8}")

        if D not in convergence_info:
            convergence_info[D] = {}
        convergence_info[D][k] = {
            "n_unique_b": n_b,
            "final_loss": final_loss,
            "candidate_accuracy": cand_acc,
            "z_shuffle_loss": z_shuf,
            "z_gap": z_gap,
            "total_steps": total_steps,
            "early_stopped": early,
        }

    # ── τ detection ───────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("τ DETECTION")
    print(f"{'='*80}")
    print(f"{'D':<8} {'K':<6} {'n_b':<8} {'τ':>8} {'log K':>8} {'Converged?':>12}")
    print("-" * 54)

    for D, k, n_b, name in all_configs:
        h = load_history(name, OUTPUT_DIR)
        if h is None:
            continue
        log_k = math.log(k)
        tau = detect_tau(h, log_k)
        converged = tau is not None
        if D not in results:
            results[D] = {}
        results[D][k] = {"n_unique_b": n_b, "tau": tau}
        tau_str = str(tau) if tau else "N/C"
        print(f"  {D:<8} {k:<6} {n_b:<8} {tau_str:>8} {log_k:>8.3f} {'Yes' if converged else 'NO':>12}")

    # ── Power-law fits ─────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("POWER-LAW FITS: τ ∝ K^δ at each fixed D")
    print(f"{'='*80}")
    print(f"{'D':<8} {'δ':>8} {'95% CI':>18} {'n':>4}")
    print("-" * 42)

    fit_results = {}
    for D in sorted(results.keys()):
        ks = sorted([k for k in results[D] if results[D][k]["tau"] is not None])
        if len(ks) < 2:
            print(f"  {D:<8} {'N/A':>8} (only {len(ks)} converged)")
            continue
        k_arr = np.array(ks, dtype=float)
        tau_arr = np.array([results[D][k]["tau"] for k in ks], dtype=float)
        delta, ci_lo, ci_hi = fit_power_law(k_arr, tau_arr)
        fit_results[D] = {"delta": delta, "ci_lo": ci_lo, "ci_hi": ci_hi,
                          "n": len(ks), "k_values": ks,
                          "tau_values": tau_arr.tolist()}
        if delta is not None:
            print(f"  {D:<8} {delta:>8.3f} [{ci_lo:.3f}, {ci_hi:.3f}] {len(ks):>4}")

    # ── Comparison with baselines ─────────────────────────────────────────
    print(f"\n{'='*80}")
    print("COMPARISON: same-K same-n_unique_b across experiments")
    print(f"{'='*80}")
    # K=10 n=1000 appears in both D=10000 confound AND baseline
    # K=20 n=1000 appears in both D=20000 confound AND baseline
    baseline_runs = {
        5: "landauer_dense_k5",
        10: "landauer_dense_k10",
        20: "landauer_dense_k20",
        36: "landauer_dense_k36",
    }
    print(f"  {'K':<6} {'Baseline τ':>12} {'Confound τ':>12} {'D_conf':>8} {'Match?':>8}")
    print("  " + "-" * 52)
    for k in [5, 10, 20, 36]:
        bname = baseline_runs.get(k)
        bh = load_history(bname, OUTPUT_DIR) if bname else None
        b_tau = detect_tau(bh, math.log(k)) if bh else None
        # Find confound run with same n_unique_b=1000
        c_tau = None
        c_D = None
        for D in sorted(results.keys()):
            if k in results[D] and results[D][k].get("n_unique_b") == 1000 - 0:
                # n_unique_b = D/K, so D/K=1000 → D=1000*K
                pass
            if k in results[D]:
                entry = results[D][k]
                if entry["n_unique_b"] == 1000:
                    c_tau = entry["tau"]
                    c_D = D
        b_str = str(b_tau) if b_tau else "N/A"
        c_str = str(c_tau) if c_tau else "N/A"
        d_str = str(c_D) if c_D else "N/A"
        match = "~" if (b_tau and c_tau and abs(b_tau - c_tau) / b_tau < 0.15) else ""
        print(f"  {k:<6} {b_str:>12} {c_str:>12} {d_str:>8} {match:>8}")

    # ── Figure ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: τ(K) at fixed D + baseline overlay
    ax = axes[0]
    colors_D = {10000: "#E74C3C", 20000: "#3498DB"}
    markers_D = {10000: "o", 20000: "s"}

    for D in sorted(results.keys()):
        ks = sorted([k for k in results[D] if results[D][k]["tau"] is not None])
        if not ks:
            continue
        k_arr = np.array(ks, dtype=float)
        tau_arr = np.array([results[D][k]["tau"] for k in ks], dtype=float)
        label = f"D={D:,} (fixed)"
        if D in fit_results and fit_results[D]["delta"] is not None:
            label += f" δ={fit_results[D]['delta']:.2f}"
        ax.loglog(k_arr, tau_arr, f"{markers_D.get(D, 'o')}-",
                  color=colors_D.get(D, "gray"),
                  label=label, markersize=7, linewidth=1.5)

    # Overlay baseline (n_unique_b=1000)
    bks, btaus = [], []
    for k, bname in sorted(baseline_runs.items()):
        bh = load_history(bname, OUTPUT_DIR)
        if bh:
            bt = detect_tau(bh, math.log(k))
            if bt:
                bks.append(k)
                btaus.append(bt)
    if bks:
        ax.loglog(bks, btaus, "k^--", markersize=7, linewidth=1.5,
                  label=f"Baseline (n_b=1000, D=1000K)")

    ax.set_xlabel("K (ambiguity)", fontsize=10)
    ax.set_ylabel("τ (transition time, steps)", fontsize=10)
    ax.set_title("τ(K): fixed-D vs baseline", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

    # Panel B: Loss curves for D=10000 (all K values)
    ax = axes[1]
    colors_K = {5: "#E74C3C", 10: "#3498DB", 20: "#2ECC71", 36: "#9B59B6"}
    for D, k, n_b, name in all_configs:
        if D != 10000:
            continue
        h = load_history(name, OUTPUT_DIR)
        if h is None:
            continue
        steps = h["steps"]
        loss = h.get("candidate_loss") or h.get("first_target_loss")
        if loss:
            ax.plot(steps, loss, color=colors_K.get(k, "gray"), linewidth=1.2,
                    label=f"K={k}, n_b={n_b}", alpha=0.8)
            ax.axhline(0.5 * math.log(k), color=colors_K.get(k, "gray"),
                       linestyle=":", alpha=0.3)

    ax.set_xlabel("Step", fontsize=10)
    ax.set_ylabel("Candidate loss", fontsize=10)
    ax.set_title("D=10,000: loss curves by K", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    ax.set_ylim(bottom=-0.1)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(FIG_DIR / f"fig_dataset_confound.{ext}", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved: {FIG_DIR}/fig_dataset_confound")

    # ── Save summary ───────────────────────────────────────────────────────
    summary = {
        "experiment": "dataset_confound_v2",
        "description": "Dataset-size confound control with baseline-matched config (Exp 1 v2)",
        "config_notes": "split_by_base=True, enforce_unique_a_first_char=True, warmup=0, scheduler=constant",
        "results": {str(D): {str(k): v for k, v in kv.items()}
                    for D, kv in results.items()},
        "convergence": {str(D): {str(k): v for k, v in kv.items()}
                        for D, kv in convergence_info.items()},
        "power_law_fits": {str(D): v for D, v in fit_results.items()},
    }
    summary_path = Path(OUTPUT_DIR) / "dataset_confound_v2_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
