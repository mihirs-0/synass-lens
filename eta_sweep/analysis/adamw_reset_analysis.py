#!/usr/bin/env python
"""
Q2: Adam moment reset — analysis of τ shift.

Compares:
  - Baseline (no reset): existing v_tracking runs at K=10, η=1e-3, 3 seeds
  - Reset at mid-plateau (step 1500): new adamw_reset runs
  - Reset at late-plateau (step 2200): new adamw_reset runs

Three possible outcomes per the experimental design:
  (A) τ_reset ≈ τ_baseline  → optimizer state isn't carrying meaningful drift
  (B) τ_reset ≈ τ_baseline + (reset_step - some_intercept)  → moments are
      accumulating linearly and reset removes that progress
  (C) τ_reset > budget  → moments are essentially the entire mechanism

Output: outputs/paper_figures/fig_adamw_reset.{pdf,png}
        eta_sweep/results/adamw_reset_analysis.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = REPO_ROOT / "eta_sweep" / "results"
OUT = REPO_ROOT / "outputs" / "paper_figures"


def load_run(path: Path) -> dict | None:
    """Load a run's status + log."""
    status_f = path / "status.json"
    log_f = path / "log.jsonl"
    if not status_f.exists():
        return None
    s = json.loads(status_f.read_text())
    log = [json.loads(l) for l in log_f.read_text().strip().splitlines()] if log_f.exists() else []
    return {"status": s, "log": log}


def main():
    # ---- Baselines: existing v_tracking runs ----
    baselines = {}
    for seed in (0, 1, 2):
        r = load_run(RESULTS / "v_tracking" / f"eta_0.001_K_10_seed_{seed}")
        if r is None:
            continue
        baselines[seed] = r["status"]["transition_detected_step"]

    # ---- Reset runs ----
    resets = {1500: {}, 2200: {}}
    for reset_step in (1500, 2200):
        for seed in (0, 1, 2):
            d = RESULTS / "adamw_reset" / f"eta_0.001_K_10_seed_{seed}_reset_{reset_step}"
            r = load_run(d)
            if r is None:
                continue
            resets[reset_step][seed] = {
                "tau": r["status"]["transition_detected_step"],
                "status": r["status"]["status"],
                "log": r["log"],
            }

    # ---- Tabulate ----
    print("=== Baseline τ (no reset) ===")
    for s, t in baselines.items():
        print(f"  seed {s}: τ={t}")
    print()
    for reset_step, group in resets.items():
        print(f"=== Reset at step {reset_step} ===")
        for s, info in group.items():
            base = baselines.get(s)
            tau = info["tau"]
            delay = (tau - base) if (tau and base) else None
            delay_from_reset = (tau - reset_step) if tau else None
            base_from_reset = (base - reset_step) if base else None
            print(f"  seed {s}: τ_reset={tau}  status={info['status']}  "
                  f"  baseline τ={base}  Δτ={delay}"
                  f"  τ_reset-reset_step={delay_from_reset}  base-reset={base_from_reset}")
        print()

    # ---- Save ----
    out = {
        "baselines": baselines,
        "resets": {str(rs): {str(s): {"tau": info["tau"], "status": info["status"]}
                              for s, info in g.items()} for rs, g in resets.items()},
    }
    (RESULTS / "adamw_reset_analysis.json").write_text(json.dumps(out, indent=2))

    # ---- Figure ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    log_K = np.log(10)

    # Panel (a): trajectories
    ax = axes[0]
    for seed in sorted(baselines.keys()):
        r = load_run(RESULTS / "v_tracking" / f"eta_0.001_K_10_seed_{seed}")
        if r is None:
            continue
        st = np.array([row["step"] for row in r["log"]])
        cl = np.array([row["candidate_loss"] / log_K for row in r["log"]])
        ax.plot(st, cl, color="C2", alpha=0.5, lw=1.2,
                label=f"baseline (s{seed}, τ={baselines[seed]})" if seed == 0 else None)
        ax.axvline(baselines[seed], color="C2", linestyle=":", alpha=0.3)

    colors = {1500: "C0", 2200: "C3"}
    for reset_step, group in resets.items():
        for seed, info in group.items():
            log = info["log"]
            if not log:
                continue
            st = np.array([row["step"] for row in log])
            cl = np.array([row["candidate_loss"] / log_K for row in log])
            label = f"reset@{reset_step} (s{seed}, τ={info['tau']})" if seed == 0 else None
            ax.plot(st, cl, color=colors[reset_step], alpha=0.6, lw=1.4, label=label)
            ax.axvline(reset_step, color=colors[reset_step], linestyle="--", alpha=0.5, lw=1.0)

    ax.axhline(1.0, color="grey", linestyle="--", lw=0.6, alpha=0.4)
    ax.axhline(0.5, color="black", linestyle=":", lw=0.6, alpha=0.4)
    ax.set_xlabel("step")
    ax.set_ylabel("candidate loss / log K")
    ax.set_title("(a) Trajectories: baseline (green) vs reset")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(0, 6500)
    ax.set_ylim(-0.05, 1.2)

    # Panel (b): τ comparison bar plot
    ax = axes[1]
    seeds = sorted(baselines.keys())
    x = np.arange(len(seeds))
    width = 0.27
    base_taus = [baselines[s] for s in seeds]
    r1500_taus = [resets[1500].get(s, {"tau": None})["tau"] or 0 for s in seeds]
    r2200_taus = [resets[2200].get(s, {"tau": None})["tau"] or 0 for s in seeds]
    ax.bar(x - width, base_taus, width, color="C2", alpha=0.75, label="baseline (no reset)")
    ax.bar(x,         r1500_taus, width, color="C0", alpha=0.75, label="reset @ step 1500")
    ax.bar(x + width, r2200_taus, width, color="C3", alpha=0.75, label="reset @ step 2200")
    ax.axhline(np.mean(base_taus), color="C2", linestyle=":", lw=0.7,
               label=f"baseline mean = {np.mean(base_taus):.0f}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel(r"transition step $\tau$")
    ax.set_title("(b) τ shift after Adam moment reset")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3, lw=0.4)

    fig.suptitle("Causal test: zeroing Adam's $m_t$ and $v_t$ mid-plateau", fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT / "fig_adamw_reset.pdf", dpi=200)
    fig.savefig(OUT / "fig_adamw_reset.png", dpi=160)
    print(f"\nFigure: {OUT / 'fig_adamw_reset.pdf'}")

    # ---- Decision ----
    base_mean = np.mean(base_taus)
    r1500_done = [t for t in r1500_taus if t > 0]
    r2200_done = [t for t in r2200_taus if t > 0]
    if r1500_done:
        delay_mid = np.mean(r1500_done) - base_mean
        print(f"\n=== Reset @ 1500 mean τ = {np.mean(r1500_done):.0f}; baseline mean = {base_mean:.0f}; "
              f"Δτ = {delay_mid:.0f}")
    if r2200_done:
        delay_late = np.mean(r2200_done) - base_mean
        print(f"=== Reset @ 2200 mean τ = {np.mean(r2200_done):.0f}; baseline mean = {base_mean:.0f}; "
              f"Δτ = {delay_late:.0f}")


if __name__ == "__main__":
    main()
