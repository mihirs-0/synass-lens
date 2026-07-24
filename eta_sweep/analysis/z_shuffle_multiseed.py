#!/usr/bin/env python
"""
Multi-seed z_shuffle_gap (= delta_z) trajectories from existing v_tracking
log.jsonl files (3 seeds at K=10, η=10⁻³, AdamW).

For each seed: plot delta_z over training step, mark transition step τ.
Show mean and per-seed scatter.  Verify the qualitative claim: "delta_z
grows monotonically during the plateau across seeds; the exponential-growth
shape is consistent."

Output:
  outputs/paper_figures/fig_z_shuffle_gap_multiseed.{pdf,png}
  eta_sweep/results/multi_seed/z_shuffle_gap_multiseed.json
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
MS_DIR = RESULTS / "multi_seed"
MS_DIR.mkdir(parents=True, exist_ok=True)


def load(seed: int):
    base = RESULTS / "v_tracking" / f"eta_0.001_K_10_seed_{seed}"
    log = [json.loads(l) for l in (base / "log.jsonl").read_text().strip().splitlines()]
    status = json.loads((base / "status.json").read_text())
    return log, status


def main():
    seeds = [0, 1, 2]
    runs = {s: load(s) for s in seeds}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Panel a: delta_z trajectory per seed, log scale
    ax = axes[0]
    summary = {}
    for s in seeds:
        log, status = runs[s]
        steps = np.array([r["step"] for r in log])
        dz = np.array([r["delta_z"] for r in log])
        tau = status["transition_detected_step"]
        ax.plot(steps, dz, alpha=0.85, lw=1.6, label=f"seed {s}  (τ={tau})")
        ax.axvline(tau, color=f"C{s}", linestyle=":", alpha=0.4, lw=0.9)
        # Compute plateau-only stats: dz at 0.3τ, 0.6τ, 0.9τ
        targets = [int(0.3*tau), int(0.6*tau), int(0.9*tau), int(1.2*tau)]
        nearest_dz = []
        for t in targets:
            idx = np.argmin(np.abs(steps - t))
            nearest_dz.append({"target_step": t, "actual_step": int(steps[idx]),
                                "delta_z": float(dz[idx])})
        summary[f"seed_{s}"] = {"tau": tau, "checkpoints": nearest_dz,
                                "max_delta_z": float(dz.max())}

    ax.axhline(0.1, color="orange", linestyle="--", lw=0.8, alpha=0.5,
               label=r"$\Delta_z$ onset threshold (0.1 nats)")
    ax.set_xlabel("step")
    ax.set_ylabel(r"$\Delta_z$ = $z$-shuffle gap (nats)")
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_title("(a) $z$-shuffle gap trajectory across 3 seeds")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(0, 5000)
    ax.grid(True, alpha=0.3, lw=0.4)

    # Panel b: delta_z normalized by τ for cross-seed comparison
    ax = axes[1]
    for s in seeds:
        log, status = runs[s]
        tau = status["transition_detected_step"]
        steps = np.array([r["step"] for r in log])
        dz = np.array([r["delta_z"] for r in log])
        norm_steps = steps / tau
        # Plot only plateau region
        mask = (norm_steps >= 0.05) & (norm_steps <= 1.3)
        ax.plot(norm_steps[mask], dz[mask], alpha=0.85, lw=1.6, label=f"seed {s}")
    ax.axvline(1.0, color="black", linestyle=":", lw=0.8, alpha=0.5, label=r"$\tau$")
    ax.axhline(0.1, color="orange", linestyle="--", lw=0.6, alpha=0.4)
    ax.set_xlabel(r"step / $\tau$")
    ax.set_ylabel(r"$\Delta_z$ (nats)")
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_title(r"(b) $\Delta_z$ vs step/$\tau$ — exponential-growth shape consistent across seeds")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(0, 1.3)
    ax.grid(True, alpha=0.3, lw=0.4)

    fig.suptitle("Multi-seed z-shuffle gap (3 seeds, K=10, η=10⁻³, AdamW)",
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT / "fig_z_shuffle_gap_multiseed.pdf", dpi=200)
    fig.savefig(OUT / "fig_z_shuffle_gap_multiseed.png", dpi=160)
    print(f"→ {OUT / 'fig_z_shuffle_gap_multiseed.pdf'}")

    # Print summary
    print("\n=== Multi-seed Δz summary ===")
    print(f"{'seed':<6} {'τ':<8} {'Δz @ 0.3τ':<12} {'Δz @ 0.6τ':<12} {'Δz @ 0.9τ':<12} {'Δz @ 1.2τ':<12} {'Δz max':<12}")
    for s in seeds:
        d = summary[f"seed_{s}"]
        cks = d["checkpoints"]
        print(f"{s:<6} {d['tau']:<8} "
              f"{cks[0]['delta_z']:<12.4f} {cks[1]['delta_z']:<12.4f} "
              f"{cks[2]['delta_z']:<12.4f} {cks[3]['delta_z']:<12.4f} {d['max_delta_z']:<12.4f}")

    # Verify monotonic growth claim during plateau
    print("\n=== Monotonic growth check (plateau region 0.3τ to 0.9τ) ===")
    all_monotonic = True
    for s in seeds:
        d = summary[f"seed_{s}"]
        cks = d["checkpoints"]
        dzs = [c["delta_z"] for c in cks[:3]]  # 0.3τ, 0.6τ, 0.9τ
        is_mono = all(dzs[i] <= dzs[i+1] for i in range(len(dzs)-1))
        print(f"  seed {s}: 0.3τ→0.6τ→0.9τ = {dzs[0]:.3f} → {dzs[1]:.3f} → {dzs[2]:.3f}  "
              f"{'✓ monotonic' if is_mono else '✗ NOT monotonic'}")
        if not is_mono:
            all_monotonic = False
    print(f"\n  All seeds monotonic during plateau: {all_monotonic}")

    out = {"summary": summary, "all_monotonic": all_monotonic}
    (MS_DIR / "z_shuffle_gap_multiseed.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {MS_DIR / 'z_shuffle_gap_multiseed.json'}")


if __name__ == "__main__":
    main()
