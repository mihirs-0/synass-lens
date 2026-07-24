#!/usr/bin/env python
"""
Q1: m_t (Adam first-moment) cosine similarity analysis with negative controls.

For each transitioning AdamW run:
  - Compute long-window smoothed m_t cosine similarity across the plateau
  - Negative control 1: shuffle checkpoint ordering, recompute cosines
  - Negative control 2: random window pairings (random pairs of checkpoints)
  - Compare to non-transitioning control run (different η, no transition)

Output: outputs/paper_figures/fig_m_cosine.{pdf,png}
        eta_sweep/results/m_cosine_analysis.json
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


def load_m_vectors(run_dir: Path):
    """Load all m_step_*.npy files in increasing step order."""
    files = sorted(run_dir.glob("m_vectors/m_step_*.npy"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    steps = [int(f.stem.split("_")[-1]) for f in files]
    vectors = [np.load(f) for f in files]
    return np.array(steps), np.array(vectors)


def cosine(a, b):
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / n) if n > 0 else 0.0


def consecutive_cosines(vectors):
    """Cosine similarity between consecutive snapshots."""
    return np.array([cosine(vectors[i], vectors[i+1]) for i in range(len(vectors)-1)])


def smoothed_m(vectors, window_steps, step_grid):
    """Long-window smoothed m_t: at each checkpoint t, average m_t over
    checkpoints whose step is in [t - window/2, t + window/2]."""
    smoothed = np.zeros_like(vectors)
    half = window_steps / 2
    for i, t in enumerate(step_grid):
        in_window = np.where((step_grid >= t - half) & (step_grid <= t + half))[0]
        if len(in_window) > 0:
            smoothed[i] = vectors[in_window].mean(axis=0)
    return smoothed


def smoothed_cosines(vectors, step_grid, window=1000):
    """Long-window smoothed cosine similarity between consecutive smoothed snapshots."""
    smoothed = smoothed_m(vectors, window, step_grid)
    return np.array([cosine(smoothed[i], smoothed[i+1])
                     for i in range(len(smoothed)-1)])


def shuffled_control(vectors, step_grid, window=1000, seed=42):
    """Negative control: shuffle vector ordering before smoothing.
    If the high cosine were just an artifact of long-window averaging,
    shuffled order would still give high cosines."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(vectors))
    shuffled_vecs = vectors[perm]
    return smoothed_cosines(shuffled_vecs, step_grid, window)


def random_pair_cosines(vectors, n_pairs=200, seed=42):
    """Cosine similarity between random pairs of checkpoints (no temporal proximity)."""
    rng = np.random.default_rng(seed)
    n = len(vectors)
    pairs = rng.integers(0, n, (n_pairs, 2))
    return np.array([cosine(vectors[i], vectors[j]) for i, j in pairs if i != j])


def main():
    # ---- Load transitioning seeds ----
    seeds_data = {}
    for seed in (0, 1, 2):
        run_dir = RESULTS / "mv_tracking" / f"eta_0.001_K_10_seed_{seed}"
        if not run_dir.exists():
            print(f"WARN: {run_dir} not found, skipping")
            continue
        steps, vecs = load_m_vectors(run_dir)
        if len(steps) == 0:
            print(f"WARN: no m_vectors in {run_dir}, skipping")
            continue
        # Get τ
        status = json.loads((run_dir / "status.json").read_text())
        seeds_data[seed] = {
            "steps": steps,
            "vectors": vecs,
            "tau": status.get("transition_detected_step"),
        }

    # ---- Load non-transitioning control ----
    control_dir = RESULTS / "mv_tracking_control" / "eta_0.0001_K_10_seed_0"
    if control_dir.exists():
        steps_c, vecs_c = load_m_vectors(control_dir)
        control_data = {"steps": steps_c, "vectors": vecs_c}
    else:
        print(f"WARN: control {control_dir} not available")
        control_data = None

    if not seeds_data:
        print("No transitioning data available yet. Re-run when mv_tracking finishes.")
        return

    # ---- Compute cosines ----
    print("\n=== Per-seed transitioning runs ===")
    results = {}
    for seed, d in seeds_data.items():
        steps = d["steps"]
        vecs = d["vectors"]
        tau = d["tau"]
        print(f"Seed {seed}: τ={tau}, n_checkpoints={len(steps)}, vector dim={vecs.shape[1]}")

        consec = consecutive_cosines(vecs)
        smoothed = smoothed_cosines(vecs, steps, window=1000)
        shuffled = shuffled_control(vecs, steps, window=1000, seed=seed * 7 + 1)
        random_pairs = random_pair_cosines(vecs, n_pairs=500, seed=seed * 11 + 3)

        plateau_mask = (steps[:-1] >= 200) & (steps[:-1] <= (tau - 200 if tau else len(steps)))
        post_mask = (steps[:-1] > tau) if tau else np.zeros_like(steps[:-1], bool)

        print(f"  consecutive cosine: plateau mean = {consec[plateau_mask].mean():.4f}, "
              f"post-τ mean = {consec[post_mask].mean():.4f}")
        print(f"  smoothed (1000-step) cosine: plateau mean = {smoothed[plateau_mask].mean():.4f}, "
              f"post-τ mean = {smoothed[post_mask].mean():.4f}")
        print(f"  shuffled control (smoothed): mean = {shuffled.mean():.4f}, std = {shuffled.std():.4f}")
        print(f"  random-pair cosine: mean = {random_pairs.mean():.4f}, std = {random_pairs.std():.4f}")

        results[seed] = {
            "tau": int(tau) if tau else None,
            "n_checkpoints": int(len(steps)),
            "consec_plateau_mean": float(consec[plateau_mask].mean()),
            "consec_post_mean": float(consec[post_mask].mean()) if post_mask.any() else None,
            "smoothed_plateau_mean": float(smoothed[plateau_mask].mean()),
            "smoothed_post_mean": float(smoothed[post_mask].mean()) if post_mask.any() else None,
            "shuffled_smoothed_mean": float(shuffled.mean()),
            "shuffled_smoothed_std": float(shuffled.std()),
            "random_pair_mean": float(random_pairs.mean()),
            "random_pair_std": float(random_pairs.std()),
        }

    # ---- Non-transitioning control ----
    if control_data is not None:
        steps_c = control_data["steps"]
        vecs_c = control_data["vectors"]
        consec_c = consecutive_cosines(vecs_c)
        smoothed_c = smoothed_cosines(vecs_c, steps_c, window=1000)
        print(f"\n=== Non-transitioning control (η=1e-4) ===")
        print(f"  n_checkpoints={len(steps_c)}")
        print(f"  consecutive cosine: mean = {consec_c.mean():.4f}")
        print(f"  smoothed (1000-step) cosine: mean = {smoothed_c.mean():.4f}")
        results["control_eta_1e-4"] = {
            "consec_mean": float(consec_c.mean()),
            "smoothed_mean": float(smoothed_c.mean()),
            "n_checkpoints": int(len(steps_c)),
        }

    # ---- Save ----
    out_path = RESULTS / "m_cosine_analysis.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out_path}")

    # ---- Figure ----
    if not seeds_data:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel (a): smoothed cosine trajectory across plateau, all seeds
    ax = axes[0]
    for seed, d in seeds_data.items():
        steps = d["steps"]
        vecs = d["vectors"]
        tau = d["tau"]
        smoothed = smoothed_cosines(vecs, steps, window=1000)
        # x-axis: midpoint of consecutive pair
        x = (steps[:-1] + steps[1:]) / 2
        ax.plot(x, smoothed, label=f"seed {seed} (smoothed)", lw=1.4, alpha=0.85)
        if tau:
            ax.axvline(tau, color="grey", linestyle=":", alpha=0.4)

    # overlay shuffled control band
    all_shuffled = []
    for seed, d in seeds_data.items():
        all_shuffled.extend(shuffled_control(d["vectors"], d["steps"], 1000, seed=seed*7+1))
    if all_shuffled:
        m = np.mean(all_shuffled); s = np.std(all_shuffled)
        ax.axhspan(m - 2*s, m + 2*s, color="orange", alpha=0.18,
                   label=f"shuffled control 2σ band ({m:.3f}±{2*s:.3f})")
    ax.set_xlabel("training step (midpoint of pair)")
    ax.set_ylabel(r"smoothed $m_t$ cosine similarity (1000-step window)")
    ax.set_title("(a) Long-window $m_t$ direction is coherent throughout plateau")
    ax.legend(fontsize=8.5, loc="lower left")
    ax.grid(True, alpha=0.3, lw=0.4)
    ax.set_ylim(-0.2, 1.05)

    # Panel (b): consecutive (short-window) cosine for comparison
    ax = axes[1]
    for seed, d in seeds_data.items():
        steps = d["steps"]
        vecs = d["vectors"]
        tau = d["tau"]
        consec = consecutive_cosines(vecs)
        x = (steps[:-1] + steps[1:]) / 2
        ax.plot(x, consec, label=f"seed {seed}", lw=1.2, alpha=0.85)
        if tau:
            ax.axvline(tau, color="grey", linestyle=":", alpha=0.4)
    ax.set_xlabel("training step (midpoint of pair)")
    ax.set_ylabel(r"consecutive $m_t$ cosine similarity")
    ax.set_title("(b) Consecutive (short-window) cosine fluctuates")
    ax.legend(fontsize=8.5, loc="best")
    ax.grid(True, alpha=0.3, lw=0.4)
    ax.set_ylim(-0.5, 1.05)

    fig.suptitle("Adam first-moment $m_t$: smoothed direction is coherent across the plateau, "
                 "vs. shuffled control",
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT / "fig_m_cosine.pdf", dpi=200)
    fig.savefig(OUT / "fig_m_cosine.png", dpi=160)
    print(f"Figure: {OUT / 'fig_m_cosine.pdf'}")


if __name__ == "__main__":
    main()
