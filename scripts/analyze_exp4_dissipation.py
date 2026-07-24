#!/usr/bin/env python
"""
Post-processing for Experiment 4 — compute cumulative dissipation integrals.

For each run, reads the per-eval-window q_work and q_update arrays from
``training_history.json`` and computes:

  Q_grad     = Σ η(s) ||∇L||²  (from grad_norm_sq, reconstructed LR)
  Q_update   = Σ ||Δθ||²
  Q_work     = Σ ⟨∇L, Δθ⟩
  Q_adaptive = Σ η_eff ||∇L||²

Each quantity is summed over the transition window (same definition as
compute_landauer_cost.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import math
import numpy as np

OUTPUT_DIR = Path("outputs")

K_VALUES = [10, 15, 20, 25, 36]
SEEDS = [42, 43, 44, 45, 46]
CONFIGS = ["adamw", "sgd", "adamw_noclip"]


def reconstruct_lr_schedule(peak_lr, warmup_steps, max_steps, scheduler_type="cosine"):
    lrs = np.zeros(max_steps)
    for s in range(max_steps):
        if scheduler_type == "constant":
            lrs[s] = peak_lr
        elif s < warmup_steps:
            lrs[s] = peak_lr * (s / max(warmup_steps, 1))
        else:
            progress = (s - warmup_steps) / max(max_steps - warmup_steps, 1)
            lrs[s] = peak_lr * 0.5 * (1 + math.cos(math.pi * progress))
    return lrs


def find_transition_window(steps, first_loss, log_k):
    high = 0.9 * log_k
    low = 0.1 * log_k
    t_start = t_end = None
    for i, fl in enumerate(first_loss):
        if fl > high:
            t_start = steps[i]
    for i, fl in enumerate(first_loss):
        if fl < low:
            t_end = steps[i]
            break
    return t_start, t_end


def process_run(name: str):
    exp_dir = OUTPUT_DIR / name
    h_path = exp_dir / "training_history.json"
    m_path = exp_dir / "exp4_meta.json"
    if not h_path.exists() or not m_path.exists():
        return None

    with open(h_path) as f:
        h = json.load(f)
    with open(m_path) as f:
        meta = json.load(f)

    k = meta["k"]
    log_k = math.log(k)
    steps = np.array(h["steps"])
    first_loss = np.array(h["first_target_loss"])
    grad_norm_sq = np.array(h.get("grad_norm_sq", [0.0] * len(steps)))

    t_start, t_end = find_transition_window(steps, first_loss, log_k)
    if t_start is None or t_end is None:
        return None

    # Compute cumulative Q_grad using reconstructed LR
    cfg_path = exp_dir / "config.yaml"
    if cfg_path.exists():
        import yaml
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        peak_lr = float(cfg["training"]["learning_rate"])
        warmup = int(cfg["training"].get("warmup_steps", 500))
        max_steps = int(cfg["training"].get("max_steps", 60000))
        sched = cfg["training"].get("scheduler", "cosine")
    else:
        peak_lr, warmup, max_steps, sched = 1e-3, 500, 60000, "cosine"

    lr_sched = reconstruct_lr_schedule(peak_lr, warmup, max_steps, sched)

    # Per-eval-window spacing
    delta_s = np.diff(steps, prepend=0)
    safe_steps = np.clip(steps, 0, len(lr_sched) - 1).astype(int)
    eta = lr_sched[safe_steps]

    q_grad_cum = np.cumsum(eta * grad_norm_sq * delta_s)
    q_grad_trans = float(np.interp(t_end, steps, q_grad_cum) - np.interp(t_start, steps, q_grad_cum))

    # Q_work and Q_update (cumulative sums stored in history)
    q_work_arr = np.array(h.get("q_work", []))
    q_update_arr = np.array(h.get("q_update", []))
    eta_eff_arr = np.array(h.get("eta_eff", []))

    def _interp_cum(arr, t):
        if len(arr) == 0:
            return 0.0
        cum = np.cumsum(arr)
        return float(np.interp(t, steps[:len(cum)], cum))

    q_work_trans = _interp_cum(q_work_arr, t_end) - _interp_cum(q_work_arr, t_start)
    q_update_trans = _interp_cum(q_update_arr, t_end) - _interp_cum(q_update_arr, t_start)

    # Q_adaptive = Σ eta_eff * ||∇L||²
    if len(eta_eff_arr) > 0 and len(eta_eff_arr) == len(grad_norm_sq):
        q_adaptive_cum = np.cumsum(eta_eff_arr * grad_norm_sq * delta_s)
        q_adaptive_trans = float(np.interp(t_end, steps, q_adaptive_cum) - np.interp(t_start, steps, q_adaptive_cum))
    else:
        q_adaptive_trans = None

    # Clipping frequency
    clip_arr = np.array(h.get("clipping_active", []))
    if len(clip_arr) > 0:
        mask = (steps[:len(clip_arr)] >= t_start) & (steps[:len(clip_arr)] <= t_end)
        clip_freq = float(clip_arr[mask].mean()) if mask.any() else 0.0
    else:
        clip_freq = None

    return {
        "name": name,
        "config": meta["config"],
        "k": k,
        "log_k": round(log_k, 4),
        "seed": meta["seed"],
        "t_start": int(t_start),
        "t_end": int(t_end),
        "Q_grad": round(q_grad_trans, 6),
        "Q_work": round(q_work_trans, 6),
        "Q_update": round(q_update_trans, 6),
        "Q_adaptive": round(q_adaptive_trans, 6) if q_adaptive_trans is not None else None,
        "clip_frequency": round(clip_freq, 4) if clip_freq is not None else None,
    }


def main():
    all_results = []
    for config_name in CONFIGS:
        for k in K_VALUES:
            for seed in SEEDS:
                name = f"exp4_{config_name}_k{k}_seed{seed}"
                r = process_run(name)
                if r:
                    all_results.append(r)

    out_path = OUTPUT_DIR / "exp4_dissipation_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved {len(all_results)} results to {out_path}")

    # Summary per config
    for config_name in CONFIGS:
        print(f"\n{'='*60}\n  {config_name}\n{'='*60}")
        print(f"{'K':>4} {'Q_grad':>12} {'Q_work':>12} {'Q_update':>12} {'clip%':>8}")
        for k in K_VALUES:
            runs = [r for r in all_results if r["config"] == config_name and r["k"] == k]
            if not runs:
                print(f"{k:>4}  {'N/A':>12}")
                continue
            qg = np.mean([r["Q_grad"] for r in runs])
            qw = np.mean([r["Q_work"] for r in runs])
            qu = np.mean([r["Q_update"] for r in runs])
            cf = np.mean([r["clip_frequency"] for r in runs if r["clip_frequency"] is not None])
            print(f"{k:>4} {qg:>12.4f} {qw:>12.4f} {qu:>12.4f} {cf:>8.2%}")


if __name__ == "__main__":
    main()
