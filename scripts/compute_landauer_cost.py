#!/usr/bin/env python
"""
Compute cumulative dissipation integral for the Landauer test.

For each experiment, computes:
  Q(t) = Σ_{s} η(s) × ||∇L(s)||² × Δs

where η(s) is the ACTUAL learning rate at step s (warmup + cosine decay),
||∇L(s)||² is the measured gradient norm squared, and Δs is the step spacing.

Defines the transition window from candidate_eval_results.json:
  - transition_start: last step where candidate_loss > 0.9 × log(K)
  - transition_end:   first step where candidate_loss < 0.1 × log(K)

Outputs Q_transition, Q_total, Q_plateau, Q_post for each experiment,
and a summary table testing whether Q_transition scales with log(K).

Usage:
    python scripts/compute_landauer_cost.py \
      --experiments temp_lr1e3_bs128 temp_lr1e3_bs128_k20 temp_lr1e3_bs512_k36 \
      --output-dir outputs
"""

import sys
import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml


def reconstruct_lr_schedule(peak_lr: float, warmup_steps: int, max_steps: int,
                            scheduler_type: str = "cosine") -> np.ndarray:
    """
    Reconstruct the learning rate at each training step.

    Matches src/training/trainer.py::get_lr_scheduler exactly:
      - "constant": flat LR = peak_lr for all steps
      - "cosine":   linear warmup from 0 to peak_lr, then cosine decay to 0
    """
    lrs = np.zeros(max_steps)
    for step in range(max_steps):
        if scheduler_type == "constant":
            lrs[step] = peak_lr
        elif step < warmup_steps:
            lrs[step] = peak_lr * (step / warmup_steps)
        else:
            progress = (step - warmup_steps) / (max_steps - warmup_steps)
            lrs[step] = peak_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    return lrs


def find_transition_window(steps, candidate_loss, log_k,
                           threshold_high_frac=0.9, threshold_low_frac=0.1):
    """
    Define the phase transition window from candidate_loss trajectory.

    transition_end:   first step where candidate_loss < threshold_low_frac × log(K)
    transition_start: last step BEFORE transition_end where
                      candidate_loss > threshold_high_frac × log(K)

    Finding t_end first prevents late instability spikes (common with
    constant LR / no clipping) from pushing t_start past convergence.

    Returns (transition_start, transition_end) or (None, None) if not found.
    """
    threshold_high = threshold_high_frac * log_k
    threshold_low = threshold_low_frac * log_k

    # Step 1: first convergence
    transition_end = None
    end_idx = None
    for i, cl in enumerate(candidate_loss):
        if cl < threshold_low:
            transition_end = steps[i]
            end_idx = i
            break

    if transition_end is None:
        return None, None

    # Step 2: last plateau step before convergence
    transition_start = None
    for i in range(end_idx):
        if candidate_loss[i] > threshold_high:
            transition_start = steps[i]

    return transition_start, transition_end


def find_continual_transition_window(steps, candidate_loss,
                                     threshold_high_frac=0.9,
                                     threshold_low_frac=0.1):
    """
    Transition window for continual learning experiments.

    Unlike from-scratch training where the reference is log(K), here the
    reference is the actual initial candidate loss after the distribution
    shift. This handles partial reassignment (f < 1.0) where the initial
    loss is approximately f * log(K), not the full log(K).

    transition_start: last step where candidate_loss > 0.9 × initial_loss
    transition_end:   first step where candidate_loss < max(0.1 × initial_loss, 0.05)

    Returns (transition_start, transition_end) or (None, None) if no transition.
    """
    if not steps or not candidate_loss:
        return None, None

    initial_loss = candidate_loss[0]
    if initial_loss < 0.1:
        return None, None

    threshold_high = threshold_high_frac * initial_loss
    threshold_low = max(threshold_low_frac * initial_loss, 0.05)

    transition_start = None
    transition_end = None

    for i, cl in enumerate(candidate_loss):
        if cl > threshold_high:
            transition_start = steps[i]

    for i, cl in enumerate(candidate_loss):
        if cl < threshold_low:
            transition_end = steps[i]
            break

    return transition_start, transition_end


def find_plateau_onset(steps, candidate_loss, log_k):
    """
    Find the step where the model first reaches the metastable plateau.

    The plateau corresponds to having learned the marginal P(A|B) (uniform
    over K candidates) but not yet using z for disambiguation.  At the
    plateau, candidate_loss ≈ log(K).

    We define plateau_onset as the first step where candidate_loss drops
    below 1.1 × log(K).  This separates the initial transient (embedding
    warm-up, learning token statistics) from the true metastable period.

    Returns the step, or None if the trajectory never reaches the plateau.
    """
    threshold = 1.1 * log_k
    for i, cl in enumerate(candidate_loss):
        if cl < threshold:
            return steps[i]
    return None


def compute_dissipation(
    grad_steps: list,
    grad_norm_sq: list,
    lr_schedule: np.ndarray,
) -> tuple:
    """
    Compute cumulative dissipation Q(t) = Σ η(s) × ||∇L(s)||² × Δs.

    Uses the actual LR at each checkpoint step. Δs is the spacing between
    consecutive checkpoint steps.

    Returns:
        steps: array of checkpoint steps
        Q_cumulative: array of cumulative dissipation at each checkpoint
        dissipation_per_step: array of per-interval dissipation contribution
    """
    steps = np.array(grad_steps)
    norms_sq = np.array(grad_norm_sq)

    # Get actual LR at each checkpoint step
    # Clip steps to valid range for the schedule
    max_step = len(lr_schedule) - 1
    safe_steps = np.clip(steps, 0, max_step).astype(int)
    eta_at_checkpoints = lr_schedule[safe_steps]

    # Compute step spacings (Δs)
    # First interval: from 0 to first checkpoint
    delta_s = np.diff(steps, prepend=0)

    # Per-interval dissipation: η(s) × ||∇L(s)||² × Δs
    dissipation = eta_at_checkpoints * norms_sq * delta_s

    # Cumulative sum
    Q_cumulative = np.cumsum(dissipation)

    return steps, Q_cumulative, dissipation


def process_experiment(experiment_name: str, output_dir: str) -> dict:
    """Process a single experiment and return its Landauer metrics."""
    exp_dir = Path(output_dir) / experiment_name

    # Load config
    config_path = exp_dir / "config.yaml"
    if not config_path.exists():
        print(f"  ERROR: Config not found at {config_path}")
        return None

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    k = int(cfg["data"]["k"])
    log_k = math.log(k)
    log2_k = math.log2(k)
    peak_lr = float(cfg["training"]["learning_rate"])
    batch_size = int(cfg["training"]["batch_size"])
    warmup_steps = int(cfg["training"].get("warmup_steps", 500))
    max_steps = int(cfg["training"].get("max_steps", 10000))
    T_eff_peak = peak_lr / batch_size

    # Load gradient norms
    grad_path = exp_dir / "gradient_norm_results.json"
    if not grad_path.exists():
        print(f"  ERROR: Gradient norms not found at {grad_path}")
        return None

    with open(grad_path, "r") as f:
        grad_data = json.load(f)

    grad_steps = grad_data["steps"]
    grad_norm_sq = grad_data["total_grad_norm_sq"]

    # Detect if this is a continual learning experiment
    is_continual = "continual" in cfg

    # Load candidate eval (optional but preferred)
    cand_path = exp_dir / "candidate_eval_results.json"
    has_candidate = cand_path.exists()
    transition_start = None
    transition_end = None

    plateau_onset = None

    # For continual experiments, also try training_history.json which has
    # new_candidate_loss tracked at higher frequency
    history_path = exp_dir / "training_history.json"
    has_history = history_path.exists()

    if is_continual and has_history:
        with open(history_path, "r") as f:
            history = json.load(f)
        hist_steps = history.get("steps", [])
        hist_new_cand = history.get("new_candidate_loss", [])

        if hist_steps and hist_new_cand:
            transition_start, transition_end = find_continual_transition_window(
                hist_steps, hist_new_cand
            )
            print(f"  [Continual] Transition window (from history): "
                  f"{transition_start} -> {transition_end}")
            print(f"  [Continual] Initial candidate_loss: {hist_new_cand[0]:.4f}")

    if transition_start is None and has_candidate:
        with open(cand_path, "r") as f:
            cand_data = json.load(f)

        cand_steps = cand_data["steps"]
        cand_loss = cand_data["candidate_loss"]

        if is_continual:
            transition_start, transition_end = find_continual_transition_window(
                cand_steps, cand_loss
            )
            print(f"  [Continual] Transition window (from candidate_eval): "
                  f"{transition_start} -> {transition_end}")
        else:
            transition_start, transition_end = find_transition_window(
                cand_steps, cand_loss, log_k
            )
            plateau_onset = find_plateau_onset(cand_steps, cand_loss, log_k)
            print(f"  Plateau onset: {plateau_onset}")
            print(f"  Transition window (from candidate_loss): "
                  f"{transition_start} -> {transition_end}")

    if transition_start is None and not is_continual:
        # Fallback: use binding_onset_step from gradient_norm_results
        onset = grad_data.get("binding_onset_step")
        if onset is not None:
            transition_start = onset
            transition_end = min(onset + 1000, grad_steps[-1])
            print(f"  Transition window (fallback from binding_onset): "
                  f"{transition_start} -> {transition_end}")
        else:
            print("  WARNING: No transition window found!")

    # Reconstruct LR schedule
    scheduler_type = cfg["training"].get("scheduler", "cosine")
    lr_schedule = reconstruct_lr_schedule(peak_lr, warmup_steps, max_steps,
                                          scheduler_type=scheduler_type)

    # Compute dissipation
    steps, Q_cum, dissipation = compute_dissipation(grad_steps, grad_norm_sq, lr_schedule)

    # Extract dissipation values
    Q_total = float(Q_cum[-1])

    # Find Q at transition boundaries by interpolation
    Q_transition = None
    Q_plateau = None
    Q_post = None
    transition_duration = None

    if transition_start is not None and transition_end is not None:
        # Interpolate Q at transition_start and transition_end
        Q_at_start = float(np.interp(transition_start, steps, Q_cum))
        Q_at_end = float(np.interp(transition_end, steps, Q_cum))

        Q_transition = Q_at_end - Q_at_start
        Q_plateau = Q_at_start
        Q_post = Q_total - Q_at_end
        transition_duration = transition_end - transition_start

    # Compute S statistics (S = ||∇L||²)
    # Plateau S: mean grad norm squared before transition
    if transition_start is not None:
        plateau_mask = np.array(grad_steps) < transition_start
        if np.any(plateau_mask):
            plateau_S_mean = float(np.mean(np.array(grad_norm_sq)[plateau_mask]))
        else:
            plateau_S_mean = float(grad_norm_sq[0])
    else:
        plateau_S_mean = float(np.mean(grad_norm_sq))

    peak_S = float(max(grad_norm_sq))
    peak_S_step = int(grad_steps[grad_norm_sq.index(peak_S)] if isinstance(grad_norm_sq, list) else grad_steps[np.argmax(grad_norm_sq)])

    # Build note about experiment conditions
    notes = []
    if batch_size != 128:
        notes.append(f"batch_size={batch_size} (differs from K=10,K=20 which use bs=128)")
    if not has_candidate:
        notes.append("transition window from binding_onset fallback (no candidate_eval)")
    note = "; ".join(notes) if notes else ""

    # Plateau duration: time spent in the metastable state, excluding the
    # initial transient where the model is still learning basic token stats.
    if plateau_onset is not None and transition_start is not None:
        plateau_duration = transition_start - plateau_onset
    else:
        plateau_duration = None

    result = {
        "name": experiment_name,
        "K": k,
        "log_K": round(log_k, 4),
        "log2_K": round(log2_k, 4),
        "peak_lr": peak_lr,
        "batch_size": batch_size,
        "T_eff_peak": T_eff_peak,
        "warmup_steps": warmup_steps,
        "max_steps": max_steps,
        "plateau_onset": plateau_onset,
        "transition_start": transition_start,
        "transition_end": transition_end,
        "transition_duration": transition_duration,
        "plateau_duration": plateau_duration,
        "Q_transition": Q_transition,
        "Q_total": Q_total,
        "Q_plateau": Q_plateau,
        "Q_post": Q_post,
        "plateau_S_mean": plateau_S_mean,
        "peak_S": peak_S,
        "peak_S_step": peak_S_step,
        "Q_transition_over_logK": round(Q_transition / log_k, 6) if Q_transition is not None else None,
        "Q_transition_over_log2K": round(Q_transition / log2_k, 6) if Q_transition is not None else None,
        "is_continual": is_continual,
        "note": note,
        # Full trajectories for plotting
        "Q_trajectory_steps": steps.tolist(),
        "Q_trajectory": Q_cum.tolist(),
        "dissipation_per_interval": dissipation.tolist(),
    }

    if is_continual:
        continual_cfg = cfg["continual"]
        result["continual"] = {
            "base_experiment": continual_cfg.get("base_experiment"),
            "variant": continual_cfg.get("variant"),
            "fraction": continual_cfg.get("fraction"),
            "reassign_seed": continual_cfg.get("reassign_seed"),
            "original_k": continual_cfg.get("original_k"),
            "divergence": continual_cfg.get("divergence", {}),
        }

    return result


def print_summary_table(experiments: list):
    """Print a nicely formatted summary table."""
    print()
    print("=" * 80)
    print("Landauer Dissipation Test — First Pass (3 data points, single seed)")
    print("=" * 80)
    print()

    # Header
    header = f"{'K':>4}  {'log(K)':>7}  {'BS':>4}  {'T_eff':>10}  {'Q_trans':>12}  {'Q_trans/logK':>14}  {'plat_S':>10}  {'peak_S':>10}"
    print(header)
    print("-" * len(header))

    ratios = []
    for exp in experiments:
        q_trans = exp["Q_transition"]
        ratio = exp["Q_transition_over_logK"]
        if q_trans is not None:
            ratios.append(ratio)
            print(
                f"{exp['K']:>4}  {exp['log_K']:>7.3f}  {exp['batch_size']:>4}  "
                f"{exp['T_eff_peak']:>10.2e}  {q_trans:>12.6f}  {ratio:>14.6f}  "
                f"{exp['plateau_S_mean']:>10.4f}  {exp['peak_S']:>10.4f}"
            )
        else:
            print(
                f"{exp['K']:>4}  {exp['log_K']:>7.3f}  {exp['batch_size']:>4}  "
                f"{exp['T_eff_peak']:>10.2e}  {'N/A':>12}  {'N/A':>14}  "
                f"{exp['plateau_S_mean']:>10.4f}  {exp['peak_S']:>10.4f}"
            )

    print("-" * len(header))

    if len(ratios) >= 2:
        mean_ratio = np.mean(ratios)
        std_ratio = np.std(ratios)
        cv = std_ratio / mean_ratio if mean_ratio != 0 else float("inf")
        print(f"\n  Q_trans/log(K): mean = {mean_ratio:.6f}, std = {std_ratio:.6f}, CV = {cv:.3f}")
        print()
        if cv < 0.3:
            print("  → CONSISTENT with Landauer scaling: Q_trans/log(K) ≈ constant")
            print(f"    (CV = {cv:.3f} < 0.3, but only {len(ratios)} data points)")
        elif cv < 0.5:
            print(f"  → WEAKLY consistent with Landauer scaling (CV = {cv:.3f})")
            print(f"    (marginal; {len(ratios)} points, single seed)")
        else:
            print(f"  → NOT consistent with Landauer scaling (CV = {cv:.3f})")
            print(f"    Q_trans/log(K) varies too much across K values")
    else:
        print("\n  Insufficient data points for scaling test")

    # Note about different batch sizes
    batch_sizes = set(exp["batch_size"] for exp in experiments)
    if len(batch_sizes) > 1:
        print(f"\n  ⚠ CAVEAT: Experiments use different batch sizes {sorted(batch_sizes)}.")
        print("    T_eff = lr/bs differs, so Q values are not on identical footing.")
        print("    This is a confound. A clean test would use matched T_eff.")

    print()
    print("=" * 80)


def run_threshold_robustness(experiments_names, output_dir):
    """
    Run threshold sensitivity analysis: compute Q_transition for multiple
    threshold pairs and report slope c and R² for each.

    Threshold pairs tested: (0.9, 0.1), (0.85, 0.15), (0.80, 0.20)
    """
    threshold_pairs = [
        (0.9, 0.1),
        (0.85, 0.15),
        (0.80, 0.20),
    ]

    print()
    print("=" * 80)
    print("Threshold Robustness Analysis")
    print("=" * 80)
    print()

    robustness_results = []

    for hi, lo in threshold_pairs:
        label = f"({hi:.2f}, {lo:.2f})"
        ks = []
        q_trans_values = []

        for exp_name in experiments_names:
            exp_dir = Path(output_dir) / exp_name
            config_path = exp_dir / "config.yaml"
            if not config_path.exists():
                continue

            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)

            k = int(cfg["data"]["k"])
            log_k = math.log(k)
            peak_lr = float(cfg["training"]["learning_rate"])
            warmup_steps = int(cfg["training"].get("warmup_steps", 500))
            max_steps = int(cfg["training"].get("max_steps", 10000))
            scheduler_type = cfg["training"].get("scheduler", "cosine")

            grad_path = exp_dir / "gradient_norm_results.json"
            cand_path = exp_dir / "candidate_eval_results.json"
            if not grad_path.exists() or not cand_path.exists():
                continue

            with open(grad_path, "r") as f:
                grad_data = json.load(f)
            with open(cand_path, "r") as f:
                cand_data = json.load(f)

            t_start, t_end = find_transition_window(
                cand_data["steps"], cand_data["candidate_loss"], log_k,
                threshold_high_frac=hi, threshold_low_frac=lo,
            )
            if t_start is None or t_end is None:
                continue

            lr_schedule = reconstruct_lr_schedule(peak_lr, warmup_steps, max_steps,
                                                  scheduler_type=scheduler_type)
            steps, Q_cum, _ = compute_dissipation(
                grad_data["steps"], grad_data["total_grad_norm_sq"], lr_schedule
            )

            Q_at_start = float(np.interp(t_start, steps, Q_cum))
            Q_at_end = float(np.interp(t_end, steps, Q_cum))
            q_t = Q_at_end - Q_at_start

            ks.append(k)
            q_trans_values.append(q_t)

        if len(ks) < 2:
            print(f"  Threshold {label}: insufficient data ({len(ks)} points)")
            continue

        log_ks = np.log(np.array(ks, dtype=float))
        q_arr = np.array(q_trans_values)
        coeffs = np.polyfit(log_ks, q_arr, 1)
        slope, intercept = coeffs
        predicted = np.polyval(coeffs, log_ks)
        ss_res = np.sum((q_arr - predicted) ** 2)
        ss_tot = np.sum((q_arr - np.mean(q_arr)) ** 2)
        r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        print(f"  Threshold {label}: slope c = {slope:.6f}, R² = {r_sq:.4f}  "
              f"({len(ks)} K values)")
        robustness_results.append({
            "threshold_high": hi,
            "threshold_low": lo,
            "slope": round(slope, 6),
            "intercept": round(intercept, 6),
            "R2": round(r_sq, 4),
            "n_points": len(ks),
            "K_values": ks,
            "Q_transition_values": [round(q, 6) for q in q_trans_values],
        })

    if len(robustness_results) >= 2:
        slopes = [r["slope"] for r in robustness_results]
        mean_slope = np.mean(slopes)
        max_dev = max(abs(s - mean_slope) / mean_slope for s in slopes) * 100
        print(f"\n  Mean slope: {mean_slope:.6f}")
        print(f"  Max deviation from mean: {max_dev:.1f}%")
        if max_dev < 15:
            print("  -> Results are ROBUST to threshold choice.")
        else:
            print(f"  -> Moderate sensitivity to threshold choice ({max_dev:.1f}% spread).")

    # Save
    rob_path = Path(output_dir) / "threshold_robustness.json"
    with open(rob_path, "w") as f:
        json.dump(robustness_results, f, indent=2)
    print(f"\n  Saved to: {rob_path}")
    print()

    return robustness_results


def main():
    parser = argparse.ArgumentParser(
        description="Compute cumulative dissipation integral for Landauer test"
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        required=True,
        help="Experiment names to process",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Base output directory",
    )
    parser.add_argument(
        "--threshold-robustness",
        action="store_true",
        help="Run threshold sensitivity analysis with multiple threshold pairs",
    )
    args = parser.parse_args()

    results = []
    for exp_name in args.experiments:
        print(f"\nProcessing: {exp_name}")
        result = process_experiment(exp_name, args.output_dir)
        if result is not None:
            results.append(result)
        else:
            print(f"  SKIPPED (missing data)")

    # Sort by K
    results.sort(key=lambda x: x["K"])

    # Save full results
    output_path = Path(args.output_dir) / "landauer_results.json"
    output = {"experiments": results}

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved results to: {output_path}")

    # Print summary
    print_summary_table(results)

    # Threshold robustness
    if args.threshold_robustness:
        run_threshold_robustness(args.experiments, args.output_dir)


if __name__ == "__main__":
    main()
