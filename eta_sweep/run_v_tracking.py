#!/usr/bin/env python
"""
v-tracking single-cell trainer.

Same training loop as run_single.py but with two added per-checkpoint logs:

  v_log.jsonl  — Adam optimizer-state v (= exp_avg_sq) statistics, aggregated
                 by parameter (keyed by full param name).  Per param per
                 checkpoint we log:
                     mean, median, max of v
                     mean of 1/sqrt(v + eps)            ← preconditioner factor
                     L2-norm-sq of grad                 ← computed at log time
                     mean of |g|/sqrt(v+eps)            ← effective per-coord step

  v_aggregates.jsonl — same metrics aggregated to module groups defined in
                       MODULE_GROUPS below.  Cheaper to plot from.

Decision criterion (for path 1 vs path 2 of the NeurIPS submission):
- z-pipeline groups {L0_attn, L0_mlp, L2_mlp, L3_attn} should show monotone
  growth of mean(1/sqrt(v)) across the plateau and a sharp jump in
  mean(|g|/sqrt(v)) at τ.
- Control groups should not.

Cells: K=10, η=1e-3, seeds {0,1,2}, AdamW.  Budget 5,000 steps (τ≈1850).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import numpy as np
import torch
from torch.utils.data import DataLoader

ETA_SWEEP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = ETA_SWEEP_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data import collate_fn, create_datasets_from_config, create_tokenizer_from_config  # noqa: E402
from src.model import create_model_from_config  # noqa: E402
from src.training.trainer import compute_loss, get_lr_scheduler  # noqa: E402

from eta_sweep.config import CellConfig, RESULTS_DIR  # noqa: E402
from eta_sweep.run_single import (  # noqa: E402
    _select_device,
    _set_all_seeds,
    _git_commit,
    build_legacy_cfg,
    build_held_out_batch,
    compute_held_out_grad_norm_sq,
    compute_candidate_loss_and_delta_z,
    RunStatusTracker,
)


# ---------------------------------------------------------------------------
# Module groupings — z-pipeline vs control (from mech interp findings)
# ---------------------------------------------------------------------------
#
# z-pipeline: L0 (z-encoding), L2 (answer crystallization), L3 (sharpening).
# L1 is the bridge — included as `bridge`, not z-pipeline.

MODULE_GROUPS: Dict[str, List[str]] = {
    "L0_attn":  ["blocks.0.attn."],
    "L0_mlp":   ["blocks.0.mlp."],
    "L0_ln":    ["blocks.0.ln1.", "blocks.0.ln2."],
    "L1_attn":  ["blocks.1.attn."],
    "L1_mlp":   ["blocks.1.mlp."],
    "L1_ln":    ["blocks.1.ln1.", "blocks.1.ln2."],
    "L2_attn":  ["blocks.2.attn."],
    "L2_mlp":   ["blocks.2.mlp."],
    "L2_ln":    ["blocks.2.ln1.", "blocks.2.ln2."],
    "L3_attn":  ["blocks.3.attn."],
    "L3_mlp":   ["blocks.3.mlp."],
    "L3_ln":    ["blocks.3.ln1.", "blocks.3.ln2."],
    "embed":    ["embed.W_E", "pos_embed.W_pos"],
    "unembed":  ["unembed.W_U", "unembed.b_U"],
    "ln_final": ["ln_final."],
}

Z_PIPELINE_GROUPS = {"L0_attn", "L0_mlp", "L2_mlp", "L3_attn"}


def group_for_param(param_name: str) -> Optional[str]:
    for group, prefixes in MODULE_GROUPS.items():
        for prefix in prefixes:
            if param_name.startswith(prefix) or param_name == prefix.rstrip("."):
                return group
    return None


# ---------------------------------------------------------------------------
# v-statistics computation
# ---------------------------------------------------------------------------

def collect_v_stats(
    optimizer: torch.optim.Optimizer,
    named_params: List[Tuple[str, torch.nn.Parameter]],
    eps: float = 1e-8,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """For every parameter in `named_params`, read Adam's exp_avg_sq from the
    optimizer state and compute summary statistics.  Also aggregates to the
    module groups in MODULE_GROUPS.

    Returns (per_param_stats, per_group_stats).
    """
    per_param: Dict[str, Dict[str, float]] = {}
    # accumulate weighted sums for group aggregation (weight = numel)
    group_acc: Dict[str, Dict[str, float]] = {}

    for name, p in named_params:
        if p.grad is None:
            continue
        state = optimizer.state.get(p, {})
        v = state.get("exp_avg_sq")
        if v is None:
            # Adam state not yet initialized (step 0); skip
            continue

        v_flat = v.detach().reshape(-1).to(torch.float32)
        g_flat = p.grad.detach().reshape(-1).to(torch.float32)
        precond = 1.0 / (v_flat.sqrt() + eps)
        eff_step = g_flat.abs() * precond  # |g| / (sqrt(v) + eps)

        n = v_flat.numel()
        v_mean = float(v_flat.mean().item())
        v_median = float(v_flat.median().item())
        v_max = float(v_flat.max().item())
        precond_mean = float(precond.mean().item())
        eff_step_mean = float(eff_step.mean().item())
        g_l2_sq = float((g_flat ** 2).sum().item())

        stats = {
            "n": n,
            "v_mean": v_mean,
            "v_median": v_median,
            "v_max": v_max,
            "precond_mean": precond_mean,
            "eff_step_mean": eff_step_mean,
            "grad_l2_sq": g_l2_sq,
        }
        per_param[name] = stats

        group = group_for_param(name)
        if group is None:
            continue
        if group not in group_acc:
            group_acc[group] = {
                "n_total": 0,
                "v_mean_w": 0.0,
                "v_max_max": 0.0,
                "precond_mean_w": 0.0,
                "eff_step_mean_w": 0.0,
                "grad_l2_sq_sum": 0.0,
            }
        a = group_acc[group]
        a["n_total"] += n
        a["v_mean_w"] += v_mean * n
        a["v_max_max"] = max(a["v_max_max"], v_max)
        a["precond_mean_w"] += precond_mean * n
        a["eff_step_mean_w"] += eff_step_mean * n
        a["grad_l2_sq_sum"] += g_l2_sq

    per_group: Dict[str, Dict[str, float]] = {}
    for group, a in group_acc.items():
        n = max(a["n_total"], 1)
        per_group[group] = {
            "n": a["n_total"],
            "v_mean": a["v_mean_w"] / n,
            "v_max": a["v_max_max"],
            "precond_mean": a["precond_mean_w"] / n,
            "eff_step_mean": a["eff_step_mean_w"] / n,
            "grad_l2_sq": a["grad_l2_sq_sum"],
            "is_z_pipeline": group in Z_PIPELINE_GROUPS,
        }

    return per_param, per_group


# ---------------------------------------------------------------------------
# Training loop with v-tracking
# ---------------------------------------------------------------------------

def run_v_tracking(cc: CellConfig, output_subdir: str = "v_tracking") -> Dict[str, object]:
    out_dir = RESULTS_DIR / output_subdir / cc.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "log.jsonl"
    v_log_path = out_dir / "v_log.jsonl"
    v_agg_path = out_dir / "v_aggregates.jsonl"
    status_path = out_dir / "status.json"
    config_path = out_dir / "config.json"
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    with open(config_path, "w") as f:
        json.dump({
            "cell": asdict(cc),
            "git_commit": _git_commit(),
            "module_groups": MODULE_GROUPS,
            "z_pipeline_groups": sorted(Z_PIPELINE_GROUPS),
        }, f, indent=2)

    _set_all_seeds(cc.seed)
    device = _select_device()
    print(f"[{cc.run_name}] device={device}  max_steps={cc.max_steps}")

    cfg = build_legacy_cfg(cc)
    tokenizer = create_tokenizer_from_config(cfg)
    train_dataset, _, mapping_data = create_datasets_from_config(cfg, tokenizer)
    log_k = math.log(cc.k)

    train_loader = DataLoader(
        train_dataset, batch_size=cc.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )

    model = create_model_from_config(cfg, tokenizer)

    if cc.optimizer.lower() != "adamw":
        raise ValueError("v-tracking only meaningful for AdamW (no exp_avg_sq under SGD)")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cc.eta, weight_decay=cc.weight_decay,
    )
    scheduler = get_lr_scheduler(
        optimizer, warmup_steps=cc.warmup_steps,
        max_steps=cc.max_steps, scheduler_type=cc.scheduler,
    )

    held_out = build_held_out_batch(
        train_dataset, cc.held_out_grad_batch_size,
        cc.seed * 2654435761 % (2**31), device,
    )

    tracker = RunStatusTracker(cc)
    named_params = list(model.named_parameters())

    log_fp = open(log_path, "w", buffering=1)
    v_fp = open(v_log_path, "w", buffering=1)
    agg_fp = open(v_agg_path, "w", buffering=1)

    def _write(fp, record):
        fp.write(json.dumps(record) + "\n")

    t0 = time.time()
    step = 0
    running_loss = 0.0
    running_grad_norm_sq = 0.0
    running_n = 0
    status: Optional[str] = None

    try:
        while step < cc.max_steps and status is None:
            for batch in train_loader:
                if step >= cc.max_steps or status is not None:
                    break
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss, _, _ = compute_loss(model, batch)
                loss.backward()
                train_grad_norm_sq = sum(
                    float(p.grad.data.norm(2).item() ** 2)
                    for p in model.parameters() if p.grad is not None
                )
                if cc.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cc.grad_clip)
                optimizer.step()
                scheduler.step()

                running_loss += float(loss.item())
                running_grad_norm_sq += train_grad_norm_sq
                running_n += 1
                step += 1

                if step < cc.switch_step:
                    do_log = (step % cc.checkpoint_every_early == 0)
                else:
                    do_log = (step % cc.checkpoint_every_late == 0)
                if not do_log:
                    continue

                # The v-stats need .grad to be present.  We just stepped, so
                # zero_grad has not been called yet; .grad still holds the
                # gradient that produced this update.  Re-run the held-out
                # forward/backward AFTER capturing v-stats so the held-out
                # |g|² is computed without polluting param.grad for v-stats.
                #
                # Order:
                #   1. capture v-stats with current (training-batch) grads
                #   2. zero_grad
                #   3. compute held-out grad norm sq
                #   4. zero_grad again
                #   5. continue training

                per_param_v, per_group_v = collect_v_stats(optimizer, named_params)

                # Held-out grad norm (overwrites .grad — safe because we
                # already collected v-stats)
                g_held = compute_held_out_grad_norm_sq(model, held_out)

                c_clean, c_shuf, delta_z = compute_candidate_loss_and_delta_z(
                    model=model, tokenizer=tokenizer, mapping_data=mapping_data,
                    n_examples=cc.candidate_eval_n, task=cc.task,
                    device=device, seed=step,
                )

                record = {
                    "step": step,
                    "train_loss": running_loss / max(running_n, 1),
                    "candidate_loss": c_clean,
                    "candidate_loss_shuffled_z": c_shuf,
                    "delta_z": delta_z,
                    "grad_norm_sq_held_out": g_held,
                    "grad_norm_sq_training": running_grad_norm_sq / max(running_n, 1),
                    "lr": optimizer.param_groups[0]["lr"],
                    "wall_clock_s": time.time() - t0,
                }
                _write(log_fp, record)
                _write(v_fp, {"step": step, "per_param": per_param_v})
                _write(agg_fp, {"step": step, "per_group": per_group_v})

                running_loss = 0.0
                running_grad_norm_sq = 0.0
                running_n = 0

                status = tracker.update(step, record)
                if status is not None:
                    break

                if step % cc.weight_checkpoint_every == 0:
                    torch.save(model.state_dict(),
                               ckpt_dir / f"model_step_{step:07d}.pt")

        if status is None:
            status = "inconclusive"
        torch.save(model.state_dict(), ckpt_dir / f"model_step_{step:07d}.pt")

    except Exception as exc:  # noqa: BLE001
        print(f"[{cc.run_name}] CRASHED at step {step}: {exc}")
        traceback.print_exc()
        status = "crashed"
    finally:
        log_fp.close()
        v_fp.close()
        agg_fp.close()

    summary = {
        "status": status,
        "final_step": step,
        "wall_clock_s": time.time() - t0,
        "run_name": cc.run_name,
        "eta": cc.eta,
        "k": cc.k,
        "seed": cc.seed,
        "transition_detected_step": tracker._transition_detected_step,
        "experiment": "v_tracking_mv",
    }
    with open(status_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[{cc.run_name}] done: status={status}  step={step}  "
          f"τ={summary['transition_detected_step']}  "
          f"wall={summary['wall_clock_s']:.1f}s")
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eta", type=float, default=1e-3)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--output-subdir", type=str, default="v_tracking")
    args = p.parse_args()

    fields = asdict(CellConfig(eta=args.eta, k=args.k, seed=args.seed))

    class _Override(CellConfig):
        @property
        def max_steps(self_inner):  # type: ignore[override]
            return int(args.max_steps)

    cc = _Override(**fields)
    run_v_tracking(cc, output_subdir=args.output_subdir)


if __name__ == "__main__":
    main()
