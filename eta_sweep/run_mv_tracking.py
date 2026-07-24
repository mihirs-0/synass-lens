#!/usr/bin/env python
"""
m_t + v_t tracking — extends run_v_tracking with first-moment logging.

In addition to per-group v_t aggregates, saves the full flattened m_t
vector at each checkpoint as .npy.  Enables post-hoc:
  - long-window smoothed m_t cosine similarity across plateau
  - shuffled-checkpoint controls (negative control 1)
  - random-window-pair controls (negative control 2)
  - cross-seed m_t comparison

Output: eta_sweep/results/mv_tracking/eta_<η>_K_<K>_seed_<s>/
        log.jsonl, v_aggregates.jsonl (existing format)
        m_vectors/m_step_<step>.npy (NEW)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Optional

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
    _select_device, _set_all_seeds, _git_commit, build_legacy_cfg,
    build_held_out_batch, compute_held_out_grad_norm_sq,
    compute_candidate_loss_and_delta_z, RunStatusTracker,
)
from eta_sweep.run_v_tracking import (  # noqa: E402
    MODULE_GROUPS, Z_PIPELINE_GROUPS, group_for_param, collect_v_stats,
)


def flatten_m(optimizer, named_params):
    """Flatten Adam's exp_avg across all parameters into one numpy vector."""
    parts = []
    for name, p in named_params:
        state = optimizer.state.get(p, {})
        m = state.get("exp_avg")
        if m is None:
            parts.append(np.zeros(p.numel(), dtype=np.float32))
        else:
            parts.append(m.detach().cpu().numpy().astype(np.float32).ravel())
    return np.concatenate(parts)


def run_mv(cc: CellConfig, output_subdir: str = "mv_tracking") -> dict:
    out_dir = RESULTS_DIR / output_subdir / cc.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    m_dir = out_dir / "m_vectors"
    m_dir.mkdir(exist_ok=True)
    log_path = out_dir / "log.jsonl"
    v_agg_path = out_dir / "v_aggregates.jsonl"
    status_path = out_dir / "status.json"
    config_path = out_dir / "config.json"
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    with open(config_path, "w") as f:
        json.dump({"cell": asdict(cc), "git_commit": _git_commit(),
                   "logs": "m_t flattened vectors per checkpoint, plus v_t aggregates"}, f, indent=2)

    _set_all_seeds(cc.seed)
    device = _select_device()
    print(f"[{cc.run_name}] device={device}  max_steps={cc.max_steps}")

    cfg = build_legacy_cfg(cc)
    tokenizer = create_tokenizer_from_config(cfg)
    train_dataset, _, mapping_data = create_datasets_from_config(cfg, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=cc.batch_size,
                              shuffle=True, collate_fn=collate_fn, num_workers=0)
    model = create_model_from_config(cfg, tokenizer)

    if cc.optimizer.lower() != "adamw":
        raise ValueError("mv-tracking is for AdamW only")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cc.eta, weight_decay=cc.weight_decay)
    scheduler = get_lr_scheduler(optimizer, warmup_steps=cc.warmup_steps,
                                 max_steps=cc.max_steps, scheduler_type=cc.scheduler)
    held_out = build_held_out_batch(train_dataset, cc.held_out_grad_batch_size,
                                    cc.seed * 2654435761 % (2**31), device)

    tracker = RunStatusTracker(cc)
    named_params = list(model.named_parameters())

    log_fp = open(log_path, "w", buffering=1)
    agg_fp = open(v_agg_path, "w", buffering=1)
    def _w(fp, r): fp.write(json.dumps(r) + "\n")

    t0 = time.time()
    step = 0
    running_loss, running_grad_sq, running_n = 0.0, 0.0, 0
    status: Optional[str] = None

    try:
        while step < cc.max_steps and status is None:
            for batch in train_loader:
                if step >= cc.max_steps or status is not None:
                    break
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss, _, _ = compute_loss(model, batch)
                loss.backward()
                tg_sq = sum(float(p.grad.data.norm(2).item()**2)
                            for p in model.parameters() if p.grad is not None)
                optimizer.step()
                scheduler.step()
                running_loss += float(loss.item())
                running_grad_sq += tg_sq
                running_n += 1
                step += 1

                do_log = (step % cc.checkpoint_every_early == 0) if step < cc.switch_step \
                         else (step % cc.checkpoint_every_late == 0)
                if not do_log:
                    continue

                # === NEW: save full m_t at this checkpoint ===
                m_flat = flatten_m(optimizer, named_params)
                np.save(m_dir / f"m_step_{step:07d}.npy", m_flat)

                _, per_group_v = collect_v_stats(optimizer, named_params)
                g_held = compute_held_out_grad_norm_sq(model, held_out)
                c_clean, c_shuf, dz = compute_candidate_loss_and_delta_z(
                    model=model, tokenizer=tokenizer, mapping_data=mapping_data,
                    n_examples=cc.candidate_eval_n, task=cc.task, device=device, seed=step)

                record = {
                    "step": step, "train_loss": running_loss / max(running_n, 1),
                    "candidate_loss": c_clean, "candidate_loss_shuffled_z": c_shuf,
                    "delta_z": dz, "grad_norm_sq_held_out": g_held,
                    "grad_norm_sq_training": running_grad_sq / max(running_n, 1),
                    "lr": optimizer.param_groups[0]["lr"],
                    "wall_clock_s": time.time() - t0,
                    "m_norm": float(np.linalg.norm(m_flat)),
                }
                _w(log_fp, record)
                _w(agg_fp, {"step": step, "per_group": per_group_v})

                running_loss, running_grad_sq, running_n = 0.0, 0.0, 0
                status = tracker.update(step, record)
                if status is not None:
                    break
                if step % cc.weight_checkpoint_every == 0:
                    torch.save(model.state_dict(), ckpt_dir / f"model_step_{step:07d}.pt")

        if status is None:
            status = "inconclusive"
        torch.save(model.state_dict(), ckpt_dir / f"model_step_{step:07d}.pt")
    except Exception as exc:
        print(f"[{cc.run_name}] CRASHED at step {step}: {exc}")
        traceback.print_exc()
        status = "crashed"
    finally:
        log_fp.close()
        agg_fp.close()

    summary = {
        "status": status, "final_step": step, "wall_clock_s": time.time() - t0,
        "run_name": cc.run_name, "eta": cc.eta, "k": cc.k, "seed": cc.seed,
        "transition_detected_step": tracker._transition_detected_step,
        "experiment": "mv_tracking",
    }
    with open(status_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{cc.run_name}] done: status={status}  τ={summary['transition_detected_step']}  "
          f"wall={summary['wall_clock_s']:.1f}s")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eta", type=float, default=1e-3)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--output-subdir", type=str, default="mv_tracking")
    args = p.parse_args()

    fields = asdict(CellConfig(eta=args.eta, k=args.k, seed=args.seed))
    class _Override(CellConfig):
        @property
        def max_steps(self_inner): return int(args.max_steps)
    cc = _Override(**fields)
    run_mv(cc, output_subdir=args.output_subdir)


if __name__ == "__main__":
    main()
