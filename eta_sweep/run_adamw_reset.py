#!/usr/bin/env python
"""
AdamW + moment reset at a specified plateau step.

At the reset step, zero out optimizer.state[p]['exp_avg'] and 'exp_avg_sq'
for all parameters, then continue training.  Measures whether the
transition step τ shifts.

Three outcomes possible:
  - τ unchanged: optimizer state is not carrying meaningful info at reset
  - τ delayed by ≈ warmup time: moments accumulating gradually, reset removes that
  - τ pushed past budget: moments are essentially the entire mechanism

Output: eta_sweep/results/adamw_reset/eta_<η>_K_<K>_seed_<s>_reset_<step>/
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


def reset_adamw_state(optimizer):
    """Zero all exp_avg and exp_avg_sq, reset step counter."""
    n_reset = 0
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state.get(p, {})
            if "exp_avg" in state:
                state["exp_avg"].zero_()
            if "exp_avg_sq" in state:
                state["exp_avg_sq"].zero_()
            if "step" in state:
                state["step"] = torch.tensor(0.0) if isinstance(state["step"], torch.Tensor) else 0
            n_reset += 1
    return n_reset


def run_reset(cc: CellConfig, reset_step: int, output_subdir: str = "adamw_reset") -> dict:
    cell_name = f"{cc.run_name}_reset_{reset_step}"
    out_dir = RESULTS_DIR / output_subdir / cell_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.jsonl"
    status_path = out_dir / "status.json"
    config_path = out_dir / "config.json"
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    with open(config_path, "w") as f:
        json.dump({"cell": asdict(cc), "reset_step": reset_step,
                   "git_commit": _git_commit()}, f, indent=2)

    _set_all_seeds(cc.seed)
    device = _select_device()
    print(f"[{cell_name}] device={device}  reset_at={reset_step}")

    cfg = build_legacy_cfg(cc)
    tokenizer = create_tokenizer_from_config(cfg)
    train_dataset, _, mapping_data = create_datasets_from_config(cfg, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=cc.batch_size,
                              shuffle=True, collate_fn=collate_fn, num_workers=0)
    model = create_model_from_config(cfg, tokenizer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cc.eta, weight_decay=cc.weight_decay)
    scheduler = get_lr_scheduler(optimizer, warmup_steps=cc.warmup_steps,
                                 max_steps=cc.max_steps, scheduler_type=cc.scheduler)
    held_out = build_held_out_batch(train_dataset, cc.held_out_grad_batch_size,
                                    cc.seed * 2654435761 % (2**31), device)

    tracker = RunStatusTracker(cc)
    log_fp = open(log_path, "w", buffering=1)
    def _log(r): log_fp.write(json.dumps(r) + "\n")

    t0 = time.time()
    step = 0
    running_loss, running_grad_sq, running_n = 0.0, 0.0, 0
    status: Optional[str] = None
    reset_done = False

    try:
        while step < cc.max_steps and status is None:
            for batch in train_loader:
                if step >= cc.max_steps or status is not None:
                    break

                # === RESET HOOK: fire at reset_step ===
                if not reset_done and step == reset_step:
                    n_reset = reset_adamw_state(optimizer)
                    print(f"[{cell_name}] RESET at step {step}: zeroed state for {n_reset} params")
                    reset_done = True

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
                    "reset_done": reset_done,
                }
                _log(record)
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
        print(f"[{cell_name}] CRASHED at step {step}: {exc}")
        traceback.print_exc()
        status = "crashed"
    finally:
        log_fp.close()

    summary = {
        "status": status, "final_step": step, "wall_clock_s": time.time() - t0,
        "run_name": cell_name, "eta": cc.eta, "k": cc.k, "seed": cc.seed,
        "reset_step": reset_step, "reset_done": reset_done,
        "transition_detected_step": tracker._transition_detected_step,
        "experiment": "adamw_moment_reset",
    }
    with open(status_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{cell_name}] done: status={status}  τ={summary['transition_detected_step']}  "
          f"wall={summary['wall_clock_s']:.1f}s")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eta", type=float, default=1e-3)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--reset-step", type=int, required=True)
    p.add_argument("--max-steps", type=int, default=6000)
    p.add_argument("--output-subdir", type=str, default="adamw_reset")
    args = p.parse_args()

    fields = asdict(CellConfig(eta=args.eta, k=args.k, seed=args.seed))
    class _Override(CellConfig):
        @property
        def max_steps(self_inner): return int(args.max_steps)
    cc = _Override(**fields)
    run_reset(cc, reset_step=args.reset_step, output_subdir=args.output_subdir)


if __name__ == "__main__":
    main()
