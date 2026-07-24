#!/usr/bin/env python
"""
Manual-preconditioning sufficiency experiment.

SGD + per-coordinate 1/sqrt(EMA(g²)) preconditioner = RMSProp(momentum=0,
centered=False).  Isolates Adam's second-moment estimator from its first-
moment estimator: if this transitions where vanilla SGD does not, the
second-moment preconditioner is the sufficient mechanism for plateau→escape.

Cells: K=10, η=1e-3, seeds {0,1,2}, alpha=0.999 (matches Adam's β₂),
weight_decay=0.01, batch=128, 5000 steps.

Output: eta_sweep/results/manual_precond/eta_<η>_K_<K>_seed_<s>/
        with the same log.jsonl format as run_single.py so the existing
        analysis can be reused.
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
    _select_device,
    _set_all_seeds,
    _git_commit,
    build_legacy_cfg,
    build_held_out_batch,
    compute_held_out_grad_norm_sq,
    compute_candidate_loss_and_delta_z,
    RunStatusTracker,
)


def run_manual_precond(
    cc: CellConfig,
    alpha: float = 0.999,
    eps: float = 1e-8,
    output_subdir: str = "manual_precond",
) -> dict:
    out_dir = RESULTS_DIR / output_subdir / cc.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.jsonl"
    status_path = out_dir / "status.json"
    config_path = out_dir / "config.json"
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    with open(config_path, "w") as f:
        json.dump({
            "cell": asdict(cc),
            "git_commit": _git_commit(),
            "optimizer_name": "rmsprop",
            "alpha": alpha,
            "eps": eps,
            "rationale": "SGD + 1/sqrt(EMA(g²)) preconditioner. Tests whether "
                         "Adam's second-moment estimator alone is sufficient "
                         "for plateau→escape (vs vanilla SGD which does not "
                         "transition under matched conditions).",
        }, f, indent=2)

    _set_all_seeds(cc.seed)
    device = _select_device()
    print(f"[{cc.run_name}] device={device}  optimizer=RMSprop(alpha={alpha})  max_steps={cc.max_steps}")

    cfg = build_legacy_cfg(cc)
    tokenizer = create_tokenizer_from_config(cfg)
    train_dataset, _, mapping_data = create_datasets_from_config(cfg, tokenizer)

    train_loader = DataLoader(
        train_dataset, batch_size=cc.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    model = create_model_from_config(cfg, tokenizer)

    # RMSprop: SGD with per-coordinate 1/sqrt(EMA(g²)) preconditioner.
    #   alpha = smoothing constant (== Adam's β₂)
    #   momentum = 0  → strips first-moment contribution
    #   centered = False → matches Adam's exp_avg_sq (no mean subtraction)
    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=cc.eta,
        alpha=alpha,
        eps=eps,
        momentum=0.0,
        weight_decay=cc.weight_decay,
        centered=False,
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
    log_fp = open(log_path, "w", buffering=1)

    def _log(record):
        log_fp.write(json.dumps(record) + "\n")

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
                _log(record)

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

    summary = {
        "status": status,
        "final_step": step,
        "wall_clock_s": time.time() - t0,
        "run_name": cc.run_name,
        "eta": cc.eta,
        "k": cc.k,
        "seed": cc.seed,
        "transition_detected_step": tracker._transition_detected_step,
        "experiment": "manual_precond_mv",
        "optimizer": "rmsprop",
        "alpha": alpha,
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
    p.add_argument("--alpha", type=float, default=0.999,
                   help="RMSprop smoothing constant; 0.999 matches Adam β₂")
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--stuck-patience", type=int, default=None,
                   help="Override stuck_patience (default 2000)")
    p.add_argument("--output-subdir", type=str, default="manual_precond")
    args = p.parse_args()

    fields = asdict(CellConfig(eta=args.eta, k=args.k, seed=args.seed))
    if args.stuck_patience is not None:
        fields["stuck_patience"] = int(args.stuck_patience)

    class _Override(CellConfig):
        @property
        def max_steps(self_inner):  # type: ignore[override]
            return int(args.max_steps)

    cc = _Override(**fields)
    run_manual_precond(
        cc,
        alpha=args.alpha,
        output_subdir=args.output_subdir,
    )


if __name__ == "__main__":
    main()
