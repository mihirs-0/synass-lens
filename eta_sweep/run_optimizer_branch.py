#!/usr/bin/env python
"""
Branch from a common pre-plateau checkpoint with different optimizer rules.

Trains baseline AdamW(β₁=0.9, β₂=0.999) for `branch_step` steps, then
swaps in a new optimizer with FRESH STATE (m_t = v_t = 0) for the
remaining budget.  Tests which per-step update rules can drive escape
from a common starting point in weight space, isolating the local-rule
question from the accumulated-state question.

Branch configs:
  adamw_b1_0    — AdamW β₁=0,    β₂=0.999  (no first-moment averaging)
  adamw_b1_0.5  — AdamW β₁=0.5,  β₂=0.999  (partial)
  adamw_b1_0.9  — AdamW β₁=0.9,  β₂=0.999  (control, same as baseline)
  adamw_b1_0.99 — AdamW β₁=0.99, β₂=0.999  (heavy)
  rmsprop       — RMSProp α=0.999, momentum=0       (no first moment)
  sgd_mom       — SGD momentum=0.9 at η=0.01        (first moment, no scaling)

Output: eta_sweep/results/optimizer_branch/<config>_seed<s>/
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


def make_branch_optimizer(branch_config: str, model: torch.nn.Module, base_eta: float, weight_decay: float):
    """Construct a fresh optimizer per branch config."""
    # Specific configs first to avoid prefix collisions
    if branch_config == "adamw_b1_0_rmsprop_eps":
        from eta_sweep.custom_optimizers import AdamWRMSPropEps
        return AdamWRMSPropEps(model.parameters(), lr=base_eta,
                               betas=(0.0, 0.999), weight_decay=weight_decay), base_eta
    if branch_config.startswith("adamw_b1_") and not branch_config.endswith("_nobc"):
        b1 = float(branch_config.split("_")[-1])
        return torch.optim.AdamW(model.parameters(), lr=base_eta,
                                 betas=(b1, 0.999), weight_decay=weight_decay), base_eta
    if branch_config.endswith("_nobc"):
        # AdamW without bias correction (β₁ from string)
        from eta_sweep.custom_optimizers import AdamWNoBiasCorrection
        # parse e.g. adamw_b1_0_nobc → β₁=0
        b1_str = branch_config.replace("adamw_b1_", "").replace("_nobc", "")
        b1 = float(b1_str)
        return AdamWNoBiasCorrection(model.parameters(), lr=base_eta,
                                     betas=(b1, 0.999), weight_decay=weight_decay), base_eta
    if branch_config == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=base_eta,
                                   alpha=0.999, momentum=0.0,
                                   weight_decay=weight_decay), base_eta
    if branch_config == "rmsprop_decoupled_bc":
        from eta_sweep.custom_optimizers import RMSPropDecoupledBiasCorrected
        return RMSPropDecoupledBiasCorrected(model.parameters(), lr=base_eta,
                                             alpha=0.999, weight_decay=weight_decay), base_eta
    if branch_config == "rmsprop_bc_only":
        from eta_sweep.custom_optimizers import RMSPropBiasCorrected
        return RMSPropBiasCorrected(model.parameters(), lr=base_eta,
                                    alpha=0.999, weight_decay=weight_decay), base_eta
    if branch_config == "rmsprop_decoupled_only":
        from eta_sweep.custom_optimizers import RMSPropDecoupledOnly
        return RMSPropDecoupledOnly(model.parameters(), lr=base_eta,
                                    alpha=0.999, weight_decay=weight_decay), base_eta
    if branch_config == "adam_coupled_b1_0":
        from eta_sweep.custom_optimizers import AdamCoupledWD
        return AdamCoupledWD(model.parameters(), lr=base_eta,
                             betas=(0.0, 0.999), weight_decay=weight_decay), base_eta
    if branch_config == "adamw_b1_0_rmsprop_eps":
        from eta_sweep.custom_optimizers import AdamWRMSPropEps
        return AdamWRMSPropEps(model.parameters(), lr=base_eta,
                               betas=(0.0, 0.999), weight_decay=weight_decay), base_eta
    if branch_config == "rmsprop_adamw_eps":
        from eta_sweep.custom_optimizers import RMSPropAdamWEps
        return RMSPropAdamWEps(model.parameters(), lr=base_eta,
                               alpha=0.999, weight_decay=weight_decay), base_eta
    if branch_config == "sgd_mom":
        sgd_eta = 0.01  # 10x AdamW base — matched to the §6 four-cell ablation
        return torch.optim.SGD(model.parameters(), lr=sgd_eta,
                               momentum=0.9, weight_decay=weight_decay), sgd_eta
    raise ValueError(f"unknown branch config: {branch_config}")


def run_branch(cc: CellConfig, branch_config: str, branch_step: int,
               output_subdir: str = "optimizer_branch") -> dict:
    cell_name = f"{branch_config}_seed{cc.seed}"
    out_dir = RESULTS_DIR / output_subdir / cell_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.jsonl"
    status_path = out_dir / "status.json"
    config_path = out_dir / "config.json"
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    with open(config_path, "w") as f:
        json.dump({"cell": asdict(cc), "branch_config": branch_config,
                   "branch_step": branch_step, "git_commit": _git_commit()}, f, indent=2)

    _set_all_seeds(cc.seed)
    device = _select_device()
    print(f"[{cell_name}] device={device}  branch_at={branch_step}  config={branch_config}")

    cfg = build_legacy_cfg(cc)
    tokenizer = create_tokenizer_from_config(cfg)
    train_dataset, _, mapping_data = create_datasets_from_config(cfg, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=cc.batch_size,
                              shuffle=True, collate_fn=collate_fn, num_workers=0)
    model = create_model_from_config(cfg, tokenizer)

    # Phase 1: baseline AdamW(β₁=0.9, β₂=0.999) for `branch_step` steps
    baseline_eta = cc.eta
    optimizer = torch.optim.AdamW(model.parameters(), lr=baseline_eta,
                                  betas=(0.9, 0.999), weight_decay=cc.weight_decay)
    # Constant LR (no scheduler) to keep things simple
    held_out = build_held_out_batch(train_dataset, cc.held_out_grad_batch_size,
                                    cc.seed * 2654435761 % (2**31), device)

    tracker = RunStatusTracker(cc)
    log_fp = open(log_path, "w", buffering=1)
    def _log(r): log_fp.write(json.dumps(r) + "\n")

    t0 = time.time()
    step = 0
    running_loss, running_grad_sq, running_n = 0.0, 0.0, 0
    status: Optional[str] = None
    branch_done = False
    current_eta = baseline_eta

    try:
        while step < cc.max_steps and status is None:
            for batch in train_loader:
                if step >= cc.max_steps or status is not None:
                    break

                # === BRANCH HOOK: at branch_step, swap optimizer ===
                if not branch_done and step == branch_step:
                    optimizer, current_eta = make_branch_optimizer(
                        branch_config, model, baseline_eta, cc.weight_decay,
                    )
                    print(f"[{cell_name}] BRANCHED at step {step} → {branch_config} (η={current_eta})")
                    branch_done = True

                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss, _, _ = compute_loss(model, batch)
                loss.backward()
                tg_sq = sum(float(p.grad.data.norm(2).item()**2)
                            for p in model.parameters() if p.grad is not None)
                optimizer.step()
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
                    "lr": current_eta, "wall_clock_s": time.time() - t0,
                    "branch_done": branch_done,
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
        "branch_config": branch_config, "branch_step": branch_step,
        "transition_detected_step": tracker._transition_detected_step,
        "experiment": "optimizer_branch",
    }
    with open(status_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{cell_name}] done: status={status}  τ={summary['transition_detected_step']}  "
          f"wall={summary['wall_clock_s']:.1f}s")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--branch-config", type=str, required=True,
                   choices=["adamw_b1_0", "adamw_b1_0.5", "adamw_b1_0.9",
                            "adamw_b1_0.99", "rmsprop", "sgd_mom",
                            "adamw_b1_0_nobc", "rmsprop_decoupled_bc",
                            "rmsprop_bc_only", "rmsprop_decoupled_only",
                            "adam_coupled_b1_0", "adamw_b1_0_rmsprop_eps",
                            "rmsprop_adamw_eps"])
    p.add_argument("--branch-step", type=int, default=1500)
    p.add_argument("--max-steps", type=int, default=6000)
    p.add_argument("--eta", type=float, default=1e-3)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--output-subdir", type=str, default="optimizer_branch")
    p.add_argument("--weight-decay", type=float, default=None,
                   help="Override weight decay (default: from CellConfig, typically 0.01)")
    p.add_argument("--stuck-patience", type=int, default=None,
                   help="Override stuck_patience (raise very high to disable early-stuck cutoff)")
    args = p.parse_args()

    fields = asdict(CellConfig(eta=args.eta, k=args.k, seed=args.seed))
    if args.weight_decay is not None:
        fields["weight_decay"] = args.weight_decay
    if args.stuck_patience is not None:
        fields["stuck_patience"] = args.stuck_patience
    class _Override(CellConfig):
        @property
        def max_steps(self_inner): return int(args.max_steps)
    cc = _Override(**fields)
    run_branch(cc, branch_config=args.branch_config, branch_step=args.branch_step,
               output_subdir=args.output_subdir)


if __name__ == "__main__":
    main()
