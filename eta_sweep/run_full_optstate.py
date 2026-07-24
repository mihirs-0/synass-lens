"""K=10 AdamW retrain that saves the FULL optimizer state (m_t, v_t per param)
per checkpoint.  Used for H4: cos(m_t, m_{t-1}) vs. cos(m_t/√v_t, ...).

Outputs to eta_sweep/results/full_optstate/eta_0.001_K_10_seed_0/.
"""
from __future__ import annotations
import argparse, json, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data import collate_fn, create_datasets_from_config, create_tokenizer_from_config
from src.model import create_model_from_config
from src.training.trainer import compute_loss, get_lr_scheduler
from eta_sweep.config import CellConfig
from eta_sweep.run_single import build_legacy_cfg


def save_opt_state(optimizer, named_params, out_path):
    state = {}
    for name, p in named_params:
        st = optimizer.state.get(p, {})
        if "exp_avg" in st and "exp_avg_sq" in st:
            state[name] = {
                "m": st["exp_avg"].detach().cpu().clone(),
                "v": st["exp_avg_sq"].detach().cpu().clone(),
                "step_count": int(st.get("step", torch.tensor(0)).item()) if torch.is_tensor(st.get("step", 0)) else int(st.get("step", 0)),
            }
    torch.save(state, out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-steps", type=int, default=8000)
    p.add_argument("--every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--eta", type=float, default=1e-3)
    p.add_argument("--out", type=str, default="full_optstate")
    a = p.parse_args()

    cc = CellConfig(eta=a.eta, k=a.k, seed=a.seed, optimizer="adamw")
    out_dir = REPO / "eta_sweep" / "results" / a.out / cc.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"; ckpt_dir.mkdir(exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({
        "eta": cc.eta, "K": cc.k, "seed": cc.seed,
        "max_steps": a.max_steps, "every": a.every,
        "weight_decay": cc.weight_decay, "warmup": cc.warmup_steps,
    }, indent=2))

    cfg = build_legacy_cfg(cc)
    cfg.training.max_steps = a.max_steps
    np.random.seed(cc.seed); torch.manual_seed(cc.seed); random.seed(cc.seed)

    tokenizer = create_tokenizer_from_config(cfg)
    train_ds, _, _ = create_datasets_from_config(cfg, tokenizer)
    loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  max_steps={a.max_steps}  every={a.every}  K={cc.k}  η={cc.eta}")

    model = create_model_from_config(cfg, tokenizer).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cc.eta, weight_decay=cc.weight_decay)
    scheduler = get_lr_scheduler(optimizer, warmup_steps=cc.warmup_steps, max_steps=a.max_steps, scheduler_type=cc.scheduler)

    named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    log = open(out_dir / "log.jsonl", "w")
    step = 0
    t0 = time.time()
    log_K = float(np.log(cc.k))

    # Save step 0
    torch.save(model.state_dict(), ckpt_dir / f"model_step_{step:07d}.pt")

    done = False
    while not done:
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss, _acc, first_target_loss = compute_loss(model, batch)
            loss.backward()
            grad_norm_sq = sum(float((p.grad.detach()**2).sum().item()) for _, p in named_params if p.grad is not None)
            optimizer.step()
            scheduler.step()
            step += 1

            if step % a.every == 0:
                rec = {
                    "step": step,
                    "train_loss": float(loss.item()),
                    "first_target_loss": float(first_target_loss),
                    "grad_norm_sq": grad_norm_sq,
                    "lr": optimizer.param_groups[0]["lr"],
                }
                log.write(json.dumps(rec) + "\n"); log.flush()
                torch.save(model.state_dict(), ckpt_dir / f"model_step_{step:07d}.pt")
                save_opt_state(optimizer, named_params, ckpt_dir / f"opt_step_{step:07d}.pt")
                if step % (a.every * 4) == 0:
                    print(f"step={step:>5d}  ftl={first_target_loss:.4f}  /logK={first_target_loss/log_K:.3f}  elapsed={time.time()-t0:.0f}s")

            if step >= a.max_steps:
                done = True; break

    log.close()
    print(f"\ndone.  total steps={step}  elapsed={time.time()-t0:.0f}s")
    print(f"output: {out_dir}")


if __name__ == "__main__":
    main()
