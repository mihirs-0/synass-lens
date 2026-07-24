"""Run probe + Δz + patching on stuck-regime checkpoints (non-AdamW arms).

Confirms that in stuck regimes (cand_loss ≈ log K), the calibrated upstream
diagnostics (probe, Δz, patch recovery) also fail to progress.

Outputs eta_sweep/results/stuck_regime_diagnostics.json.
"""
from __future__ import annotations
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from eta_sweep.config import CellConfig, RESULTS_DIR  # noqa: E402
from eta_sweep.analysis.mech_interp_mbc import (  # noqa: E402
    activation_patching, behavioral, attention_to_z, build_eval_batch,
    collect_residuals, linear_probes, _select_device, build_legacy_cfg,
    get_probe_labels,
)
from src.data import (  # noqa: E402
    create_tokenizer_from_config, create_datasets_from_config,
)
from src.model import create_model_from_config  # noqa: E402


@dataclass
class StuckCell:
    label: str
    branch: str   # 'rmsprop', 'sgd_mom', etc.
    eta: float
    k: int
    seed: int
    optimizer_branch_dir: Path  # full path

    @property
    def ckpt_dir(self) -> Path:
        return self.optimizer_branch_dir / "checkpoints"


def list_stuck_cells():
    base = RESULTS_DIR / "optimizer_branch"
    cells = []
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        status_p = child / "status.json"
        if not status_p.exists():
            continue
        try:
            st = json.load(open(status_p))
        except Exception:
            continue
        if st.get("status") != "stuck":
            continue
        # We want a few representative non-AdamW arms
        bc = st.get("branch_config", "")
        if bc.startswith("rmsprop") or bc.startswith("sgd"):
            cells.append(StuckCell(
                label=child.name,
                branch=bc,
                eta=st.get("eta"),
                k=st.get("k"),
                seed=st.get("seed"),
                optimizer_branch_dir=child,
            ))
    return cells


def load_stuck_cell(cell: StuckCell, device: str):
    cc = CellConfig(eta=cell.eta, k=cell.k, seed=cell.seed)
    cfg = build_legacy_cfg(cc)
    tokenizer = create_tokenizer_from_config(cfg)
    train_ds, _, mapping_data = create_datasets_from_config(cfg, tokenizer)
    return cc, cfg, tokenizer, train_ds, mapping_data


def make_model_at(cell: StuckCell, cfg, tokenizer, ckpt_step: int, device):
    model = create_model_from_config(cfg, tokenizer)
    sd = torch.load(
        cell.ckpt_dir / f"model_step_{ckpt_step:07d}.pt",
        map_location=device,
    )
    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()
    return model


def main():
    device = _select_device()
    print(f"device: {device}")

    cells = list_stuck_cells()
    print(f"found {len(cells)} stuck cells")
    for c in cells:
        print(f"  - {c.label} ({c.branch}, K={c.k}, η={c.eta}, seed={c.seed})")

    out = {"device": device, "cells": {}}
    t_start = time.time()

    for cell in cells:
        # Pick the latest checkpoint
        ckpts = sorted(int(p.stem.split("_")[-1])
                       for p in cell.ckpt_dir.glob("model_step_*.pt"))
        if not ckpts:
            continue
        step = ckpts[-1]
        print(f"\n>>> {cell.label} step={step}")
        try:
            cc, cfg, tokenizer, train_ds, mapping_data = load_stuck_cell(
                cell, device)
            model = make_model_at(cell, cfg, tokenizer, step, device)
        except Exception as e:
            print(f"  load error: {e}")
            out["cells"][cell.label] = {"error": str(e)}
            continue

        eval_batch, eval_idx = build_eval_batch(train_ds, 192, seed=0,
                                                device=device)
        beh = behavioral(model, tokenizer, mapping_data,
                         n_examples=64, device=device, seed=0)
        # Probe at target_start residual
        feats = collect_residuals(model, eval_batch, device)
        arr = feats["resid_pre_target_start"]
        labels = get_probe_labels(eval_batch, train_ds, mapping_data,
                                  eval_idx)["z_id"][:192]
        probe_per_layer = []
        for L in range(arr.shape[1]):
            X = arr[:, L, :]
            r = linear_probes(X, labels, n_train=128, n_test=48,
                              seed=0, C=0.1)
            probe_per_layer.append(r["test_acc"])
        # Patching
        pat = activation_patching(model, tokenizer, mapping_data, device,
                                  n_pairs=24, seed=0)
        atz = attention_to_z(model, eval_batch, device)
        log_k = math.log(cell.k)
        cell_out = {
            "branch": cell.branch,
            "eta": cell.eta, "k": cell.k, "seed": cell.seed,
            "step": step,
            "log_k": log_k,
            "candidate_loss": beh.get("candidate_loss"),
            "candidate_loss_over_logK": beh.get("candidate_loss") / log_k,
            "candidate_accuracy": beh.get("candidate_accuracy"),
            "delta_z": beh.get("delta_z"),
            "probe_per_layer": probe_per_layer,
            "patch_recovery_per_layer": pat.get("recovery_per_layer"),
            "patch_p_clean": pat.get("p_target_clean_mean"),
            "patch_p_shuf": pat.get("p_target_shuf_mean"),
            "max_attention_to_z": float(atz["attention_to_z"].max()),
        }
        print(f"  cand/logK={cell_out['candidate_loss_over_logK']:.3f} "
              f"top1={cell_out['candidate_accuracy']:.3f} "
              f"Δz={cell_out['delta_z']:.3f} "
              f"R={cell_out['patch_recovery_per_layer']}")
        out["cells"][cell.label] = cell_out
        del model
        if device == "mps":
            torch.mps.empty_cache()

    out["total_wall_clock_s"] = time.time() - t_start
    out_path = RESULTS_DIR / "stuck_regime_diagnostics.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=lambda o: o.tolist()
                  if hasattr(o, "tolist") else str(o))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
