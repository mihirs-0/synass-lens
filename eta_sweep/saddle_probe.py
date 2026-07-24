#!/usr/bin/env python
"""
Saddle probe — cross-codebase reproduction.

Measurements at three plateau checkpoints (≈0.3τ, 0.6τ, 0.9τ):
  M1: full gradient L2 norm (avg over 16 batches of 128)
  M2: Hessian smallest eigenvalue λ_min and eigenvector v_min (Lanczos via HVP)
  M3: cos(∇L, v_min) and angle in degrees
  M4: per-example gradient coherence over n=500 examples
  M5: z_shuffle_gap (loss with z permuted within B-group)
  M6: |∇L · v_min| (gradient projection onto smallest eigenvector)

Uses canonical setup: K=10, η=1e-3, AdamW, seed 0.

If any measurement fails, the failure is reported in the JSON.  No silent
fallbacks.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from dataclasses import asdict
from typing import Optional, Tuple, Dict, List

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ETA_SWEEP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = ETA_SWEEP_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data import (CharTokenizer, MappingData, collate_fn,
                      create_datasets_from_config, create_tokenizer_from_config)
from src.model import create_model_from_config
from src.training.trainer import compute_loss

from eta_sweep.config import CellConfig, RESULTS_DIR
from eta_sweep.run_single import _select_device, _set_all_seeds, build_legacy_cfg


OUT_DIR = RESULTS_DIR / "saddle_probe"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def flat_grad(model: torch.nn.Module) -> torch.Tensor:
    """Flatten current .grad of model parameters into a single vector."""
    return torch.cat([p.grad.detach().reshape(-1) for p in model.parameters()
                      if p.grad is not None])


def n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_full_grad(model, loader, n_batches: int, device: str) -> torch.Tensor:
    """Average gradient over n_batches drawn from loader.  Returns flat tensor."""
    model.zero_grad(set_to_none=True)
    accum = None
    n_seen = 0
    it = iter(loader)
    for _ in range(n_batches):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        loss, _, _ = compute_loss(model, batch)
        model.zero_grad(set_to_none=True)
        loss.backward()
        g = flat_grad(model)
        if accum is None:
            accum = g.clone()
        else:
            accum += g
        n_seen += 1
    return accum / n_seen


def hvp(model, loader_iter_factory, n_batches: int, device: str, vector: torch.Tensor):
    """Hessian–vector product H @ vector via double-backprop, averaged over n_batches.
    Returns flat tensor on cpu (float32)."""
    accum = None
    seen = 0
    it = loader_iter_factory()
    for _ in range(n_batches):
        try:
            batch = next(it)
        except StopIteration:
            it = loader_iter_factory()
            batch = next(it)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        model.zero_grad(set_to_none=True)
        loss, _, _ = compute_loss(model, batch)

        # Compute first-order grads with create_graph=True for double-backprop.
        params = [p for p in model.parameters() if p.requires_grad]
        grads = torch.autograd.grad(loss, params, create_graph=True)
        flat = torch.cat([g.reshape(-1) for g in grads])
        # dot product with vector
        v = vector.to(flat.device, dtype=flat.dtype)
        gv = (flat * v).sum()
        # Second backward to get H @ v
        Hv_grads = torch.autograd.grad(gv, params, retain_graph=False)
        Hv = torch.cat([g.detach().reshape(-1) for g in Hv_grads])
        if accum is None:
            accum = Hv.clone()
        else:
            accum += Hv
        seen += 1
        # Free graph
        del grads, flat, gv, Hv_grads, Hv
    return (accum / seen).detach().cpu().to(torch.float32)


def lanczos_smallest(model, loader_factory, n_hvp_batches: int, device: str,
                     n_params: int, k: int = 1, ncv: int = 60,
                     tol: float = 1e-5, shift: float = 1e3,
                     max_iter: int = 200) -> Tuple[float, np.ndarray, dict]:
    """Lanczos iteration for smallest algebraic eigenvalue using a shift-and-invert
    style: solve for largest of (shift - H), then return (shift - largest).
    But scipy's eigsh with sigma is overkill on MPS.  Use plain Lanczos on -H
    and ask for largest, then negate.

    Implementation: scipy.sparse.linalg.eigsh on a LinearOperator that returns
    -H @ v (i.e., negative Hessian times v).  Use which='LA' for largest of -H,
    which corresponds to smallest of H.

    Returns (lambda_min, v_min flat numpy, info dict).
    """
    from scipy.sparse.linalg import LinearOperator, eigsh

    n_evals = {"count": 0}
    last_time = {"t": time.time()}

    def matvec(v):
        n_evals["count"] += 1
        v_t = torch.from_numpy(v.astype(np.float32))
        Hv = hvp(model, loader_factory, n_hvp_batches, device, v_t)
        # Return -H @ v so largest of -H = smallest of H
        return (-Hv).numpy().astype(np.float64)

    op = LinearOperator(shape=(n_params, n_params), matvec=matvec, dtype=np.float64)
    t0 = time.time()
    eigvals, eigvecs = eigsh(op, k=k, which='LA', ncv=ncv, tol=tol, maxiter=max_iter)
    elapsed = time.time() - t0
    # eigvals are eigvals of -H (sorted ascending in scipy's convention for 'LA' it's actually unsorted).
    # Take the largest of -H → smallest of H.
    idx = int(np.argmax(eigvals))
    lambda_neg_H = float(eigvals[idx])
    lambda_min = -lambda_neg_H
    v_min = eigvecs[:, idx]
    info = {
        "n_hvp_calls": n_evals["count"],
        "lanczos_wall_s": float(elapsed),
        "ncv": ncv, "tol": tol, "max_iter": max_iter,
    }
    return lambda_min, v_min, info


def per_example_gradient_coherence(model, loader, n_examples: int, device: str) -> dict:
    """Compute coherence = ||mean(g_i)||² / mean(||g_i||²) over n_examples.

    Implementation: walk single-example batches one at a time.  For each, do
    a forward+backward, capture flat gradient, accumulate sum and sum of
    squared norms.  Memory-efficient (does not store all per-example gradients
    explicitly, just running sums).  But we DO need each individual gradient
    for the mean, so store them.
    """
    model.zero_grad(set_to_none=True)
    grads_list = []
    n_seen = 0
    it = iter(loader)
    while n_seen < n_examples:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        # Process one example at a time within the batch
        bsz = batch[next(iter(batch))].shape[0] if isinstance(batch, dict) else batch.shape[0]
        for i in range(bsz):
            if n_seen >= n_examples:
                break
            single = {k: (v[i:i+1] if isinstance(v, torch.Tensor) else v)
                      for k, v in batch.items()}
            single = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in single.items()}
            model.zero_grad(set_to_none=True)
            loss, _, _ = compute_loss(model, single)
            loss.backward()
            g = flat_grad(model).detach().cpu().to(torch.float32)
            grads_list.append(g)
            n_seen += 1
    G = torch.stack(grads_list)  # (n, D)
    mean_g = G.mean(dim=0)
    norm_mean_sq = float((mean_g**2).sum())
    norms_sq = (G**2).sum(dim=1)
    mean_norm_sq = float(norms_sq.mean())
    coherence = norm_mean_sq / mean_norm_sq if mean_norm_sq > 0 else float("nan")
    return {
        "coherence": coherence,
        "coherence_x_n": coherence * n_examples,
        "random_baseline": 1.0 / n_examples,
        "n_examples": n_examples,
        "norm_mean_sq": norm_mean_sq,
        "mean_norm_sq": mean_norm_sq,
    }


def z_shuffle_gap_full(model, loader, n_batches: int, device: str, tokenizer,
                       sep_token_id: int, z_positions: List[int]) -> dict:
    """Difference in loss between clean batch and z-shuffled batch.

    Implementation: run n_batches.  For each, compute loss on clean batch,
    then create a copy with z tokens at z_positions permuted ACROSS examples
    in the batch (so each example gets some other example's z), compute loss.
    Return mean of (shuffled - clean).
    """
    clean_losses = []
    shuffled_losses = []
    it = iter(loader)
    rng = random.Random(123)
    for _ in range(n_batches):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        with torch.no_grad():
            loss_c, _, _ = compute_loss(model, batch)
            clean_losses.append(float(loss_c.item()))

            # Shuffle z positions across batch
            ids = batch["input_ids"].clone()
            bsz = ids.shape[0]
            perm = list(range(bsz))
            rng.shuffle(perm)
            for pos in z_positions:
                ids[:, pos] = batch["input_ids"][perm, pos]
            shuffled_batch = {k: (ids if k == "input_ids" else v)
                              for k, v in batch.items()}
            loss_s, _, _ = compute_loss(model, shuffled_batch)
            shuffled_losses.append(float(loss_s.item()))
    return {
        "clean_mean": float(np.mean(clean_losses)),
        "shuffled_mean": float(np.mean(shuffled_losses)),
        "z_shuffle_gap": float(np.mean(shuffled_losses) - np.mean(clean_losses)),
        "n_batches": n_batches,
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def find_or_train_checkpoints(target_steps: List[int], cell_seed: int = 0):
    """Look for checkpoints near target_steps in v_tracking.  If any are
    missing, retrain to capture them.  Returns dict step → state_dict path."""
    v_dir = RESULTS_DIR / "v_tracking" / f"eta_0.001_K_10_seed_{cell_seed}" / "checkpoints"
    available = {}
    if v_dir.exists():
        for p in v_dir.glob("model_step_*.pt"):
            step = int(p.stem.split("_")[-1])
            available[step] = p
    print(f"Available v_tracking checkpoints: {sorted(available.keys())}")

    # Find closest available step to each target
    found = {}
    missing = []
    for t in target_steps:
        if not available:
            missing.append(t)
            continue
        nearest = min(available.keys(), key=lambda s: abs(s - t))
        if abs(nearest - t) <= 200:
            found[t] = (nearest, available[nearest])
        else:
            missing.append(t)
    print(f"Found nearest checkpoints: {found}")
    print(f"Missing: {missing}")

    # If anything missing, retrain to capture
    if missing:
        print(f"Retraining seed={cell_seed} to capture missing checkpoints {missing}")
        from eta_sweep.run_single import run_single
        from copy import deepcopy
        max_target = max(target_steps) + 100
        cc = CellConfig(eta=0.001, k=10, seed=cell_seed)
        # Override checkpoint cadence to capture targets
        # The run_single weight_checkpoint_every is 2000.  We override here by
        # creating a custom CellConfig with a smaller cadence that hits our targets.
        cc_dict = asdict(cc)
        cc_dict["weight_checkpoint_every"] = 250
        class _CC(CellConfig):
            @property
            def max_steps(self_inner):
                return max_target
        cc = _CC(**cc_dict)
        run_single(cc, output_subdir="saddle_probe_ckpts")
        # Now look in the new dir
        new_dir = RESULTS_DIR / "saddle_probe_ckpts" / f"eta_0.001_K_10_seed_{cell_seed}" / "checkpoints"
        for p in new_dir.glob("model_step_*.pt"):
            step = int(p.stem.split("_")[-1])
            available[step] = p
        for t in missing:
            nearest = min(available.keys(), key=lambda s: abs(s - t))
            found[t] = (nearest, available[nearest])

    return found


def probe_at_checkpoint(model, ckpt_path: Path, loader, batch_size: int, device: str,
                        target_step: int, actual_step: int) -> dict:
    """Run all 6 measurements at this checkpoint."""
    print(f"\n{'='*60}")
    print(f"=== Probing checkpoint at step {actual_step} (target {target_step}) ===")
    print(f"{'='*60}")
    sd = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(sd)
    model.train()  # use train mode for grad computations
    model.zero_grad(set_to_none=True)

    out = {"target_step": target_step, "actual_step": actual_step,
           "ckpt_path": str(ckpt_path)}

    # M1: full gradient norm
    print(f"\n[M1] Full gradient norm (avg 16 batches × 128)...")
    t0 = time.time()
    try:
        full_grad = compute_full_grad(model, loader, n_batches=16, device=device)
        grad_norm = float(full_grad.norm().item())
        out["M1_grad_norm"] = grad_norm
        print(f"   ‖∇L‖ = {grad_norm:.6f}  (interp: {'NEAR-CRITICAL' if grad_norm < 0.01 else 'NOT critical' if grad_norm > 0.1 else 'AMBIGUOUS'})")
    except Exception as e:
        out["M1_error"] = f"{type(e).__name__}: {e}"
        print(f"   M1 FAILED: {e}")
        traceback.print_exc()
        full_grad = None
    print(f"   wall: {time.time() - t0:.1f}s")

    # M2: Lanczos for smallest eigenvalue
    print(f"\n[M2] Hessian λ_min via Lanczos...")
    t0 = time.time()
    try:
        npar = n_params(model)
        loader_factory = lambda: iter(loader)
        # Use 4 batches per HVP for stability and speed (vs 16 for grad)
        lambda_min, v_min, lanczos_info = lanczos_smallest(
            model, loader_factory, n_hvp_batches=2, device=device,
            n_params=npar, k=1, ncv=40, tol=1e-4, max_iter=120,
        )
        out["M2_lambda_min"] = lambda_min
        out["M2_lanczos_info"] = lanczos_info
        print(f"   λ_min = {lambda_min:.6f}  (interp: {'NEGATIVE — saddle direction exists' if lambda_min < 0 else 'POSITIVE — local minimum'})")
        print(f"   HVP calls: {lanczos_info['n_hvp_calls']}")
    except Exception as e:
        out["M2_error"] = f"{type(e).__name__}: {e}"
        print(f"   M2 FAILED: {e}")
        traceback.print_exc()
        v_min = None
        lambda_min = None
    print(f"   wall: {time.time() - t0:.1f}s")

    # M3: cosine of grad with v_min
    if full_grad is not None and v_min is not None:
        print(f"\n[M3] cos(∇L, v_min)...")
        try:
            g = full_grad.detach().cpu().to(torch.float32)
            v = torch.from_numpy(v_min).to(torch.float32)
            cosine = float(((g * v).sum() / (g.norm() * v.norm() + 1e-12)).item())
            angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
            out["M3_cos_grad_vmin"] = cosine
            out["M3_angle_deg"] = angle_deg
            print(f"   cos = {cosine:.6f},  angle = {angle_deg:.2f}°")
            print(f"   (interp: {'PERPENDICULAR — saddle escape direction not active' if abs(cosine) < 0.1 else 'NOT perpendicular'})")
        except Exception as e:
            out["M3_error"] = f"{type(e).__name__}: {e}"
            print(f"   M3 FAILED: {e}")
    else:
        out["M3_error"] = "skipped: M1 or M2 failed"

    # M6: |grad · v_min|
    if full_grad is not None and v_min is not None:
        try:
            g = full_grad.detach().cpu().to(torch.float32)
            v = torch.from_numpy(v_min).to(torch.float32)
            proj = float(abs((g * v).sum()).item())
            out["M6_grad_proj_vmin"] = proj
            print(f"\n[M6] |∇L · v_min| = {proj:.6f}")
        except Exception as e:
            out["M6_error"] = f"{type(e).__name__}: {e}"

    # M4: per-example gradient coherence
    print(f"\n[M4] Per-example gradient coherence (n=500)...")
    t0 = time.time()
    try:
        coh_info = per_example_gradient_coherence(model, loader, n_examples=500, device=device)
        out["M4"] = coh_info
        print(f"   coherence = {coh_info['coherence']:.6f}")
        print(f"   coherence × n = {coh_info['coherence_x_n']:.4f}  "
              f"(random baseline = 1.0; predicted plateau = 1.2–1.6)")
    except Exception as e:
        out["M4_error"] = f"{type(e).__name__}: {e}"
        print(f"   M4 FAILED: {e}")
        traceback.print_exc()
    print(f"   wall: {time.time() - t0:.1f}s")

    # M5: z_shuffle_gap
    print(f"\n[M5] z_shuffle_gap...")
    t0 = time.time()
    try:
        # z is at positions 8, 9 in the input (per the README / task layout)
        gap_info = z_shuffle_gap_full(model, loader, n_batches=8, device=device,
                                       tokenizer=None, sep_token_id=None,
                                       z_positions=[8, 9])
        out["M5"] = gap_info
        gap = gap_info["z_shuffle_gap"]
        print(f"   z_shuffle_gap = {gap:.6f} nats")
        print(f"   (clean = {gap_info['clean_mean']:.4f}, shuffled = {gap_info['shuffled_mean']:.4f})")
    except Exception as e:
        out["M5_error"] = f"{type(e).__name__}: {e}"
        print(f"   M5 FAILED: {e}")
        traceback.print_exc()
    print(f"   wall: {time.time() - t0:.1f}s")

    return out


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-suffix", type=str, default="",
                   help="suffix added to output JSON filename for multi-seed")
    args = p.parse_args()
    seed = args.seed
    target_steps = [750, 1500, 2250]

    # Set up data + model from canonical config
    cc = CellConfig(eta=0.001, k=10, seed=seed)
    _set_all_seeds(cc.seed)
    device = _select_device()
    print(f"device: {device}")

    cfg = build_legacy_cfg(cc)
    tokenizer = create_tokenizer_from_config(cfg)
    train_dataset, _, mapping_data = create_datasets_from_config(cfg, tokenizer)
    loader = DataLoader(train_dataset, batch_size=cc.batch_size, shuffle=True,
                        collate_fn=collate_fn, num_workers=0)
    model = create_model_from_config(cfg, tokenizer)
    model = model.to(device)

    # Locate or capture checkpoints
    found = find_or_train_checkpoints(target_steps, cell_seed=seed)
    print(f"\nUsing checkpoints: {[(t, n) for t, (n, _) in found.items()]}")

    results = {
        "config": {"K": 10, "eta": 1e-3, "seed": seed,
                   "batch_size": cc.batch_size,
                   "n_params": n_params(model), "device": device},
        "target_steps": target_steps,
        "checkpoints": [],
    }
    t_run = time.time()
    for target_step in target_steps:
        actual_step, ckpt_path = found[target_step]
        ckpt_result = probe_at_checkpoint(model, ckpt_path, loader,
                                           cc.batch_size, device,
                                           target_step, actual_step)
        results["checkpoints"].append(ckpt_result)
        # Save incrementally
        suffix = args.out_suffix or f"_seed{seed}"
    (OUT_DIR / f"saddle_probe_results{suffix}.json").write_text(json.dumps(results, indent=2))

    results["total_wall_s"] = time.time() - t_run
    suffix = args.out_suffix or f"_seed{seed}"
    (OUT_DIR / f"saddle_probe_results{suffix}.json").write_text(json.dumps(results, indent=2))
    print(f"\n=== Total wall: {results['total_wall_s']:.1f}s ===")
    print(f"Saved: {OUT_DIR / f'saddle_probe_results{suffix}.json'}")

    # Print summary table
    print("\n" + "="*100)
    print("SUMMARY TABLE")
    print("="*100)
    print(f"{'metric':<25} {'step ' + str(target_steps[0]):<22} {'step ' + str(target_steps[1]):<22} {'step ' + str(target_steps[2]):<22}")
    print("-"*100)
    for metric, fmt, key in [
        ("M1: ‖∇L‖", "{:.4f}", "M1_grad_norm"),
        ("M2: λ_min", "{:.4f}", "M2_lambda_min"),
        ("M3: cos(∇L,v_min)", "{:.4f}", "M3_cos_grad_vmin"),
        ("M3: angle (deg)", "{:.2f}", "M3_angle_deg"),
        ("M4: coherence × n", "{:.4f}", None),
        ("M5: z_shuffle_gap", "{:.4f}", None),
        ("M6: |∇L·v_min|", "{:.4f}", "M6_grad_proj_vmin"),
    ]:
        cells = []
        for c in results["checkpoints"]:
            if metric == "M4: coherence × n":
                v = c.get("M4", {}).get("coherence_x_n")
            elif metric == "M5: z_shuffle_gap":
                v = c.get("M5", {}).get("z_shuffle_gap")
            else:
                v = c.get(key)
            if v is None:
                cells.append("FAILED")
            else:
                cells.append(fmt.format(v))
        print(f"{metric:<25} {cells[0]:<22} {cells[1]:<22} {cells[2]:<22}")
    print("="*100)


if __name__ == "__main__":
    main()
