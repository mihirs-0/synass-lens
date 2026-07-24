#!/usr/bin/env python
"""
Phase D — Trap or bystander: surgically degrade (or strengthen) the
constant-output machinery at a mid-plateau checkpoint, resume training,
measure escape time vs matched controls.

Arms (each resumed with >= 3 resume-seeds; mappings pinned to seed 42, the
resume seed controls torch/data-order only):
  control   : untouched weights
  targeted  : zero MLP W_out rows of the top-N neurons by |pure relevance|
              of the plain correct-logit metric at the READOUT position
              (the constant-output machinery located in run 2)
  random    : zero W_out rows of N random neurons (damage-matched control)
  reverse   : 300 steps of prior-only training (labels resampled from the
              GLOBAL answer pool -> only q* is learnable), then normal resume

Sanity gate (stop condition c): after surgery the model must still be a
degraded CONSTANT machine, not a re-randomized one — KL(model||q*) rises,
input-sensitivity stays ~ 0, CE stays below log|V|. Dose-response over N
is printed; the chosen N must pass the gate.

Escape time: first step (resume-relative) where first-target full-vocab CE
crosses 0.5 * C-shelf. Read from each arm's training history.

Usage:
  python scripts/phantom_d.py --plateau-step 600 --n-surgery 64 \
      --resume-seeds 101 202 303 --max-steps 6000
"""

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf
from src.data import create_tokenizer_from_config
from src.data.dataset import MappingData, generate_mappings
from scripts.experiment_helpers import make_config, run_single_experiment
from scripts.relp_core import (
    load_model, select_device, run_relp, READ_POS,
)
from scripts.relp2_core import (
    build_pair_batch, to_device, regime_metrics, register_regime_metrics,
    position_distributions, mean_pairwise_jsd_grid, empirical_shelves,
)

OUT = Path("results/phantom")


def constant_diagnostics(model, ids, pb, q1, device):
    """KL(model||q*) at pos 1, JSD_B sensitivity, full CE pos 1."""
    with torch.no_grad():
        logits = model(ids)
        rm = regime_metrics(logits, pb)
        d = position_distributions(logits)[:, 0, :]
    qs = q1.unsqueeze(0).expand_as(d) + 1e-6
    qs = qs / qs.sum(-1, keepdim=True)
    kl = float((d * (torch.log(d + 1e-12) - torch.log(qs))).sum(-1).mean())
    jsd_b = mean_pairwise_jsd_grid(d.cpu(), pb.n_b, pb.k, "B", max_pairs=1500)
    return {"kl_qstar": kl, "jsd_B": jsd_b, "ce_pos1": float(rm["ce_pos1"].mean())}


def find_constant_machinery(model, ids, pb, n_layers, d_mlp):
    """Rank (layer, neuron) by |grad x act| of m_plain at the readout position."""
    acts, grads, _ = run_relp(model, ids, pb, "m_plain", linearize=True)
    score = torch.zeros(n_layers, d_mlp)
    for l in range(n_layers):
        rel = (grads[f"mlp_post_{l}"] * acts[f"mlp_post_{l}"])[:, READ_POS, :]
        score[l] = rel.abs().mean(dim=0).cpu()
    return score


def zero_neurons(model, pairs):
    with torch.no_grad():
        for (l, n) in pairs:
            model.blocks[l].mlp.W_out.data[n, :] = 0.0


def constant_logit_delta(model, ids):
    """The learned order-0 solution, output-side: mean logits at the readout
    position, centered. Subtracting this from b_U flattens the constant."""
    with torch.no_grad():
        logits = model(ids)
    c = logits[:, READ_POS, :].mean(dim=0)
    return c - c.mean()


def bias_surgery(model, delta):
    with torch.no_grad():
        model.b_U.data -= delta.to(model.b_U.device, model.b_U.dtype)


def prior_only_mapping(md, seed=7):
    """Labels resampled from the GLOBAL answer pool: only q* is learnable."""
    rng = random.Random(seed)
    all_a = [a for pairs in md.mappings.values() for (_z, a) in pairs]
    examples = [{"b": ex["b"], "z": ex["z"], "a": rng.choice(all_a)}
                for ex in md.examples]
    return MappingData(mappings=md.mappings, examples=examples,
                       n_unique_b=md.n_unique_b, n_unique_a=md.n_unique_a,
                       k=md.k, task=md.task)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plateau-step", type=int, default=600)
    ap.add_argument("--n-surgery", type=int, default=64)
    ap.add_argument("--dose-response", type=int, nargs="*", default=[16, 32, 64, 128])
    ap.add_argument("--resume-seeds", type=int, nargs="+", default=[101, 202, 303])
    ap.add_argument("--max-steps", type=int, default=6000)
    ap.add_argument("--arms", nargs="+",
                    default=["control", "targeted", "random", "reverse"])
    ap.add_argument("--dose-only", action="store_true")
    args = ap.parse_args()

    register_regime_metrics()
    device = select_device()
    base_cfg = OmegaConf.load("outputs/landauer_dense_k10/config.yaml")
    tok = create_tokenizer_from_config(base_cfg)
    pb = build_pair_batch(base_cfg, tok, n_b_eval=64, seed=1234)
    ids, pb = to_device(pb, device)
    shelves = empirical_shelves(base_cfg, tok)
    q1 = torch.tensor(shelves["q_star"][1], device=device)
    c_shelf = shelves["H_qstar"][1]
    n_layers, d_mlp = 4, 512
    ckpt_dir = Path("outputs/landauer_dense_k10/checkpoints")

    md42 = generate_mappings(
        n_unique_b=1000, k=10, b_length=6, a_length=4, z_length=2,
        vocab_chars=base_cfg.data.vocab_chars, seed=42, task="bz_to_a",
        enforce_unique_a_first_char_per_b=True, disambiguation_prefix_length=1)

    # ---- locate machinery + dose response at the plateau checkpoint ----
    model0 = load_model(base_cfg, tok, ckpt_dir, args.plateau_step, device)
    model0.requires_grad_(True)
    diag0 = constant_diagnostics(model0, ids, pb, q1, device)
    print(f"pre-surgery @ step {args.plateau_step}: {diag0} (C shelf {c_shelf:.3f}, "
          f"log|V| {shelves['log_vocab']:.3f})", flush=True)
    score = find_constant_machinery(model0, ids, pb, n_layers, d_mlp)
    order = torch.argsort(score.flatten(), descending=True)

    dose = {}
    for N in args.dose_response:
        m = load_model(base_cfg, tok, ckpt_dir, args.plateau_step, device)
        pairs = [(int(i) // d_mlp, int(i) % d_mlp) for i in order[:N]]
        zero_neurons(m, pairs)
        dose[f"neurons_{N}"] = constant_diagnostics(m, ids, pb, q1, device)
        print(f"  targeted neurons N={N}: {dose[f'neurons_{N}']}", flush=True)
        del m
    # neuron surgery cannot reach the constant (distributed) — the effective
    # order-0 solution is output-side: special-token suppression + slight char
    # skew in the readout logit geometry. Exact undo = bias surgery.
    delta = constant_logit_delta(model0, ids)
    m = load_model(base_cfg, tok, ckpt_dir, args.plateau_step, device)
    bias_surgery(m, delta)
    dose["bias_targeted"] = constant_diagnostics(m, ids, pb, q1, device)
    print(f"  bias surgery (||delta||={float(delta.norm()):.2f}): "
          f"{dose['bias_targeted']}", flush=True)
    del m
    g = torch.Generator().manual_seed(0)
    rvec = torch.randn(delta.shape, generator=g).to(delta.device)
    rvec = rvec - rvec.mean()
    rvec = rvec * (delta.norm() / rvec.norm())
    m = load_model(base_cfg, tok, ckpt_dir, args.plateau_step, device)
    bias_surgery(m, rvec)
    dose["bias_random"] = constant_diagnostics(m, ids, pb, q1, device)
    print(f"  bias random (norm-matched): {dose['bias_random']}", flush=True)
    del m, model0

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "d_dose_response.json", "w") as f:
        json.dump({"pre": diag0, "dose": {str(k): v for k, v in dose.items()},
                   "plateau_step": args.plateau_step,
                   "C_shelf": c_shelf, "log_vocab": shelves["log_vocab"]}, f, indent=1)
    if args.dose_only:
        return

    # sanity gate: bias surgery must undo the order-0 solution specifically —
    # KL to q* rises, sensitivity stays ~0, CE back near init (log|V|), i.e.
    # a degraded CONSTANT machine, not a re-randomized model
    chosen = dose["bias_targeted"]
    gate = (chosen["kl_qstar"] > diag0["kl_qstar"] + 0.3
            and chosen["jsd_B"] < 0.05
            and chosen["ce_pos1"] < shelves["log_vocab"] + 0.25)
    print(f"sanity gate (bias surgery): {'PASS' if gate else 'FAIL'}", flush=True)
    if not gate:
        print("stop condition (c): fix surgery before collecting timings")
        return

    # ---- resume arms ----
    timings = []
    for seed in args.resume_seeds:
        for arm in args.arms:
            name = f"phantom_d_{arm}_s{seed}"
            m = load_model(base_cfg, tok, ckpt_dir, args.plateau_step, device)
            if arm == "targeted":
                bias_surgery(m, delta)
            elif arm == "random":
                gr = torch.Generator().manual_seed(seed)
                rv = torch.randn(delta.shape, generator=gr).to(delta.device)
                rv = rv - rv.mean()
                rv = rv * (delta.norm() / rv.norm())
                bias_surgery(m, rv)
            elif arm == "reverse":
                cfg_r = make_config(name + "_pre", k=10, seed=seed,
                                    max_steps=300, checkpoint_every=100000,
                                    eval_every=100)
                run_single_experiment(cfg_r, mapping_data=prior_only_mapping(md42),
                                      output_dir="outputs", model=m)
                m.train()
            cfg = make_config(name, k=10, seed=seed, max_steps=args.max_steps,
                              checkpoint_every=1000, eval_every=50)
            run_single_experiment(cfg, mapping_data=md42, output_dir="outputs", model=m)
            hist = json.load(open(Path("outputs") / name / "training_history.json"))
            steps = np.array(hist["steps"]); ftl = np.array(hist["first_target_loss"])
            half = 0.5 * c_shelf
            below = np.where(ftl < half)[0]
            t_escape = int(steps[below[0]]) if len(below) else None
            timings.append({"arm": arm, "seed": seed, "t_escape_half_shelf": t_escape})
            print(f"[{name}] escape(half-shelf) = {t_escape}", flush=True)
            del m
            if device == "mps":
                torch.mps.empty_cache()
            with open(OUT / "d_timings.json", "w") as f:
                json.dump(timings, f, indent=1)
    print("done.")


if __name__ == "__main__":
    main()
