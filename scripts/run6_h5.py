#!/usr/bin/env python
"""
Run 6 / H5 — Basin bookkeeping.
(a) does the constant machine reconstitute rapidly after bias surgery?
    600-step resumes (control vs targeted), checkpoints every 100,
    KL(output || q*) per checkpoint.
(b) Adam moments: design fact — all phantom_d arms resumed with FRESH
    optimizers; no arm preserved moments. Recorded, no experiment.
(c) bias surgery edits b_U only (downstream of every LN): residual stream
    is untouched by construction; verified numerically.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import create_tokenizer_from_config
from src.data.dataset import generate_mappings
from scripts.experiment_helpers import make_config, run_single_experiment
from scripts.phantom_d import (constant_diagnostics, constant_logit_delta,
                               bias_surgery)
from scripts.relp_core import load_model, select_device
from scripts.relp2_core import build_pair_batch, to_device

OUT = Path("results/adversarial")
PLATEAU = 600


def main():
    device = select_device()
    cfg = make_config("h5", k=10, max_steps=50000)
    tok = create_tokenizer_from_config(cfg)
    md = generate_mappings(
        n_unique_b=1000, k=10, b_length=6, a_length=4, z_length=2,
        vocab_chars=cfg.data.vocab_chars, seed=42, task="bz_to_a",
        enforce_unique_a_first_char_per_b=True, disambiguation_prefix_length=1)
    pb = build_pair_batch(cfg, tok, n_b_eval=128, seed=1234)
    ids, pb = to_device(pb, device)
    q1 = torch.zeros(len(tok.token_to_id), device=device)
    for ex in md.examples:
        q1[tok.token_to_id[ex["a"][0]]] += 1
    q1 = q1 / q1.sum()

    ckdir = Path("outputs") / "landauer_dense_k10" / "checkpoints"
    base = load_model(cfg, tok, ckdir, PLATEAU, device)
    delta = constant_logit_delta(base, ids)

    # (c) residual-stream invariance under the surgery
    with torch.no_grad():
        _, c0 = base.run_with_cache(ids[:256], return_type=None,
                                    names_filter=lambda n:
                                    n.endswith("hook_resid_post"))
        pre = {l: c0["resid_post", l][:, 10, :].norm(dim=1).mean().item()
               for l in range(4)}
    bias_surgery(base, delta)
    with torch.no_grad():
        _, c1 = base.run_with_cache(ids[:256], return_type=None,
                                    names_filter=lambda n:
                                    n.endswith("hook_resid_post"))
        post = {l: c1["resid_post", l][:, 10, :].norm(dim=1).mean().item()
                for l in range(4)}
    norms = {f"L{l}": {"pre": pre[l], "post": post[l],
                       "ratio": post[l] / pre[l]} for l in range(4)}
    del base, c0, c1

    # (a) 600-step resumes
    traces = {}
    for arm in ("control", "targeted"):
        m = load_model(cfg, tok, ckdir, PLATEAU, device)
        if arm == "targeted":
            bias_surgery(m, delta)
        traces[arm] = {"step0": constant_diagnostics(m, ids, pb, q1, device)}
        rcfg = make_config(f"h5_{arm}", k=10, seed=101, max_steps=600,
                           checkpoint_every=100, eval_every=50)
        run_single_experiment(rcfg, mapping_data=md, output_dir="outputs",
                              model=m)
        del m
        for step in range(100, 700, 100):
            mm = load_model(cfg, tok, Path("outputs") / f"h5_{arm}" /
                            "checkpoints", step, device)
            traces[arm][f"step{step}"] = constant_diagnostics(
                mm, ids, pb, q1, device)
            del mm
        if device == "mps":
            torch.mps.empty_cache()
        print(f"[{arm}] " + " ".join(
            f'{k}:KL={v["kl_qstar"]:.3f}' for k, v in traces[arm].items()),
            flush=True)

    instant = all(
        traces["targeted"][f"step{s}"]["kl_qstar"] <=
        2 * traces["control"][f"step{s}"]["kl_qstar"]
        for s in (200,))
    res = {"a_reconstitution": traces,
           "a_instant_repair_by_200": bool(instant),
           "b_adam_fact": "all phantom_d arms resumed with fresh AdamW "
                          "optimizers (run_single_experiment constructs a new "
                          "optimizer); moments were never preserved in any "
                          "arm; a surgery+moment-reset arm is identical to "
                          "the existing targeted arm",
           "c_resid_norms": norms,
           "c_flag_renormalize": bool(any(
               not 0.8 <= v["ratio"] <= 1.25 for v in norms.values()))}
    with open(OUT / "h5_basin.json", "w") as f:
        json.dump(res, f, indent=1)
    print("H5:", json.dumps({k: res[k] for k in
                             ("a_instant_repair_by_200",
                              "c_flag_renormalize")}))


if __name__ == "__main__":
    main()
