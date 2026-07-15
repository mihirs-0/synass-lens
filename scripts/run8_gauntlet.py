#!/usr/bin/env python
"""
Run 8 / Experiment 3 — the gauntlet. Predictions frozen in
predictions_run8.json[predictions_gauntlet] BEFORE these runs/analyses.

  ga --inject N : continuous-optimizer dose-600 at step N of substrate (a),
                  resume to escape (mirror of substrate c)
  gb            : kink seed-divergence of |C2| vs output statistics
  gc            : 8b cross-condition threshold test
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import create_tokenizer_from_config
from src.data.dataset import DisambiguationDataset
from src.model import create_model_from_config
from scripts.experiment_helpers import make_config
from scripts.run8_substrate import md42, STATES, run_train
from scripts.run8_lib import (load_state, StateRestorer, StateSaver,
                              flat_from_state)
from scripts.phantom_d import prior_only_mapping
from scripts.run8_candidates import grad_at, state_files
from src.training.checkpoint import load_checkpoint, list_checkpoints

OUT = Path("results/clock")
HALF = 1.7908


def cmd_ga(inject):
    st = load_state(STATES / "a" / f"state_{inject:06d}.pt")
    name = f"clock_ga_{inject}"
    cfg0 = make_config(name + "_dose", k=10, seed=42, max_steps=600,
                       checkpoint_every=600, eval_every=50)
    tok = create_tokenizer_from_config(cfg0)
    model = create_model_from_config(cfg0, tok)
    model.load_state_dict(st["model"])
    base = md42(cfg0)
    run_train(cfg0, model, prior_only_mapping(base, seed=7),
              StateSaver(STATES / f"ga{inject}_dose", 600, also_at=(600,)),
              StateRestorer(st["opt"]))
    last = load_state(STATES / f"ga{inject}_dose" / "state_000600.pt")
    cfg1 = make_config(name, k=10, seed=42, max_steps=6000,
                       checkpoint_every=6000, eval_every=25)
    run_train(cfg1, model, base, None, StateRestorer(last["opt"]))
    h = json.load(open(f"outputs/{name}/training_history.json"))
    s = np.array(h["steps"]); ftl = np.array(h["first_target_loss"])
    below = np.where(ftl < HALF)[0]
    esc = int(s[below[0]]) if len(below) else None
    abs_esc = inject + 600 + esc if esc else None
    delay = abs_esc - 2075 if abs_esc else None
    with open(OUT / f"ga_{inject}.json", "w") as f:
        json.dump({"inject": inject, "resume_escape": esc,
                   "abs_escape": abs_esc, "delay_vs_2075": delay}, f)
    print(f"[ga {inject}] abs_escape={abs_esc} delay={delay}")


def c2_series(states_dir, hist_name, step_stride, device, batch_ids, ref):
    sf = state_files(Path(states_dir))
    h = json.load(open(f"outputs/{hist_name}/training_history.json"))
    s = np.array(h["steps"]); ftl = np.array(h["first_target_loss"])
    below = np.where(ftl < HALF)[0]
    esc = int(s[below[0]]) if len(below) else None
    grid = sorted(sf)
    pre = min(grid, key=lambda g: abs(g - (esc - 250)))
    post = min(grid, key=lambda g: abs(g - (esc + 500)))
    u = flat_from_state(load_state(sf[post])["model"], ref) - \
        flat_from_state(load_state(sf[pre])["model"], ref)
    u = u / u.norm()
    rows = []
    for step in grid:
        if step % step_stride or step > esc + 500:
            continue
        st = load_state(sf[step])
        ref.load_state_dict(st["model"])
        g = grad_at(ref.to(device), batch_ids, device)
        rows.append({"step": step, "absC2": float(abs((g @ u) /
                                                      (g.norm() + 1e-12)))})
        ref.cpu()
    return esc, rows


def cmd_gb():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = make_config("gb", k=10, seed=42, max_steps=12000)
    tok = create_tokenizer_from_config(cfg)
    ref = create_model_from_config(cfg, tok)
    md = md42(cfg)
    ds = DisambiguationDataset(mapping_data=md, tokenizer=tok, split="train",
                               probe_fraction=0.0, seed=42, task="bz_to_a")
    rng = np.random.RandomState(5)
    ids = torch.stack([ds[i]["input_ids"] for i in
                       rng.choice(len(ds), 2048, False)]).to(device)
    out = {}
    for seed in (42, 43):
        esc, rows = c2_series(STATES / f"b_s{seed}", f"clock_b_s{seed}",
                              200, device, ids, ref)
        out[f"s{seed}"] = {"escape": esc, "rows": rows}
        print(f"[gb s{seed}] escape={esc}, {len(rows)} pts", flush=True)
    with open(OUT / "gb_kink.json", "w") as f:
        json.dump(out, f)


def cmd_gc():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    from scripts.relp2_e8 import get_design
    cfg, md = get_design("8b_skewz")[0:2]
    tok = create_tokenizer_from_config(cfg)
    ref = create_model_from_config(cfg, tok)
    ds = DisambiguationDataset(mapping_data=md, tokenizer=tok, split="train",
                               probe_fraction=0.0, seed=42, task="bz_to_a")
    rng = np.random.RandomState(5)
    ids = torch.stack([ds[i]["input_ids"] for i in
                       rng.choice(len(ds), 2048, False)]).to(device)
    ckdir = Path("outputs/e8_8b_skewz/checkpoints")
    # u from own checkpoints: pre=400, post=1200 (grid limit documented)
    def theta(step):
        m = create_model_from_config(cfg, tok)
        load_checkpoint(m, None, ckdir, step=step)
        return torch.cat([p.detach().flatten().cpu().float()
                          for p in m.parameters()]), m
    th_pre, _ = theta(400)
    th_post, _ = theta(1200)
    u = th_post - th_pre
    u = u / u.norm()
    rows = []
    for step in range(100, 1300, 100):
        _, m = theta(step)
        g = grad_at(m.to(device), ids, device)
        rows.append({"step": step, "absC2": float(abs((g @ u) /
                                                      (g.norm() + 1e-12)))})
        print(f"[gc] {step}: |C2|={rows[-1]['absC2']:.4f}", flush=True)
    cross = next((r["step"] for r in rows if r["absC2"] >= 0.030), None)
    with open(OUT / "gc_8b.json", "w") as f:
        json.dump({"rows": rows, "crossing_0.030": cross,
                   "observed_escape": 650,
                   "pass_pm15pct": bool(cross is not None and
                                        552 <= cross <= 748)}, f)
    print(f"[gc] crossing={cross} (pass band [552,748])")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["ga", "gb", "gc"])
    ap.add_argument("--inject", type=int)
    a = ap.parse_args()
    if a.mode == "ga":
        cmd_ga(a.inject)
    elif a.mode == "gb":
        cmd_gb()
    else:
        cmd_gc()
