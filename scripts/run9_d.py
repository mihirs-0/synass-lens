#!/usr/bin/env python
"""
Run 9 / Experiment D — the prospective demo.
  train : K=34 standard init seed 42, max 60000, states every 250
  watch : online detector (frozen in predictions_run9.json[predictions_D_frozen]):
          S(t) = -cos(g, theta_t - theta_{t-500}), fire on 3 consecutive
          samples < -0.015; writes prospective_call.json AT FIRE TIME.
          Pass check is steps-based: the call depends only on states up to
          t_fire+500, so 'call before escape' <=> escape > t_fire + 500.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import create_tokenizer_from_config
from src.data.dataset import DisambiguationDataset, generate_mappings
from src.model import create_model_from_config
from scripts.experiment_helpers import make_config
from scripts.run8_substrate import STATES, run_train
from scripts.run8_lib import StateSaver, load_state, flat_from_state
from scripts.run8_candidates import grad_at

OUT = Path("results/production")
SDIR = STATES / "prod_k34"
HALF = 1.7908


def k40_cfg_md():
    cfg = make_config("prod_k34", k=34, seed=42, max_steps=60000,
                      checkpoint_every=60000, eval_every=25)
    tok = create_tokenizer_from_config(cfg)
    md = generate_mappings(
        n_unique_b=1000, k=34, b_length=6, a_length=4, z_length=2,
        vocab_chars=cfg.data.vocab_chars, seed=42, task="bz_to_a",
        enforce_unique_a_first_char_per_b=True, disambiguation_prefix_length=1)
    return cfg, tok, md


def cmd_train():
    cfg, tok, md = k40_cfg_md()
    torch.manual_seed(42)
    model = create_model_from_config(cfg, tok)
    run_train(cfg, model, md, StateSaver(SDIR, 250))
    h = json.load(open("outputs/prod_k34/training_history.json"))
    s = np.array(h["steps"]); ftl = np.array(h["first_target_loss"])
    below = np.where(ftl < HALF)[0]
    print("[k40] escape =", int(s[below[0]]) if len(below) else None)


def cmd_watch():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg, tok, md = k40_cfg_md()
    ref = create_model_from_config(cfg, tok)
    ds = DisambiguationDataset(mapping_data=md, tokenizer=tok, split="train",
                               probe_fraction=0.0, seed=42, task="bz_to_a")
    rng = np.random.RandomState(5)
    ids = torch.stack([ds[i]["input_ids"] for i in
                       rng.choice(len(ds), 2048, False)]).to(device)
    seen, series, consec, fired = set(), [], 0, False
    log = OUT / "d_watch_log.json"
    idle = 0
    while idle < 60:                      # ~30 min of no new states -> stop
        files = {int(p.stem.split("_")[1]): p for p in
                 SDIR.glob("state_*.pt")} if SDIR.exists() else {}
        new = sorted(s for s in files if s not in seen and s >= 750
                     and (s - 500) in files)
        if not new:
            idle += 1
            time.sleep(30)
            continue
        idle = 0
        for t in new:
            seen.add(t)
            try:
                st = load_state(files[t])
                th = flat_from_state(st["model"], ref)
                th0 = flat_from_state(load_state(files[t - 500])["model"],
                                      ref)
            except Exception:
                seen.discard(t)          # partial write; retry next poll
                break
            d = th - th0
            d = d / (d.norm() + 1e-12)
            ref.load_state_dict(st["model"])
            g = grad_at(ref.to(device), ids, device)
            ref.cpu()
            S = float(-(g @ d) / (g.norm() + 1e-12))
            series.append({"step": t, "S": S})
            with open(log, "w") as f:
                json.dump(series, f)
            consec = consec + 1 if S < -0.015 else 0
            print(f"[watch] {t}: S={S:+.4f} consec={consec}", flush=True)
            if consec >= 3 and not fired:
                fired = True
                call = {"call_written_at": datetime.now().astimezone()
                        .isoformat(timespec="seconds"),
                        "t_fire": t,
                        "last_state_used": t,
                        "predicted_escape_window": [t, int(1.35 * t)],
                        "pass_requires": f"escape > {t} (call causally "
                                         f"prior) and escape <= {int(1.35*t)}"}
                with open(OUT / "prospective_call.json", "w") as f:
                    json.dump(call, f, indent=1)
                print(f"[watch] *** CALL WRITTEN: fire at {t}, window "
                      f"[{t}, {int(1.35*t)}] ***", flush=True)
    print("[watch] done (idle timeout or training ended)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "watch"])
    a = ap.parse_args()
    (cmd_train if a.mode == "train" else cmd_watch)()
