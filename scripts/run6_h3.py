#!/usr/bin/env python
"""
Run 6 / H3 — Gradient starvation, measured.
Attack: "'the signal arrives' is a metaphor." Per checkpoint on 8e (XOR,
parked) and 8f (easy control): cosine between the training-loss gradient and
the gradient of the z-sensitivity metric m_z, in parameter space.
Thresholds: predictions_run6.json (H3).
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import create_tokenizer_from_config
from src.data.dataset import DisambiguationDataset
from scripts.relp_core import load_model, select_device
from scripts.relp2_core import build_pair_batch, to_device, regime_metrics
from scripts.relp2_e12 import get_design

OUT = Path("results/adversarial")
STEPS = {"8f_easy": [100, 300, 600, 1000, 1500, 2000, 3000, 4000, 6000,
                     10000, 15000],
         "8e_xor": [100, 300, 600, 1000, 2000, 4000, 8000, 12000, 16000,
                    20000, 24200]}


def flat_grad(model):
    return torch.cat([p.grad.flatten() for p in model.parameters()
                      if p.grad is not None])


def main():
    device = select_device()
    res = {}
    for name in ("8f_easy", "8e_xor"):
        cfg, md = get_design(name)
        tok = create_tokenizer_from_config(cfg)
        # fixed training batch (2048 examples, seed 5)
        ds = DisambiguationDataset(mapping_data=md, tokenizer=tok,
                                   split="train", probe_fraction=0.0,
                                   seed=42, task="bz_to_a")
        rng = np.random.RandomState(5)
        idx = rng.choice(len(ds), 2048, replace=False)
        ids_tr = torch.stack([ds[i]["input_ids"] for i in idx]).to(device)
        # eval grid for m_z (n_b=32 keeps the graph small)
        pb = build_pair_batch(cfg, tok, n_b_eval=32, seed=1234,
                              mapping_data=md)
        ids_ev, pb = to_device(pb, device)
        ckdir = Path("outputs") / f"e12_{name}" / "checkpoints"
        rows = []
        for step in STEPS[name]:
            model = load_model(cfg, tok, ckdir, step, device)
            model.zero_grad()
            logits = model(ids_tr)
            ce = torch.nn.functional.cross_entropy(
                logits[:, 10:14].reshape(-1, logits.shape[-1]),
                ids_tr[:, 11:15].reshape(-1))
            ce.backward()
            g_loss = flat_grad(model).detach().clone()
            model.zero_grad()
            rm = regime_metrics(model(ids_ev), pb)
            rm["m_z_pos1"].mean().backward()
            g_mz = flat_grad(model).detach().clone()
            cos = float(torch.abs(g_loss @ g_mz) /
                        (g_loss.norm() * g_mz.norm() + 1e-12))
            rows.append({"step": step, "cos_loss_mz": cos,
                         "g_loss_norm": float(g_loss.norm()),
                         "g_mz_norm": float(g_mz.norm()),
                         "train_ce": float(ce)})
            print(f"{name} {step}: cos={cos:.4f} ce={float(ce):.3f}",
                  flush=True)
            del model, g_loss, g_mz
            if device == "mps":
                torch.mps.empty_cache()
        res[name] = rows

    f_max = max(r["cos_loss_mz"] for r in res["8f_easy"])
    early_8f = all(r["cos_loss_mz"] >= 0.02 for r in res["8f_easy"]
                   if r["step"] <= 2000)
    park_8e = all(r["cos_loss_mz"] <= 0.05 * f_max for r in res["8e_xor"]
                  if r["step"] >= 1000)
    res["verdict"] = {"max_8f": f_max, "pass_8f_early": bool(early_8f),
                      "pass_8e_starved": bool(park_8e),
                      "pass": bool(early_8f and park_8e)}
    with open(OUT / "h3_starvation.json", "w") as f:
        json.dump(res, f, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, color in (("8f_easy", "tab:green"), ("8e_xor", "tab:red")):
        ax.plot([r["step"] for r in res[name]],
                [r["cos_loss_mz"] for r in res[name]], "o-", color=color,
                label=f"{name}: |cos(grad loss, grad m_z)|")
    ax.set_xscale("symlog", linthresh=300)
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("selector-direction gradient signal")
    ax.set_title("H3 — gradient starvation, measured: the selector signal the\n"
                 "loss gradient carries (easy vs XOR selector)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "h3_starvation.png", dpi=150)
    print("H3 verdict:", json.dumps(res["verdict"]))


if __name__ == "__main__":
    main()
