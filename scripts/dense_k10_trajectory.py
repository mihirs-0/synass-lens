"""Dense K=10 phase trajectory: run patching + within-fiber-swap at ~20
evenly-spaced checkpoints from the canonical landauer_dense_k10 run.

Combines with dense behavioral data from training_history.json.

Output: eta_sweep/results/dense_k10_trajectory.json
"""
from __future__ import annotations
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import yaml  # noqa: E402

from eta_sweep.analysis.mech_interp_mbc import (  # noqa: E402
    activation_patching, _select_device,
)
from src.data import (  # noqa: E402
    create_tokenizer_from_config, create_datasets_from_config,
)
from src.model import create_model_from_config  # noqa: E402
from src.analysis.candidate_eval import score_candidate_sequences  # noqa: E402
from scripts.within_fiber_swap import within_fiber_swap_eval  # noqa: E402


LANDAUER_DIR = REPO_ROOT / "outputs" / "landauer_dense_k10"


def load_yaml_cfg(yaml_path):
    raw = yaml.safe_load(open(yaml_path))

    def to_ns(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: to_ns(v) for k, v in d.items()})
        if isinstance(d, list):
            return [to_ns(v) for v in d]
        return d
    return to_ns(raw)


def make_model_at_step(cfg, tokenizer, step, device):
    model = create_model_from_config(cfg, tokenizer)
    sd_path = LANDAUER_DIR / "checkpoints" / f"step_{step:06d}" / "model.pt"
    sd = torch.load(sd_path, map_location=device)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()
    return model


def main():
    device = _select_device()
    print(f"device: {device}")

    cfg = load_yaml_cfg(LANDAUER_DIR / "config.yaml")
    tokenizer = create_tokenizer_from_config(cfg)
    train_ds, _, mapping_data = create_datasets_from_config(cfg, tokenizer)
    log_k = math.log(cfg.data.k)

    # Pick ~20 checkpoints, denser around transition (~step 1850 for K=10)
    target_steps = [
        100, 200, 400, 600, 800, 1000, 1200, 1400, 1600,
        1800, 1900, 2000, 2100, 2200, 2400, 2600, 2800,
        3200, 4000, 5000, 8000, 12000, 25000, 50000,
    ]
    available = sorted(int(p.name.split("_")[1])
                       for p in (LANDAUER_DIR / "checkpoints").glob("step_*"))
    chosen = sorted(set(min(available, key=lambda s: abs(s - t))
                        for t in target_steps))
    print(f"chosen {len(chosen)} checkpoints: {chosen}")

    # Dense behavioral from training_history.json
    th = json.load(open(LANDAUER_DIR / "training_history.json"))
    behavioral = {}
    for i, step in enumerate(th["steps"]):
        behavioral[step] = {
            "train_loss": th["train_loss"][i],
            "candidate_loss": th["candidate_loss"][i],
            "candidate_accuracy": th["candidate_accuracy"][i],
            "loss_z_shuffled": th["loss_z_shuffled"][i],
            "delta_z": (th["loss_z_shuffled"][i] - th["candidate_loss"][i]
                        if th["loss_z_shuffled"][i] is not None else None),
        }

    out = {
        "device": device, "k": cfg.data.k, "log_k": log_k,
        "checkpoints": [],
        "dense_behavioral": behavioral,
    }
    t_start = time.time()

    for step in chosen:
        t0 = time.time()
        try:
            model = make_model_at_step(cfg, tokenizer, step, device)
        except Exception as e:
            print(f"  step={step}: load error {e}")
            continue

        pat = activation_patching(model, tokenizer, mapping_data, device,
                                   n_pairs=24, seed=0)
        wfs = within_fiber_swap_eval(model, tokenizer, mapping_data,
                                      device, n_examples=64, seed=0)

        # Find nearest behavioral row
        nearest = min(behavioral.keys(), key=lambda s: abs(s - step))
        beh = behavioral.get(nearest, {})
        cand_loss = beh.get("candidate_loss")
        top1 = beh.get("candidate_accuracy")
        dz = beh.get("delta_z")

        rec = pat.get("recovery_per_layer", [None]*4) or [None]*4

        ck_out = {
            "step": step,
            "step_over_tau": None,  # filled below
            "candidate_loss": cand_loss,
            "candidate_loss_over_logK": (cand_loss / log_k
                                         if cand_loss else None),
            "candidate_accuracy": top1,
            "delta_z": dz,
            "patch_R_L0": rec[0], "patch_R_L1": rec[1],
            "patch_R_L2": rec[2], "patch_R_L3": rec[3],
            "patch_p_clean": pat.get("p_target_clean_mean"),
            "patch_p_shuf": pat.get("p_target_shuf_mean"),
            "swap_clean_acc": wfs.get("clean_accuracy"),
            "swap_swap_acc": wfs.get("swap_accuracy"),
            "swap_predicts_clean_target_rate":
                wfs.get("swap_predicts_clean_target_rate"),
            "swap_differs_from_clean": wfs.get(
                "swap_predicts_differs_from_clean_rate"),
        }
        out["checkpoints"].append(ck_out)
        print(f"  step={step:>5}  cand/logK={(cand_loss/log_k if cand_loss else 0):.3f}  "
              f"top1={(top1 or 0):.2f}  Δz={(dz or 0):>5.2f}  "
              f"R_L0={(rec[0] or 0):>5.2f}  swap_acc={(wfs.get('swap_accuracy') or 0):.2f}  "
              f"differs={(wfs.get('swap_predicts_differs_from_clean_rate') or 0):.2f}  "
              f"({time.time()-t0:.1f}s)")
        del model
        if device == "mps":
            torch.mps.empty_cache()

    out["total_wall_clock_s"] = time.time() - t_start

    # Estimate τ as the step where candidate_loss / log_k first dips below 0.5
    tau = None
    for ck in out["checkpoints"]:
        ratio = ck.get("candidate_loss_over_logK")
        if ratio is not None and ratio < 0.5:
            tau = ck["step"]
            break
    if tau is None:
        tau = 2500  # canonical
    out["tau_estimate"] = tau
    for ck in out["checkpoints"]:
        ck["step_over_tau"] = ck["step"] / tau

    out_path = REPO_ROOT / "eta_sweep" / "results" / "dense_k10_trajectory.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o))
    print(f"\nwrote {out_path}; τ ≈ {tau}")


if __name__ == "__main__":
    main()
