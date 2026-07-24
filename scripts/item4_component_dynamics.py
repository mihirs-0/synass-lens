#!/usr/bin/env python
"""Item 4 — component-level training dynamics (seed 42 only).

Extends Block 3 to measure per-checkpoint ablation Δ for:
    - L2 attention-only ablation
    - L2 MLP-only ablation
    - L3 attention-only ablation
    - L3 MLP-only ablation
Plus the same component-level decodability snapshots as Block 3 (lens).

Question: does L2's MLP-criticality emerge before or after L3's attention-
criticality in training? If the division of labor (L2 MLP, L3 attn) has
a temporal ordering, the paper gets a cleaner mechanistic arc.
"""

import sys
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import create_tokenizer_from_config, collate_fn
from src.data.dataset import generate_mappings, DisambiguationDataset
from src.model import create_model_from_config
from src.training.checkpoint import load_checkpoint, list_checkpoints
from scripts.experiment_helpers import make_config

SEED = 42
K = 10
N_UNIQUE_B = 1000
N_EVAL = 512
BATCH_SIZE = 128
DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")

SAMPLE_STEPS = [100, 500, 1000, 1500, 1800, 2000, 2200, 2400, 2600, 2800,
                3000, 3500, 4000, 5000, 7500, 10000, 15000, 20000, 30000, 50000]

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "followup" / "item4"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def setup():
    cfg = make_config(experiment_name="i4", k=K, seed=SEED, n_unique_b=N_UNIQUE_B)
    tokenizer = create_tokenizer_from_config(cfg)
    mapping_data = generate_mappings(
        n_unique_b=N_UNIQUE_B, k=K, b_length=6, a_length=4, z_length=2,
        vocab_chars="abcdefghijklmnopqrstuvwxyz0123456789",
        seed=SEED, task="bz_to_a",
        enforce_unique_a_first_char_per_b=True, disambiguation_prefix_length=1,
    )
    ds = DisambiguationDataset(
        mapping_data=mapping_data, tokenizer=tokenizer,
        split="train", probe_fraction=0.0, seed=SEED, task="bz_to_a",
    )
    ds.tokenized = ds.tokenized[:N_EVAL]; ds.examples = ds.examples[:N_EVAL]
    loader = torch.utils.data.DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn,
        num_workers=0,
    )
    return cfg, tokenizer, loader


def base_loss(model, loader):
    total = 0.0; n = 0
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(DEVICE)
            lab = batch["labels"].to(DEVICE)
            logits = model(ids)
            sl = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            slb = lab[:, 1:].contiguous().view(-1)
            mask = slb != -100
            total += F.cross_entropy(sl[mask], slb[mask], reduction="sum").item()
            n += mask.sum().item()
    return total / max(n, 1)


def loss_with_hook(model, loader, hook_name, fn):
    total = 0.0; n = 0
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(DEVICE)
            lab = batch["labels"].to(DEVICE)
            logits = model.run_with_hooks(ids, fwd_hooks=[(hook_name, fn)])
            sl = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            slb = lab[:, 1:].contiguous().view(-1)
            mask = slb != -100
            total += F.cross_entropy(sl[mask], slb[mask], reduction="sum").item()
            n += mask.sum().item()
    return total / max(n, 1)


def eval_ckpt(cfg, tokenizer, loader, step):
    model = create_model_from_config(cfg, tokenizer)
    model = model.to(DEVICE)
    load_checkpoint(model, None, Path("outputs") / "landauer_dense_k10" / "checkpoints",
                    step=step)
    model.eval()
    bl = base_loss(model, loader)

    zero = lambda v, hook: torch.zeros_like(v)
    results = {}
    for layer in [2, 3]:
        l_attn = loss_with_hook(model, loader,
                                 f"blocks.{layer}.hook_attn_out", zero) - bl
        l_mlp = loss_with_hook(model, loader,
                                f"blocks.{layer}.hook_mlp_out", zero) - bl
        results[f"L{layer}_attn_ablation"] = l_attn
        results[f"L{layer}_mlp_ablation"] = l_mlp

    # Lens per layer for reference
    lens = []
    for L in range(4):
        hook = f"blocks.{L}.hook_resid_post"
        ps = []
        with torch.no_grad():
            for batch in loader:
                ids = batch["input_ids"].to(DEVICE)
                lab = batch["labels"].to(DEVICE)
                tgt = batch["target_start_positions"].to(DEVICE)
                bs = ids.shape[0]; idx = torch.arange(bs, device=DEVICE)
                _, cache = model.run_with_cache(ids, names_filter=[hook])
                r = cache[hook][idx, tgt - 1, :]
                logits = model.unembed(model.ln_final(r))
                probs = F.softmax(logits, dim=-1)
                correct = lab[idx, tgt]
                ps.append(probs[idx, correct].cpu().numpy())
        lens.append(float(np.concatenate(ps).mean()))
    results["lens_per_layer"] = lens
    results["baseline_loss"] = bl

    del model
    if DEVICE == "mps":
        try: torch.mps.empty_cache()
        except Exception: pass
    return results


def find_onset(values, steps, threshold, sustain=3):
    """Smallest step at which values are >= threshold for 'sustain' consecutive points."""
    for i in range(len(values) - sustain + 1):
        if all(v >= threshold for v in values[i:i + sustain]):
            return int(steps[i])
    for i in range(len(values)):
        if all(v >= threshold for v in values[i:]):
            return int(steps[i])
    return None


def main():
    cfg, tokenizer, loader = setup()
    available = sorted(list_checkpoints(Path("outputs") / "landauer_dense_k10"
                                         / "checkpoints"))
    selected = sorted(set(min(available, key=lambda x: abs(x - s))
                          for s in SAMPLE_STEPS))
    per_ckpt = []
    print(f"Evaluating {len(selected)} checkpoints")
    for i, step in enumerate(selected):
        print(f"  [{i+1}/{len(selected)}] step {step}...", end="", flush=True)
        r = eval_ckpt(cfg, tokenizer, loader, step)
        r["step"] = step
        per_ckpt.append(r)
        print(f"  L2_attn={r['L2_attn_ablation']:.2f}  L2_mlp={r['L2_mlp_ablation']:.2f}  "
              f"L3_attn={r['L3_attn_ablation']:.2f}  L3_mlp={r['L3_mlp_ablation']:.2f}")

    steps = [c["step"] for c in per_ckpt]
    # Onset thresholds: component is "active" when its ablation Δ >= 0.3 nats sustained
    onset = {}
    for key in ["L2_attn_ablation", "L2_mlp_ablation",
                "L3_attn_ablation", "L3_mlp_ablation"]:
        vals = [c[key] for c in per_ckpt]
        onset[key] = find_onset(vals, steps, threshold=0.3, sustain=3)

    # Division-of-labor check:
    # Expected: L2_mlp > L2_attn (MLP dominates at L2 late)
    #           L3_attn > L3_mlp (attn dominates at L3 late)
    final = per_ckpt[-1]
    l2_mlp_dominates = final["L2_mlp_ablation"] > final["L2_attn_ablation"]
    l3_attn_dominates = final["L3_attn_ablation"] > final["L3_mlp_ablation"]

    # Does L2_mlp onset BEFORE L3_attn onset?
    if onset["L2_mlp_ablation"] is not None and onset["L3_attn_ablation"] is not None:
        ordering = "L2_mlp_first" if onset["L2_mlp_ablation"] < onset["L3_attn_ablation"] else (
            "L3_attn_first" if onset["L3_attn_ablation"] < onset["L2_mlp_ablation"] else "simultaneous"
        )
    else:
        ordering = "inconclusive"

    results = {
        "config": {"seed": SEED, "K": K, "n_checkpoints": len(selected),
                   "onset_threshold": 0.3, "sustain": 3},
        "checkpoints": per_ckpt,
        "onsets": onset,
        "final_state": {
            "L2_attn_dominates_at_final": final["L2_attn_ablation"] > final["L2_mlp_ablation"],
            "L2_mlp_dominates_at_final": l2_mlp_dominates,
            "L3_attn_dominates_at_final": l3_attn_dominates,
            "L3_mlp_dominates_at_final": final["L3_mlp_ablation"] > final["L3_attn_ablation"],
        },
        "temporal_ordering_L2mlp_vs_L3attn": ordering,
    }

    # Verdict
    division_of_labor_correct = l2_mlp_dominates and l3_attn_dominates
    if division_of_labor_correct and ordering != "inconclusive":
        verdict = "supports"
    elif division_of_labor_correct:
        verdict = "partially_supports"  # pattern holds but temporal ordering inconclusive
    else:
        verdict = "weakens"
    results["verdict"] = verdict

    with open(OUT_DIR / "item4_component_dynamics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Item 4: component-level training dynamics ===")
    print(f"\nTrajectories (step, L2_attn, L2_mlp, L3_attn, L3_mlp):")
    for c in per_ckpt:
        print(f"  {c['step']:>6d}  {c['L2_attn_ablation']:+.2f}  "
              f"{c['L2_mlp_ablation']:+.2f}  {c['L3_attn_ablation']:+.2f}  "
              f"{c['L3_mlp_ablation']:+.2f}")

    print(f"\nOnsets (step where Δ >= 0.3 sustained):")
    for k, v in onset.items():
        print(f"  {k}: {v}")
    print(f"\nFinal-state division of labor:")
    print(f"  L2 MLP dominates L2 attn? {l2_mlp_dominates}  "
          f"({final['L2_mlp_ablation']:+.2f} > {final['L2_attn_ablation']:+.2f})")
    print(f"  L3 attn dominates L3 MLP? {l3_attn_dominates}  "
          f"({final['L3_attn_ablation']:+.2f} > {final['L3_mlp_ablation']:+.2f})")
    print(f"\nTemporal ordering (L2_mlp vs L3_attn): {ordering}")
    print(f"Verdict: {verdict}")
    print(f"Wrote {OUT_DIR/'item4_component_dynamics.json'}")


if __name__ == "__main__":
    main()
