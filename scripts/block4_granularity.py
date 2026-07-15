#!/usr/bin/env python
"""Block 4 — granularity ladder (full-layer → component → head).

For seed 42, K=10, final checkpoint, compute ablation Δ at three granularities:
  (G1) full layer (attention + MLP + skip → all go to 0)
  (G2) component: attention-only and MLP-only, per layer
  (G3) head: per-head within each layer's attention

For each granularity, compute:
  - Spearman ρ between normalized per-unit ablation Δ and the per-unit
    "decodability lift" (logit-lens ΔP(correct) attributable to that unit,
    computed as P_after - P_before where applicable).
  - Peak distance between ablation peak unit and decodability peak unit.
  - Original L0/L2 ratio (full-layer only; component/head levels report per-layer aggregates).

Expected: finer granularity -> necessity ranking aligns better with sufficiency / decodability.
"""

import sys
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import create_tokenizer_from_config, collate_fn
from src.data.dataset import generate_mappings, DisambiguationDataset
from src.model import create_model_from_config
from src.training.checkpoint import load_checkpoint, list_checkpoints
from scripts.experiment_helpers import make_config

K = 10
N_UNIQUE_B = 1000
SEED = 42
N_LAYERS = 4
N_HEADS = 4
N_EVAL = 512
BATCH_SIZE = 128
DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "followup" / "block4"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_model():
    cfg = make_config(experiment_name="b4", k=K, seed=SEED, n_unique_b=N_UNIQUE_B)
    tokenizer = create_tokenizer_from_config(cfg)
    mapping_data = generate_mappings(
        n_unique_b=N_UNIQUE_B, k=K, b_length=6, a_length=4, z_length=2,
        vocab_chars="abcdefghijklmnopqrstuvwxyz0123456789",
        seed=SEED, task="bz_to_a",
        enforce_unique_a_first_char_per_b=True, disambiguation_prefix_length=1,
    )
    ckpt_dir = Path("outputs") / "landauer_dense_k10" / "checkpoints"
    steps = sorted(list_checkpoints(ckpt_dir))
    model = create_model_from_config(cfg, tokenizer)
    model = model.to(DEVICE)
    load_checkpoint(model, None, ckpt_dir, step=steps[-1])
    model.eval()
    ds = DisambiguationDataset(
        mapping_data=mapping_data, tokenizer=tokenizer,
        split="train", probe_fraction=0.0, seed=SEED, task="bz_to_a",
    )
    ds.tokenized = ds.tokenized[:N_EVAL]
    ds.examples = ds.examples[:N_EVAL]
    loader = torch.utils.data.DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn,
        num_workers=0,
    )
    return model, loader


def baseline_loss(model, loader):
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


def loss_with_hooks(model, loader, hooks):
    total = 0.0; n = 0
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(DEVICE)
            lab = batch["labels"].to(DEVICE)
            logits = model.run_with_hooks(ids, fwd_hooks=hooks)
            sl = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            slb = lab[:, 1:].contiguous().view(-1)
            mask = slb != -100
            total += F.cross_entropy(sl[mask], slb[mask], reduction="sum").item()
            n += mask.sum().item()
    return total / max(n, 1)


def ablate_full_layer(model, loader, layer, bl):
    pre = f"blocks.{layer}.hook_resid_pre"
    post = f"blocks.{layer}.hook_resid_post"
    store = {}
    def cap(v, hook): store["p"] = v.clone(); return v
    def byp(v, hook): return store["p"]
    total = 0.0; n = 0
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(DEVICE)
            lab = batch["labels"].to(DEVICE)
            store.clear()
            logits = model.run_with_hooks(ids, fwd_hooks=[(pre, cap), (post, byp)])
            sl = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            slb = lab[:, 1:].contiguous().view(-1)
            mask = slb != -100
            total += F.cross_entropy(sl[mask], slb[mask], reduction="sum").item()
            n += mask.sum().item()
    return (total / max(n, 1)) - bl


def ablate_attn_only(model, loader, layer, bl):
    # Zero out attn output for this layer, keep MLP + skip
    hook = f"blocks.{layer}.hook_attn_out"
    def zero(v, hook): return torch.zeros_like(v)
    return loss_with_hooks(model, loader, [(hook, zero)]) - bl


def ablate_mlp_only(model, loader, layer, bl):
    hook = f"blocks.{layer}.hook_mlp_out"
    def zero(v, hook): return torch.zeros_like(v)
    return loss_with_hooks(model, loader, [(hook, zero)]) - bl


def ablate_head(model, loader, layer, head, bl):
    hook = f"blocks.{layer}.attn.hook_z"
    def zhead(v, hook):
        v[:, :, head, :] = 0.0
        return v
    return loss_with_hooks(model, loader, [(hook, zhead)]) - bl


def logit_lens_Pcorrect(model, loader, layer):
    hook = f"blocks.{layer}.hook_resid_post"
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
    return float(np.concatenate(ps).mean())


def primary_gap_metrics(necessity, sufficiency):
    nec = np.array(necessity); suf = np.array(sufficiency)
    n_nec = nec / (np.max(np.abs(nec)) + 1e-9)
    n_suf = suf / (np.max(suf) + 1e-9)
    rho, _ = stats.spearmanr(n_nec, n_suf)
    xs = np.linspace(0, 1, len(nec))
    area = float(np.trapz(np.abs(n_nec - n_suf), xs))
    peak_dist = int(abs(np.argmax(nec) - np.argmax(suf)))
    return {"spearman": float(rho), "area": area, "peak_dist": peak_dist}


def main():
    model, loader = load_model()
    bl = baseline_loss(model, loader)

    # Per-layer lens as the "decodability" reference
    lens_per_layer = np.array([logit_lens_Pcorrect(model, loader, L)
                               for L in range(N_LAYERS)])

    # G1: full-layer ablation
    abl_layer = np.array([ablate_full_layer(model, loader, L, bl)
                          for L in range(N_LAYERS)])

    # G2: component ablation per layer
    abl_attn = np.array([ablate_attn_only(model, loader, L, bl) for L in range(N_LAYERS)])
    abl_mlp = np.array([ablate_mlp_only(model, loader, L, bl) for L in range(N_LAYERS)])

    # G3: per-head ablation
    abl_head = np.zeros((N_LAYERS, N_HEADS))
    for L in range(N_LAYERS):
        for H in range(N_HEADS):
            abl_head[L, H] = ablate_head(model, loader, L, H, bl)

    # Per-layer aggregate of component / head ablation to compare with lens
    # Component: take max(attn, mlp) per layer (the critical-component signal)
    abl_component_max = np.maximum(abl_attn, abl_mlp)
    # Head: sum of per-head ablations per layer
    abl_head_sum = abl_head.sum(axis=1)
    # Head max
    abl_head_max = abl_head.max(axis=1)

    results = {
        "config": {"K": K, "seed": SEED, "device": DEVICE, "n_eval": N_EVAL},
        "baseline_loss": bl,
        "lens_per_layer": lens_per_layer.tolist(),
        "full_layer_ablation": abl_layer.tolist(),
        "attn_only_ablation": abl_attn.tolist(),
        "mlp_only_ablation": abl_mlp.tolist(),
        "head_ablation_matrix": abl_head.tolist(),
        "component_max_per_layer": abl_component_max.tolist(),
        "head_sum_per_layer": abl_head_sum.tolist(),
        "head_max_per_layer": abl_head_max.tolist(),
    }
    # Gap metrics at each granularity (aggregate to per-layer where needed)
    results["metrics_by_granularity"] = {
        "G1_full_layer": primary_gap_metrics(abl_layer, lens_per_layer),
        "G2_component_attn": primary_gap_metrics(abl_attn, lens_per_layer),
        "G2_component_mlp": primary_gap_metrics(abl_mlp, lens_per_layer),
        "G2_component_max": primary_gap_metrics(abl_component_max, lens_per_layer),
        "G3_head_sum": primary_gap_metrics(abl_head_sum, lens_per_layer),
        "G3_head_max": primary_gap_metrics(abl_head_max, lens_per_layer),
    }
    # Verdict: finer granularity should produce |ρ| closer to 0 (less mis-alignment)
    # or smaller area, indicating necessity aligns with decodability better.
    g1_rho = results["metrics_by_granularity"]["G1_full_layer"]["spearman"]
    g2_rho = results["metrics_by_granularity"]["G2_component_max"]["spearman"]
    g3_rho = results["metrics_by_granularity"]["G3_head_max"]["spearman"]
    g1_area = results["metrics_by_granularity"]["G1_full_layer"]["area"]
    g2_area = results["metrics_by_granularity"]["G2_component_max"]["area"]
    g3_area = results["metrics_by_granularity"]["G3_head_max"]["area"]

    # "Closes the gap" = |ρ| decreases and area decreases as granularity gets finer
    rho_closes = (abs(g1_rho) >= abs(g2_rho)) and (abs(g2_rho) >= abs(g3_rho))
    area_closes = (g1_area >= g2_area) and (g2_area >= g3_area)

    if rho_closes or area_closes:
        verdict = "supports" if (rho_closes and area_closes) else "partially_supports"
    else:
        verdict = "weakens"

    results["verdict"] = verdict
    results["gap_closure"] = {
        "rho_closes_with_granularity": rho_closes,
        "area_closes_with_granularity": area_closes,
    }

    out = OUT_DIR / "block4_granularity.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Block 4: granularity ladder ===")
    print(f"  lens per layer (L0..L3):        {lens_per_layer.round(3).tolist()}")
    print(f"  full-layer ablation (Δ nats):    {abl_layer.round(3).tolist()}")
    print(f"  attn-only ablation (Δ nats):     {abl_attn.round(3).tolist()}")
    print(f"  mlp-only ablation (Δ nats):      {abl_mlp.round(3).tolist()}")
    print(f"  component-max per layer:         {abl_component_max.round(3).tolist()}")
    print(f"  head-max per layer:              {abl_head_max.round(3).tolist()}")
    print(f"\n  G1 full-layer:  ρ={g1_rho:+.2f}  area={g1_area:.3f}")
    print(f"  G2 component:   ρ={g2_rho:+.2f}  area={g2_area:.3f}")
    print(f"  G3 head:        ρ={g3_rho:+.2f}  area={g3_area:.3f}")
    print(f"  |ρ| closes with granularity? {rho_closes}")
    print(f"  area closes with granularity? {area_closes}")
    print(f"  Verdict: {verdict}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
