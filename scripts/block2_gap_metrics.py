#!/usr/bin/env python
"""Block 2 — pre-committed gap metrics across K sweep.

For each K in {3, 5, 7, 10, 13, 17, 20, 25, 30, 36} (seed 42 final ckpt):
  - full-layer ablation Δ per layer (necessity profile)
  - logit-lens P(correct) per layer (decodability profile)
  - target-position patching flip rate per layer (sufficiency profile)

Compute per-K gap metrics:
  PRIMARY:
    (P1) Spearman ρ between normalized ablation and normalized lens
    (P2) area between normalized necessity and sufficiency curves
  SUPPLEMENTARY:
    (S1) L0/L2 ablation ratio (legacy)
    (S2) peak distance (ablation peak vs lens peak)
    (S3) center-of-mass difference
    (S4) mass-before-onset (fraction of necessity mass before first decodable layer)

For K=10 also pull Block 0's 4-seed data so the K=10 row reports seed variance.
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

K_VALUES = [3, 5, 7, 10, 13, 17, 20, 25, 30, 36]
SEED = 42
N_UNIQUE_B = 1000
N_LAYERS = 4
N_EVAL = 512
BATCH_SIZE = 128
DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "followup" / "block2"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_for_K(k):
    cfg = make_config(experiment_name=f"b2_k{k}", k=k, seed=SEED,
                      n_unique_b=N_UNIQUE_B)
    tokenizer = create_tokenizer_from_config(cfg)
    mapping_data = generate_mappings(
        n_unique_b=N_UNIQUE_B, k=k, b_length=6, a_length=4, z_length=2,
        vocab_chars="abcdefghijklmnopqrstuvwxyz0123456789",
        seed=SEED, task="bz_to_a",
        enforce_unique_a_first_char_per_b=True, disambiguation_prefix_length=1,
    )
    ckpt_dir = Path("outputs") / f"landauer_dense_k{k}" / "checkpoints"
    steps = sorted(list_checkpoints(ckpt_dir))
    model = create_model_from_config(cfg, tokenizer)
    model = model.to(DEVICE)
    load_checkpoint(model, None, ckpt_dir, step=steps[-1])
    model.eval()
    return cfg, tokenizer, mapping_data, model


def build_loader(mapping_data, tokenizer, n_eval=N_EVAL):
    ds = DisambiguationDataset(
        mapping_data=mapping_data, tokenizer=tokenizer,
        split="train", probe_fraction=0.0, seed=SEED, task="bz_to_a",
    )
    ds.tokenized = ds.tokenized[:n_eval]
    ds.examples = ds.examples[:n_eval]
    return torch.utils.data.DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn,
        num_workers=0,
    )


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


def layer_ablation_delta(model, loader, layer, bl):
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


def logit_lens_P(model, loader, layer):
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


def build_pairs(mapping_data, tokenizer, n_pairs=256):
    pairs = []
    for b_str, targets in mapping_data.mappings.items():
        if len(targets) < 2:
            continue
        z_c, a_c = targets[0]
        z_x, a_x = targets[1]
        c = tokenizer.encode_sequence(b_str, z_c, a_c, task="bz_to_a")
        x = tokenizer.encode_sequence(b_str, z_x, a_x, task="bz_to_a")
        c["first_target_id"] = tokenizer.encode(a_c)[0]
        x["first_target_id"] = tokenizer.encode(a_x)[0]
        pairs.append((c, x))
        if len(pairs) >= n_pairs:
            break
    return pairs


def tgt_flip_rate(model, pairs, layer, bs=64):
    hook = f"blocks.{layer}.hook_resid_post"
    flips = []
    for start in range(0, len(pairs), bs):
        bp = pairs[start:start + bs]
        n = len(bp)
        max_len = max(max(len(p[0]["input_ids"]) for p in bp),
                      max(len(p[1]["input_ids"]) for p in bp))
        clean_ids = torch.zeros(n, max_len, dtype=torch.long)
        corrupt_ids = torch.zeros(n, max_len, dtype=torch.long)
        tgt_starts = []; corrupt_tok = []
        for i, (c, x) in enumerate(bp):
            clean_ids[i, :len(c["input_ids"])] = c["input_ids"]
            corrupt_ids[i, :len(x["input_ids"])] = x["input_ids"]
            tgt_starts.append(c["target_start_position"])
            corrupt_tok.append(x["first_target_id"])
        clean_ids = clean_ids.to(DEVICE); corrupt_ids = corrupt_ids.to(DEVICE)
        corrupt_tok = torch.tensor(corrupt_tok, device=DEVICE)
        tgt_pos = torch.tensor(tgt_starts, device=DEVICE)
        pred_pos = tgt_pos - 1
        bi = torch.arange(n, device=DEVICE)
        with torch.no_grad():
            _, cc = model.run_with_cache(corrupt_ids, names_filter=[hook])
            c_resid = cc[hook].clone()
            def patch(v, hook):
                for i in range(v.shape[0]):
                    v[i, pred_pos[i], :] = c_resid[i, pred_pos[i], :]
                return v
            pl = model.run_with_hooks(clean_ids, fwd_hooks=[(hook, patch)])
            argmax = pl[bi, pred_pos, :].argmax(dim=-1)
            flip = (argmax == corrupt_tok).float()
            flips.extend(flip.cpu().tolist())
            del cc
    return float(np.mean(flips))


def compute_profiles(model, loader, pairs):
    bl = baseline_loss(model, loader)
    abl = np.array([layer_ablation_delta(model, loader, L, bl) for L in range(N_LAYERS)])
    lens = np.array([logit_lens_P(model, loader, L) for L in range(N_LAYERS)])
    flip = np.array([tgt_flip_rate(model, pairs, L) for L in range(N_LAYERS)])
    return {"ablation": abl.tolist(), "lens": lens.tolist(),
            "flip": flip.tolist(), "baseline_loss": bl}


def normalize(p):
    return p / (np.max(np.abs(p)) + 1e-9)


def gap_metrics(abl, lens, flip):
    abl = np.array(abl); lens = np.array(lens); flip = np.array(flip)
    n = len(abl)
    xs = np.linspace(0, 1, n)
    n_abl = normalize(abl); n_lens = normalize(lens); n_flip = normalize(flip)

    # PRIMARY
    rho_lens, _ = stats.spearmanr(n_abl, n_lens)
    rho_flip, _ = stats.spearmanr(n_abl, n_flip)
    area_ns = float(np.trapz(np.abs(n_abl - n_flip), xs))

    # SUPPLEMENTARY
    ratio_l0l2 = float(abl[0] / max(abl[2], 1e-9))
    peak_dist_lens = float(abs(np.argmax(abl) - np.argmax(lens)))
    peak_dist_flip = float(abs(np.argmax(abl) - np.argmax(flip)))
    com_abl = float(np.sum(xs * n_abl) / (np.sum(n_abl) + 1e-9))
    com_lens = float(np.sum(xs * n_lens) / (np.sum(n_lens) + 1e-9))
    com_flip = float(np.sum(xs * n_flip) / (np.sum(n_flip) + 1e-9))
    com_diff_lens = abs(com_abl - com_lens)
    com_diff_flip = abs(com_abl - com_flip)
    # Mass-before-onset: fraction of total abl mass in layers before first decodable layer.
    # Onset = first layer where lens >= 0.5 (or last layer if never reached).
    first_dec = int(np.argmax(lens >= 0.5)) if np.any(lens >= 0.5) else n - 1
    mass_before_onset = float(np.sum(n_abl[:first_dec]) / (np.sum(n_abl) + 1e-9))

    return {
        "primary": {
            "spearman_abl_lens": float(rho_lens),
            "area_necessity_sufficiency": area_ns,
        },
        "supplementary": {
            "ratio_L0_L2": ratio_l0l2,
            "spearman_abl_flip": float(rho_flip),
            "peak_dist_lens": peak_dist_lens,
            "peak_dist_flip": peak_dist_flip,
            "com_diff_lens": float(com_diff_lens),
            "com_diff_flip": float(com_diff_flip),
            "mass_before_onset": mass_before_onset,
            "first_decodable_layer": first_dec,
        },
    }


def main():
    results = {"config": {"K_values": K_VALUES, "seed": SEED, "device": DEVICE,
                          "n_eval": N_EVAL, "primary_metrics": [
                              "spearman_abl_lens",
                              "area_necessity_sufficiency",
                          ]},
               "per_K": {}}

    for k in K_VALUES:
        print(f"\n=== K={k} ===")
        cfg, tokenizer, mapping_data, model = load_for_K(k)
        loader = build_loader(mapping_data, tokenizer)
        pairs = build_pairs(mapping_data, tokenizer, n_pairs=256)
        profiles = compute_profiles(model, loader, pairs)
        metrics = gap_metrics(profiles["ablation"], profiles["lens"],
                              profiles["flip"])
        print(f"  ablation: {profiles['ablation']}")
        print(f"  lens:     {profiles['lens']}")
        print(f"  flip:     {profiles['flip']}")
        print(f"  PRIMARY  ρ(abl,lens)={metrics['primary']['spearman_abl_lens']:+.3f}  "
              f"area={metrics['primary']['area_necessity_sufficiency']:.3f}")
        print(f"  SUPP     L0/L2={metrics['supplementary']['ratio_L0_L2']:.2f}  "
              f"peak_dist_lens={metrics['supplementary']['peak_dist_lens']:.0f}  "
              f"mass_before={metrics['supplementary']['mass_before_onset']:.3f}")
        results["per_K"][k] = {"profiles": profiles, **metrics}
        del model
        if DEVICE == "mps":
            try: torch.mps.empty_cache()
            except Exception: pass

    # K=10 seed variance from Block 0
    b0_path = Path("outputs") / "followup" / "block0" / "block0_stability.json"
    if b0_path.exists():
        with open(b0_path) as f:
            b0 = json.load(f)
        per_seed = {}
        for run in b0["runs"]:
            s = run["seed"]
            if s not in per_seed:
                # Pick first config per seed (they're identical up to shuffle noise)
                per_seed[s] = {
                    "ablation": run["ablation_delta_per_layer"],
                    "lens": run["lens_p_correct_per_layer"],
                    "flip": run["tgt_flip_rate_per_layer"],
                }
        seedwise_k10 = []
        for s, p in per_seed.items():
            m = gap_metrics(p["ablation"], p["lens"], p["flip"])
            seedwise_k10.append({"seed": s, **p, **m})
        results["K10_seedwise"] = seedwise_k10

        # Aggregate across seeds for K=10 primary metrics
        rhos = [r["primary"]["spearman_abl_lens"] for r in seedwise_k10]
        areas = [r["primary"]["area_necessity_sufficiency"] for r in seedwise_k10]
        ratios = [r["supplementary"]["ratio_L0_L2"] for r in seedwise_k10]
        results["K10_seedagg"] = {
            "spearman_abl_lens": {"mean": float(np.mean(rhos)), "std": float(np.std(rhos, ddof=0))},
            "area_necessity_sufficiency": {"mean": float(np.mean(areas)), "std": float(np.std(areas, ddof=0))},
            "ratio_L0_L2": {"mean": float(np.mean(ratios)), "std": float(np.std(ratios, ddof=0))},
        }

    out = OUT_DIR / "block2_gap_metrics.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    # Verdict
    per_K = results["per_K"]
    rhos = [per_K[k]["primary"]["spearman_abl_lens"] for k in K_VALUES]
    areas = [per_K[k]["primary"]["area_necessity_sufficiency"] for k in K_VALUES]
    all_rho_neg = all(r < -0.5 for r in rhos)
    all_area_high = all(a > 0.3 for a in areas)
    print("\n=== Block 2 summary ===")
    print(f"  Per-K Spearman ρ(abl, lens): {[f'{r:+.2f}' for r in rhos]}")
    print(f"  Per-K area (nec, suf):       {[f'{a:.2f}' for a in areas]}")
    print(f"  all ρ <= -0.5: {all_rho_neg}")
    print(f"  all area > 0.3: {all_area_high}")
    if all_rho_neg and all_area_high:
        verdict = "supports"
    elif all(r < 0 for r in rhos):
        verdict = "partially_supports"
    else:
        verdict = "weakens"
    results["verdict"] = verdict
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Verdict: {verdict}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
