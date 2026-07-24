#!/usr/bin/env python
"""Block 0 — evaluation-stability check.

For each of the 4 seeds, load the final checkpoint and compute three
quantities under N_CONFIGS evaluation configurations:
    (1) per-layer full-layer ablation Delta (nats),
    (2) per-layer logit-lens P(correct) at the prediction position,
    (3) per-layer target-position patching flip rate.

An "eval configuration" varies:
    (a) the shuffle seed used to order eval prompts (3 shuffles),
    (b) the batch size (64 vs 256).

Each (seed, config) run produces the three metric scalars:
    m1 = L0/L2 ablation ratio
    m2 = Spearman rho between normalized ablation and normalized logit lens
    m3 = area between normalized necessity (ablation) and sufficiency (tgt flip)

Variance is decomposed into eval-noise variance (within-seed across configs)
and seed variance (across seeds). Success if eval-noise std < 25% of seed std
on all three metrics.
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
LOG_K = math.log(K)
N_UNIQUE_B = 1000
N_LAYERS = 4
SEEDS = [42, 123, 456, 789]
SHUFFLE_SEEDS = [0, 1, 2]
BATCH_SIZES = [64, 256]
N_EVAL = 512

DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "followup" / "block0"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def get_cfg_tokenizer(seed):
    cfg = make_config(experiment_name=f"b0_s{seed}", k=K, seed=seed,
                      n_unique_b=N_UNIQUE_B)
    tokenizer = create_tokenizer_from_config(cfg)
    return cfg, tokenizer


def get_ckpt_dir(seed):
    root = Path("outputs") / ("landauer_dense_k10" if seed == 42
                              else f"landauer_dense_k10_seed{seed}")
    return root / "checkpoints"


def load_model(seed):
    cfg, tokenizer = get_cfg_tokenizer(seed)
    ckpt_dir = get_ckpt_dir(seed)
    steps = sorted(list_checkpoints(ckpt_dir))
    model = create_model_from_config(cfg, tokenizer)
    model = model.to(DEVICE)
    load_checkpoint(model, None, ckpt_dir, step=steps[-1])
    model.eval()
    return cfg, tokenizer, model, steps[-1]


def build_dataset(seed):
    mapping_data = generate_mappings(
        n_unique_b=N_UNIQUE_B, k=K, b_length=6, a_length=4, z_length=2,
        vocab_chars="abcdefghijklmnopqrstuvwxyz0123456789",
        seed=seed, task="bz_to_a",
        enforce_unique_a_first_char_per_b=True, disambiguation_prefix_length=1,
    )
    return mapping_data


def build_loader(mapping_data, tokenizer, shuffle_seed, batch_size, n_eval=N_EVAL):
    ds = DisambiguationDataset(
        mapping_data=mapping_data, tokenizer=tokenizer,
        split="train", probe_fraction=0.0, seed=shuffle_seed, task="bz_to_a",
    )
    # Shuffle deterministically according to shuffle_seed, then truncate
    rng = np.random.default_rng(shuffle_seed)
    n_total = len(ds.tokenized)
    perm = rng.permutation(n_total)[:n_eval]
    ds.tokenized = [ds.tokenized[i] for i in perm]
    ds.examples = [ds.examples[i] for i in perm]
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
        num_workers=0,
    )
    return loader


def baseline_loss(model, loader):
    total = 0.0; n = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            logits = model(input_ids)
            sl = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            slb = labels[:, 1:].contiguous().view(-1)
            mask = slb != -100
            loss = F.cross_entropy(sl[mask], slb[mask], reduction="sum")
            total += loss.item()
            n += mask.sum().item()
    return total / max(n, 1)


def layer_ablation_delta(model, loader, layer, base_loss):
    pre_hook = f"blocks.{layer}.hook_resid_pre"
    post_hook = f"blocks.{layer}.hook_resid_post"
    store = {}
    def cap(v, hook): store["p"] = v.clone(); return v
    def byp(v, hook): return store["p"]
    total = 0.0; n = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            store.clear()
            logits = model.run_with_hooks(
                input_ids,
                fwd_hooks=[(pre_hook, cap), (post_hook, byp)],
            )
            sl = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            slb = labels[:, 1:].contiguous().view(-1)
            mask = slb != -100
            loss = F.cross_entropy(sl[mask], slb[mask], reduction="sum")
            total += loss.item()
            n += mask.sum().item()
    return (total / max(n, 1)) - base_loss


def logit_lens_p_correct(model, loader, layer):
    hook = "blocks.0.hook_resid_pre" if layer == 0 else f"blocks.{layer-1}.hook_resid_post"
    # Actually we want P(correct) at each layer's resid_post (after layer L)
    # Use lens_post = resid_post[layer]
    hook = f"blocks.{layer}.hook_resid_post"
    correct_probs = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            tgt_starts = batch["target_start_positions"].to(DEVICE)
            bs = input_ids.shape[0]
            batch_idx = torch.arange(bs, device=DEVICE)
            _, cache = model.run_with_cache(input_ids,
                                            names_filter=[hook])
            resid = cache[hook]
            pred_pos = tgt_starts - 1
            r = resid[batch_idx, pred_pos, :]
            # Apply final LN + unembed
            r_ln = model.ln_final(r)
            logits = model.unembed(r_ln)
            probs = F.softmax(logits, dim=-1)
            correct_tok = labels[batch_idx, tgt_starts]
            p = probs[batch_idx, correct_tok]
            correct_probs.append(p.cpu().numpy())
    return float(np.concatenate(correct_probs).mean())


def build_pairs(mapping_data, tokenizer, n_pairs, rng):
    pairs = []
    keys = list(mapping_data.mappings.keys())
    rng.shuffle(keys)
    for b_str in keys:
        targets = mapping_data.mappings[b_str]
        if len(targets) < 2:
            continue
        z_clean, a_clean = targets[0]
        z_corrupt, a_corrupt = targets[1]
        clean = tokenizer.encode_sequence(b_str, z_clean, a_clean, task="bz_to_a")
        corrupt = tokenizer.encode_sequence(b_str, z_corrupt, a_corrupt, task="bz_to_a")
        clean["first_target_id"] = tokenizer.encode(a_clean)[0]
        corrupt["first_target_id"] = tokenizer.encode(a_corrupt)[0]
        pairs.append((clean, corrupt))
        if len(pairs) >= n_pairs:
            break
    return pairs


def tgt_position_flip_rate(model, pairs, layer, batch_size):
    hook_name = f"blocks.{layer}.hook_resid_post"
    flips = []
    for start in range(0, len(pairs), batch_size):
        bp = pairs[start:start + batch_size]
        bs = len(bp)
        max_len = max(max(len(p[0]["input_ids"]) for p in bp),
                      max(len(p[1]["input_ids"]) for p in bp))
        clean_ids = torch.zeros(bs, max_len, dtype=torch.long)
        corrupt_ids = torch.zeros(bs, max_len, dtype=torch.long)
        tgt_starts = []
        corrupt_first_tok = []
        for i, (c, x) in enumerate(bp):
            clean_ids[i, :len(c["input_ids"])] = c["input_ids"]
            corrupt_ids[i, :len(x["input_ids"])] = x["input_ids"]
            tgt_starts.append(c["target_start_position"])
            corrupt_first_tok.append(x["first_target_id"])
        clean_ids = clean_ids.to(DEVICE)
        corrupt_ids = corrupt_ids.to(DEVICE)
        corrupt_tok = torch.tensor(corrupt_first_tok, device=DEVICE)
        tgt_pos = torch.tensor(tgt_starts, device=DEVICE)
        pred_pos = tgt_pos - 1
        batch_idx = torch.arange(bs, device=DEVICE)
        with torch.no_grad():
            _, corrupt_cache = model.run_with_cache(
                corrupt_ids, names_filter=[hook_name])
            corrupt_resid = corrupt_cache[hook_name].clone()
            def make_patch(c_resid, p_pos):
                def hook_fn(value, hook):
                    for i in range(value.shape[0]):
                        value[i, p_pos[i], :] = c_resid[i, p_pos[i], :]
                    return value
                return hook_fn
            patched_logits = model.run_with_hooks(
                clean_ids,
                fwd_hooks=[(hook_name, make_patch(corrupt_resid, pred_pos))],
            )
            patched_at = patched_logits[batch_idx, pred_pos, :]
            argmax = patched_at.argmax(dim=-1)
            flip = (argmax == corrupt_tok).float()
            flips.extend(flip.cpu().tolist())
            del corrupt_cache
    return float(np.mean(flips))


def primary_metrics(ablation, lens_p, tgt_flip):
    """Compute the three headline metrics from per-layer profiles.
    ablation: list of 4 ablation deltas (L0, L1, L2, L3).
    lens_p:   list of 4 P(correct) values at resid_post per layer.
    tgt_flip: list of 4 target-position flip rates per layer.
    """
    abl = np.array(ablation)
    lens = np.array(lens_p)
    flip = np.array(tgt_flip)
    # 1. Ratio L0/L2
    ratio = float(abl[0] / max(abl[2], 1e-9))
    # 2. Spearman between normalized ablation and lens
    n_abl = abl / (np.max(np.abs(abl)) + 1e-9)
    n_lens = lens / (np.max(lens) + 1e-9)
    rho, _ = stats.spearmanr(n_abl, n_lens)
    # 3. Area between normalized necessity and sufficiency curves (trapezoidal)
    n_flip = flip / (np.max(flip) + 1e-9)
    # depths are [0,1,2,3] → rescaled to [0,1]
    xs = np.linspace(0, 1, len(abl))
    diff = np.abs(n_abl - n_flip)
    area = float(np.trapz(diff, xs))
    return {"ratio_L0_L2": ratio, "spearman_abl_lens": float(rho),
            "area_necessity_sufficiency": area}


def run_one_config(seed, shuffle_seed, batch_size, cfg, tokenizer, model,
                   mapping_data):
    """Return dict with per-layer ablation, lens, and flip + 3 metrics."""
    loader = build_loader(mapping_data, tokenizer, shuffle_seed, batch_size,
                          n_eval=N_EVAL)
    bl = baseline_loss(model, loader)
    ablation = [layer_ablation_delta(model, loader, L, bl) for L in range(N_LAYERS)]
    lens_p = [logit_lens_p_correct(model, loader, L) for L in range(N_LAYERS)]
    rng = np.random.default_rng(shuffle_seed + 1000)
    pairs = build_pairs(mapping_data, tokenizer, n_pairs=256, rng=rng)
    tgt_flip = [tgt_position_flip_rate(model, pairs, L, batch_size)
                for L in range(N_LAYERS)]
    metrics = primary_metrics(ablation, lens_p, tgt_flip)
    return {
        "ablation_delta_per_layer": ablation,
        "lens_p_correct_per_layer": lens_p,
        "tgt_flip_rate_per_layer": tgt_flip,
        "baseline_loss": bl,
        **metrics,
    }


def main():
    results = {
        "config": {
            "seeds": SEEDS, "shuffle_seeds": SHUFFLE_SEEDS,
            "batch_sizes": BATCH_SIZES, "n_eval": N_EVAL,
            "device": DEVICE, "K": K,
        },
        "runs": [],
    }
    for seed in SEEDS:
        try:
            cfg, tokenizer, model, final_step = load_model(seed)
        except Exception as e:
            print(f"[skip] seed {seed}: {e}")
            continue
        mapping_data = build_dataset(seed)
        print(f"\n=== seed {seed} (final step {final_step}) ===")
        for shuf in SHUFFLE_SEEDS:
            for bs in BATCH_SIZES:
                key = f"seed{seed}_shuf{shuf}_bs{bs}"
                print(f"  {key}...", end="", flush=True)
                run = run_one_config(seed, shuf, bs, cfg, tokenizer, model,
                                     mapping_data)
                run.update({"seed": seed, "shuffle_seed": shuf, "batch_size": bs,
                            "key": key, "final_step": final_step})
                results["runs"].append(run)
                print(f"  ratio={run['ratio_L0_L2']:.2f} "
                      f"ρ={run['spearman_abl_lens']:+.2f} "
                      f"area={run['area_necessity_sufficiency']:.3f}")
        # Free
        del model
        if DEVICE == "mps":
            try: torch.mps.empty_cache()
            except Exception: pass
        elif DEVICE == "cuda":
            torch.cuda.empty_cache()

    # Variance decomposition
    import pandas as pd
    df = pd.DataFrame(results["runs"])
    metric_cols = ["ratio_L0_L2", "spearman_abl_lens", "area_necessity_sufficiency"]
    decomposition = {}
    for m in metric_cols:
        # within-seed (eval-noise) std: average across seeds of within-seed std
        within = df.groupby("seed")[m].std(ddof=0)
        # across-seed std: std of per-seed means
        means = df.groupby("seed")[m].mean()
        eval_std = float(within.mean())
        seed_std = float(means.std(ddof=0))
        ratio = float(eval_std / (seed_std + 1e-12))
        decomposition[m] = {
            "eval_noise_std": eval_std,
            "seed_std": seed_std,
            "eval_over_seed_ratio": ratio,
            "per_seed_means": {int(k): float(v) for k, v in means.items()},
            "per_seed_within_std": {int(k): float(v) for k, v in within.items()},
        }
    results["variance_decomposition"] = decomposition

    # Verdict: eval-noise std < 25% of seed std for all three metrics
    verdicts = {}
    for m, d in decomposition.items():
        verdicts[m] = "supports" if d["eval_over_seed_ratio"] < 0.25 else (
            "partially_supports" if d["eval_over_seed_ratio"] < 0.5 else "weakens"
        )
    results["per_metric_verdict"] = verdicts
    overall = (
        "supports" if all(v == "supports" for v in verdicts.values())
        else "partially_supports" if any(v != "weakens" for v in verdicts.values())
        else "weakens"
    )
    results["verdict"] = overall

    out_path = OUT_DIR / "block0_stability.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Block 0] verdict: {overall}")
    for m, d in decomposition.items():
        print(f"  {m:32s}  eval-std={d['eval_noise_std']:.3f}  "
              f"seed-std={d['seed_std']:.3f}  "
              f"eval/seed={d['eval_over_seed_ratio']:.2f}  "
              f"→ {verdicts[m]}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
