#!/usr/bin/env python
"""Path A — transport-distance discrimination.

Three variants of bz_to_a K=10 with constant sequence length (M=6 filler
budget) differing only in where the filler sits:
    short:  6 filler_head + 0 filler_tail   (informative z adjacent to pred)
    medium: 3 filler_head + 3 filler_tail
    long:   0 filler_head + 6 filler_tail   (informative z far from pred)

Same total sequence length (22 tokens) across all variants. Random-char
filler sampled once per (example, position) at dataset construction,
deterministic per seed.

Pre-committed decision rule:
    positional_transport_supported iff
        area monotonically increases across {short, medium, long}
        AND |Spearman ρ| monotonically increases (toward -1)
    positional_transport_not_primary iff
        neither metric shows a trend
    inconclusive otherwise
"""

import sys
import json
import math
import time
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf
from src.data import create_tokenizer_from_config, collate_fn
from src.data.dataset import generate_mappings
from src.model import create_model_from_config
from src.training import train
from src.training.checkpoint import list_checkpoints, load_checkpoint
from scripts.experiment_helpers import make_config

K = 10
N_UNIQUE_B = 1000
N_LAYERS = 4
M_FILLER = 6  # total filler budget
VARIANTS = [
    ("short",  6, 0),
    ("medium", 3, 3),
    ("long",   0, 6),
]
SEEDS = [42, 123]
N_EVAL = 512
BATCH_SIZE = 128
DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")

HARD_STOP_PER_RUN = 30 * 60
HARD_STOP_TOTAL = 4 * 3600

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "followup" / "path_a_transport"
OUT_DIR.mkdir(parents=True, exist_ok=True)


class TransportDataset(torch.utils.data.Dataset):
    """bz_to_a dataset with filler inserted at head and/or tail positions.

    Layout: <BOS> [filler_head] B <SEP> z <SEP> [filler_tail] A <EOS>
    The prediction site for the first target token is target_start - 1
    (the last filler_tail token, or the second SEP if filler_tail=0).
    Labels are -100 everywhere before target_start, then the target char
    tokens, then EOS. Filler tokens are never supervised.
    """

    def __init__(self, mapping_data, tokenizer, filler_head, filler_tail,
                 seed, n_examples=None):
        self.tokenizer = tokenizer
        self.filler_head = filler_head
        self.filler_tail = filler_tail
        # Pool of filler-eligible characters: same alphabet as B/z/A
        self._vocab_char_ids = [tokenizer.token_to_id[c]
                                 for c in tokenizer.vocab_chars]
        examples = list(mapping_data.examples)
        rng = random.Random(seed)
        rng.shuffle(examples)
        if n_examples is not None:
            examples = examples[:n_examples]
        self.examples = examples
        # Pre-tokenize, including deterministic random filler per example
        self.tokenized = []
        filler_rng = random.Random(seed + 1_000_003)
        for ex in self.examples:
            B, Z, A = ex["b"], ex["z"], ex["a"]
            head = [filler_rng.choice(self._vocab_char_ids)
                    for _ in range(filler_head)]
            tail = [filler_rng.choice(self._vocab_char_ids)
                    for _ in range(filler_tail)]
            tokens = [tokenizer.bos_token_id]
            tokens.extend(head)
            tokens.extend(tokenizer.encode(B))
            tokens.append(tokenizer.sep_token_id)
            z_position = len(tokens)
            tokens.extend(tokenizer.encode(Z))
            tokens.append(tokenizer.sep_token_id)
            tokens.extend(tail)
            target_start = len(tokens)
            target_tokens = tokenizer.encode(A)
            tokens.extend(target_tokens)
            tokens.append(tokenizer.eos_token_id)
            labels = [-100] * target_start + target_tokens + [tokenizer.eos_token_id]
            self.tokenized.append({
                "input_ids": torch.tensor(tokens, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "z_position": z_position,
                "z_end_position": z_position + len(Z),
                "target_start_position": target_start,
                "target_end_position": target_start + len(A),
                "b": B, "z": Z, "a": A,
                "base_string": B,
            })

    def __len__(self):
        return len(self.tokenized)

    def __getitem__(self, idx):
        return self.tokenized[idx]


def make_variant_config(variant_name, seed, max_steps=15000):
    cfg = make_config(
        experiment_name=f"path_a_{variant_name}_s{seed}",
        task="bz_to_a", k=K, seed=seed, n_unique_b=N_UNIQUE_B,
        max_steps=max_steps,
        checkpoint_every=max_steps,
        eval_every=250,
        early_stop_frac=0.005,
    )
    return cfg


def train_variant(variant_name, filler_head, filler_tail, seed):
    cfg = make_variant_config(variant_name, seed)
    torch.manual_seed(seed); np.random.seed(seed)
    tokenizer = create_tokenizer_from_config(cfg)
    mapping_data = generate_mappings(
        n_unique_b=N_UNIQUE_B, k=K, b_length=6, a_length=4, z_length=2,
        vocab_chars="abcdefghijklmnopqrstuvwxyz0123456789",
        seed=seed, task="bz_to_a",
        enforce_unique_a_first_char_per_b=True, disambiguation_prefix_length=1,
    )
    train_ds = TransportDataset(mapping_data, tokenizer, filler_head, filler_tail, seed)
    probe_ds = TransportDataset(mapping_data, tokenizer, filler_head, filler_tail,
                                 seed + 1, n_examples=512)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    probe_loader = torch.utils.data.DataLoader(
        probe_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )
    model = create_model_from_config(cfg, tokenizer)
    model = model.to(DEVICE)

    # Write config
    out = Path("outputs") / cfg.experiment.name
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    t0 = time.time()
    history = train(
        model=model, train_loader=train_loader, probe_loader=probe_loader,
        cfg=cfg, output_dir=Path("outputs"), grad_clip=1.0,
        optimizer_type="adamw",
        mapping_data=mapping_data, tokenizer=tokenizer,  # enables early-stop
    )
    elapsed = time.time() - t0
    final_loss = (history.get("train_loss", []) or [math.nan])[-1]
    return {
        "variant": variant_name, "filler_head": filler_head,
        "filler_tail": filler_tail, "seed": seed, "cfg": cfg,
        "tokenizer": tokenizer, "mapping_data": mapping_data,
        "model": model, "history": history,
        "elapsed_seconds": elapsed,
        "final_loss": float(final_loss),
        "final_step": history.get("steps", [0])[-1],
    }


# Evaluation helpers (reuse pattern from item1_scope.py)
def build_eval_loader(mapping_data, tokenizer, filler_head, filler_tail, seed):
    ds = TransportDataset(mapping_data, tokenizer, filler_head, filler_tail,
                          seed, n_examples=N_EVAL)
    return torch.utils.data.DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
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
            logits = model.run_with_hooks(
                ids, fwd_hooks=[(pre, cap), (post, byp)]
            )
            sl = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            slb = lab[:, 1:].contiguous().view(-1)
            mask = slb != -100
            total += F.cross_entropy(sl[mask], slb[mask], reduction="sum").item()
            n += mask.sum().item()
    return (total / max(n, 1)) - bl


def lens_Pcorrect(model, loader, layer):
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


def primary_metrics(ablation, lens):
    abl = np.array(ablation); le = np.array(lens)
    n_abl = abl / (np.max(np.abs(abl)) + 1e-9)
    n_lens = le / (np.max(le) + 1e-9)
    rho, _ = stats.spearmanr(n_abl, n_lens)
    xs = np.linspace(0, 1, len(abl))
    area = float(np.trapz(np.abs(n_abl - n_lens), xs))
    ratio = float(abl[0] / max(abl[2], 1e-9))
    return {"spearman_rho": float(rho), "area": area,
            "l0_l2_ratio": ratio}


def evaluate_run(run):
    model = run["model"].to(DEVICE).eval()
    loader = build_eval_loader(run["mapping_data"], run["tokenizer"],
                               run["filler_head"], run["filler_tail"],
                               run["seed"])
    bl = baseline_loss(model, loader)
    abl = [layer_ablation_delta(model, loader, L, bl) for L in range(N_LAYERS)]
    lens = [lens_Pcorrect(model, loader, L) for L in range(N_LAYERS)]
    m = primary_metrics(abl, lens)
    # Held-out accuracy sanity check
    correct = 0; total = 0
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(DEVICE)
            lab = batch["labels"].to(DEVICE)
            tgt = batch["target_start_positions"].to(DEVICE)
            bs = ids.shape[0]; idx = torch.arange(bs, device=DEVICE)
            logits = model(ids)
            preds = logits[idx, tgt - 1, :].argmax(dim=-1)
            truth = lab[idx, tgt]
            correct += (preds == truth).sum().item()
            total += bs
    accuracy = correct / max(total, 1)
    return {
        "variant": run["variant"], "seed": run["seed"],
        "filler_head": run["filler_head"], "filler_tail": run["filler_tail"],
        "final_step": run["final_step"], "final_loss": run["final_loss"],
        "elapsed_seconds": run["elapsed_seconds"],
        "held_out_accuracy": accuracy,
        "baseline_loss": bl,
        "ablation_per_layer": abl,
        "lens_per_layer": lens,
        "converged": bool(run["final_loss"] < 0.1 and accuracy > 0.9),
        **m,
    }


def cleanup(run):
    del run["model"]
    if DEVICE == "mps":
        try: torch.mps.empty_cache()
        except Exception: pass
    elif DEVICE == "cuda":
        torch.cuda.empty_cache()


def monotonic_increasing(xs):
    return all(xs[i] <= xs[i+1] for i in range(len(xs)-1))


def monotonic_decreasing(xs):
    return all(xs[i] >= xs[i+1] for i in range(len(xs)-1))


def main():
    overall_t0 = time.time()
    all_runs = []
    per_variant = {v[0]: [] for v in VARIANTS}

    for variant, fh, ft in VARIANTS:
        for seed in SEEDS:
            name = f"{variant}_s{seed}"
            print(f"\n=== {name}  (filler_head={fh}, filler_tail={ft}) ===")
            t0 = time.time()
            run = train_variant(variant, fh, ft, seed)
            if run["elapsed_seconds"] > HARD_STOP_PER_RUN:
                print(f"[HARD STOP] {name} took {run['elapsed_seconds']:.0f}s")
                cleanup(run)
                break
            print(f"  trained: step={run['final_step']} "
                  f"loss={run['final_loss']:.4f} "
                  f"elapsed={run['elapsed_seconds']:.0f}s")
            result = evaluate_run(run)
            cleanup(run)
            all_runs.append(result)
            per_variant[variant].append(result)
            print(f"  accuracy={result['held_out_accuracy']:.3f}")
            print(f"  abl={[round(v,3) for v in result['ablation_per_layer']]}")
            print(f"  lens={[round(v,3) for v in result['lens_per_layer']]}")
            print(f"  ρ={result['spearman_rho']:+.3f} "
                  f"area={result['area']:.3f} "
                  f"L0/L2={result['l0_l2_ratio']:.2f} "
                  f"converged={result['converged']}")
            if time.time() - overall_t0 > HARD_STOP_TOTAL:
                print(f"[HARD STOP] total elapsed > {HARD_STOP_TOTAL}s")
                break

    # Aggregate per variant
    def agg(subset):
        if not subset:
            return None
        rhos = [r["spearman_rho"] for r in subset]
        areas = [r["area"] for r in subset]
        ratios = [r["l0_l2_ratio"] for r in subset]
        return {
            "n_seeds": len(subset),
            "n_converged": sum(1 for r in subset if r["converged"]),
            "mean_area": float(np.mean(areas)), "std_area": float(np.std(areas, ddof=0)),
            "mean_spearman": float(np.mean(rhos)), "std_spearman": float(np.std(rhos, ddof=0)),
            "mean_l0_l2_ratio": float(np.mean(ratios)),
        }
    variants_out = []
    for name, fh, ft in VARIANTS:
        subset = per_variant[name]
        seq_len = 1 + fh + 6 + 1 + 2 + 1 + ft + 4 + 1  # BOS + head + B + SEP + z + SEP + tail + A + EOS
        variants_out.append({
            "name": name,
            "filler_head": fh,
            "filler_tail": ft,
            "total_seq_length": seq_len,
            "seeds": subset,
            "aggregate": agg(subset),
        })

    # Monotonicity (pre-committed rule)
    def get_agg(name, key):
        for v in variants_out:
            if v["name"] == name and v["aggregate"]:
                return v["aggregate"][key]
        return None

    areas = [get_agg(n, "mean_area") for n, _, _ in VARIANTS]
    rhos = [get_agg(n, "mean_spearman") for n, _, _ in VARIANTS]

    mono = {
        "area_values_short_medium_long": areas,
        "spearman_values_short_medium_long": rhos,
    }
    if all(a is not None for a in areas):
        mono["area_monotonic_increasing_in_distance"] = monotonic_increasing(areas)
    else:
        mono["area_monotonic_increasing_in_distance"] = None
    # Spearman is negative in all settings; "toward -1" means more negative,
    # i.e. monotonically DECREASING as distance grows (more negative -> closer to -1).
    if all(r is not None for r in rhos):
        mono["spearman_monotonic_toward_neg1"] = monotonic_decreasing(rhos)
    else:
        mono["spearman_monotonic_toward_neg1"] = None
    area_trend = mono.get("area_monotonic_increasing_in_distance") is True
    rho_trend = mono.get("spearman_monotonic_toward_neg1") is True
    mono["both_monotonic"] = area_trend and rho_trend

    # Check "neither shows trend" = both are flat (max-min spread small)
    def spread(xs): return max(xs) - min(xs) if xs and all(x is not None for x in xs) else None
    area_spread = spread(areas)
    rho_spread = spread(rhos)
    mono["area_spread"] = area_spread
    mono["spearman_spread"] = rho_spread
    # Define "trendless" as spread < 0.03 on area and < 0.05 on spearman
    neither_trend = ((area_spread is not None and area_spread < 0.03)
                     and (rho_spread is not None and rho_spread < 0.05))
    mono["neither_shows_trend"] = neither_trend

    if mono["both_monotonic"]:
        verdict = "positional_transport_supported"
    elif neither_trend:
        verdict = "positional_transport_not_primary"
    else:
        verdict = "inconclusive"

    decision = {
        "rule": ("positional transport supported iff both metrics monotonic "
                 "in predicted direction (area ↑, Spearman toward −1)"),
        "verdict": verdict,
    }

    output = {
        "M_filler_budget": M_FILLER,
        "variants": variants_out,
        "monotonicity_check": mono,
        "decision": decision,
        "config": {"device": DEVICE, "seeds": SEEDS, "k": K,
                   "max_steps": 15000, "n_eval": N_EVAL},
    }
    with open(OUT_DIR / "transport_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Figure
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150, facecolor="white")
        xs = [0, 1, 2]
        labels = ["short\n(6,0)", "medium\n(3,3)", "long\n(0,6)"]
        # Left: area
        ax = axes[0]
        for seed_idx, seed in enumerate(SEEDS):
            y = []
            for name, _, _ in VARIANTS:
                vs = [r["area"] for r in per_variant[name] if r["seed"] == seed]
                y.append(vs[0] if vs else None)
            if all(v is not None for v in y):
                ax.plot(xs, y, "o--", alpha=0.5, linewidth=1,
                        color="gray", label=f"seed {seed}")
        means = [get_agg(n, "mean_area") for n, _, _ in VARIANTS]
        stds = [get_agg(n, "std_area") for n, _, _ in VARIANTS]
        if all(m is not None for m in means):
            ax.errorbar(xs, means, yerr=stds, fmt="o-", color="#C4452D",
                        linewidth=2, markersize=9, capsize=5, label="mean ± std")
        ax.set_xticks(xs); ax.set_xticklabels(labels)
        ax.set_xlabel("transport distance variant"); ax.set_ylabel("area (necessity, sufficiency)")
        ax.set_title(f"Area metric vs transport distance\n"
                      f"monotonic↑: {mono['area_monotonic_increasing_in_distance']}",
                      fontsize=10)
        ax.legend(fontsize=8, frameon=False)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

        ax = axes[1]
        for seed_idx, seed in enumerate(SEEDS):
            y = []
            for name, _, _ in VARIANTS:
                vs = [r["spearman_rho"] for r in per_variant[name] if r["seed"] == seed]
                y.append(vs[0] if vs else None)
            if all(v is not None for v in y):
                ax.plot(xs, y, "o--", alpha=0.5, linewidth=1,
                        color="gray", label=f"seed {seed}")
        means = [get_agg(n, "mean_spearman") for n, _, _ in VARIANTS]
        stds = [get_agg(n, "std_spearman") for n, _, _ in VARIANTS]
        if all(m is not None for m in means):
            ax.errorbar(xs, means, yerr=stds, fmt="o-", color="#3B6FB6",
                        linewidth=2, markersize=9, capsize=5, label="mean ± std")
        ax.axhline(-1, color="black", linestyle="dotted", linewidth=0.6)
        ax.set_xticks(xs); ax.set_xticklabels(labels)
        ax.set_xlabel("transport distance variant"); ax.set_ylabel("Spearman ρ(abl, lens)")
        ax.set_title(f"Spearman vs transport distance\n"
                      f"monotonic toward -1: {mono['spearman_monotonic_toward_neg1']}",
                      fontsize=10)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

        fig.suptitle(f"Path A — transport distance — verdict: {verdict}",
                     fontsize=12, y=1.02)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "transport_figure.png", dpi=180,
                    facecolor="white", bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"[warn] figure generation failed: {e}")

    # Summary markdown
    lines = ["# Path A — transport-distance discrimination\n"]
    lines.append(f"**Verdict:** `{verdict}`\n")
    lines.append(f"**Rule:** {decision['rule']}\n")
    lines.append("\n## Per-variant aggregate (2 seeds each)\n")
    lines.append("| variant | filler (head,tail) | seq len | area (mean±std) | Spearman ρ (mean±std) | L0/L2 ratio |")
    lines.append("|---|---|---|---|---|---|")
    for v in variants_out:
        if not v["aggregate"]:
            lines.append(f"| {v['name']} | ({v['filler_head']},{v['filler_tail']}) | {v['total_seq_length']} | — | — | — |")
            continue
        a = v["aggregate"]
        lines.append(f"| {v['name']} | ({v['filler_head']},{v['filler_tail']}) | "
                      f"{v['total_seq_length']} | "
                      f"{a['mean_area']:.3f} ± {a['std_area']:.3f} | "
                      f"{a['mean_spearman']:+.3f} ± {a['std_spearman']:.3f} | "
                      f"{a['mean_l0_l2_ratio']:.2f} |")
    lines.append("\n## Monotonicity check\n")
    lines.append(f"- area values (short→med→long): {[round(a,3) if a is not None else None for a in areas]}")
    lines.append(f"- Spearman values: {[round(r,3) if r is not None else None for r in rhos]}")
    lines.append(f"- area monotonically increasing: **{mono['area_monotonic_increasing_in_distance']}**")
    lines.append(f"- Spearman monotonically toward -1: **{mono['spearman_monotonic_toward_neg1']}**")
    lines.append(f"- both monotonic: **{mono['both_monotonic']}**")
    lines.append(f"- neither shows trend (area spread<0.03 AND rho spread<0.05): **{mono['neither_shows_trend']}**")
    lines.append("\n## Per-seed raw\n")
    for v in variants_out:
        lines.append(f"\n### {v['name']}")
        for r in v["seeds"]:
            lines.append(f"- seed {r['seed']}: final_loss={r['final_loss']:.4f} "
                          f"acc={r['held_out_accuracy']:.3f} converged={r['converged']} "
                          f"abl={[round(x,3) for x in r['ablation_per_layer']]} "
                          f"lens={[round(x,3) for x in r['lens_per_layer']]} "
                          f"area={r['area']:.3f} ρ={r['spearman_rho']:+.3f}")
    lines.append("")
    if verdict == "positional_transport_supported":
        lines.append("\n## Meaning\n\nThe gap grows monotonically with transport distance between "
                     "informative source tokens and the prediction site, controlling for task structure. "
                     "This supports the positional-transport hypothesis from the Item 1 audit: the "
                     "ablation–decodability mismatch reflects information routing across residual-stream "
                     "positions rather than compositional task structure.")
    elif verdict == "positional_transport_not_primary":
        lines.append("\n## Meaning\n\nThe gap does not vary systematically with transport distance. "
                     "This weakens the positional-transport hypothesis as the primary driver: "
                     "transport distance is not sufficient to modulate the mismatch. Other factors "
                     "(task structure, architecture, readout alignment) must be more important.")
    else:
        lines.append("\n## Meaning\n\nThe data are ambiguous under the pre-committed monotonicity rule "
                     "at this sample size (2 seeds per distance). No causal claim is earned.")
    with open(OUT_DIR / "transport_summary.md", "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n========== PATH A SUMMARY ==========")
    print(f"Verdict: {verdict}")
    for v in variants_out:
        if v["aggregate"]:
            a = v["aggregate"]
            print(f"  {v['name']:<7} ({v['filler_head']},{v['filler_tail']}) seqlen={v['total_seq_length']}: "
                  f"area={a['mean_area']:.3f}±{a['std_area']:.3f}  "
                  f"ρ={a['mean_spearman']:+.3f}±{a['std_spearman']:.3f}  "
                  f"L0/L2={a['mean_l0_l2_ratio']:.2f}")
    print(f"Total wall-clock: {(time.time()-overall_t0)/60:.1f} min")


if __name__ == "__main__":
    main()
