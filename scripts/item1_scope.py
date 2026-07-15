#!/usr/bin/env python
"""Item 1 — scope conditions: structural perturbations.

Trains two task variants from scratch and compares their
necessity--sufficiency gap against the primary (bz_to_a, K=10) setting:

  Condition 1 (control):        b_to_a,  K=1,  4 seeds (42, 123, 456, 789)
  Condition 2 (perturbation):   az_to_b, K=10, 2 seeds (42, 123)

On each final checkpoint, compute:
  - per-layer full-layer ablation Δ (zero-ablation)
  - per-layer logit lens P(correct) at the prediction position
  - primary gap metrics: Spearman ρ (normalised profiles) and area between
    |norm_ablation - norm_lens| curves.

Pre-committed decision rule: scope claim EARNED iff
  control area < 0.15 AND control Spearman > -0.3.
Evaluated on the 4-seed mean for the control.

Hard stops: any single training run > 2 hours, or total wall-clock > 4 hours.
"""

import sys
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import create_tokenizer_from_config, collate_fn
from src.data.dataset import generate_mappings, DisambiguationDataset
from src.model import create_model_from_config
from src.training.checkpoint import list_checkpoints, load_checkpoint
from scripts.experiment_helpers import make_config, run_single_experiment

N_UNIQUE_B = 1000
N_LAYERS = 4
N_EVAL = 512
BATCH_SIZE = 128
DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")

HARD_STOP_PER_RUN_SECONDS = 2 * 3600
HARD_STOP_TOTAL_SECONDS = 4 * 3600

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "followup" / "item1"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def train_run(exp_name, task, k, seed, max_steps=5000):
    # Both tasks are easy:
    #   b_to_a K=1: 1000 pairs, 8 batches/epoch -> ~625 epochs at 5k steps.
    #   az_to_b K=10: 10000 ex, 80 batches/epoch -> ~60 epochs at 5k steps.
    # The trainer's early-stop is gated on candidate_eval which is only on
    # for bz_to_a, so we cannot rely on it firing. 5k steps is enough for
    # convergence on these trivial variants.
    cfg = make_config(
        experiment_name=exp_name,
        task=task, k=k, seed=seed, n_unique_b=N_UNIQUE_B,
        max_steps=max_steps, checkpoint_every=max_steps,  # final only
        eval_every=500, early_stop_frac=0.005,
    )
    torch.manual_seed(seed); np.random.seed(seed)
    mapping_data = generate_mappings(
        n_unique_b=N_UNIQUE_B, k=k, b_length=6, a_length=4, z_length=2,
        vocab_chars="abcdefghijklmnopqrstuvwxyz0123456789",
        seed=seed, task=task,
        enforce_unique_a_first_char_per_b=(k > 1),
        disambiguation_prefix_length=1,
    )
    t0 = time.time()
    model, history, _, tokenizer = run_single_experiment(cfg, mapping_data=mapping_data)
    elapsed = time.time() - t0
    final_loss = (history.get("train_losses", []) or [math.nan])[-1]
    return {
        "exp_name": exp_name, "task": task, "k": k, "seed": seed,
        "final_loss": float(final_loss),
        "final_step": history.get("steps", [0])[-1],
        "elapsed_seconds": elapsed,
        "cfg": cfg, "tokenizer": tokenizer, "mapping_data": mapping_data,
        "model": model,
    }


def build_eval_loader(mapping_data, tokenizer, seed, task):
    ds = DisambiguationDataset(
        mapping_data=mapping_data, tokenizer=tokenizer,
        split="train", probe_fraction=0.0, seed=seed, task=task,
    )
    n = min(N_EVAL, len(ds.tokenized))
    ds.tokenized = ds.tokenized[:n]; ds.examples = ds.examples[:n]
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
    abl = np.array(ablation)
    le = np.array(lens)
    n_abl = abl / (np.max(np.abs(abl)) + 1e-9)
    n_lens = le / (np.max(le) + 1e-9)
    rho, _ = stats.spearmanr(n_abl, n_lens)
    xs = np.linspace(0, 1, len(abl))
    area = float(np.trapz(np.abs(n_abl - n_lens), xs))
    ratio = float(abl[0] / max(abl[2], 1e-9))
    return {"spearman_rho": float(rho), "area": area, "l0_over_l2_ratio": ratio}


def evaluate_run(run):
    model = run["model"].to(DEVICE).eval()
    loader = build_eval_loader(run["mapping_data"], run["tokenizer"],
                               run["seed"], run["task"])
    bl = baseline_loss(model, loader)
    abl = [layer_ablation_delta(model, loader, L, bl) for L in range(N_LAYERS)]
    le = [lens_Pcorrect(model, loader, L) for L in range(N_LAYERS)]
    m = primary_metrics(abl, le)
    return {
        "seed": run["seed"], "task": run["task"], "k": run["k"],
        "final_step": run["final_step"], "final_loss": run["final_loss"],
        "elapsed_seconds": run["elapsed_seconds"],
        "baseline_loss": bl,
        "ablation_per_layer": abl,
        "lens_per_layer": le,
        **m,
        "converged": run["final_loss"] < 1.0,
    }


def cleanup(run):
    del run["model"]
    if DEVICE == "mps":
        try: torch.mps.empty_cache()
        except Exception: pass
    elif DEVICE == "cuda":
        torch.cuda.empty_cache()


def main():
    overall_start = time.time()

    jobs = [
        # Control: b_to_a, K=1, 4 seeds
        ("b_to_a", 1, 42),
        ("b_to_a", 1, 123),
        ("b_to_a", 1, 456),
        ("b_to_a", 1, 789),
        # Perturbation: az_to_b, K=10, 2 seeds
        ("az_to_b", 10, 42),
        ("az_to_b", 10, 123),
    ]

    all_results = []
    for task, k, seed in jobs:
        exp_name = f"item1_{task}_k{k}_s{seed}"
        print(f"\n=== {exp_name} ===")
        per_run_start = time.time()
        try:
            run = train_run(exp_name, task, k, seed, max_steps=5000)
        except Exception as e:
            print(f"[ERROR] training failed: {e}")
            all_results.append({
                "seed": seed, "task": task, "k": k,
                "error": str(e), "converged": False,
            })
            continue

        if run["elapsed_seconds"] > HARD_STOP_PER_RUN_SECONDS:
            print(f"[HARD STOP] {exp_name} took {run['elapsed_seconds']:.0f}s > "
                  f"{HARD_STOP_PER_RUN_SECONDS}s limit")
            cleanup(run); break

        print(f"  final_step={run['final_step']}  "
              f"final_loss={run['final_loss']:.4f}  "
              f"elapsed={run['elapsed_seconds']:.0f}s")
        result = evaluate_run(run)
        cleanup(run)
        all_results.append(result)
        print(f"  abl={[f'{v:.3f}' for v in result['ablation_per_layer']]}")
        print(f"  lens={[f'{v:.3f}' for v in result['lens_per_layer']]}")
        print(f"  ρ={result['spearman_rho']:+.3f}  area={result['area']:.3f}  "
              f"L0/L2={result['l0_over_l2_ratio']:.2f}")

        if time.time() - overall_start > HARD_STOP_TOTAL_SECONDS:
            print(f"[HARD STOP] total elapsed > {HARD_STOP_TOTAL_SECONDS}s")
            break

    # Aggregate
    def aggregate(subset):
        if not subset:
            return None
        rhos = [r["spearman_rho"] for r in subset if "spearman_rho" in r]
        areas = [r["area"] for r in subset if "area" in r]
        ratios = [r["l0_over_l2_ratio"] for r in subset if "l0_over_l2_ratio" in r]
        return {
            "n_runs": len(subset),
            "n_converged": sum(1 for r in subset if r.get("converged", False)),
            "mean_spearman": float(np.mean(rhos)) if rhos else None,
            "std_spearman": float(np.std(rhos, ddof=0)) if rhos else None,
            "mean_area": float(np.mean(areas)) if areas else None,
            "std_area": float(np.std(areas, ddof=0)) if areas else None,
            "mean_ratio": float(np.mean(ratios)) if ratios else None,
            "std_ratio": float(np.std(ratios, ddof=0)) if ratios else None,
        }

    control = [r for r in all_results if r.get("task") == "b_to_a"]
    perturb = [r for r in all_results if r.get("task") == "az_to_b"]
    ctrl_agg = aggregate(control)
    pert_agg = aggregate(perturb)

    # Pre-committed rule
    decision = {"rule": "control area < 0.15 AND control Spearman > -0.3"}
    if ctrl_agg is not None:
        ca = ctrl_agg["mean_area"]
        cr = ctrl_agg["mean_spearman"]
        decision.update({
            "control_area": ca, "control_spearman": cr,
            "control_passes_area": (ca is not None and ca < 0.15),
            "control_passes_spearman": (cr is not None and cr > -0.3),
        })
        decision["control_passes"] = (
            decision.get("control_passes_area", False)
            and decision.get("control_passes_spearman", False)
        )
        decision["verdict"] = (
            "scope_claim_earned" if decision["control_passes"]
            else "scope_claim_dropped"
        )
    else:
        decision["verdict"] = "scope_claim_dropped"
        decision["note"] = "control runs did not complete"
    if pert_agg is not None:
        decision["perturbation_area"] = pert_agg["mean_area"]
        decision["perturbation_spearman"] = pert_agg["mean_spearman"]

    # Primary reference from Block 2 K=10 seed 42
    try:
        with open(OUT_DIR.parent / "block2" / "block2_gap_metrics.json") as f:
            b2 = json.load(f)
        primary = b2["per_K"]["10"]
        decision["primary_reference_k10_seed42"] = {
            "area": primary["primary"]["area_necessity_sufficiency"],
            "spearman": primary["primary"]["spearman_abl_lens"],
            "l0_over_l2_ratio": primary["supplementary"]["ratio_L0_L2"],
        }
    except Exception as e:
        decision["primary_reference_k10_seed42"] = None

    output = {
        "config": {"device": DEVICE, "n_eval": N_EVAL, "max_steps": 30000,
                   "batch_size": BATCH_SIZE},
        "condition_1_control": {
            "task": "b_to_a, K=1",
            "seeds": [42, 123, 456, 789],
            "per_seed": control,
            "aggregate": ctrl_agg,
        },
        "condition_2_perturbation": {
            "task": "az_to_b, K=10",
            "seeds": [42, 123],
            "per_seed": perturb,
            "aggregate": pert_agg,
        },
        "decision": decision,
    }
    with open(OUT_DIR / "item1_scope.json", "w") as f:
        json.dump(output, f, indent=2)

    # Summary markdown
    lines = ["# Item 1: scope-conditions summary\n"]
    lines.append(f"- Verdict: **{decision.get('verdict')}**")
    lines.append(f"- Rule: `{decision.get('rule')}`")
    if ctrl_agg:
        lines.append(f"- Control mean area: {ctrl_agg['mean_area']:.3f} "
                     f"± {ctrl_agg['std_area']:.3f}")
        lines.append(f"- Control mean Spearman ρ: "
                     f"{ctrl_agg['mean_spearman']:+.3f} ± {ctrl_agg['std_spearman']:.3f}")
    if pert_agg:
        lines.append(f"- Perturbation mean area: {pert_agg['mean_area']:.3f} "
                     f"± {pert_agg['std_area']:.3f}")
        lines.append(f"- Perturbation mean Spearman ρ: "
                     f"{pert_agg['mean_spearman']:+.3f} ± {pert_agg['std_spearman']:.3f}")
    if decision.get("primary_reference_k10_seed42"):
        pr = decision["primary_reference_k10_seed42"]
        lines.append(f"- Primary (K=10, seed 42) reference: area {pr['area']:.3f}, "
                     f"ρ {pr['spearman']:+.2f}, L0/L2 ratio {pr['l0_over_l2_ratio']:.2f}")
    lines.append("")
    lines.append("## Per-seed")
    for block_name, block in [("control (b_to_a, K=1)", control),
                               ("perturbation (az_to_b, K=10)", perturb)]:
        lines.append(f"\n### {block_name}")
        for r in block:
            if "error" in r:
                lines.append(f"- seed {r['seed']}: ERROR ({r['error']})")
                continue
            lines.append(f"- seed {r['seed']}: final_loss={r['final_loss']:.3f}, "
                         f"abl={[round(v,3) for v in r['ablation_per_layer']]}, "
                         f"lens={[round(v,3) for v in r['lens_per_layer']]}, "
                         f"ρ={r['spearman_rho']:+.2f}, area={r['area']:.3f}")
    with open(OUT_DIR / "item1_summary.md", "w") as f:
        f.write("\n".join(lines) + "\n")

    # Figure
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), dpi=150, facecolor="white",
                                 sharey=False)
        pr = decision.get("primary_reference_k10_seed42")
        primary_abl = None; primary_lens = None
        if pr:
            p = b2["per_K"]["10"]["profiles"]
            primary_abl = p["ablation"]; primary_lens = p["lens"]

        def plot_panel(ax, title, abl, lens, area):
            layers = np.arange(len(abl))
            n_abl = np.array(abl) / (max(abs(np.array(abl)).max(), 1e-9))
            n_lens = np.array(lens) / (max(np.array(lens).max(), 1e-9))
            ax.plot(layers, n_abl, "o-", color="#C4452D", linewidth=1.8,
                    label="necessity (abl Δ, norm.)")
            ax.plot(layers, n_lens, "s-", color="#3B6FB6", linewidth=1.8,
                    label="sufficiency (lens, norm.)")
            ax.fill_between(layers, n_abl, n_lens, color="gray", alpha=0.18,
                            label=f"area = {area:.2f}")
            ax.set_xticks(layers); ax.set_xticklabels([f"L{L}" for L in layers])
            ax.set_ylim(-0.05, 1.08)
            ax.set_title(title, fontsize=10)
            ax.legend(fontsize=7, frameon=False, loc="center left")
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

        if primary_abl is not None:
            plot_panel(axes[0], f"Primary (K=10, seed 42)\narea={pr['area']:.2f}  ρ={pr['spearman']:+.2f}",
                       primary_abl, primary_lens, pr["area"])
        else:
            axes[0].set_title("Primary reference unavailable")

        if control and "ablation_per_layer" in control[0]:
            # Mean profiles
            abls = np.array([r["ablation_per_layer"] for r in control
                             if "ablation_per_layer" in r]).mean(axis=0)
            lenss = np.array([r["lens_per_layer"] for r in control
                              if "lens_per_layer" in r]).mean(axis=0)
            plot_panel(axes[1], f"Control (b_to_a, K=1, 4 seeds)\narea={ctrl_agg['mean_area']:.2f}  ρ={ctrl_agg['mean_spearman']:+.2f}",
                       abls.tolist(), lenss.tolist(), ctrl_agg["mean_area"])
        if perturb and "ablation_per_layer" in perturb[0]:
            abls = np.array([r["ablation_per_layer"] for r in perturb
                             if "ablation_per_layer" in r]).mean(axis=0)
            lenss = np.array([r["lens_per_layer"] for r in perturb
                              if "lens_per_layer" in r]).mean(axis=0)
            plot_panel(axes[2], f"Perturbation (az_to_b, 2 seeds)\narea={pert_agg['mean_area']:.2f}  ρ={pert_agg['mean_spearman']:+.2f}",
                       abls.tolist(), lenss.tolist(), pert_agg["mean_area"])
        fig.suptitle(f"Item 1 scope conditions — verdict: {decision.get('verdict')}",
                     fontsize=11, y=1.02)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "item1_figure.png", dpi=180, facecolor="white",
                    bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"[warn] figure generation failed: {e}")

    # Stdout summary
    print("\n========== ITEM 1 SUMMARY ==========")
    print(f"Verdict: {decision.get('verdict')}")
    if ctrl_agg:
        print(f"Control (b_to_a, K=1, {ctrl_agg['n_runs']} seeds, "
              f"{ctrl_agg['n_converged']} converged): "
              f"area = {ctrl_agg['mean_area']:.3f} ± {ctrl_agg['std_area']:.3f}, "
              f"ρ = {ctrl_agg['mean_spearman']:+.3f}")
    if pert_agg:
        print(f"Perturbation (az_to_b, K=10, {pert_agg['n_runs']} seeds): "
              f"area = {pert_agg['mean_area']:.3f} ± {pert_agg['std_area']:.3f}, "
              f"ρ = {pert_agg['mean_spearman']:+.3f}")
    if decision.get("primary_reference_k10_seed42"):
        pr = decision["primary_reference_k10_seed42"]
        print(f"Primary (K=10, seed 42) reference: area {pr['area']:.3f}, "
              f"ρ {pr['spearman']:+.2f}")
    print(f"Total elapsed: {(time.time()-overall_start)/60:.1f} min")


if __name__ == "__main__":
    main()
