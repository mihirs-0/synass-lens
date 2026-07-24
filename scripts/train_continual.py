#!/usr/bin/env python
"""
Training script for continual learning experiments.

Loads a converged checkpoint, applies a distribution shift (reassignment,
K-expansion, or K-contraction), and continues training while measuring
gradient dissipation, candidate loss, and forgetting dynamics.

Usage:
    python scripts/train_continual.py \
        --base-experiment landauer_k20 \
        --variant reassign \
        --fraction 0.5 \
        --reassign-seed 137 \
        --name continual_reassign_f0.5 \
        --max-steps 30000 \
        --lr 1e-3 --bs 128 \
        --scheduler constant \
        --seed 42
"""

import sys
import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import (
    CharTokenizer,
    MappingData,
    generate_mappings,
    DisambiguationDataset,
    collate_fn,
    create_tokenizer_from_config,
)
from src.data.continual import (
    reassign_mappings,
    expand_k,
    contract_k,
    compute_mapping_divergence,
    mappings_to_examples,
    verify_reassignment,
)
from src.model import create_model_from_config
from src.training.trainer import compute_loss, shuffle_z_in_batch, get_lr_scheduler
from src.training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    list_checkpoints,
    get_checkpoint_dir,
)
from src.analysis.candidate_eval import run_candidate_eval


def _to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


def load_base_config(base_experiment: str, output_dir: str) -> SimpleNamespace:
    config_path = Path(output_dir) / base_experiment / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Base experiment config not found: {config_path}")
    with open(config_path, "r") as f:
        return _to_namespace(yaml.safe_load(f))


def find_best_checkpoint(checkpoint_dir: Path) -> int:
    """Return the latest checkpoint step."""
    steps = list_checkpoints(checkpoint_dir)
    if not steps:
        raise FileNotFoundError(f"No checkpoints in {checkpoint_dir}")
    return steps[-1]


def generate_original_mappings(cfg: SimpleNamespace) -> MappingData:
    """Regenerate the exact mappings used in the base experiment."""
    return generate_mappings(
        n_unique_b=cfg.data.n_unique_b,
        k=cfg.data.k,
        b_length=cfg.data.b_length,
        a_length=cfg.data.a_length,
        z_length=cfg.data.z_length,
        vocab_chars=cfg.data.vocab_chars,
        seed=cfg.experiment.seed,
        task=cfg.data.task,
        enforce_unique_a_first_char_per_b=getattr(
            cfg.data, "enforce_unique_a_first_char_per_b", False
        ),
        disambiguation_prefix_length=int(
            getattr(cfg.data, "disambiguation_prefix_length", 1)
        ),
    )


def build_dataset_from_mappings(
    mappings_dict, original_mapping_data, tokenizer, cfg, split="train"
):
    """Build a DisambiguationDataset from a modified mappings dict."""
    examples = mappings_to_examples(mappings_dict)
    k = len(next(iter(mappings_dict.values())))
    n_unique_a = len({a for pairs in mappings_dict.values() for _, a in pairs})

    mapping_data = MappingData(
        mappings=mappings_dict,
        examples=examples,
        n_unique_b=len(mappings_dict),
        n_unique_a=n_unique_a,
        k=k,
        task=original_mapping_data.task,
    )

    dataset = DisambiguationDataset(
        mapping_data=mapping_data,
        tokenizer=tokenizer,
        split=split,
        probe_fraction=0.0,
        seed=cfg.experiment.seed,
        task=cfg.data.task,
        split_by_base=False,
        label_noise_prob=0.0,
    )
    return dataset, mapping_data


def evaluate_candidate_loss(model, tokenizer, mapping_data, device, seed=42, n=32):
    """Quick candidate eval wrapper."""
    model.eval()
    with torch.no_grad():
        result = run_candidate_eval(
            model=model,
            tokenizer=tokenizer,
            mapping_data=mapping_data,
            n_examples=n,
            task=mapping_data.task,
            device=device,
            seed=seed,
        )
    return result["candidate_loss"]


def split_mapping_data(mapping_data, changed_bs):
    """Split MappingData into changed and unchanged subsets."""
    changed_mappings = {b: v for b, v in mapping_data.mappings.items() if b in changed_bs}
    unchanged_mappings = {b: v for b, v in mapping_data.mappings.items() if b not in changed_bs}

    def _make_sub(mappings_dict):
        if not mappings_dict:
            return None
        examples = mappings_to_examples(mappings_dict)
        k = len(next(iter(mappings_dict.values())))
        n_unique_a = len({a for pairs in mappings_dict.values() for _, a in pairs})
        return MappingData(
            mappings=mappings_dict, examples=examples,
            n_unique_b=len(mappings_dict), n_unique_a=n_unique_a,
            k=k, task=mapping_data.task,
        )

    return _make_sub(changed_mappings), _make_sub(unchanged_mappings)


def main():
    parser = argparse.ArgumentParser(
        description="Continual learning training for Landauer experiments"
    )
    parser.add_argument(
        "--base-experiment", type=str, required=True,
        help="Name of the converged base experiment to load",
    )
    parser.add_argument(
        "--variant", type=str, required=True,
        choices=["reassign", "expand", "contract"],
        help="Type of distribution shift",
    )
    parser.add_argument("--fraction", type=float, default=0.5,
                        help="Reassignment fraction (variant=reassign)")
    parser.add_argument("--reassign-seed", type=int, default=137,
                        help="Seed for reassignment randomness")
    parser.add_argument("--target-k", type=int, default=20,
                        help="Target K for expand/contract variants")
    parser.add_argument("--name", type=str, required=True,
                        help="Experiment name for outputs")
    parser.add_argument("--max-steps", type=int, default=30000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--scheduler", type=str, default="constant",
                        choices=["constant", "cosine"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--n-eval-examples", type=int, default=32)
    parser.add_argument(
        "--load-optimizer-state", action="store_true",
        help="Load optimizer state from base checkpoint (ablation)",
    )
    args = parser.parse_args()

    device = _select_device()
    print(f"Using device: {device}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # --- Load base experiment ---
    print(f"\n{'='*60}")
    print(f"Continual Learning Experiment: {args.name}")
    print(f"Base experiment: {args.base_experiment}")
    print(f"Variant: {args.variant}")
    if args.variant == "reassign":
        print(f"Fraction: {args.fraction}")
        print(f"Reassign seed: {args.reassign_seed}")
    elif args.variant in ("expand", "contract"):
        print(f"Target K: {args.target_k}")
    print(f"Load optimizer state: {args.load_optimizer_state}")
    print(f"{'='*60}\n")

    base_cfg = load_base_config(args.base_experiment, args.output_dir)
    tokenizer = create_tokenizer_from_config(base_cfg)

    # Regenerate original mappings from the base experiment's config
    original_mapping_data = generate_original_mappings(base_cfg)
    original_mappings = original_mapping_data.mappings
    k_original = original_mapping_data.k

    print(f"Original mappings: {len(original_mappings)} B groups, K={k_original}")

    # --- Apply distribution shift ---
    if args.variant == "reassign":
        new_mappings, reassigned_bs = reassign_mappings(
            original_mappings, args.fraction, seed=args.reassign_seed
        )
        verify_reassignment(
            original_mappings, new_mappings, reassigned_bs, args.fraction
        )
        divergence = compute_mapping_divergence(original_mappings, new_mappings)
        new_k = k_original

    elif args.variant == "expand":
        new_mappings = expand_k(
            original_mappings,
            new_k=args.target_k,
            z_length=base_cfg.data.z_length,
            a_length=base_cfg.data.a_length,
            vocab_chars=base_cfg.data.vocab_chars,
            seed=args.reassign_seed,
        )
        reassigned_bs = set()
        divergence = compute_mapping_divergence(original_mappings, new_mappings)
        new_k = args.target_k

    elif args.variant == "contract":
        new_mappings = contract_k(
            original_mappings, new_k=args.target_k, seed=args.reassign_seed
        )
        reassigned_bs = set()
        divergence = compute_mapping_divergence(original_mappings, new_mappings)
        new_k = args.target_k

    print(f"\nDivergence stats:")
    print(f"  Changed pairs: {divergence['changed_pairs']}/{divergence['total_pairs']}"
          f" = {divergence['fraction_changed']:.3f}")
    print(f"  Groups changed: {divergence['n_groups_changed']}/{divergence['n_groups_total']}")

    # --- Build datasets ---
    # New (post-shift) dataset for training
    new_dataset, new_mapping_data = build_dataset_from_mappings(
        new_mappings, original_mapping_data, tokenizer, base_cfg
    )
    # Old (pre-shift) dataset for forgetting measurement
    old_dataset, old_mapping_data = build_dataset_from_mappings(
        original_mappings, original_mapping_data, tokenizer, base_cfg
    )

    new_loader = DataLoader(
        new_dataset, batch_size=args.bs, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    old_loader = DataLoader(
        old_dataset, batch_size=args.bs, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    print(f"New dataset size: {len(new_dataset)}")
    print(f"Old dataset size: {len(old_dataset)}")

    # --- Create model and load checkpoint ---
    model = create_model_from_config(base_cfg, tokenizer)
    checkpoint_dir = Path(args.output_dir) / args.base_experiment / "checkpoints"
    best_step = find_best_checkpoint(checkpoint_dir)
    print(f"\nLoading base checkpoint from step {best_step}")

    # Set up optimizer before loading (needed for load_checkpoint interface)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01,
    )

    if args.load_optimizer_state:
        load_checkpoint(model, optimizer, checkpoint_dir, best_step)
        print("Loaded model weights AND optimizer state")
        # Update LR in loaded optimizer to match current args
        for pg in optimizer.param_groups:
            pg["lr"] = args.lr
    else:
        load_checkpoint(model, None, checkpoint_dir, best_step)
        print("Loaded model weights only (fresh optimizer)")

    model.to(device)

    # --- Measure initial performance ---
    print("\nInitial performance (before continual training):")
    initial_new_candidate_loss = evaluate_candidate_loss(
        model, tokenizer, new_mapping_data, device, seed=args.seed,
        n=args.n_eval_examples,
    )
    initial_old_candidate_loss = evaluate_candidate_loss(
        model, tokenizer, old_mapping_data, device, seed=args.seed,
        n=args.n_eval_examples,
    )
    print(f"  New data candidate_loss: {initial_new_candidate_loss:.4f}"
          f" (high = model is confidently wrong on reassigned groups)")
    print(f"  Old data candidate_loss: {initial_old_candidate_loss:.4f}"
          f" (should be ~0 if base converged)")

    # --- Setup output directory ---
    output_dir = Path(args.output_dir)
    exp_dir = output_dir / args.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    new_checkpoint_dir = get_checkpoint_dir(str(output_dir), args.name)

    # Save config
    config_dict = {
        "experiment": {"name": args.name, "seed": args.seed},
        "data": {
            "n_unique_b": base_cfg.data.n_unique_b,
            "k": new_k,
            "task": base_cfg.data.task,
            "b_length": base_cfg.data.b_length,
            "a_length": base_cfg.data.a_length,
            "z_length": base_cfg.data.z_length,
            "vocab_chars": base_cfg.data.vocab_chars,
            "probe_fraction": 0.0,
            "split_by_base": False,
            "enforce_unique_a_first_char_per_b": getattr(
                base_cfg.data, "enforce_unique_a_first_char_per_b", False
            ),
            "disambiguation_prefix_length": int(
                getattr(base_cfg.data, "disambiguation_prefix_length", 1)
            ),
            "label_noise_prob": 0.0,
        },
        "tokenizer": {
            "pad_token": base_cfg.tokenizer.pad_token,
            "bos_token": base_cfg.tokenizer.bos_token,
            "eos_token": base_cfg.tokenizer.eos_token,
            "sep_token": base_cfg.tokenizer.sep_token,
        },
        "model": {
            "n_layers": base_cfg.model.n_layers,
            "n_heads": base_cfg.model.n_heads,
            "d_model": base_cfg.model.d_model,
            "d_head": base_cfg.model.d_head,
            "d_mlp": base_cfg.model.d_mlp,
            "act_fn": base_cfg.model.act_fn,
        },
        "training": {
            "batch_size": args.bs,
            "learning_rate": args.lr,
            "weight_decay": 0.01,
            "max_steps": args.max_steps,
            "warmup_steps": 0,
            "scheduler": args.scheduler,
            "checkpoint_every": args.checkpoint_every,
            "eval_every": args.eval_every,
        },
        "output": {"base_dir": args.output_dir},
        "continual": {
            "base_experiment": args.base_experiment,
            "base_checkpoint_step": best_step,
            "variant": args.variant,
            "fraction": args.fraction if args.variant == "reassign" else None,
            "reassign_seed": args.reassign_seed,
            "target_k": args.target_k if args.variant in ("expand", "contract") else None,
            "load_optimizer_state": args.load_optimizer_state,
            "original_k": k_original,
            "new_k": new_k,
            "divergence": {
                "changed_pairs": divergence["changed_pairs"],
                "total_pairs": divergence["total_pairs"],
                "fraction_changed": divergence["fraction_changed"],
                "n_groups_changed": divergence["n_groups_changed"],
            },
            "initial_new_candidate_loss": initial_new_candidate_loss,
            "initial_old_candidate_loss": initial_old_candidate_loss,
        },
    }

    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    # Save reassigned B groups for downstream split eval
    if reassigned_bs:
        config_dict["continual"]["reassigned_bs"] = sorted(reassigned_bs)
        with open(exp_dir / "reassigned_bs.json", "w") as f:
            json.dump(sorted(reassigned_bs), f)

    # Save divergence details
    with open(exp_dir / "divergence.json", "w") as f:
        serializable_divergence = {
            k: v for k, v in divergence.items() if k != "per_group_changes"
        }
        serializable_divergence["n_per_group_changes"] = len(
            divergence.get("per_group_changes", {})
        )
        json.dump(serializable_divergence, f, indent=2)

    # --- Setup scheduler ---
    scheduler = get_lr_scheduler(optimizer, 0, args.max_steps, args.scheduler)

    # --- Build split eval data (changed vs unchanged B groups) ---
    old_changed_md, old_unchanged_md = None, None
    if reassigned_bs:
        old_changed_md, old_unchanged_md = split_mapping_data(
            old_mapping_data, reassigned_bs
        )
        if old_changed_md:
            print(f"Split eval: {old_changed_md.n_unique_b} changed, "
                  f"{old_unchanged_md.n_unique_b if old_unchanged_md else 0} unchanged B groups")

    # --- Training loop ---
    history = {
        "steps": [],
        "train_loss": [],
        "train_accuracy": [],
        "first_target_loss": [],
        "loss_z_shuffled": [],
        "new_candidate_loss": [],
        "old_candidate_loss": [],
        "old_unchanged_candidate_loss": [],
        "old_changed_candidate_loss": [],
        "gradient_norm": [],
        "new_z_gap": [],
        "old_z_gap": [],
    }

    model.train()
    step = 0
    epoch = 0
    running_loss = 0.0
    running_acc = 0.0
    running_first_loss = 0.0
    running_grad_norm = 0.0
    n_batches = 0

    pbar = tqdm(total=args.max_steps, desc="Continual Training")

    while step < args.max_steps:
        epoch += 1
        for batch in new_loader:
            if step >= args.max_steps:
                break

            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            optimizer.zero_grad()
            loss, train_acc, first_target_loss = compute_loss(model, batch)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            running_acc += train_acc
            running_first_loss += first_target_loss
            running_grad_norm += float(grad_norm) ** 2
            n_batches += 1
            step += 1

            # --- Periodic evaluation ---
            if step % args.eval_every == 0:
                avg_train_loss = running_loss / n_batches
                avg_train_acc = running_acc / n_batches
                avg_first_loss = running_first_loss / n_batches
                avg_grad_norm_sq = running_grad_norm / n_batches

                # Z-shuffle diagnostic on current batch
                with torch.no_grad():
                    model.eval()
                    shuffled = shuffle_z_in_batch(batch)
                    _, _, shuffled_first = compute_loss(model, shuffled)
                    model.train()

                # Candidate eval on new and old data
                new_cand_loss = evaluate_candidate_loss(
                    model, tokenizer, new_mapping_data, device,
                    seed=args.seed, n=args.n_eval_examples,
                )
                old_cand_loss = evaluate_candidate_loss(
                    model, tokenizer, old_mapping_data, device,
                    seed=args.seed, n=args.n_eval_examples,
                )

                # Split eval: changed vs unchanged B groups
                old_unchanged_cand = 0.0
                old_changed_cand = 0.0
                if old_unchanged_md is not None:
                    old_unchanged_cand = evaluate_candidate_loss(
                        model, tokenizer, old_unchanged_md, device,
                        seed=args.seed, n=min(args.n_eval_examples, old_unchanged_md.n_unique_b),
                    )
                if old_changed_md is not None:
                    old_changed_cand = evaluate_candidate_loss(
                        model, tokenizer, old_changed_md, device,
                        seed=args.seed, n=min(args.n_eval_examples, old_changed_md.n_unique_b),
                    )

                # Z-gap on old data
                with torch.no_grad():
                    model.eval()
                    old_batch = next(iter(old_loader))
                    old_batch = {
                        k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in old_batch.items()
                    }
                    _, _, old_clean_first = compute_loss(model, old_batch)
                    old_shuffled = shuffle_z_in_batch(old_batch)
                    _, _, old_shuf_first = compute_loss(model, old_shuffled)
                    old_z_gap = old_shuf_first - old_clean_first
                    model.train()

                new_z_gap = shuffled_first - first_target_loss

                history["steps"].append(step)
                history["train_loss"].append(avg_train_loss)
                history["train_accuracy"].append(avg_train_acc)
                history["first_target_loss"].append(avg_first_loss)
                history["loss_z_shuffled"].append(float(shuffled_first))
                history["new_candidate_loss"].append(float(new_cand_loss))
                history["old_candidate_loss"].append(float(old_cand_loss))
                history["old_unchanged_candidate_loss"].append(float(old_unchanged_cand))
                history["old_changed_candidate_loss"].append(float(old_changed_cand))
                history["gradient_norm"].append(float(avg_grad_norm_sq))
                history["new_z_gap"].append(float(new_z_gap))
                history["old_z_gap"].append(float(old_z_gap))

                pbar.set_postfix({
                    "loss": f"{avg_train_loss:.4f}",
                    "new_cand": f"{new_cand_loss:.4f}",
                    "old_cand": f"{old_cand_loss:.4f}",
                    "acc": f"{avg_train_acc:.2%}",
                })

                running_loss = 0.0
                running_acc = 0.0
                running_first_loss = 0.0
                running_grad_norm = 0.0
                n_batches = 0

            # --- Checkpoint ---
            if step % args.checkpoint_every == 0:
                t_loss = history["train_loss"][-1] if history["train_loss"] else loss.item()
                t_acc = history["train_accuracy"][-1] if history["train_accuracy"] else train_acc

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    train_loss=t_loss,
                    train_accuracy=t_acc,
                    checkpoint_dir=new_checkpoint_dir,
                )

                # Save history periodically
                with open(exp_dir / "training_history.json", "w") as f:
                    json.dump(history, f, indent=2)

            pbar.update(1)

    pbar.close()

    # --- Final checkpoint and history ---
    final_loss = history["train_loss"][-1] if history["train_loss"] else 0.0
    final_acc = history["train_accuracy"][-1] if history["train_accuracy"] else 0.0
    save_checkpoint(
        model=model, optimizer=optimizer, step=step,
        train_loss=final_loss, train_accuracy=final_acc,
        checkpoint_dir=new_checkpoint_dir,
    )

    with open(exp_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"Continual learning training complete: {args.name}")
    print(f"  Final new_candidate_loss: {history['new_candidate_loss'][-1]:.4f}")
    print(f"  Final old_candidate_loss: {history['old_candidate_loss'][-1]:.4f}")
    print(f"  Initial new_candidate_loss: {initial_new_candidate_loss:.4f}")
    print(f"  Outputs saved to: {exp_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
