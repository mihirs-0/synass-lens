#!/usr/bin/env python
"""
Run candidate-set evaluation across checkpoints for an experiment.
"""

import sys
from pathlib import Path
import argparse
import json
import math
from types import SimpleNamespace

import torch
import yaml
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import create_tokenizer_from_config, create_datasets_from_config, collate_fn
from src.model import create_model_from_config
from src.training.checkpoint import list_checkpoints, load_checkpoint
from src.analysis.candidate_eval import (
    run_candidate_eval,
    compute_z_usage_metrics,
    detect_binding_onset,
)


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


def _to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def _select_steps(all_steps, every_n: int):
    if every_n <= 1:
        return list(all_steps)
    selected = {all_steps[0], all_steps[-1]}
    for idx, step in enumerate(all_steps):
        if idx % every_n == 0:
            selected.add(step)
    return [step for step in all_steps if step in selected]


def main():
    parser = argparse.ArgumentParser(description="Run candidate evaluation on checkpoints")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment name")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Base output directory")
    parser.add_argument("--n-examples", type=int, default=32, help="Examples per checkpoint")
    parser.add_argument("--every-n", type=int, default=1, help="Evaluate every N-th checkpoint")
    parser.add_argument("--gap-threshold", type=float, default=0.5, help="z-gap threshold")
    parser.add_argument("--consecutive", type=int, default=3, help="Consecutive checkpoints for onset")
    args = parser.parse_args()

    experiment_dir = Path(args.output_dir) / args.experiment
    config_path = experiment_dir / "config.yaml"
    if not config_path.exists():
        print(f"Error: Config not found at {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        cfg = _to_namespace(yaml.safe_load(f))

    device = _select_device()
    print(f"Using device: {device}")

    torch.manual_seed(cfg.experiment.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.experiment.seed)

    tokenizer = create_tokenizer_from_config(cfg)
    train_dataset, probe_dataset, mapping_data = create_datasets_from_config(cfg, tokenizer)
    if len(probe_dataset) == 0:
        print("Note: probe_fraction is 0; using train dataset for z-usage batch.")
        probe_dataset = train_dataset

    probe_loader = DataLoader(
        probe_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    try:
        eval_batch = next(iter(probe_loader))
    except StopIteration:
        print("Error: No data available for evaluation batch.")
        sys.exit(1)

    checkpoint_dir = experiment_dir / "checkpoints"
    all_steps = list_checkpoints(checkpoint_dir)
    if not all_steps:
        print(f"Error: No checkpoints found in {checkpoint_dir}")
        sys.exit(1)

    steps = _select_steps(all_steps, args.every_n)
    print(f"Evaluating {len(steps)} checkpoints: {steps[:5]}...{steps[-5:]}")

    results = {
        "steps": [],
        "candidate_loss": [],
        "candidate_accuracy": [],
        "candidate_top3_accuracy": [],
        "loss_clean": [],
        "loss_z_shuffled": [],
        "z_gap": [],
        "mean_correct_log_prob": [],
        "mean_incorrect_log_prob": [],
    }

    for step in steps:
        print(f"Evaluating checkpoint step {step}...")
        model = create_model_from_config(cfg, tokenizer)
        load_checkpoint(model, None, checkpoint_dir, step)
        model.to(device)
        model.eval()

        candidate_metrics = run_candidate_eval(
            model=model,
            tokenizer=tokenizer,
            mapping_data=mapping_data,
            n_examples=args.n_examples,
            task=cfg.data.task,
            device=device,
            seed=cfg.experiment.seed,
        )
        z_metrics = compute_z_usage_metrics(model, eval_batch, device=device)

        results["steps"].append(step)
        results["candidate_loss"].append(candidate_metrics["candidate_loss"])
        results["candidate_accuracy"].append(candidate_metrics["candidate_accuracy"])
        results["candidate_top3_accuracy"].append(candidate_metrics["candidate_top3_accuracy"])
        results["mean_correct_log_prob"].append(candidate_metrics["mean_correct_log_prob"])
        results["mean_incorrect_log_prob"].append(candidate_metrics["mean_incorrect_log_prob"])
        results["loss_clean"].append(z_metrics["loss_clean"])
        results["loss_z_shuffled"].append(z_metrics["loss_z_shuffled"])
        results["z_gap"].append(z_metrics["z_gap"])

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    binding_onset_step = detect_binding_onset(
        results["z_gap"],
        results["steps"],
        gap_threshold=args.gap_threshold,
        consecutive_required=args.consecutive,
    )

    plateau_avg = None
    post_binding_avg = None
    if results["candidate_loss"]:
        if binding_onset_step is None:
            plateau_avg = float(sum(results["candidate_loss"]) / len(results["candidate_loss"]))
        else:
            onset_idx = results["steps"].index(binding_onset_step)
            if onset_idx > 0:
                plateau_avg = float(sum(results["candidate_loss"][:onset_idx]) / onset_idx)
            else:
                plateau_avg = float(results["candidate_loss"][0])
            post_binding_avg = float(
                sum(results["candidate_loss"][onset_idx:]) / len(results["candidate_loss"][onset_idx:])
            )

    output = {
        **results,
        "k": int(mapping_data.k),
        "log_k": float(math.log(mapping_data.k)),
        "n_candidate_examples": int(args.n_examples),
        "binding_onset_step": binding_onset_step,
        "binding_onset_config": {
            "gap_threshold": args.gap_threshold,
            "consecutive_required": args.consecutive,
        },
        "candidate_loss_plateau_avg": plateau_avg,
        "candidate_loss_post_binding_avg": post_binding_avg,
        "config": {
            "learning_rate": float(cfg.training.learning_rate),
            "batch_size": int(cfg.training.batch_size),
            "t_eff": float(cfg.training.learning_rate) / float(cfg.training.batch_size),
            "k": int(cfg.data.k),
            "task": str(cfg.data.task),
            "experiment_name": str(cfg.experiment.name),
        },
    }

    results_path = experiment_dir / "candidate_eval_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)

    print("\nCandidate evaluation summary")
    print("=" * 60)
    print(f"K: {output['k']} | log(K): {output['log_k']:.4f}")
    print(f"Binding onset step: {output['binding_onset_step']}")
    print(f"Plateau avg candidate_loss: {output['candidate_loss_plateau_avg']}")
    print(f"Post-binding avg candidate_loss: {output['candidate_loss_post_binding_avg']}")
    if output["candidate_loss"]:
        print(f"Final candidate_loss: {output['candidate_loss'][-1]:.4f}")
        print(f"Final candidate_accuracy: {output['candidate_accuracy'][-1]:.4f}")
        print(f"Final z_gap: {output['z_gap'][-1]:.4f}")
    print(f"Saved results to: {results_path}")


if __name__ == "__main__":
    main()
