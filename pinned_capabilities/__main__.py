"""Small administrative CLI; experiment runners are added gate by gate."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .config import MBCExperimentConfig, MetricConfig, ProtocolConfig
from .experiment import JSONLWriter, MBCExperiment
from .gate0 import deep_linear_control
from .manifest import freeze_manifest
from .references import empirical_constant_machine
from .reference_run import run_reference_ensemble


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m pinned_capabilities")
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze", help="freeze the default protocol manifest")
    freeze.add_argument("path", type=Path)
    freeze.add_argument("--repo", type=Path, default=Path.cwd())
    show = subparsers.add_parser("show-config", help="print the default protocol configuration")
    deep_linear = subparsers.add_parser(
        "deep-linear", help="run the analytic Gate 0 positive control"
    )
    deep_linear.add_argument(
        "--learning-rates", type=float, nargs="+", default=(0.5, 0.9, 1.0, 1.1, 1.5)
    )
    smoke = subparsers.add_parser("smoke", help="run a tiny end-to-end MBC suite check")
    smoke.add_argument("--steps", type=int, default=4)
    smoke.add_argument(
        "--output", type=Path, default=Path("pinned_capabilities/results/smoke")
    )
    reference_smoke = subparsers.add_parser(
        "reference-smoke", help="exercise reference generation on a small non-oracle system"
    )
    reference_smoke.add_argument(
        "--output", type=Path, default=Path("pinned_capabilities/results/reference_smoke")
    )
    reference_smoke.add_argument("--max-steps", type=int, default=2_000)
    reference = subparsers.add_parser(
        "reference", help="run a manifest-frozen full behavioral reference ensemble"
    )
    reference.add_argument("--seeds", type=int, nargs="+", required=True)
    reference.add_argument("--max-steps", type=int, default=40_000)
    reference.add_argument(
        "--output", type=Path, default=Path("pinned_capabilities/results/reference_v1_2_3")
    )
    reference.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    reference.add_argument("--learning-rate", type=float, default=None)
    args = parser.parse_args()
    config = ProtocolConfig()
    if args.command == "freeze":
        manifest = freeze_manifest(config, args.path, repo=args.repo)
        print(json.dumps(manifest, indent=2, sort_keys=True))
    elif args.command == "show-config":
        print(json.dumps(config.to_dict(), indent=2, sort_keys=True))
    elif args.command == "deep-linear":
        print(json.dumps(deep_linear_control(args.learning_rates), indent=2, sort_keys=True))
    elif args.command == "reference":
        experiment_config = replace(config.experiment, device=args.device)
        if args.learning_rate is not None:
            experiment_config = replace(
                experiment_config, learning_rate=args.learning_rate
            )
        frozen = {
            "kind": "behavioral_reference",
            "protocol_version": config.protocol_version,
            "experiment": experiment_config,
            "metric": config.metric,
            "seeds": tuple(args.seeds),
            "acquisition_budget": args.max_steps,
        }
        freeze_manifest(frozen, args.output / "manifest.json", repo=Path.cwd())
        summary = run_reference_ensemble(
            experiment_config,
            config.metric,
            seeds=args.seeds,
            acquisition_budget=args.max_steps,
            output_dir=args.output,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
    elif args.command == "smoke":
        experiment_config = MBCExperimentConfig(
            seed=0,
            n_unique_b=12,
            k=3,
            b_length=3,
            a_length=2,
            z_length=1,
            n_layers=1,
            n_heads=1,
            d_model=16,
            d_head=16,
            d_mlp=32,
            batch_size=6,
            probe_b_count=5,
            device="cpu",
        )
        metric = MetricConfig(
            eval_every=max(1, args.steps), quartets_per_probe=24, probe_seeds=(17_071, 91_237)
        )
        freeze_manifest(
            {"kind": "smoke", "experiment": experiment_config, "metric": metric},
            args.output / "manifest.json",
            repo=Path.cwd(),
        )
        experiment = MBCExperiment(experiment_config, metric)
        writer = JSONLWriter(args.output / "metrics.jsonl")
        initial = experiment.evaluate()
        writer.write({"kind": "initial", **initial})
        experiment.advance(args.steps, writer)
        final = experiment.evaluate()
        writer.write({"kind": "final", **final})
        print(
            json.dumps(
                {
                    "initial": initial,
                    "final": final,
                    "q_star": empirical_constant_machine(
                        experiment.mapping, experiment.tokenizer
                    ),
                    "output": str(args.output),
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        experiment_config = MBCExperimentConfig(
            seed=0,
            n_unique_b=24,
            k=3,
            b_length=3,
            a_length=2,
            z_length=1,
            n_layers=1,
            n_heads=1,
            d_model=32,
            d_head=32,
            d_mlp=64,
            batch_size=24,
            learning_rate=1e-2,
            probe_b_count=10,
            device="cpu",
        )
        metric = MetricConfig(
            eval_every=20,
            quartets_per_probe=64,
            solved_hold_steps=100,
            solved_summary_window=50,
        )
        freeze_manifest(
            {
                "kind": "reference_smoke_not_gate_evidence",
                "experiment": experiment_config,
                "metric": metric,
                "seeds": (101, 103),
                "acquisition_budget": args.max_steps,
            },
            args.output / "manifest.json",
            repo=Path.cwd(),
        )
        summary = run_reference_ensemble(
            experiment_config,
            metric,
            seeds=(101, 103),
            acquisition_budget=args.max_steps,
            output_dir=args.output,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
