"""Small administrative CLI; experiment runners are added gate by gate."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from .config import MBCExperimentConfig, MetricConfig, ProtocolConfig
from .experiment import JSONLWriter, MBCExperiment
from .gate0 import deep_linear_control
from .gate0_boundary import geometric_erasure_bisection, run_acquisition_branch
from .local_measurement import measure_checkpoint_local_stability
from .manifest import freeze_manifest
from .memory_surgery_run import run_memory_factorial
from .references import empirical_constant_machine, load_reference_bands
from .reference_run import run_reference_ensemble
from .state import StateThresholds
from .state_preparation import prepare_expressed_checkpoint, prepare_suppressed_checkpoint
from .hysteresis_run import run_hysteresis_cycles


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
    prepare = subparsers.add_parser(
        "prepare-suppressed", help="construct a validated high-rate suppressed checkpoint"
    )
    prepare.add_argument("--reference", type=Path, required=True)
    prepare.add_argument("--seed", type=int, required=True)
    prepare.add_argument("--learning-rate", type=float, required=True)
    prepare.add_argument("--max-steps", type=int, default=20_000)
    prepare.add_argument("--save-at-step", type=int, default=None)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    prepare_expressed = subparsers.add_parser(
        "prepare-expressed", help="construct a stably expressed checkpoint at a fixed step"
    )
    prepare_expressed.add_argument("--reference", type=Path, required=True)
    prepare_expressed.add_argument("--seed", type=int, required=True)
    prepare_expressed.add_argument("--learning-rate", type=float, required=True)
    prepare_expressed.add_argument("--save-at-step", type=int, required=True)
    prepare_expressed.add_argument("--output", type=Path, required=True)
    prepare_expressed.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    hysteresis = subparsers.add_parser(
        "hysteresis", help="run transactionally resumable two-cycle Gate 1 sweeps"
    )
    hysteresis.add_argument("--reference", type=Path, required=True)
    hysteresis.add_argument("--snapshot", type=Path, required=True)
    hysteresis.add_argument("--seed", type=int, required=True)
    hysteresis.add_argument("--high-learning-rate", type=float, required=True)
    hysteresis.add_argument("--low-learning-rate", type=float, required=True)
    hysteresis.add_argument("--dwell-multiplier", type=int, default=1)
    hysteresis.add_argument("--output", type=Path, required=True)
    hysteresis.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    local = subparsers.add_parser(
        "local-stability", help="measure registered Gate 0 predictors at one checkpoint"
    )
    local.add_argument("--snapshot", type=Path, required=True)
    local.add_argument("--seed", type=int, required=True)
    local.add_argument("--learning-rate", type=float, required=True)
    local.add_argument("--output", type=Path, required=True)
    local.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    erasure = subparsers.add_parser(
        "erasure-boundary", help="run a manifest-frozen geometric Gate 0 erasure bisection"
    )
    erasure.add_argument("--reference", type=Path, required=True)
    erasure.add_argument("--snapshot", type=Path, required=True)
    erasure.add_argument("--seed", type=int, required=True)
    erasure.add_argument("--lower-learning-rate", type=float, required=True)
    erasure.add_argument("--upper-learning-rate", type=float, required=True)
    erasure.add_argument("--output", type=Path, required=True)
    erasure.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    acquisition = subparsers.add_parser(
        "acquisition", help="run one censored Gate 0 acquisition branch"
    )
    acquisition.add_argument("--reference", type=Path, required=True)
    acquisition.add_argument("--snapshot", type=Path, required=True)
    acquisition.add_argument("--seed", type=int, required=True)
    acquisition.add_argument("--learning-rate", type=float, required=True)
    acquisition.add_argument("--output", type=Path, required=True)
    acquisition.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    memory = subparsers.add_parser(
        "memory-factorial", help="run matched-step weights x optimizer-state surgery"
    )
    memory.add_argument("--reference", type=Path, required=True)
    memory.add_argument("--expressed-snapshot", type=Path, required=True)
    memory.add_argument("--suppressed-snapshot", type=Path, required=True)
    memory.add_argument("--seed", type=int, required=True)
    memory.add_argument("--learning-rate", type=float, required=True)
    memory.add_argument("--challenge-steps", type=int, default=10_000)
    memory.add_argument("--output", type=Path, required=True)
    memory.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
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
    elif args.command == "prepare-suppressed":
        bands = load_reference_bands(args.reference)
        thresholds = StateThresholds(**asdict(config.state))
        experiment_config = replace(
            config.experiment, seed=args.seed, device=args.device
        )
        frozen = {
            "kind": "suppressed_state_preparation",
            "protocol_version": config.protocol_version,
            "experiment": experiment_config,
            "metric": config.metric,
            "state": config.state,
            "reference_path": str(args.reference),
            "learning_rate": args.learning_rate,
            "maximum_steps": args.max_steps,
            "save_at_step": args.save_at_step,
        }
        freeze_manifest(frozen, args.output / "manifest.json", repo=Path.cwd())
        result = prepare_suppressed_checkpoint(
            experiment_config,
            config.metric,
            bands,
            thresholds,
            learning_rate=args.learning_rate,
            maximum_steps=args.max_steps,
            output_dir=args.output,
            save_at_step=args.save_at_step,
        )
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
    elif args.command == "prepare-expressed":
        bands = load_reference_bands(args.reference)
        thresholds = StateThresholds(**asdict(config.state))
        experiment_config = replace(
            config.experiment, seed=args.seed, device=args.device
        )
        frozen = {
            "kind": "expressed_state_preparation",
            "protocol_version": config.protocol_version,
            "experiment": experiment_config,
            "metric": config.metric,
            "state": config.state,
            "reference_path": str(args.reference),
            "learning_rate": args.learning_rate,
            "save_at_step": args.save_at_step,
        }
        freeze_manifest(frozen, args.output / "manifest.json", repo=Path.cwd())
        result = prepare_expressed_checkpoint(
            experiment_config,
            config.metric,
            bands,
            thresholds,
            learning_rate=args.learning_rate,
            save_at_step=args.save_at_step,
            output_dir=args.output,
        )
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
    elif args.command == "hysteresis":
        bands = load_reference_bands(args.reference)
        thresholds = StateThresholds(**asdict(config.state))
        experiment_config = replace(
            config.experiment, seed=args.seed, device=args.device
        )
        frozen = {
            "kind": "gate1_hysteresis",
            "protocol_version": config.protocol_version,
            "experiment": experiment_config,
            "metric": config.metric,
            "state": config.state,
            "gate1": config.gate1,
            "reference_path": str(args.reference),
            "suppressed_snapshot": str(args.snapshot),
            "high_learning_rate": args.high_learning_rate,
            "low_learning_rate": args.low_learning_rate,
            "dwell_multiplier": args.dwell_multiplier,
        }
        freeze_manifest(frozen, args.output / "manifest.json", repo=Path.cwd())
        summary = run_hysteresis_cycles(
            experiment_config,
            config.metric,
            config.gate1,
            bands,
            thresholds,
            args.snapshot,
            high_learning_rate=args.high_learning_rate,
            low_learning_rate=args.low_learning_rate,
            output_dir=args.output,
            dwell_multiplier=args.dwell_multiplier,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
    elif args.command == "local-stability":
        experiment_config = replace(
            config.experiment, seed=args.seed, device=args.device
        )
        frozen = {
            "kind": "gate0_local_stability",
            "protocol_version": config.protocol_version,
            "experiment": experiment_config,
            "metric": config.metric,
            "gate0": config.gate0,
            "snapshot": str(args.snapshot),
            "learning_rate": args.learning_rate,
        }
        freeze_manifest(frozen, args.output / "manifest.json", repo=Path.cwd())
        result = measure_checkpoint_local_stability(
            experiment_config,
            config.metric,
            config.gate0,
            args.snapshot,
            args.output / "local_stability.json",
            learning_rate=args.learning_rate,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "erasure-boundary":
        bands = load_reference_bands(args.reference)
        thresholds = StateThresholds(**asdict(config.state))
        experiment_config = replace(
            config.experiment, seed=args.seed, device=args.device
        )
        frozen = {
            "kind": "gate0_erasure_boundary",
            "protocol_version": config.protocol_version,
            "experiment": experiment_config,
            "metric": config.metric,
            "state": config.state,
            "gate0": config.gate0,
            "reference_path": str(args.reference),
            "snapshot": str(args.snapshot),
            "lower_learning_rate": args.lower_learning_rate,
            "upper_learning_rate": args.upper_learning_rate,
        }
        freeze_manifest(frozen, args.output / "manifest.json", repo=Path.cwd())
        summary = geometric_erasure_bisection(
            experiment_config,
            config.metric,
            config.gate0,
            bands,
            args.snapshot,
            lower_learning_rate=args.lower_learning_rate,
            upper_learning_rate=args.upper_learning_rate,
            output_dir=args.output,
            thresholds=thresholds,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
    elif args.command == "acquisition":
        bands = load_reference_bands(args.reference)
        thresholds = StateThresholds(**asdict(config.state))
        experiment_config = replace(
            config.experiment, seed=args.seed, device=args.device
        )
        frozen = {
            "kind": "gate0_acquisition",
            "protocol_version": config.protocol_version,
            "experiment": experiment_config,
            "metric": config.metric,
            "state": config.state,
            "gate0": config.gate0,
            "reference_path": str(args.reference),
            "snapshot": str(args.snapshot),
            "learning_rate": args.learning_rate,
        }
        freeze_manifest(frozen, args.output / "manifest.json", repo=Path.cwd())
        result = run_acquisition_branch(
            experiment_config,
            config.metric,
            config.gate0,
            bands,
            args.snapshot,
            args.learning_rate,
            args.output,
            thresholds=thresholds,
        )
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
    elif args.command == "memory-factorial":
        bands = load_reference_bands(args.reference)
        thresholds = StateThresholds(**asdict(config.state))
        experiment_config = replace(
            config.experiment, seed=args.seed, device=args.device
        )
        frozen = {
            "kind": "gate1_memory_factorial",
            "protocol_version": config.protocol_version,
            "experiment": experiment_config,
            "metric": config.metric,
            "state": config.state,
            "reference_path": str(args.reference),
            "expressed_snapshot": str(args.expressed_snapshot),
            "suppressed_snapshot": str(args.suppressed_snapshot),
            "learning_rate": args.learning_rate,
            "challenge_steps": args.challenge_steps,
        }
        freeze_manifest(frozen, args.output / "manifest.json", repo=Path.cwd())
        summary = run_memory_factorial(
            experiment_config,
            config.metric,
            bands,
            thresholds,
            expressed_snapshot=args.expressed_snapshot,
            suppressed_snapshot=args.suppressed_snapshot,
            learning_rate=args.learning_rate,
            challenge_steps=args.challenge_steps,
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
