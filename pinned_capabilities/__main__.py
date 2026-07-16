"""Small administrative CLI; experiment runners are added gate by gate."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from .config import MBCExperimentConfig, MetricConfig, ProtocolConfig
from .experiment import JSONLWriter, MBCExperiment
from .gate0 import deep_linear_control
from .gate0_calibration import (
    adjudicate_gate0_calibration,
    adjudicate_gate0_erasure_precheck,
    bind_passed_gate0_calibration,
)
from .gate0_analysis import (
    CellKey,
    Direction,
    Gate0Action,
    analyze_gate0_payload,
    bind_gate0_analysis_report,
    gate0_analysis_to_dict,
    gate0_payload_registry_complete,
    gate0_payload_source_paths,
    validate_scan_artifact,
)
from .gate0_autopsy import run_autopsy
from .gate0e_control import write_positive_control
from .gate0e_escape import GATE0E_GRID, GATE0E_STREAMS, run_escape_curve
from .gate0e_verdict import gate0e_verdict
from .gate0e_null import (
    REFRESH_RATE,
    REFRESH_STEPS,
    freeze_null_predictions,
    radius_table,
    run_null_scan,
)
from .gate0_boundary import (
    geometric_erasure_bisection,
    run_acquisition_branch,
    run_acquisition_scan,
    run_erasure_scan,
)
from .local_measurement import (
    measure_checkpoint_local_stability,
    measure_checkpoint_local_stability_scan,
)
from .manifest import freeze_manifest
from .memory_surgery_run import run_memory_factorial
from .provenance import bind_file, bind_nearest_manifest, bind_reference, bind_snapshot
from .references import empirical_constant_machine, load_reference_bands
from .reference_run import run_reference_ensemble
from .reversibility_run import run_transition_timing
from .state import StateThresholds
from .state_preparation import prepare_expressed_checkpoint, prepare_suppressed_checkpoint
from .hysteresis_run import run_hysteresis_cycles


def _runtime_experiment(config: ProtocolConfig, args: argparse.Namespace) -> MBCExperimentConfig:
    updates = {"seed": args.seed, "device": args.device}
    if getattr(args, "batch_size", None) is not None:
        updates["batch_size"] = args.batch_size
    return replace(config.experiment, **updates)


def _add_batch_size(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="override the frozen default batch size for a registered mechanism cell",
    )


def _reference_input(path: Path) -> dict:
    return {
        "artifact": bind_reference(path),
        "source_manifest": bind_nearest_manifest(path),
    }


def _snapshot_input(path: Path, experiment_config: MBCExperimentConfig) -> dict:
    artifact = bind_snapshot(
        path,
        expected_seed=experiment_config.seed,
        expected_config=experiment_config,
    )
    if artifact["checks"]["seed"] != "matched":
        raise ValueError("suite snapshots must carry matched seed metadata")
    config_check = artifact["checks"]["config"]
    if config_check != "matched" and not (
        experiment_config.seed == 100 and config_check == "unavailable"
    ):
        raise ValueError("suite snapshots must carry matched experiment configuration")
    return {
        "artifact": artifact,
        "source_manifest": bind_nearest_manifest(path),
    }


def _calibration_input(path: Path, config: ProtocolConfig) -> dict:
    return {
        **bind_passed_gate0_calibration(
            path,
            expected_gate=config.gate0,
            expected_protocol_version=config.protocol_version,
        ),
        "source_manifest": bind_nearest_manifest(path),
    }


def _require_official_calibration(
    args: argparse.Namespace, config: ProtocolConfig
) -> dict | None:
    if args.seed not in config.gate0.seeds:
        return None
    path = getattr(args, "calibration", None)
    if path is None:
        raise ValueError("official Gate 0 work requires a passed calibration artifact")
    return _calibration_input(path, config)


def _add_calibration(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="passed Gate 0 calibration report required for official seeds",
    )


def _add_local_prerequisite(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--local-scan",
        type=Path,
        default=None,
        help="completed matching local scan required before official fate is opened",
    )


def _add_first_cell_analysis(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--first-cell-analysis",
        type=Path,
        default=None,
        help="partial Gate 0 analysis authorizing collection beyond the first cell",
    )


def _first_cell_analysis_input(path: Path | None, config: ProtocolConfig) -> dict:
    if path is None:
        raise ValueError("work beyond the first Gate 0 cell requires its analysis artifact")
    expected_gate0 = json.loads(json.dumps(asdict(config.gate0)))
    bound = bind_gate0_analysis_report(
        path,
        expected_protocol_version=config.protocol_version,
        expected_gate0_config=expected_gate0,
        registered_seeds=config.gate0.seeds,
        registered_batch_sizes=config.gate0.batch_sizes,
        required_action=Gate0Action.COLLECT,
    )
    report = json.loads(path.read_text())
    eligible = False
    for assessment in report.get("cell_assessments", []):
        key = assessment.get("key", {})
        if (
            key.get("seed") == config.gate0.first_cell_seed
            and key.get("batch_size") == config.gate0.first_cell_batch_size
            and key.get("direction") == "erasure"
            and assessment.get("conclusion") in {"reduction", "ambiguous"}
        ):
            eligible = True
            break
    if not eligible:
        raise ValueError("first-cell analysis lacks a valid completed erasure assessment")
    return bound


def _is_first_cell(config: ProtocolConfig, experiment_config: MBCExperimentConfig) -> bool:
    return (
        experiment_config.seed == config.gate0.first_cell_seed
        and experiment_config.batch_size == config.gate0.first_cell_batch_size
    )


def _add_gate0_analysis(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--gate0-analysis",
        type=Path,
        required=True,
        help="Gate 0 report whose action must authorize Gate 1",
    )


def _gate0_continue_input(path: Path, config: ProtocolConfig) -> dict:
    return bind_gate0_analysis_report(
        path,
        expected_protocol_version=config.protocol_version,
        expected_gate0_config=json.loads(json.dumps(asdict(config.gate0))),
        registered_seeds=config.gate0.seeds,
        registered_batch_sizes=config.gate0.batch_sizes,
        required_action=Gate0Action.CONTINUE,
    )


def _local_prerequisite_input(
    path: Path | None,
    *,
    snapshot_path: Path,
    experiment_config: MBCExperimentConfig,
    config: ProtocolConfig,
) -> dict:
    if path is None:
        raise ValueError("official empirical fate requires its completed local scan")
    expected_gate0 = json.loads(json.dumps(asdict(config.gate0)))
    artifact = validate_scan_artifact(
        path,
        key=CellKey(
            experiment_config.seed,
            experiment_config.batch_size,
            Direction.ERASURE,
        ),
        role="local",
        expected_rates=config.gate0.learning_rates,
        expected_protocol_version=config.protocol_version,
        expected_gate0_config=expected_gate0,
    )
    if artifact.payload.get("all_augmented_eigenpairs_certified") is not True:
        raise ValueError("local prerequisite contains uncertified eigenpairs")
    manifest_config = artifact.manifest_config
    current_snapshot = _snapshot_input(snapshot_path, experiment_config)
    bound_snapshot = manifest_config.get("snapshot_input", {})
    if bound_snapshot.get("artifact", {}).get("sha256") != current_snapshot["artifact"]["sha256"]:
        raise ValueError("local prerequisite does not bind the empirical snapshot")
    calibration = manifest_config.get("calibration_input")
    if not isinstance(calibration, dict):
        raise ValueError("local prerequisite has no passed calibration binding")
    calibration_artifact = calibration.get("artifact", {})
    calibration_path = calibration_artifact.get("path")
    if not isinstance(calibration_path, str):
        raise ValueError("local prerequisite calibration binding has no path")
    current_calibration = _calibration_input(Path(calibration_path), config)
    if calibration_artifact.get("sha256") != current_calibration["artifact"]["sha256"]:
        raise ValueError("local prerequisite calibration changed after measurement")
    return {
        "artifact": bind_file(path),
        "source_manifest": bind_nearest_manifest(path),
    }


def _validate_gate0_scan_controls(
    config: ProtocolConfig,
    args: argparse.Namespace,
    experiment_config: MBCExperimentConfig,
    *,
    allow_calibration_seed: bool,
) -> None:
    allowed = set(config.gate0.seeds)
    if allow_calibration_seed:
        allowed.add(config.gate0.calibration_seed)
    if args.seed not in allowed:
        raise ValueError(f"seed {args.seed} is not registered for this Gate 0 scan")
    if experiment_config.batch_size not in config.gate0.batch_sizes:
        raise ValueError("Gate 0 scan must use a registered batch size")
    if tuple(args.learning_rates) != config.gate0.learning_rates:
        raise ValueError("Gate 0 scan must use the frozen learning-rate grid")
    if (
        args.seed == config.gate0.calibration_seed
        and experiment_config.batch_size != config.gate0.calibration_batch_size
    ):
        raise ValueError("Gate 0 calibration must use the frozen batch size")


def _validate_expressed_preparation_controls(
    config: ProtocolConfig,
    args: argparse.Namespace,
    experiment_config: MBCExperimentConfig,
) -> None:
    if args.seed not in config.gate0.seeds:
        return
    if experiment_config.batch_size not in config.gate0.batch_sizes:
        raise ValueError("official expressed preparation needs a registered batch size")
    if args.learning_rate != config.gate0.expressed_preparation_learning_rate:
        raise ValueError("official expressed preparation must use the frozen learning rate")
    if args.save_at_step != config.gate0.state_preparation_step:
        raise ValueError("official expressed preparation must use the frozen state age")


def _validate_suppressed_preparation_controls(
    config: ProtocolConfig,
    args: argparse.Namespace,
    experiment_config: MBCExperimentConfig,
    calibration_input: dict | None,
) -> None:
    if args.seed not in config.gate0.seeds:
        return
    if experiment_config.batch_size not in config.gate0.batch_sizes:
        raise ValueError("official suppressed preparation needs a registered batch size")
    if calibration_input is None:
        raise ValueError("official suppressed preparation requires calibration")
    erasing_rate = calibration_input["frozen_erasure_bracket"]["upper_learning_rate"]
    if args.learning_rate != erasing_rate:
        raise ValueError("official suppressed preparation must use the calibrated erasing rate")
    if (
        args.max_steps != config.gate0.state_preparation_step
        or args.save_at_step != config.gate0.state_preparation_step
    ):
        raise ValueError("official suppressed preparation must use the frozen state age")


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
    deep_linear.add_argument(
        "--output", type=Path, default=None,
        help="optional manifest-frozen positive-control output directory",
    )
    gate0_calibrate = subparsers.add_parser(
        "gate0-calibrate", help="adjudicate the frozen non-gate calibration"
    )
    gate0_calibrate.add_argument("--positive-control", type=Path, required=True)
    gate0_calibrate.add_argument("--erasure-scan", type=Path, required=True)
    gate0_calibrate.add_argument(
        "--local-scan",
        type=Path,
        default=None,
        help="omit to emit the empirical-bracket precheck before local compute",
    )
    gate0_calibrate.add_argument("--snapshot", type=Path, required=True)
    gate0_calibrate.add_argument("--output", type=Path, required=True)
    gate0_calibrate.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
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
        "--output", type=Path, default=Path("pinned_capabilities/results/reference_v1_4_0")
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
    _add_batch_size(prepare)
    _add_calibration(prepare)
    _add_first_cell_analysis(prepare)
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
    _add_batch_size(prepare_expressed)
    _add_calibration(prepare_expressed)
    _add_first_cell_analysis(prepare_expressed)
    prepare_expressed.add_argument(
        "--gate0e-autopsy",
        type=Path,
        default=None,
        help="v1.5 alternative authorization: committed Branch A autopsy artifact",
    )
    prepare_expressed.add_argument(
        "--gate0e-control",
        type=Path,
        default=None,
        help="v1.5 alternative authorization: passing positive-control artifact",
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
    _add_batch_size(hysteresis)
    _add_gate0_analysis(hysteresis)
    local = subparsers.add_parser(
        "local-stability", help="measure registered Gate 0 predictors at one checkpoint"
    )
    local.add_argument("--snapshot", type=Path, required=True)
    local.add_argument("--seed", type=int, required=True)
    local.add_argument("--learning-rate", type=float, required=True)
    local.add_argument("--output", type=Path, required=True)
    local.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    _add_batch_size(local)
    _add_calibration(local)
    local_scan = subparsers.add_parser(
        "local-stability-scan",
        help="run a resumable multi-rate Gate 0 local-stability scan",
    )
    local_scan.add_argument("--snapshot", type=Path, required=True)
    local_scan.add_argument("--seed", type=int, required=True)
    local_scan.add_argument("--learning-rates", type=float, nargs="+", required=True)
    local_scan.add_argument(
        "--workers", type=int, default=1,
        help="number of spawned rate-cell workers (execution only; default: 1)",
    )
    local_scan.add_argument("--output", type=Path, required=True)
    local_scan.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    _add_batch_size(local_scan)
    _add_calibration(local_scan)
    _add_first_cell_analysis(local_scan)
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
    _add_batch_size(erasure)
    _add_calibration(erasure)
    _add_gate0_analysis(erasure)
    erasure_scan = subparsers.add_parser(
        "erasure-scan", help="run a resumable coarse Gate 0 erasure bracket scan"
    )
    erasure_scan.add_argument("--reference", type=Path, required=True)
    erasure_scan.add_argument("--snapshot", type=Path, required=True)
    erasure_scan.add_argument("--seed", type=int, required=True)
    erasure_scan.add_argument("--learning-rates", type=float, nargs="+", required=True)
    erasure_scan.add_argument(
        "--workers", type=int, default=1,
        help="number of spawned rate-cell workers (execution only; default: 1)",
    )
    erasure_scan.add_argument("--output", type=Path, required=True)
    erasure_scan.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    _add_batch_size(erasure_scan)
    _add_calibration(erasure_scan)
    _add_local_prerequisite(erasure_scan)
    _add_first_cell_analysis(erasure_scan)
    acquisition = subparsers.add_parser(
        "acquisition", help="run one censored Gate 0 acquisition branch"
    )
    acquisition.add_argument("--reference", type=Path, required=True)
    acquisition.add_argument("--snapshot", type=Path, required=True)
    acquisition.add_argument("--seed", type=int, required=True)
    acquisition.add_argument("--learning-rate", type=float, required=True)
    acquisition.add_argument("--output", type=Path, required=True)
    acquisition.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    _add_batch_size(acquisition)
    _add_calibration(acquisition)
    acquisition_scan = subparsers.add_parser(
        "acquisition-scan",
        help="run a resumable multi-rate Gate 0 acquisition scan",
    )
    acquisition_scan.add_argument("--reference", type=Path, required=True)
    acquisition_scan.add_argument("--snapshot", type=Path, required=True)
    acquisition_scan.add_argument("--seed", type=int, required=True)
    acquisition_scan.add_argument("--learning-rates", type=float, nargs="+", required=True)
    acquisition_scan.add_argument(
        "--workers", type=int, default=1,
        help="number of spawned rate-cell workers (execution only; default: 1)",
    )
    acquisition_scan.add_argument("--output", type=Path, required=True)
    acquisition_scan.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    _add_batch_size(acquisition_scan)
    _add_calibration(acquisition_scan)
    _add_local_prerequisite(acquisition_scan)
    _add_first_cell_analysis(acquisition_scan)
    gate0_analyze = subparsers.add_parser(
        "gate0-analyze", help="apply the frozen total Gate 0 decision rule"
    )
    gate0_analyze.add_argument("--evidence", type=Path, required=True)
    gate0_analyze.add_argument("--output", type=Path, required=True)
    gate0_autopsy = subparsers.add_parser(
        "gate0-autopsy",
        help="apply the frozen amendment v1.5 classifier to the sealed cells",
    )
    gate0_autopsy.add_argument("--scan-dir", type=Path, required=True)
    gate0_autopsy.add_argument("--reference", type=Path, required=True)
    gate0_autopsy.add_argument("--output", type=Path, required=True)
    gate0e_control = subparsers.add_parser(
        "gate0e-control",
        help="run the Gate 0-E noise-escape positive control",
    )
    gate0e_control.add_argument("--output", type=Path, required=True)
    gate0e_curve = subparsers.add_parser(
        "gate0e-curve",
        help="run a Branch A escape curve from one expressed snapshot",
    )
    gate0e_curve.add_argument("--seed", type=int, required=True)
    gate0e_curve.add_argument("--device", default="cpu")
    gate0e_curve.add_argument("--snapshot", type=Path, required=True)
    gate0e_curve.add_argument("--reference", type=Path, required=True)
    gate0e_curve.add_argument("--autopsy", type=Path, required=True)
    gate0e_curve.add_argument("--control", type=Path, required=True)
    gate0e_curve.add_argument("--output", type=Path, required=True)
    gate0e_curve.add_argument("--workers", type=int, default=1)
    gate0e_curve.add_argument("--torch-threads", type=int, default=2)
    gate0e_curve.add_argument("--streams", type=int, default=None)
    gate0e_curve.add_argument("--weight-decay", type=float, default=None)
    gate0e_curve.add_argument(
        "--rates",
        type=float,
        nargs="*",
        default=None,
        help="override only for the single permitted half-shift or registered subsets",
    )
    gate0e_curve.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="committed gate-seed prediction artifact; required for official seeds",
    )
    _add_batch_size(gate0e_curve)
    gate0e_null = subparsers.add_parser(
        "gate0e-null-scan",
        help="moment refresh plus certified local-stability scan for one condition",
    )
    gate0e_null.add_argument("--seed", type=int, required=True)
    gate0e_null.add_argument("--device", default="cpu")
    gate0e_null.add_argument("--snapshot", type=Path, required=True)
    gate0e_null.add_argument("--autopsy", type=Path, required=True)
    gate0e_null.add_argument("--output", type=Path, required=True)
    gate0e_null.add_argument("--workers", type=int, default=1)
    gate0e_null.add_argument(
        "--rates", type=float, nargs="*", default=None,
        help="challenge rates; defaults to the active Branch A grid",
    )
    _add_batch_size(gate0e_null)
    gate0e_freeze = subparsers.add_parser(
        "gate0e-freeze-predictions",
        help="compute c* from the dev pair and commit gate-seed predictions",
    )
    gate0e_freeze.add_argument("--dev-curve", type=Path, required=True)
    gate0e_freeze.add_argument("--dev-null", type=Path, required=True)
    gate0e_freeze.add_argument(
        "--gate-null",
        action="append",
        required=True,
        metavar="SEED:BATCH:PATH",
        help="repeatable seed:batch:path triples for condition null scans",
    )
    gate0e_freeze.add_argument("--output", type=Path, required=True)
    gate0e_verdict_parser = subparsers.add_parser(
        "gate0e-verdict",
        help="apply the frozen amendment v1.5 section 2.5 decision rule",
    )
    gate0e_verdict_parser.add_argument("--predictions", type=Path, required=True)
    gate0e_verdict_parser.add_argument(
        "--gate-curve", action="append", required=True, metavar="SEED:PATH"
    )
    gate0e_verdict_parser.add_argument(
        "--batch-curve", action="append", required=True, metavar="SEED:BATCH:PATH"
    )
    gate0e_verdict_parser.add_argument(
        "--contrast-rates", type=float, nargs=2, required=True
    )
    gate0e_verdict_parser.add_argument("--wd-curve", type=Path, default=None)
    gate0e_verdict_parser.add_argument("--output", type=Path, required=True)
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
    _add_batch_size(memory)
    _add_gate0_analysis(memory)
    timing = subparsers.add_parser(
        "transition-time", help="time acquisition or recovery under one fixed low-rate rule"
    )
    timing.add_argument("--reference", type=Path, required=True)
    timing.add_argument("--snapshot", type=Path, default=None)
    timing.add_argument("--arm", type=str, required=True)
    timing.add_argument("--seed", type=int, required=True)
    timing.add_argument("--learning-rate", type=float, required=True)
    timing.add_argument("--max-steps", type=int, default=40_000)
    timing.add_argument("--reset-optimizer", action="store_true")
    timing.add_argument("--output", type=Path, required=True)
    timing.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    _add_batch_size(timing)
    _add_gate0_analysis(timing)
    args = parser.parse_args()
    config = ProtocolConfig()
    if args.command == "freeze":
        manifest = freeze_manifest(config, args.path, repo=args.repo)
        print(json.dumps(manifest, indent=2, sort_keys=True))
    elif args.command == "show-config":
        print(json.dumps(config.to_dict(), indent=2, sort_keys=True))
    elif args.command == "deep-linear":
        if tuple(args.learning_rates) != config.gate0.positive_control_rates:
            raise ValueError("deep-linear must use the frozen positive-control grid")
        result = deep_linear_control(args.learning_rates)
        if args.output is not None:
            freeze_manifest(
                {
                    "kind": "gate0_positive_control",
                    "protocol_version": config.protocol_version,
                    "learning_rates": tuple(args.learning_rates),
                    "target": 1.0,
                },
                args.output / "manifest.json",
                repo=Path.cwd(),
            )
            result_path = args.output / "positive_control.json"
            temporary = result_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            temporary.replace(result_path)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "gate0-calibrate":
        experiment_config = replace(
            config.experiment,
            seed=config.gate0.calibration_seed,
            batch_size=config.gate0.calibration_batch_size,
            device=args.device,
        )
        snapshot_input = _snapshot_input(args.snapshot, experiment_config)
        frozen = {
            "kind": (
                "gate0_calibration_adjudication"
                if args.local_scan is not None
                else "gate0_calibration_erasure_precheck"
            ),
            "protocol_version": config.protocol_version,
            "experiment": experiment_config,
            "gate0": config.gate0,
            "positive_control_input": {
                "artifact": bind_file(args.positive_control),
                "source_manifest": bind_nearest_manifest(args.positive_control),
            },
            "erasure_scan_input": {
                "artifact": bind_file(args.erasure_scan),
                "source_manifest": bind_nearest_manifest(args.erasure_scan),
            },
            "local_scan_input": (
                {
                    "artifact": bind_file(args.local_scan),
                    "source_manifest": bind_nearest_manifest(args.local_scan),
                }
                if args.local_scan is not None
                else None
            ),
            "snapshot": str(args.snapshot),
            "snapshot_input": snapshot_input,
        }
        freeze_manifest(frozen, args.output / "manifest.json", repo=Path.cwd())
        if args.local_scan is None:
            report = adjudicate_gate0_erasure_precheck(
                positive_control_path=args.positive_control,
                erasure_scan_path=args.erasure_scan,
                snapshot_path=args.snapshot,
                experiment=experiment_config,
                gate=config.gate0,
                protocol_version=config.protocol_version,
                output_path=args.output / "calibration_precheck.json",
            )
        else:
            report = adjudicate_gate0_calibration(
                positive_control_path=args.positive_control,
                erasure_scan_path=args.erasure_scan,
                local_scan_path=args.local_scan,
                snapshot_path=args.snapshot,
                experiment=experiment_config,
                gate=config.gate0,
                protocol_version=config.protocol_version,
                output_path=args.output / "calibration.json",
            )
        print(json.dumps(report, indent=2, sort_keys=True))
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
        experiment_config = _runtime_experiment(config, args)
        calibration_input = _require_official_calibration(args, config)
        _validate_suppressed_preparation_controls(
            config, args, experiment_config, calibration_input
        )
        first_cell_analysis = (
            _first_cell_analysis_input(args.first_cell_analysis, config)
            if args.seed in config.gate0.seeds
            else None
        )
        frozen = {
            "kind": "suppressed_state_preparation",
            "protocol_version": config.protocol_version,
            "experiment": experiment_config,
            "metric": config.metric,
            "state": config.state,
            "reference_path": str(args.reference),
            "reference_input": _reference_input(args.reference),
            "calibration_input": calibration_input,
            "first_cell_analysis": first_cell_analysis,
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
        experiment_config = _runtime_experiment(config, args)
        _validate_expressed_preparation_controls(config, args, experiment_config)
        calibration_input = None
        gate0e_authorization = None
        first_cell_analysis = None
        if args.seed in config.gate0.seeds and args.calibration is None:
            if args.gate0e_autopsy is None or args.gate0e_control is None:
                raise ValueError(
                    "official expressed preparation requires either a passed"
                    " calibration artifact or the v1.5 Branch A authorization"
                    " (autopsy plus passing positive control)"
                )
            autopsy_report = json.loads(args.gate0e_autopsy.read_text())
            if (
                autopsy_report.get("kind") != "gate0_autopsy_v1_5"
                or autopsy_report["activation"].get("primary_branch") != "A"
            ):
                raise ValueError(
                    "v1.5 state preparation requires the committed Branch A autopsy"
                )
            control_report = json.loads(args.gate0e_control.read_text())
            if control_report.get("kind") != "gate0e_positive_control" or not control_report.get("passed"):
                raise ValueError(
                    "v1.5 state preparation requires a passing positive control"
                )
            gate0e_authorization = {
                "autopsy_input": bind_file(args.gate0e_autopsy),
                "control_input": bind_file(args.gate0e_control),
                "design_note": "AMENDMENT_v1_5_DESIGN_NOTES.md section 2:"
                " state preparation only; no fate run is authorized by this input",
            }
        else:
            calibration_input = _require_official_calibration(args, config)
            first_cell_analysis = (
                _first_cell_analysis_input(args.first_cell_analysis, config)
                if args.seed in config.gate0.seeds
                and not _is_first_cell(config, experiment_config)
                else None
            )
        frozen = {
            "kind": "expressed_state_preparation",
            "protocol_version": config.protocol_version,
            "experiment": experiment_config,
            "metric": config.metric,
            "state": config.state,
            "reference_path": str(args.reference),
            "reference_input": _reference_input(args.reference),
            "calibration_input": calibration_input,
            "gate0e_authorization": gate0e_authorization,
            "first_cell_analysis": first_cell_analysis,
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
        experiment_config = _runtime_experiment(config, args)
        gate0_input = _gate0_continue_input(args.gate0_analysis, config)
        frozen = {
            "kind": "gate1_hysteresis",
            "protocol_version": config.protocol_version,
            "experiment": experiment_config,
            "metric": config.metric,
            "state": config.state,
            "gate1": config.gate1,
            "reference_path": str(args.reference),
            "reference_input": _reference_input(args.reference),
            "suppressed_snapshot": str(args.snapshot),
            "suppressed_snapshot_input": _snapshot_input(args.snapshot, experiment_config),
            "gate0_analysis_input": gate0_input,
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
        experiment_config = _runtime_experiment(config, args)
        if args.seed in config.gate0.seeds:
            raise ValueError(
                "single-rate local-stability is exploratory and disabled for official Gate 0 seeds"
            )
        calibration_input = _require_official_calibration(args, config)
        frozen = {
            "kind": "gate0_local_stability",
            "protocol_version": config.protocol_version,
            "experiment": experiment_config,
            "metric": config.metric,
            "gate0": config.gate0,
            "snapshot": str(args.snapshot),
            "snapshot_input": _snapshot_input(args.snapshot, experiment_config),
            "calibration_input": calibration_input,
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
    elif args.command == "local-stability-scan":
        experiment_config = _runtime_experiment(config, args)
        _validate_gate0_scan_controls(
            config, args, experiment_config, allow_calibration_seed=True
        )
        calibration_input = _require_official_calibration(args, config)
        first_cell_analysis = (
            _first_cell_analysis_input(args.first_cell_analysis, config)
            if args.seed in config.gate0.seeds and not _is_first_cell(config, experiment_config)
            else None
        )
        frozen = {
            "kind": "gate0_local_stability_scan",
            "protocol_version": config.protocol_version,
            "experiment": experiment_config,
            "metric": config.metric,
            "gate0": config.gate0,
            "snapshot": str(args.snapshot),
            "snapshot_input": _snapshot_input(args.snapshot, experiment_config),
            "calibration_input": calibration_input,
            "first_cell_analysis": first_cell_analysis,
            "learning_rates": tuple(args.learning_rates),
        }
        freeze_manifest(frozen, args.output / "manifest.json", repo=Path.cwd())
        summary = measure_checkpoint_local_stability_scan(
            experiment_config,
            config.metric,
            config.gate0,
            args.snapshot,
            args.learning_rates,
            args.output,
            workers=args.workers,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
    elif args.command == "erasure-boundary":
        bands = load_reference_bands(args.reference)
        thresholds = StateThresholds(**asdict(config.state))
        experiment_config = _runtime_experiment(config, args)
        calibration_input = _require_official_calibration(args, config)
        gate0_input = _gate0_continue_input(args.gate0_analysis, config)
        frozen = {
            "kind": "gate0_erasure_boundary",
            "protocol_version": config.protocol_version,
            "experiment": experiment_config,
            "metric": config.metric,
            "state": config.state,
            "gate0": config.gate0,
            "reference_path": str(args.reference),
            "reference_input": _reference_input(args.reference),
            "snapshot": str(args.snapshot),
            "snapshot_input": _snapshot_input(args.snapshot, experiment_config),
            "calibration_input": calibration_input,
            "gate0_analysis_input": gate0_input,
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
    elif args.command == "erasure-scan":
        bands = load_reference_bands(args.reference)
        thresholds = StateThresholds(**asdict(config.state))
        experiment_config = _runtime_experiment(config, args)
        _validate_gate0_scan_controls(
            config, args, experiment_config, allow_calibration_seed=True
        )
        calibration_input = _require_official_calibration(args, config)
        first_cell_analysis = (
            _first_cell_analysis_input(args.first_cell_analysis, config)
            if args.seed in config.gate0.seeds and not _is_first_cell(config, experiment_config)
            else None
        )
        local_prerequisite = (
            _local_prerequisite_input(
                args.local_scan,
                snapshot_path=args.snapshot,
                experiment_config=experiment_config,
                config=config,
            )
            if args.seed in config.gate0.seeds
            else None
        )
        frozen = {
            "kind": "gate0_erasure_scan",
            "protocol_version": config.protocol_version,
            "experiment": experiment_config,
            "metric": config.metric,
            "state": config.state,
            "gate0": config.gate0,
            "reference_path": str(args.reference),
            "reference_input": _reference_input(args.reference),
            "snapshot": str(args.snapshot),
            "snapshot_input": _snapshot_input(args.snapshot, experiment_config),
            "calibration_input": calibration_input,
            "local_prerequisite": local_prerequisite,
            "first_cell_analysis": first_cell_analysis,
            "learning_rates": tuple(args.learning_rates),
        }
        freeze_manifest(frozen, args.output / "manifest.json", repo=Path.cwd())
        summary = run_erasure_scan(
            experiment_config,
            config.metric,
            config.gate0,
            bands,
            args.snapshot,
            args.learning_rates,
            args.output,
            thresholds=thresholds,
            workers=args.workers,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
    elif args.command == "acquisition":
        bands = load_reference_bands(args.reference)
        thresholds = StateThresholds(**asdict(config.state))
        experiment_config = _runtime_experiment(config, args)
        if args.seed in config.gate0.seeds:
            raise ValueError(
                "single-rate acquisition is exploratory and disabled for official Gate 0 seeds"
            )
        calibration_input = _require_official_calibration(args, config)
        frozen = {
            "kind": "gate0_acquisition",
            "protocol_version": config.protocol_version,
            "experiment": experiment_config,
            "metric": config.metric,
            "state": config.state,
            "gate0": config.gate0,
            "reference_path": str(args.reference),
            "reference_input": _reference_input(args.reference),
            "snapshot": str(args.snapshot),
            "snapshot_input": _snapshot_input(args.snapshot, experiment_config),
            "calibration_input": calibration_input,
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
    elif args.command == "acquisition-scan":
        bands = load_reference_bands(args.reference)
        thresholds = StateThresholds(**asdict(config.state))
        experiment_config = _runtime_experiment(config, args)
        _validate_gate0_scan_controls(
            config, args, experiment_config, allow_calibration_seed=False
        )
        calibration_input = _require_official_calibration(args, config)
        first_cell_analysis = _first_cell_analysis_input(args.first_cell_analysis, config)
        local_prerequisite = _local_prerequisite_input(
            args.local_scan,
            snapshot_path=args.snapshot,
            experiment_config=experiment_config,
            config=config,
        )
        frozen = {
            "kind": "gate0_acquisition_scan",
            "protocol_version": config.protocol_version,
            "experiment": experiment_config,
            "metric": config.metric,
            "state": config.state,
            "gate0": config.gate0,
            "reference_path": str(args.reference),
            "reference_input": _reference_input(args.reference),
            "snapshot": str(args.snapshot),
            "snapshot_input": _snapshot_input(args.snapshot, experiment_config),
            "calibration_input": calibration_input,
            "local_prerequisite": local_prerequisite,
            "first_cell_analysis": first_cell_analysis,
            "learning_rates": tuple(args.learning_rates),
        }
        freeze_manifest(frozen, args.output / "manifest.json", repo=Path.cwd())
        summary = run_acquisition_scan(
            experiment_config,
            config.metric,
            config.gate0,
            bands,
            args.snapshot,
            args.learning_rates,
            args.output,
            thresholds=thresholds,
            workers=args.workers,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
    elif args.command == "gate0-analyze":
        evidence = json.loads(args.evidence.read_text())
        if "finalize" in evidence:
            raise ValueError("Gate 0 completion is derived, not an evidence-file control")
        registry_complete = gate0_payload_registry_complete(
            evidence,
            registered_seeds=config.gate0.seeds,
            registered_batch_sizes=config.gate0.batch_sizes,
        )
        scan_inputs = [
            {
                "artifact": bind_file(path),
                "source_manifest": bind_nearest_manifest(path),
            }
            for path in gate0_payload_source_paths(evidence)
        ]
        frozen = {
            "kind": "gate0_decision_analysis",
            "protocol_version": config.protocol_version,
            "gate0": config.gate0,
            "evidence_input": bind_file(args.evidence),
            "scan_inputs": scan_inputs,
            "registry_complete": registry_complete,
        }
        freeze_manifest(frozen, args.output / "manifest.json", repo=Path.cwd())
        analysis = analyze_gate0_payload(
            evidence,
            registered_seeds=config.gate0.seeds,
            registered_batch_sizes=config.gate0.batch_sizes,
            reduction_match_factor=config.gate0.reduction_match_factor,
            reduction_miss_factor=config.gate0.reduction_miss_factor,
            residual_shift_fraction=config.gate0.residual_batch_shift_fraction,
            require_scan_sources=True,
            expected_learning_rates=config.gate0.learning_rates,
            expected_protocol_version=config.protocol_version,
            expected_gate0_config=json.loads(json.dumps(asdict(config.gate0))),
        )
        report = {
            "schema_version": 1,
            "kind": "gate0_decision_analysis",
            "protocol_version": config.protocol_version,
            "registry_complete": registry_complete,
            **gate0_analysis_to_dict(analysis),
        }
        result_path = args.output / "gate0_analysis.json"
        temporary = result_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        temporary.replace(result_path)
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.command == "gate0-autopsy":
        report = run_autopsy(args.scan_dir, args.reference, args.output)
        summary = {
            "labels": report["activation"]["labels"],
            "activation": {
                key: report["activation"][key]
                for key in ("primary_branch", "include_branch_c", "grid_cap", "t_hold_steps")
            },
            "output": str(args.output),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    elif args.command == "gate0e-control":
        report = write_positive_control(args.output)
        print(
            json.dumps(
                {
                    "passed": report["passed"],
                    "arrhenius": report["arrhenius"],
                    "predicted_arrhenius_slope": report["predicted_arrhenius_slope"],
                    "arrhenius_slope_ratio": report["arrhenius_slope_ratio"],
                    "monotonicity_flags": report["monotonicity_flags"],
                    "fractions": {
                        rate: cell["primary_fraction"]
                        for rate, cell in report["cells"].items()
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "gate0e-null-scan":
        autopsy_report = json.loads(args.autopsy.read_text())
        if autopsy_report.get("kind") != "gate0_autopsy_v1_5":
            raise ValueError("gate0e-null-scan requires the committed v1.5 autopsy artifact")
        if autopsy_report["activation"].get("primary_branch") != "A":
            raise ValueError("Branch A is not active; gate0e-null-scan is not authorized")
        grid = list(GATE0E_GRID)
        if autopsy_report["activation"].get("grid_cap") is not None:
            cap = float(autopsy_report["activation"]["grid_cap"])
            grid = [rate for rate in grid if rate <= cap]
        rates = sorted(float(rate) for rate in args.rates) if args.rates else grid
        experiment_config = _runtime_experiment(config, args)
        frozen = {
            "kind": "gate0e_null_scan",
            "protocol_version": config.protocol_version,
            "amendment": "v1.5.0",
            "experiment": experiment_config,
            "metric": config.metric,
            "gate0": config.gate0,
            "learning_rates": rates,
            "refresh_steps": REFRESH_STEPS,
            "refresh_rate": REFRESH_RATE,
            "snapshot_input": _snapshot_input(args.snapshot, experiment_config),
            "autopsy_input": bind_file(args.autopsy),
        }
        freeze_manifest(frozen, args.output / "manifest.json", repo=Path.cwd())
        summary = run_null_scan(
            experiment_config,
            config.metric,
            config.gate0,
            args.snapshot,
            rates,
            args.output,
            workers=args.workers,
        )
        print(
            json.dumps(
                {
                    "theta_relative_drift": summary["refresh"]["theta_relative_drift"],
                    "radii": {
                        f"{row['learning_rate']:g}": row["radius"]
                        for row in radius_table(summary)
                    },
                    "all_certified": summary["scan"][
                        "all_augmented_eigenpairs_certified"
                    ],
                    "output": str(args.output),
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "gate0e-freeze-predictions":
        gate_paths = {}
        for entry in args.gate_null:
            seed_text, batch_text, path_text = entry.split(":", 2)
            gate_paths[f"{int(seed_text)}:{int(batch_text)}"] = Path(path_text)
        sealed = freeze_null_predictions(
            args.dev_curve, args.dev_null, gate_paths, args.output
        )
        print(
            json.dumps(
                {
                    "dev_eta50": sealed["dev_eta50"],
                    "c_star": sealed["c_star"],
                    "predictions": {
                        seed: entry["prediction"]
                        for seed, entry in sealed["predictions"].items()
                    },
                    "batch_discriminator": {
                        key: sealed["batch_discriminator"][key]
                        for key in (
                            "status",
                            "noise_predicted_direction",
                            "v_conditioned_direction",
                            "pooled_dlnD_dlnB",
                            "slope_interval_95",
                        )
                    },
                    "result_sha256": sealed["result_sha256"],
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "gate0e-verdict":
        predictions_artifact = json.loads(args.predictions.read_text())
        gate_curves = {}
        for entry in args.gate_curve:
            seed_text, _, path_text = entry.partition(":")
            gate_curves[int(seed_text)] = json.loads(Path(path_text).read_text())
        batch_curves = {}
        batch_inputs = {}
        for entry in args.batch_curve:
            seed_text, batch_text, path_text = entry.split(":", 2)
            key = f"{int(seed_text)}:{int(batch_text)}"
            batch_curves[key] = json.loads(Path(path_text).read_text())
            batch_inputs[key] = bind_file(Path(path_text))
        wd_curve = None
        if args.wd_curve is not None:
            wd_curve = json.loads(args.wd_curve.read_text())
        report = gate0e_verdict(
            predictions_artifact,
            gate_curves,
            batch_curves,
            args.contrast_rates,
            wd_curve,
        )
        report["inputs"] = {
            "predictions": bind_file(args.predictions),
            "gate_curves": {
                entry.partition(":")[0]: bind_file(Path(entry.partition(":")[2]))
                for entry in args.gate_curve
            },
            "batch_curves": batch_inputs,
            "wd_curve": None if args.wd_curve is None else bind_file(args.wd_curve),
        }
        args.output.mkdir(parents=True, exist_ok=True)
        result_path = args.output / "verdict.json"
        temporary = result_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
        temporary.replace(result_path)
        print(
            json.dumps(
                {
                    "outcome": report["outcome"],
                    "action": report["action"],
                    "claim_status": report["claim_status"],
                    "miss_count": report["miss_count"],
                    "batch_matches": report["batch_shift"]["matches_prediction"],
                    "output": str(result_path),
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "gate0e-curve":
        autopsy_report = json.loads(args.autopsy.read_text())
        if autopsy_report.get("kind") != "gate0_autopsy_v1_5":
            raise ValueError("gate0e-curve requires the committed v1.5 autopsy artifact")
        activation = autopsy_report["activation"]
        if activation.get("primary_branch") != "A":
            raise ValueError("Branch A is not active; gate0e-curve is not authorized")
        control_report = json.loads(args.control.read_text())
        if control_report.get("kind") != "gate0e_positive_control" or not control_report.get("passed"):
            raise ValueError("gate0e-curve requires a passing positive-control artifact")
        if args.seed in config.gate0.seeds and args.predictions is None:
            raise ValueError(
                "official gate seeds require the committed prediction artifact"
            )
        hold_steps = int(activation["t_hold_steps"])
        grid = list(GATE0E_GRID)
        if activation.get("grid_cap") is not None:
            grid = [rate for rate in grid if rate <= float(activation["grid_cap"])]
        if args.rates:
            requested = sorted(float(rate) for rate in args.rates)
            permitted = {round(rate, 12) for rate in grid} | {
                round(rate * 0.5, 12) for rate in grid
            }
            if not all(round(rate, 12) in permitted for rate in requested):
                raise ValueError(
                    "requested rates must come from the frozen grid or its single half-shift"
                )
            grid = requested
        streams = args.streams if args.streams is not None else GATE0E_STREAMS
        bands = load_reference_bands(args.reference)
        thresholds = StateThresholds(**asdict(config.state))
        experiment_config = _runtime_experiment(config, args)
        frozen = {
            "kind": "gate0e_escape_curve",
            "protocol_version": config.protocol_version,
            "amendment": "v1.5.0",
            "experiment": experiment_config,
            "metric": config.metric,
            "thresholds": thresholds,
            "learning_rates": grid,
            "streams_per_rate": streams,
            "hold_steps": hold_steps,
            "weight_decay_override": args.weight_decay,
            "snapshot_input": _snapshot_input(args.snapshot, experiment_config),
            "reference_input": _reference_input(args.reference),
            "autopsy_input": bind_file(args.autopsy),
            "control_input": bind_file(args.control),
            "predictions_input": None
            if args.predictions is None
            else bind_file(args.predictions),
        }
        freeze_manifest(frozen, args.output / "manifest.json", repo=Path.cwd())
        summary = run_escape_curve(
            experiment_config,
            config.metric,
            bands,
            args.snapshot,
            args.output,
            learning_rates=grid,
            streams=streams,
            hold_steps=hold_steps,
            weight_decay_override=args.weight_decay,
            thresholds=thresholds,
            workers=args.workers,
            torch_threads=args.torch_threads,
        )
        print(
            json.dumps(
                {
                    "curve_valid": summary["statistics"]["curve_valid"],
                    "eta50": summary["statistics"]["eta50"],
                    "fractions": {
                        str(cell["learning_rate"]): cell["primary_fraction"]
                        for cell in summary["cells"]
                    },
                    "output": str(args.output),
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "memory-factorial":
        bands = load_reference_bands(args.reference)
        thresholds = StateThresholds(**asdict(config.state))
        experiment_config = _runtime_experiment(config, args)
        gate0_input = _gate0_continue_input(args.gate0_analysis, config)
        frozen = {
            "kind": "gate1_memory_factorial",
            "protocol_version": config.protocol_version,
            "experiment": experiment_config,
            "metric": config.metric,
            "state": config.state,
            "reference_path": str(args.reference),
            "reference_input": _reference_input(args.reference),
            "expressed_snapshot": str(args.expressed_snapshot),
            "expressed_snapshot_input": _snapshot_input(
                args.expressed_snapshot, experiment_config
            ),
            "suppressed_snapshot": str(args.suppressed_snapshot),
            "suppressed_snapshot_input": _snapshot_input(
                args.suppressed_snapshot, experiment_config
            ),
            "gate0_analysis_input": gate0_input,
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
    elif args.command == "transition-time":
        bands = load_reference_bands(args.reference)
        thresholds = StateThresholds(**asdict(config.state))
        experiment_config = _runtime_experiment(config, args)
        gate0_input = _gate0_continue_input(args.gate0_analysis, config)
        frozen = {
            "kind": "gate1_transition_timing",
            "protocol_version": config.protocol_version,
            "experiment": experiment_config,
            "metric": config.metric,
            "state": config.state,
            "reference_path": str(args.reference),
            "reference_input": _reference_input(args.reference),
            "start_snapshot": str(args.snapshot) if args.snapshot else None,
            "start_snapshot_input": (
                _snapshot_input(args.snapshot, experiment_config) if args.snapshot else None
            ),
            "gate0_analysis_input": gate0_input,
            "arm": args.arm,
            "learning_rate": args.learning_rate,
            "maximum_steps": args.max_steps,
            "optimizer_reset": args.reset_optimizer,
        }
        freeze_manifest(frozen, args.output / "manifest.json", repo=Path.cwd())
        result = run_transition_timing(
            experiment_config,
            config.metric,
            bands,
            thresholds,
            arm=args.arm,
            learning_rate=args.learning_rate,
            maximum_steps=args.max_steps,
            output_dir=args.output,
            start_snapshot=args.snapshot,
            optimizer_reset=args.reset_optimizer,
        )
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
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
