"""Directional empirical boundaries for Gate 0."""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .config import Gate0Config, MBCExperimentConfig, MetricConfig
from .experiment import JSONLWriter, MBCExperiment
from .manifest import content_hash
from .parameter_groups import set_learning_rates
from .provenance import sha256_file
from .snapshot import load_snapshot, save_snapshot
from .state import (
    ReferenceBands,
    StateThresholds,
    first_transition_step,
    is_expressed,
    is_flat_loss,
    plateau_bounds,
)


@dataclass(frozen=True)
class ErasureResult:
    learning_rate: float
    erased: bool
    outcome: str
    sustained_entry_step: Optional[int]
    final_c_int: float
    final_exact_match: float
    final_delta_z: float
    final_full_vocab_ce: float
    provenance: Optional[Dict[str, object]] = None
    result_sha256: Optional[str] = None


@dataclass(frozen=True)
class AcquisitionResult:
    learning_rate: float
    outcome: str
    transition_step: Optional[int]
    final_c_int: float
    final_exact_match: float
    final_delta_z: float
    final_full_vocab_ce: float
    provenance: Optional[Dict[str, object]] = None
    result_sha256: Optional[str] = None


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _atomic_write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows)
    )
    temporary.replace(path)


def _metrics_prefix_binding(path: Path) -> Dict[str, object]:
    payload = Path(path).read_bytes()
    return {
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _decode_jsonl(payload: bytes, *, label: str) -> list[dict]:
    if payload and not payload.endswith(b"\n"):
        raise ValueError(f"{label} is truncated or lacks a final newline")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not valid UTF-8") from error
    rows = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise ValueError(f"{label} contains a blank line at {line_number}")
        try:
            row = json.loads(line, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"{label} contains malformed JSON at line {line_number}") from error
        if not isinstance(row, dict):
            raise ValueError(f"{label} row {line_number} is not a JSON object")
        rows.append(row)
    return rows


def _load_bound_metrics_prefix(
    path: Path, binding: object, *, label: str
) -> list[dict]:
    if not isinstance(binding, dict):
        raise ValueError(f"{label} progress has no metrics-prefix provenance")
    size = binding.get("size_bytes")
    digest = binding.get("sha256")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError(f"{label} metrics-prefix byte length is invalid")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{label} metrics-prefix digest is invalid")
    payload = Path(path).read_bytes()
    if len(payload) < size:
        raise ValueError(f"{label} metrics log is shorter than its bound prefix")
    prefix = payload[:size]
    if hashlib.sha256(prefix).hexdigest() != digest:
        raise ValueError(f"{label} metrics-prefix digest mismatch")
    # Validate the suffix too. It may contain complete post-checkpoint rows that
    # are intentionally discarded, but a torn append must fail closed.
    _decode_jsonl(payload, label=f"{label} metrics log")
    return _decode_jsonl(prefix, label=f"{label} bound metrics prefix")


def _branch_label(learning_rate: float) -> str:
    return f"eta_{learning_rate:.12g}".replace(".", "p")


def _control_value(value: Any) -> Any:
    if is_dataclass(value):
        return _control_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): _control_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_control_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dict__"):
        return _control_value(vars(value))
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return {"object_type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _reference_sha256(reference: ReferenceBands) -> str:
    return content_hash(_control_value(reference))


def _cell_provenance(
    kind: str,
    experiment_config: MBCExperimentConfig,
    metric: MetricConfig,
    gate: Gate0Config,
    reference: ReferenceBands,
    snapshot_path: Path,
    learning_rate: float,
    thresholds: StateThresholds,
    *,
    snapshot_sha256: Optional[str] = None,
    reference_sha256: Optional[str] = None,
) -> Dict[str, object]:
    snapshot_digest = snapshot_sha256 or sha256_file(snapshot_path)
    reference_digest = reference_sha256 or _reference_sha256(reference)
    identity: Dict[str, object] = {
        "schema_version": 1,
        "kind": kind,
        "controls": {
            "experiment": _control_value(experiment_config),
            "metric": _control_value(metric),
            "gate": _control_value(gate),
            "thresholds": _control_value(thresholds),
            "learning_rate": float(learning_rate),
        },
        "sources": {
            "snapshot_sha256": snapshot_digest,
            "reference_sha256": reference_digest,
        },
    }
    return {**identity, "cell_sha256": content_hash(identity)}


def _validate_provenance(
    observed: object, expected: Dict[str, object], *, location: Path
) -> None:
    if not isinstance(observed, dict):
        raise ValueError(f"cached cell at {location} has no provenance binding")
    declared = observed.get("cell_sha256")
    unsigned = {key: value for key, value in observed.items() if key != "cell_sha256"}
    if declared != content_hash(unsigned):
        raise ValueError(f"cached cell at {location} has tampered provenance")
    if observed != expected:
        raise ValueError(f"cached cell at {location} has mismatched provenance")


def _verify_sources(
    snapshot_path: Path,
    reference: ReferenceBands,
    *,
    expected_snapshot_sha256: str,
    expected_reference_sha256: str,
) -> None:
    if sha256_file(snapshot_path) != expected_snapshot_sha256:
        raise ValueError(f"snapshot changed before worker load: {snapshot_path}")
    if _reference_sha256(reference) != expected_reference_sha256:
        raise ValueError("reference bands changed before worker load")


def _seal_result(result: ErasureResult | AcquisitionResult):
    payload = asdict(result)
    payload.pop("result_sha256", None)
    return replace(result, result_sha256=content_hash(payload))


def _validate_result_digest(
    result: ErasureResult | AcquisitionResult, *, location: Path
) -> None:
    payload = asdict(result)
    declared = payload.pop("result_sha256", None)
    if declared != content_hash(payload):
        raise ValueError(f"completed branch at {location} has a tampered result payload")


def _validated_erasure_result(
    result: ErasureResult,
    expected_provenance: Dict[str, object],
    *,
    location: Path,
) -> ErasureResult:
    if result.outcome not in {"retained", "erased", "diverged", "unresolved"}:
        raise ValueError(f"completed erasure branch at {location} has an invalid outcome")
    if result.erased != (result.outcome == "erased"):
        raise ValueError(
            f"completed erasure branch at {location} has inconsistent outcome/erased fields"
        )
    if result.erased and result.sustained_entry_step is None:
        raise ValueError(f"completed erasure branch at {location} has no erasure entry")
    gate_controls = expected_provenance.get("controls", {}).get("gate", {})
    if isinstance(gate_controls, dict) and "erase_horizon" in gate_controls:
        entry_implies_erasure = (
            result.sustained_entry_step is not None
            and result.sustained_entry_step <= int(gate_controls["erase_horizon"])
        )
        if result.erased != entry_implies_erasure:
            raise ValueError(
                f"completed erasure branch at {location} has inconsistent entry/erased fields"
            )
    _validate_provenance(result.provenance, expected_provenance, location=location)
    _validate_result_digest(result, location=location)
    return result


def _validate_expected_cell_provenance(
    provenance: Dict[str, object],
    kind: str,
    experiment_config: MBCExperimentConfig,
    metric: MetricConfig,
    gate: Gate0Config,
    reference: ReferenceBands,
    snapshot_path: Path,
    learning_rate: float,
    thresholds: StateThresholds,
    *,
    location: Path,
) -> None:
    sources = provenance.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("expected cell provenance has no source bindings")
    expected = _cell_provenance(
        kind,
        experiment_config,
        metric,
        gate,
        reference,
        snapshot_path,
        learning_rate,
        thresholds,
        snapshot_sha256=str(sources.get("snapshot_sha256")),
        reference_sha256=str(sources.get("reference_sha256")),
    )
    _validate_provenance(provenance, expected, location=location)


def _load_erasure_result(
    path: Path,
    learning_rate: float,
    expected_provenance: Dict[str, object],
) -> Optional[ErasureResult]:
    if not path.exists():
        return None
    result = ErasureResult(**json.loads(path.read_text()))
    if not math.isclose(result.learning_rate, learning_rate, rel_tol=1e-12, abs_tol=0.0):
        raise ValueError(f"completed branch at {path} has the wrong learning rate")
    return _validated_erasure_result(result, expected_provenance, location=path)


def _load_acquisition_result(
    path: Path,
    learning_rate: float,
    expected_provenance: Dict[str, object],
) -> Optional[AcquisitionResult]:
    if not path.exists():
        return None
    result = AcquisitionResult(**json.loads(path.read_text()))
    if not math.isclose(result.learning_rate, learning_rate, rel_tol=1e-12, abs_tol=0.0):
        raise ValueError(f"completed branch at {path} has the wrong learning rate")
    if result.outcome not in {"transitioned", "censored", "diverged"}:
        raise ValueError(f"completed acquisition branch at {path} has an invalid outcome")
    if result.outcome == "transitioned" and result.transition_step is None:
        raise ValueError(
            f"completed acquisition branch at {path} has an inconsistent transition"
        )
    if result.outcome == "censored" and result.transition_step is not None:
        raise ValueError(
            f"completed acquisition branch at {path} has an inconsistent transition"
        )
    _validate_provenance(result.provenance, expected_provenance, location=path)
    _validate_result_digest(result, location=path)
    return result


def _run_erasure_scan_cell(
    experiment_config: MBCExperimentConfig,
    metric: MetricConfig,
    gate: Gate0Config,
    reference: ReferenceBands,
    snapshot_path: Path,
    learning_rate: float,
    branch_dir: Path,
    thresholds: Optional[StateThresholds],
    expected_provenance: Dict[str, object],
) -> ErasureResult:
    """Spawn-safe entry point for one independent erasure rate cell."""
    sources = expected_provenance["sources"]
    _verify_sources(
        snapshot_path,
        reference,
        expected_snapshot_sha256=str(sources["snapshot_sha256"]),
        expected_reference_sha256=str(sources["reference_sha256"]),
    )
    return run_erasure_branch(
        experiment_config,
        metric,
        gate,
        reference,
        snapshot_path,
        learning_rate,
        branch_dir,
        thresholds=thresholds,
        _expected_provenance=expected_provenance,
    )


def _run_acquisition_scan_cell(
    experiment_config: MBCExperimentConfig,
    metric: MetricConfig,
    gate: Gate0Config,
    reference: ReferenceBands,
    snapshot_path: Path,
    learning_rate: float,
    branch_dir: Path,
    thresholds: Optional[StateThresholds],
    expected_provenance: Dict[str, object],
) -> AcquisitionResult:
    """Spawn-safe entry point for one independent acquisition rate cell."""
    sources = expected_provenance["sources"]
    _verify_sources(
        snapshot_path,
        reference,
        expected_snapshot_sha256=str(sources["snapshot_sha256"]),
        expected_reference_sha256=str(sources["reference_sha256"]),
    )
    return run_acquisition_branch(
        experiment_config,
        metric,
        gate,
        reference,
        snapshot_path,
        learning_rate,
        branch_dir,
        thresholds=thresholds,
        _expected_provenance=expected_provenance,
    )


def sustained_band_entry(
    rows: Sequence[Dict[str, float]],
    reference: ReferenceBands,
    *,
    branch_end_step: int,
    thresholds: Optional[StateThresholds] = None,
) -> Optional[int]:
    thresholds = thresholds or StateThresholds()
    low, high = plateau_bounds(reference, thresholds)
    for index, row in enumerate(rows):
        if not (
            low <= row["c_int"] <= high
            and is_flat_loss(
                row["full_vocab_ce"],
                reference,
                relative_tolerance=thresholds.flat_loss_relative_tolerance,
            )
        ):
            continue
        if int(rows[-1]["branch_step"]) < branch_end_step:
            return None
        if all(
            low <= future["c_int"] <= high
            and is_flat_loss(
                future["full_vocab_ce"],
                reference,
                relative_tolerance=thresholds.flat_loss_relative_tolerance,
            )
            for future in rows[index:]
        ):
            return int(row["branch_step"])
    return None


def run_erasure_branch(
    experiment_config: MBCExperimentConfig,
    metric: MetricConfig,
    gate: Gate0Config,
    reference: ReferenceBands,
    snapshot_path: Path,
    learning_rate: float,
    output_dir: Path,
    *,
    thresholds: Optional[StateThresholds] = None,
    _expected_provenance: Optional[Dict[str, object]] = None,
) -> ErasureResult:
    thresholds = thresholds or StateThresholds()
    output_dir = Path(output_dir)
    provenance = _expected_provenance or _cell_provenance(
        "gate0_erasure",
        experiment_config,
        metric,
        gate,
        reference,
        snapshot_path,
        learning_rate,
        thresholds,
    )
    _validate_expected_cell_provenance(
        provenance,
        "gate0_erasure",
        experiment_config,
        metric,
        gate,
        reference,
        snapshot_path,
        learning_rate,
        thresholds,
        location=output_dir,
    )
    sources = provenance["sources"]
    result_path = output_dir / "result.json"
    completed = _load_erasure_result(result_path, learning_rate, provenance)
    if completed is not None:
        return completed
    experiment = MBCExperiment(replace(experiment_config, learning_rate=learning_rate), metric)
    progress_path = output_dir / "progress.json"
    metrics_path = output_dir / "metrics.jsonl"
    rows = []
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        _validate_provenance(progress.get("provenance"), provenance, location=progress_path)
        if (
            progress.get("learning_rate") != learning_rate
            or int(progress.get("erase_hold_steps", -1)) != gate.erase_hold_steps
        ):
            raise ValueError("erasure-branch progress has mismatched controls")
        checkpoint_path = output_dir / progress["checkpoint_path"]
        if not checkpoint_path.exists():
            raise FileNotFoundError("erasure progress references a missing checkpoint")
        expected_checkpoint_sha256 = progress.get("checkpoint_sha256")
        if not isinstance(expected_checkpoint_sha256, str):
            raise ValueError("erasure progress has no checkpoint provenance")
        if sha256_file(checkpoint_path) != expected_checkpoint_sha256:
            raise ValueError("erasure progress checkpoint digest mismatch")
        restored = load_snapshot(
            checkpoint_path,
            model=experiment.model,
            optimizer=experiment.optimizer,
            stream=experiment.stream,
            map_location=experiment.device,
        )
        experiment.step = restored["step"]
        branch_start = int(progress["branch_start"])
        completed_steps = int(progress["completed_branch_steps"])
        active_slot = int(progress["active_slot"])
        if experiment.step - branch_start != completed_steps:
            raise ValueError("erasure checkpoint and progress disagree")
        if not metrics_path.exists():
            raise FileNotFoundError("erasure progress has no metrics log")
        logged = _load_bound_metrics_prefix(
            metrics_path,
            progress.get("metrics_prefix"),
            label="erasure",
        )
        logged = [row for row in logged if int(row["branch_step"]) <= completed_steps]
        if not logged or int(logged[-1]["branch_step"]) != completed_steps:
            raise ValueError("erasure metrics do not reach the progress checkpoint")
        _atomic_write_jsonl(metrics_path, logged)
        rows = logged
    else:
        if metrics_path.exists():
            _atomic_write_jsonl(metrics_path, [])
        _verify_sources(
            snapshot_path,
            reference,
            expected_snapshot_sha256=str(sources["snapshot_sha256"]),
            expected_reference_sha256=str(sources["reference_sha256"]),
        )
        restored = load_snapshot(
            snapshot_path,
            model=experiment.model,
            optimizer=experiment.optimizer,
            stream=experiment.stream,
            map_location=experiment.device,
        )
        experiment.step = restored["step"]
        branch_start = experiment.step
        active_slot = -1
    set_learning_rates(experiment.optimizer, learning_rate)
    writer = JSONLWriter(metrics_path)
    checkpoint_every = max(1_000, metric.eval_every)
    while experiment.step - branch_start < gate.erase_hold_steps:
        chunk = min(metric.eval_every, gate.erase_hold_steps - (experiment.step - branch_start))
        training = experiment.advance(chunk)
        row = {**training, **experiment.evaluate()}
        row["branch_step"] = float(experiment.step - branch_start)
        row["learning_rate"] = learning_rate
        rows.append(row)
        writer.write({"kind": "gate0_erasure", **row})
        branch_step = int(row["branch_step"])
        if branch_step % checkpoint_every == 0 or branch_step == gate.erase_hold_steps:
            active_slot = 1 if active_slot != 1 else 0
            checkpoint_path = output_dir / "checkpoints" / f"slot_{active_slot}.pt"
            temporary = checkpoint_path.with_suffix(".pt.tmp")
            save_snapshot(
                temporary,
                model=experiment.model,
                optimizer=experiment.optimizer,
                stream=experiment.stream,
                step=experiment.step,
                metadata={
                    "kind": "gate0_erasure_progress",
                    "learning_rate": learning_rate,
                    "branch_start": branch_start,
                },
            )
            temporary.replace(checkpoint_path)
            checkpoint_sha256 = sha256_file(checkpoint_path)
            metrics_prefix = _metrics_prefix_binding(metrics_path)
            _atomic_write_json(
                progress_path,
                {
                    "learning_rate": learning_rate,
                    "erase_hold_steps": gate.erase_hold_steps,
                    "branch_start": branch_start,
                    "completed_branch_steps": branch_step,
                    "active_slot": active_slot,
                    "checkpoint_path": str(checkpoint_path.relative_to(output_dir)),
                    "checkpoint_sha256": checkpoint_sha256,
                    "metrics_prefix": metrics_prefix,
                    "provenance": provenance,
                },
            )
    entry = sustained_band_entry(
        rows, reference, branch_end_step=gate.erase_hold_steps, thresholds=thresholds
    )
    latest = rows[-1]
    erased = entry is not None and entry <= gate.erase_horizon
    divergent = (
        not math.isfinite(latest["full_vocab_ce"])
        or latest["full_vocab_ce"] > 5.0 * reference.q_star_loss_mean
    )
    if entry is not None and entry <= gate.erase_horizon:
        outcome = "erased"
    elif divergent:
        outcome = "diverged"
    elif is_expressed(
        latest["c_int"],
        latest["exact_match"],
        latest["delta_z"],
        reference,
        thresholds,
    ):
        outcome = "retained"
    else:
        outcome = "unresolved"
    result = _seal_result(ErasureResult(
        learning_rate=learning_rate,
        erased=erased,
        outcome=outcome,
        sustained_entry_step=entry,
        final_c_int=latest["c_int"],
        final_exact_match=latest["exact_match"],
        final_delta_z=latest["delta_z"],
        final_full_vocab_ce=latest["full_vocab_ce"],
        provenance=provenance,
    ))
    _validated_erasure_result(result, provenance, location=result_path)
    _atomic_write_json(result_path, asdict(result))
    return result


def run_erasure_scan(
    experiment_config: MBCExperimentConfig,
    metric: MetricConfig,
    gate: Gate0Config,
    reference: ReferenceBands,
    snapshot_path: Path,
    learning_rates: Sequence[float],
    output_dir: Path,
    *,
    thresholds: Optional[StateThresholds] = None,
    workers: int = 1,
) -> dict:
    """Run a resumable coarse scan and expose only strict adjacent brackets.

    Rate cells are independent. With ``workers > 1`` they run in isolated spawned
    processes, while this parent remains the sole writer of the ordered scan
    summary. Completed per-rate results are always loaded before work is
    scheduled, so changing worker count on resume cannot change the trajectory.
    """
    rates = sorted(float(rate) for rate in learning_rates)
    if not rates or any(rate <= 0 for rate in rates):
        raise ValueError("erasure scan learning rates must be positive")
    if len(set(rates)) != len(rates):
        raise ValueError("erasure scan learning rates must be unique")
    if workers < 1:
        raise ValueError("erasure scan workers must be at least one")
    output_dir = Path(output_dir)
    effective_thresholds = thresholds or StateThresholds()
    snapshot_sha256 = sha256_file(snapshot_path)
    reference_sha256 = _reference_sha256(reference)
    provenances = {
        learning_rate: _cell_provenance(
            "gate0_erasure",
            experiment_config,
            metric,
            gate,
            reference,
            snapshot_path,
            learning_rate,
            effective_thresholds,
            snapshot_sha256=snapshot_sha256,
            reference_sha256=reference_sha256,
        )
        for learning_rate in rates
    }
    cached: Dict[float, ErasureResult] = {}
    missing = []
    for learning_rate in rates:
        branch_dir = output_dir / _branch_label(learning_rate)
        result_path = branch_dir / "result.json"
        result = _load_erasure_result(
            result_path, learning_rate, provenances[learning_rate]
        )
        if result is not None:
            cached[learning_rate] = result
        else:
            missing.append((learning_rate, branch_dir))

    futures = {}
    executor = None
    if workers > 1 and len(missing) > 1:
        executor = ProcessPoolExecutor(
            max_workers=min(workers, len(missing)),
            mp_context=multiprocessing.get_context("spawn"),
        )
        futures = {
            learning_rate: executor.submit(
                _run_erasure_scan_cell,
                experiment_config,
                metric,
                gate,
                reference,
                snapshot_path,
                learning_rate,
                branch_dir,
                thresholds,
                provenances[learning_rate],
            )
            for learning_rate, branch_dir in missing
        }

    results = []
    try:
        for learning_rate in rates:
            result = cached.get(learning_rate)
            if result is None:
                branch_dir = output_dir / _branch_label(learning_rate)
                if executor is None:
                    result = _run_erasure_scan_cell(
                        experiment_config,
                        metric,
                        gate,
                        reference,
                        snapshot_path,
                        learning_rate,
                        branch_dir,
                        thresholds,
                        provenances[learning_rate],
                    )
                else:
                    result = futures[learning_rate].result()
                if result.provenance is None:
                    result = replace(result, provenance=provenances[learning_rate])
                if result.result_sha256 is None:
                    result = _seal_result(result)
                _validated_erasure_result(
                    result,
                    provenances[learning_rate],
                    location=branch_dir / "result.json",
                )
                _atomic_write_json(branch_dir / "result.json", asdict(result))
            results.append(result)
            partial = {
                "complete": len(results) == len(rates),
                "requested_learning_rates": rates,
                "branches": [asdict(branch) for branch in results],
            }
            _atomic_write_json(output_dir / "scan.json", partial)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
    adjacent_brackets = [
        {
            "lower_learning_rate": lower.learning_rate,
            "upper_learning_rate": upper.learning_rate,
        }
        for lower, upper in zip(results, results[1:])
        if lower.outcome == "retained" and upper.outcome == "erased"
    ]
    summary = {
        "complete": True,
        "requested_learning_rates": rates,
        "strict_adjacent_brackets": adjacent_brackets,
        "branches": [asdict(branch) for branch in results],
    }
    _atomic_write_json(output_dir / "scan.json", summary)
    return summary


def geometric_erasure_bisection(
    experiment_config: MBCExperimentConfig,
    metric: MetricConfig,
    gate: Gate0Config,
    reference: ReferenceBands,
    snapshot_path: Path,
    *,
    lower_learning_rate: float,
    upper_learning_rate: float,
    output_dir: Path,
    thresholds: Optional[StateThresholds] = None,
) -> dict:
    if not 0 < lower_learning_rate < upper_learning_rate:
        raise ValueError("erasure bracket must be positive and ordered")
    output_dir = Path(output_dir)
    effective_thresholds = thresholds or StateThresholds()
    snapshot_sha256 = sha256_file(snapshot_path)
    reference_sha256 = _reference_sha256(reference)
    cache: Dict[float, ErasureResult] = {}

    def evaluate(learning_rate: float) -> ErasureResult:
        if learning_rate not in cache:
            branch_dir = output_dir / _branch_label(learning_rate)
            provenance = _cell_provenance(
                "gate0_erasure",
                experiment_config,
                metric,
                gate,
                reference,
                snapshot_path,
                learning_rate,
                effective_thresholds,
                snapshot_sha256=snapshot_sha256,
                reference_sha256=reference_sha256,
            )
            existing = _load_erasure_result(
                branch_dir / "result.json", learning_rate, provenance
            )
            if existing is not None:
                cache[learning_rate] = existing
            else:
                result = run_erasure_branch(
                    experiment_config,
                    metric,
                    gate,
                    reference,
                    snapshot_path,
                    learning_rate,
                    branch_dir,
                    thresholds=thresholds,
                    _expected_provenance=provenance,
                )
                if result.provenance is None:
                    result = replace(result, provenance=provenance)
                if result.result_sha256 is None:
                    result = _seal_result(result)
                cache[learning_rate] = _validated_erasure_result(
                    result, provenance, location=branch_dir / "result.json"
                )
        return cache[learning_rate]

    lower = evaluate(lower_learning_rate)
    upper = evaluate(upper_learning_rate)
    if lower.outcome != "retained" or upper.outcome != "erased":
        raise ValueError(
            "invalid erasure bracket: lower must be retained and upper must be erased"
        )
    for _ in range(gate.boundary_bisection_steps):
        midpoint = math.sqrt(lower.learning_rate * upper.learning_rate)
        result = evaluate(midpoint)
        if result.erased:
            upper = result
        elif result.outcome == "retained":
            lower = result
        else:
            raise ValueError(
                f"erasure bracket encountered invalid midpoint outcome: {result.outcome}"
            )
    summary = {
        "largest_non_erasing_learning_rate": lower.learning_rate,
        "smallest_erasing_learning_rate": upper.learning_rate,
        "multiplicative_interval": upper.learning_rate / lower.learning_rate,
        "branches": [asdict(cache[key]) for key in sorted(cache)],
    }
    _atomic_write_json(output_dir / "boundary.json", summary)
    return summary


def run_acquisition_branch(
    experiment_config: MBCExperimentConfig,
    metric: MetricConfig,
    gate: Gate0Config,
    reference: ReferenceBands,
    snapshot_path: Path,
    learning_rate: float,
    output_dir: Path,
    *,
    thresholds: Optional[StateThresholds] = None,
    _expected_provenance: Optional[Dict[str, object]] = None,
) -> AcquisitionResult:
    """Run an exactly resumable fixed-rule branch from one suppressed state."""
    thresholds = thresholds or StateThresholds()
    output_dir = Path(output_dir)
    provenance = _expected_provenance or _cell_provenance(
        "gate0_acquisition",
        experiment_config,
        metric,
        gate,
        reference,
        snapshot_path,
        learning_rate,
        thresholds,
    )
    _validate_expected_cell_provenance(
        provenance,
        "gate0_acquisition",
        experiment_config,
        metric,
        gate,
        reference,
        snapshot_path,
        learning_rate,
        thresholds,
        location=output_dir,
    )
    sources = provenance["sources"]
    result_path = output_dir / "result.json"
    completed = _load_acquisition_result(result_path, learning_rate, provenance)
    if completed is not None:
        return completed
    experiment = MBCExperiment(replace(experiment_config, learning_rate=learning_rate), metric)
    progress_path = output_dir / "progress.json"
    metrics_path = output_dir / "metrics.jsonl"
    analysis_rows = []
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        _validate_provenance(progress.get("provenance"), provenance, location=progress_path)
        if (
            progress.get("learning_rate") != learning_rate
            or int(progress.get("acquire_horizon", -1)) != gate.acquire_horizon
        ):
            raise ValueError("acquisition-branch progress has mismatched controls")
        checkpoint_path = output_dir / progress["checkpoint_path"]
        if not checkpoint_path.exists():
            raise FileNotFoundError("acquisition progress references a missing checkpoint")
        expected_checkpoint_sha256 = progress.get("checkpoint_sha256")
        if not isinstance(expected_checkpoint_sha256, str):
            raise ValueError("acquisition progress has no checkpoint provenance")
        if sha256_file(checkpoint_path) != expected_checkpoint_sha256:
            raise ValueError("acquisition progress checkpoint digest mismatch")
        restored = load_snapshot(
            checkpoint_path,
            model=experiment.model,
            optimizer=experiment.optimizer,
            stream=experiment.stream,
            map_location=experiment.device,
        )
        experiment.step = restored["step"]
        branch_start = int(progress["branch_start"])
        completed_steps = int(progress["completed_branch_steps"])
        active_slot = int(progress["active_slot"])
        if experiment.step - branch_start != completed_steps:
            raise ValueError("acquisition checkpoint and progress disagree")
        if not metrics_path.exists():
            raise FileNotFoundError("acquisition progress has no metrics log")
        logged = _load_bound_metrics_prefix(
            metrics_path,
            progress.get("metrics_prefix"),
            label="acquisition",
        )
        logged = [
            row
            for row in logged
            if row["kind"] == "gate0_acquisition_start"
            or int(row["branch_step"]) <= completed_steps
        ]
        timed = [row for row in logged if row["kind"] == "gate0_acquisition"]
        if not timed or int(timed[-1]["branch_step"]) != completed_steps:
            raise ValueError("acquisition metrics do not reach the progress checkpoint")
        _atomic_write_jsonl(metrics_path, logged)
        analysis_rows = [{**row, "step": float(row["branch_step"])} for row in timed]
        latest = timed[-1]
    else:
        if metrics_path.exists():
            _atomic_write_jsonl(metrics_path, [])
        _verify_sources(
            snapshot_path,
            reference,
            expected_snapshot_sha256=str(sources["snapshot_sha256"]),
            expected_reference_sha256=str(sources["reference_sha256"]),
        )
        restored = load_snapshot(
            snapshot_path,
            model=experiment.model,
            optimizer=experiment.optimizer,
            stream=experiment.stream,
            map_location=experiment.device,
        )
        experiment.step = restored["step"]
        branch_start = experiment.step
        active_slot = -1
        latest = experiment.evaluate()
        low, high = plateau_bounds(reference, thresholds)
        if not (
            low <= latest["c_int"] <= high
            and is_flat_loss(
                latest["full_vocab_ce"],
                reference,
                relative_tolerance=thresholds.flat_loss_relative_tolerance,
            )
        ):
            raise ValueError(
                "acquisition branch must start from a suppressed flat-loss snapshot"
            )
    set_learning_rates(experiment.optimizer, learning_rate)
    writer = JSONLWriter(metrics_path)
    if not analysis_rows:
        writer.write(
            {
                "kind": "gate0_acquisition_start",
                **latest,
                "branch_step": 0.0,
                "learning_rate": learning_rate,
            }
        )
    checkpoint_every = max(1_000, metric.eval_every)
    while experiment.step - branch_start < gate.acquire_horizon:
        chunk = min(metric.eval_every, gate.acquire_horizon - (experiment.step - branch_start))
        training = experiment.advance(chunk)
        latest = {**training, **experiment.evaluate()}
        branch_step = experiment.step - branch_start
        logged = {**latest, "branch_step": float(branch_step), "learning_rate": learning_rate}
        writer.write({"kind": "gate0_acquisition", **logged})
        analysis_rows.append({**latest, "step": float(branch_step)})
        if branch_step % checkpoint_every == 0 or branch_step == gate.acquire_horizon:
            active_slot = 1 if active_slot != 1 else 0
            checkpoint_path = output_dir / "checkpoints" / f"slot_{active_slot}.pt"
            temporary = checkpoint_path.with_suffix(".pt.tmp")
            save_snapshot(
                temporary,
                model=experiment.model,
                optimizer=experiment.optimizer,
                stream=experiment.stream,
                step=experiment.step,
                metadata={
                    "kind": "gate0_acquisition_progress",
                    "learning_rate": learning_rate,
                    "branch_start": branch_start,
                },
            )
            temporary.replace(checkpoint_path)
            checkpoint_sha256 = sha256_file(checkpoint_path)
            metrics_prefix = _metrics_prefix_binding(metrics_path)
            _atomic_write_json(
                progress_path,
                {
                    "learning_rate": learning_rate,
                    "acquire_horizon": gate.acquire_horizon,
                    "branch_start": branch_start,
                    "completed_branch_steps": branch_step,
                    "active_slot": active_slot,
                    "checkpoint_path": str(checkpoint_path.relative_to(output_dir)),
                    "checkpoint_sha256": checkpoint_sha256,
                    "metrics_prefix": metrics_prefix,
                    "provenance": provenance,
                },
            )
    transition = first_transition_step(analysis_rows, reference, thresholds)
    divergent = (
        not math.isfinite(latest["full_vocab_ce"])
        or latest["full_vocab_ce"] > 5.0 * reference.q_star_loss_mean
    )
    outcome = "diverged" if divergent else ("transitioned" if transition is not None else "censored")
    result = _seal_result(AcquisitionResult(
        learning_rate=learning_rate,
        outcome=outcome,
        transition_step=transition,
        final_c_int=latest["c_int"],
        final_exact_match=latest["exact_match"],
        final_delta_z=latest["delta_z"],
        final_full_vocab_ce=latest["full_vocab_ce"],
        provenance=provenance,
    ))
    _atomic_write_json(result_path, asdict(result))
    return result


def run_acquisition_scan(
    experiment_config: MBCExperimentConfig,
    metric: MetricConfig,
    gate: Gate0Config,
    reference: ReferenceBands,
    snapshot_path: Path,
    learning_rates: Sequence[float],
    output_dir: Path,
    *,
    thresholds: Optional[StateThresholds] = None,
    workers: int = 1,
) -> dict:
    """Run resumable acquisition rate cells and retain strict event outcomes.

    A timeout remains censored and divergence remains divergence: neither is
    silently treated as a stable non-transition. The parent writes a
    learning-rate-ordered aggregate independent of scheduling and worker count.
    """
    rates = sorted(float(rate) for rate in learning_rates)
    if not rates or any(rate <= 0 for rate in rates):
        raise ValueError("acquisition scan learning rates must be positive")
    if len(set(rates)) != len(rates):
        raise ValueError("acquisition scan learning rates must be unique")
    if workers < 1:
        raise ValueError("acquisition scan workers must be at least one")
    output_dir = Path(output_dir)
    effective_thresholds = thresholds or StateThresholds()
    snapshot_sha256 = sha256_file(snapshot_path)
    reference_sha256 = _reference_sha256(reference)
    provenances = {
        learning_rate: _cell_provenance(
            "gate0_acquisition",
            experiment_config,
            metric,
            gate,
            reference,
            snapshot_path,
            learning_rate,
            effective_thresholds,
            snapshot_sha256=snapshot_sha256,
            reference_sha256=reference_sha256,
        )
        for learning_rate in rates
    }
    cached: Dict[float, AcquisitionResult] = {}
    missing = []
    for learning_rate in rates:
        branch_dir = output_dir / _branch_label(learning_rate)
        result = _load_acquisition_result(
            branch_dir / "result.json", learning_rate, provenances[learning_rate]
        )
        if result is not None:
            cached[learning_rate] = result
        else:
            missing.append((learning_rate, branch_dir))

    futures = {}
    executor = None
    if workers > 1 and len(missing) > 1:
        executor = ProcessPoolExecutor(
            max_workers=min(workers, len(missing)),
            mp_context=multiprocessing.get_context("spawn"),
        )
        futures = {
            learning_rate: executor.submit(
                _run_acquisition_scan_cell,
                experiment_config,
                metric,
                gate,
                reference,
                snapshot_path,
                learning_rate,
                branch_dir,
                thresholds,
                provenances[learning_rate],
            )
            for learning_rate, branch_dir in missing
        }

    results = []
    try:
        for learning_rate in rates:
            result = cached.get(learning_rate)
            if result is None:
                branch_dir = output_dir / _branch_label(learning_rate)
                if executor is None:
                    result = _run_acquisition_scan_cell(
                        experiment_config,
                        metric,
                        gate,
                        reference,
                        snapshot_path,
                        learning_rate,
                        branch_dir,
                        thresholds,
                        provenances[learning_rate],
                    )
                else:
                    result = futures[learning_rate].result()
                if result.provenance is None:
                    result = replace(result, provenance=provenances[learning_rate])
                if result.result_sha256 is None:
                    result = _seal_result(result)
                _validate_provenance(
                    result.provenance,
                    provenances[learning_rate],
                    location=branch_dir / "result.json",
                )
                _validate_result_digest(result, location=branch_dir / "result.json")
                _atomic_write_json(branch_dir / "result.json", asdict(result))
            results.append(result)
            _atomic_write_json(
                output_dir / "scan.json",
                {
                    "complete": len(results) == len(rates),
                    "requested_learning_rates": rates,
                    "branches": [asdict(branch) for branch in results],
                },
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)

    transitioned = [
        result.learning_rate for result in results if result.outcome == "transitioned"
    ]
    summary = {
        "complete": True,
        "requested_learning_rates": rates,
        "largest_transitioning_learning_rate": max(transitioned, default=None),
        "outcome_counts": {
            outcome: sum(result.outcome == outcome for result in results)
            for outcome in ("transitioned", "censored", "diverged")
        },
        "branches": [asdict(branch) for branch in results],
    }
    _atomic_write_json(output_dir / "scan.json", summary)
    return summary
