"""Strict adjudication and provenance binding for Gate 0 calibration.

Calibration is the one place where a non-gate seed is allowed to choose the
empirical erasure bracket.  This module keeps that choice mechanical: invalid
or mismatched inputs raise, while valid evidence that misses a preregistered
scientific criterion produces a terminal ``stop`` report.

The returned report contains only JSON-compatible values.  It can be written
as an artifact, embedded in a frozen manifest, or reduced to a compact bound
input with :func:`bind_passed_gate0_calibration`.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .config import Gate0Config, MBCExperimentConfig
from .gate0_analysis import CellKey, Direction, validate_scan_artifact
from .provenance import (
    bind_file,
    bind_manifest,
    bind_nearest_manifest,
    bind_reference,
    bind_snapshot,
)


POSITIVE_CONTROL_LEARNING_RATES: Tuple[float, ...] = (0.5, 0.9, 1.0, 1.1, 1.5)
_POSITIVE_CONTROL_TARGET = 1.0
_NUMERICAL_TOLERANCE = 1e-10


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _load_json_object(path: Path, *, label: str) -> Dict[str, Any]:
    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not a readable JSON document: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_sequence(value: Any, *, label: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    return value


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return int(value)


def _sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        raise ValueError(f"{label} must be a 64-character hexadecimal SHA-256 digest")
    return value.lower()


def _json_normalized(value: Any) -> Any:
    """Match the tuple-to-array normalization performed by frozen manifests."""
    return json.loads(json.dumps(value, allow_nan=False))


def _gate_from_json(value: Any) -> Gate0Config:
    raw = dict(_require_mapping(value, label="Gate 0 config"))
    for name in (
        "seeds",
        "learning_rates",
        "positive_control_rates",
        "batch_sizes",
        "primary_batch_contrast",
        "secondary_batch_contrast",
    ):
        if name in raw:
            raw[name] = tuple(raw[name])
    return Gate0Config(**raw)


def _exact_rates(value: Any, expected: Tuple[float, ...], *, label: str) -> None:
    rows = _require_sequence(value, label=label)
    observed = tuple(
        _finite_number(rate, label=f"{label}[{index}]")
        for index, rate in enumerate(rows)
    )
    if observed != expected:
        raise ValueError(f"{label} does not match the frozen learning-rate grid")


def _same_path(observed: Any, expected: Path, *, label: str) -> None:
    if not isinstance(observed, str):
        raise ValueError(f"{label} must be a path string")
    if Path(observed).resolve() != Path(expected).resolve():
        raise ValueError(f"{label} does not identify the frozen calibration snapshot")


def _compare_binding(
    observed: Any,
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    observed = _require_mapping(observed, label=label)
    _same_path(observed.get("path"), Path(str(expected["path"])), label=f"{label}.path")
    observed_without_path = {key: value for key, value in observed.items() if key != "path"}
    expected_without_path = {key: value for key, value in expected.items() if key != "path"}
    if observed_without_path != expected_without_path:
        raise ValueError(f"{label} does not match the current content binding")


def _validate_scan_manifest(
    manifest_path: Path,
    *,
    scan_path: Path,
    expected_kind: str,
    protocol_version: str,
    experiment: MBCExperimentConfig,
    gate: Gate0Config,
    snapshot_path: Path,
    expected_snapshot_input: Mapping[str, Any],
) -> Dict[str, Any]:
    manifest_path = Path(manifest_path)
    scan_path = Path(scan_path)
    if manifest_path.resolve() != (scan_path.parent / "manifest.json").resolve():
        raise ValueError(f"{expected_kind} result is not paired with its own manifest")
    manifest_binding = bind_manifest(manifest_path)
    manifest = _load_json_object(manifest_path, label=f"{expected_kind} manifest")
    config = _require_mapping(manifest.get("config"), label=f"{expected_kind} manifest config")
    if config.get("kind") != expected_kind:
        raise ValueError(f"expected a {expected_kind} manifest")
    if config.get("protocol_version") != protocol_version:
        raise ValueError(f"{expected_kind} manifest has the wrong protocol version")
    if config.get("experiment") != _json_normalized(asdict(experiment)):
        raise ValueError(f"{expected_kind} manifest has the wrong experiment config")
    if config.get("gate0") != _json_normalized(asdict(gate)):
        raise ValueError(f"{expected_kind} manifest has the wrong Gate 0 config")
    _exact_rates(
        config.get("learning_rates"),
        tuple(gate.learning_rates),
        label=f"{expected_kind} manifest learning_rates",
    )
    _same_path(
        config.get("snapshot"),
        snapshot_path,
        label=f"{expected_kind} manifest snapshot",
    )
    snapshot_input = _require_mapping(
        config.get("snapshot_input"), label=f"{expected_kind} snapshot_input"
    )
    _compare_binding(
        snapshot_input.get("artifact"),
        _require_mapping(expected_snapshot_input["artifact"], label="snapshot artifact"),
        label=f"{expected_kind} snapshot artifact",
    )
    _compare_binding(
        snapshot_input.get("source_manifest"),
        _require_mapping(
            expected_snapshot_input["source_manifest"], label="snapshot source manifest"
        ),
        label=f"{expected_kind} snapshot source manifest",
    )
    return manifest_binding


def _validate_positive_control_manifest(
    manifest_path: Path,
    *,
    result_path: Path,
    protocol_version: str,
) -> Dict[str, Any]:
    manifest_path = Path(manifest_path)
    result_path = Path(result_path)
    if manifest_path.resolve() != (result_path.parent / "manifest.json").resolve():
        raise ValueError("positive-control result is not paired with its own manifest")
    binding = bind_manifest(manifest_path)
    manifest = _load_json_object(manifest_path, label="positive-control manifest")
    config = _require_mapping(manifest.get("config"), label="positive-control manifest config")
    if config.get("kind") != "gate0_positive_control":
        raise ValueError("expected a gate0_positive_control manifest")
    if config.get("protocol_version") != protocol_version:
        raise ValueError("positive-control manifest has the wrong protocol version")
    _exact_rates(
        config.get("learning_rates"),
        POSITIVE_CONTROL_LEARNING_RATES,
        label="positive-control manifest learning_rates",
    )
    target = _finite_number(config.get("target"), label="positive-control manifest target")
    if target != _POSITIVE_CONTROL_TARGET:
        raise ValueError("positive-control manifest has the wrong frozen target")
    return binding


def _validate_positive_control(payload: Mapping[str, Any]) -> Tuple[bool, Tuple[str, ...]]:
    if payload.get("system") != "depth_2_scalar_linear_network":
        raise ValueError("positive control has the wrong system")
    target = _finite_number(payload.get("target"), label="positive control target")
    analytic = _finite_number(
        payload.get("analytic_critical_learning_rate"),
        label="positive control analytic critical learning rate",
    )
    hessian = _finite_number(
        payload.get("hessian_lambda_max"), label="positive control Hessian eigenvalue"
    )
    if not math.isclose(target, _POSITIVE_CONTROL_TARGET, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("positive control target does not match the frozen target")
    if not math.isclose(analytic, 1.0 / target, rel_tol=0.0, abs_tol=_NUMERICAL_TOLERANCE):
        raise ValueError("positive control analytic boundary is inconsistent with its target")
    if not math.isclose(hessian, 2.0 * target, rel_tol=0.0, abs_tol=_NUMERICAL_TOLERANCE):
        raise ValueError("positive control Hessian eigenvalue is inconsistent with its target")

    rows = _require_sequence(payload.get("rows"), label="positive control rows")
    if len(rows) != len(POSITIVE_CONTROL_LEARNING_RATES):
        raise ValueError("positive control does not contain the frozen rate grid")
    observed_rates = []
    failures = []
    has_lower = has_upper = has_critical = False
    for index, (raw_row, expected_rate) in enumerate(
        zip(rows, POSITIVE_CONTROL_LEARNING_RATES)
    ):
        row = _require_mapping(raw_row, label=f"positive control row {index}")
        rate = _finite_number(row.get("learning_rate"), label=f"positive control rate {index}")
        observed_rates.append(rate)
        if rate != expected_rate:
            raise ValueError("positive control does not use the frozen rate grid")
        multiplier = _finite_number(
            row.get("nontrivial_multiplier"),
            label=f"positive control multiplier at {rate}",
        )
        locally_stable = row.get("locally_stable")
        if not isinstance(locally_stable, bool):
            raise ValueError("positive control stability labels must be Boolean")
        expected_multiplier = 1.0 - 2.0 * rate * target
        expected_stable = abs(expected_multiplier) < 1.0
        if not math.isclose(
            multiplier,
            expected_multiplier,
            rel_tol=_NUMERICAL_TOLERANCE,
            abs_tol=_NUMERICAL_TOLERANCE,
        ):
            failures.append(f"multiplier mismatch at learning rate {rate:g}")
        if locally_stable != expected_stable:
            failures.append(f"unit-circle classification mismatch at learning rate {rate:g}")
        has_lower = has_lower or rate < analytic
        has_upper = has_upper or rate > analytic
        has_critical = has_critical or math.isclose(
            rate, analytic, rel_tol=0.0, abs_tol=_NUMERICAL_TOLERANCE
        )
    if tuple(observed_rates) != POSITIVE_CONTROL_LEARNING_RATES:
        raise ValueError("positive control rate ordering is not frozen")
    if not (has_lower and has_upper and has_critical):
        raise ValueError("positive control grid does not straddle its analytic boundary")

    augmented = _require_mapping(
        payload.get("augmented_adam_control"), label="augmented Adam control"
    )
    if augmented.get("system") != "two_parameter_diagonal_quadratic_adamw":
        raise ValueError("augmented Adam control has the wrong tractable system")
    if _integer(
        augmented.get("augmented_dimension"), label="augmented control dimension"
    ) != 6:
        raise ValueError("augmented Adam control has the wrong state dimension")
    jacobian_error = _finite_number(
        augmented.get("jacobian_max_absolute_error"),
        label="augmented control Jacobian error",
    )
    relative_error = _finite_number(
        augmented.get("jacobian_relative_frobenius_error"),
        label="augmented control relative Jacobian error",
    )
    max_residual = _finite_number(
        augmented.get("maximum_eigenpair_relative_residual"),
        label="augmented control eigenpair residual",
    )
    if min(jacobian_error, relative_error, max_residual) < 0.0:
        raise ValueError("augmented Adam control diagnostics must be nonnegative")
    eigenvalue_count = _integer(
        augmented.get("dominant_eigenvalue_count"),
        label="augmented control eigenvalue count",
    )
    eigenvalues = _require_sequence(
        augmented.get("dominant_eigenvalues"),
        label="augmented control eigenvalues",
    )
    residuals = _require_sequence(
        augmented.get("eigenpair_relative_residuals"),
        label="augmented control eigenpair residuals",
    )
    if eigenvalue_count != 3 or len(eigenvalues) != 3 or len(residuals) != 3:
        raise ValueError("augmented Adam control must report three dominant eigenpairs")
    residual_values = tuple(
        _finite_number(value, label=f"augmented residual {index}")
        for index, value in enumerate(residuals)
    )
    if any(value < 0.0 for value in residual_values):
        raise ValueError("augmented Adam control residuals must be nonnegative")
    if not math.isclose(
        max_residual, max(residual_values), rel_tol=1e-12, abs_tol=1e-15
    ):
        raise ValueError("augmented Adam maximum residual is inconsistent")
    if jacobian_error > 1e-5:
        failures.append("augmented Adam HVP Jacobian misses centered finite differences")
    if relative_error > 1e-6:
        failures.append("augmented Adam Jacobian relative error exceeds tolerance")
    if max_residual > 1e-8:
        failures.append("augmented Adam eigensolver residual exceeds tolerance")
    return not failures, tuple(failures)


def _validate_erasure_scan(
    payload: Mapping[str, Any], gate: Gate0Config
) -> Tuple[Optional[Dict[str, float]], Tuple[Dict[str, float], ...]]:
    if payload.get("complete") is not True:
        raise ValueError("empirical erasure scan is incomplete")
    rates = tuple(gate.learning_rates)
    _exact_rates(
        payload.get("requested_learning_rates"), rates, label="erasure requested_learning_rates"
    )
    raw_branches = _require_sequence(payload.get("branches"), label="erasure branches")
    if len(raw_branches) != len(rates):
        raise ValueError("empirical erasure scan does not contain every frozen rate cell")

    outcomes = []
    for index, (raw_branch, expected_rate) in enumerate(zip(raw_branches, rates)):
        branch = _require_mapping(raw_branch, label=f"erasure branch {index}")
        rate = _finite_number(
            branch.get("learning_rate"), label=f"erasure branch {index} learning rate"
        )
        if rate != expected_rate:
            raise ValueError("empirical erasure branches are not on the frozen ordered grid")
        outcome = branch.get("outcome")
        if outcome not in {"retained", "erased", "diverged", "unresolved"}:
            raise ValueError(f"empirical erasure branch {index} has an invalid outcome")
        erased = branch.get("erased")
        if not isinstance(erased, bool) or erased != (outcome == "erased"):
            raise ValueError(f"empirical erasure branch {index} has inconsistent erasure fields")
        if outcome == "erased":
            entry = _finite_number(
                branch.get("sustained_entry_step"),
                label=f"erasure branch {index} sustained entry",
            )
            if entry < 0 or entry > gate.erase_horizon:
                raise ValueError("an erased calibration branch misses the frozen entry horizon")
        outcomes.append(str(outcome))

    derived = tuple(
        {
            "lower_learning_rate": float(lower_rate),
            "upper_learning_rate": float(upper_rate),
        }
        for lower_rate, upper_rate, lower_outcome, upper_outcome in zip(
            rates, rates[1:], outcomes, outcomes[1:]
        )
        if lower_outcome == "retained" and upper_outcome == "erased"
    )
    declared = _require_sequence(
        payload.get("strict_adjacent_brackets"), label="strict_adjacent_brackets"
    )
    if declared != list(derived):
        raise ValueError("declared erasure brackets disagree with the branch outcomes")
    selected = min(
        derived,
        key=lambda bracket: (
            bracket["lower_learning_rate"], bracket["upper_learning_rate"]
        ),
        default=None,
    )
    return selected, derived


def _validate_local_scan(payload: Mapping[str, Any], gate: Gate0Config) -> Tuple[bool, Tuple[float, ...]]:
    if payload.get("complete") is not True:
        raise ValueError("local-stability scan is incomplete")
    rates = tuple(gate.learning_rates)
    _exact_rates(
        payload.get("requested_learning_rates"), rates, label="local requested_learning_rates"
    )
    raw_measurements = _require_sequence(payload.get("measurements"), label="local measurements")
    if len(raw_measurements) != len(rates):
        raise ValueError("local-stability scan does not contain every frozen rate cell")

    uncertified = []
    for index, (raw_measurement, expected_rate) in enumerate(zip(raw_measurements, rates)):
        measurement = _require_mapping(raw_measurement, label=f"local measurement {index}")
        rate = _finite_number(
            measurement.get("learning_rate"), label=f"local measurement {index} learning rate"
        )
        if rate != expected_rate:
            raise ValueError("local-stability cells are not on the frozen ordered grid")
        spectral_radius = _finite_number(
            measurement.get("augmented_spectral_radius"),
            label=f"local spectral radius at {rate}",
        )
        if spectral_radius < 0:
            raise ValueError("local spectral radii must be nonnegative")
        max_residual = _finite_number(
            measurement.get("augmented_max_relative_residual"),
            label=f"local maximum residual at {rate}",
        )
        if max_residual < 0:
            raise ValueError("local residuals must be nonnegative")
        eigenvalues = _require_sequence(
            measurement.get("augmented_eigenvalues"),
            label=f"local eigenvalues at {rate}",
        )
        if len(eigenvalues) != gate.augmented_eigenvalue_count:
            raise ValueError("local cell has the wrong number of augmented eigenpairs")
        residuals = []
        for eigen_index, raw_eigenvalue in enumerate(eigenvalues):
            eigenvalue = _require_mapping(
                raw_eigenvalue, label=f"local eigenvalue {eigen_index} at {rate}"
            )
            residual = _finite_number(
                eigenvalue.get("relative_residual"),
                label=f"local eigenpair residual {eigen_index} at {rate}",
            )
            if residual < 0:
                raise ValueError("local eigenpair residuals must be nonnegative")
            residuals.append(residual)
        if not math.isclose(max_residual, max(residuals), rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("local maximum residual disagrees with its eigenpair residuals")
        certified = measurement.get("augmented_certified")
        expected_certified = max_residual <= gate.augmented_max_relative_residual
        if not isinstance(certified, bool) or certified != expected_certified:
            raise ValueError("local certification label disagrees with the frozen residual rule")
        if measurement.get("stream_unchanged") is not True:
            raise ValueError("local measurement advanced or failed to verify the frozen stream")
        if not certified:
            uncertified.append(rate)

    declared_all = payload.get("all_augmented_eigenpairs_certified")
    if not isinstance(declared_all, bool) or declared_all != (not uncertified):
        raise ValueError("local aggregate certification field is inconsistent")
    declared_uncertified = _require_sequence(
        payload.get("uncertified_learning_rates"), label="uncertified_learning_rates"
    )
    declared_uncertified_rates = tuple(
        _finite_number(rate, label="uncertified learning rate")
        for rate in declared_uncertified
    )
    if declared_uncertified_rates != tuple(uncertified):
        raise ValueError("local uncertified-rate list is inconsistent")
    return not uncertified, tuple(uncertified)


def adjudicate_gate0_erasure_precheck(
    *,
    positive_control_path: Path,
    erasure_scan_path: Path,
    snapshot_path: Path,
    experiment: MBCExperimentConfig,
    gate: Gate0Config,
    protocol_version: str,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Close calibration early when the frozen empirical bracket is absent.

    This is an administrative short-circuit of the already frozen rule.  It
    never opens official seeds: a successful precheck merely authorizes the
    local calibration scan, while a failed precheck emits the terminal stop
    without spending that compute.
    """

    if experiment.seed != gate.calibration_seed:
        raise ValueError("calibration precheck must use the frozen non-gate seed")
    if experiment.batch_size != gate.calibration_batch_size:
        raise ValueError("calibration precheck must use the frozen batch size")
    positive_control_path = Path(positive_control_path)
    erasure_scan_path = Path(erasure_scan_path)
    snapshot_path = Path(snapshot_path)
    snapshot_artifact = bind_snapshot(
        snapshot_path,
        expected_seed=gate.calibration_seed,
        expected_config=experiment,
    )
    if snapshot_artifact["checks"]["seed"] != "matched":
        raise ValueError("calibration snapshot must carry matched seed metadata")
    if snapshot_artifact["checks"]["config"] not in {"matched", "unavailable"}:
        raise ValueError("calibration snapshot configuration was not checked")

    positive_payload = _load_json_object(positive_control_path, label="positive control")
    positive_manifest = _validate_positive_control_manifest(
        positive_control_path.parent / "manifest.json",
        result_path=positive_control_path,
        protocol_version=protocol_version,
    )
    positive_passed, positive_failures = _validate_positive_control(positive_payload)
    artifact = validate_scan_artifact(
        erasure_scan_path,
        key=CellKey(gate.calibration_seed, gate.calibration_batch_size, Direction.ERASURE),
        role="empirical",
        expected_rates=tuple(gate.learning_rates),
        expected_protocol_version=protocol_version,
        expected_gate0_config=_json_normalized(asdict(gate)),
    )
    if artifact.snapshot_binding[0] != snapshot_artifact["sha256"]:
        raise ValueError("calibration erasure scan does not bind the adjudicated snapshot")
    selected_bracket, eligible_brackets = _validate_erasure_scan(artifact.payload, gate)
    passed = positive_passed and selected_bracket is not None
    reasons = []
    if not positive_passed:
        reasons.extend(f"positive control: {failure}" for failure in positive_failures)
    if selected_bracket is None:
        reasons.append("empirical scan has no strict adjacent retained-to-erased bracket")
    if passed:
        reasons = [
            "positive control passed both analytic and augmented numerical checks",
            "empirical scan contains a strict nondivergent retained-to-erased bracket",
        ]
    report: Dict[str, Any] = {
        "schema_version": 1,
        "kind": "gate0_calibration_erasure_precheck",
        "protocol_version": protocol_version,
        "status": "pass" if passed else "stop",
        "action": "run_local_calibration" if passed else "stop_before_gate0",
        "calibration_seed": gate.calibration_seed,
        "frozen_batch_size": gate.calibration_batch_size,
        "frozen_learning_rates": list(gate.learning_rates),
        "selected_empirical_bracket": selected_bracket,
        "eligible_empirical_brackets": list(eligible_brackets),
        "checks": {
            "positive_control_passed": positive_passed,
            "strict_empirical_bracket_found": selected_bracket is not None,
        },
        "reasons": reasons,
        "inputs": {
            "positive_control": {
                "artifact": bind_file(positive_control_path),
                "manifest": positive_manifest,
            },
            "empirical_erasure_scan": {
                "artifact": bind_file(erasure_scan_path),
                "manifest": bind_manifest(erasure_scan_path.parent / "manifest.json"),
            },
            "calibration_snapshot": {
                "artifact": snapshot_artifact,
                "source_manifest": bind_nearest_manifest(snapshot_path),
            },
        },
    }
    if output_path is not None:
        _atomic_write_json(Path(output_path), report)
    return report


def adjudicate_gate0_calibration(
    *,
    positive_control_path: Path,
    erasure_scan_path: Path,
    local_scan_path: Path,
    snapshot_path: Path,
    experiment: MBCExperimentConfig,
    gate: Gate0Config,
    protocol_version: str,
    output_path: Optional[Path] = None,
    positive_control_manifest_path: Optional[Path] = None,
    erasure_manifest_path: Optional[Path] = None,
    local_manifest_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Adjudicate the frozen seed-100 calibration and optionally write a report.

    Scientific failures return ``status == "stop"``.  Incomplete, malformed,
    or provenance-mismatched inputs raise ``ValueError`` because they cannot be
    interpreted as evidence for either outcome.
    """

    gate_rates = tuple(float(rate) for rate in gate.learning_rates)
    if tuple(sorted(gate_rates)) != gate_rates or len(set(gate_rates)) != len(gate_rates):
        raise ValueError("Gate 0 calibration grid must be sorted and unique")
    if experiment.seed != gate.calibration_seed:
        raise ValueError("calibration experiment must use the frozen non-gate seed")
    if experiment.batch_size != gate.calibration_batch_size:
        raise ValueError("calibration experiment must use the frozen batch size")
    if not protocol_version:
        raise ValueError("protocol version must be non-empty")

    positive_control_path = Path(positive_control_path)
    erasure_scan_path = Path(erasure_scan_path)
    local_scan_path = Path(local_scan_path)
    snapshot_path = Path(snapshot_path)
    positive_control_manifest_path = Path(
        positive_control_manifest_path or positive_control_path.parent / "manifest.json"
    )
    erasure_manifest_path = Path(erasure_manifest_path or erasure_scan_path.parent / "manifest.json")
    local_manifest_path = Path(local_manifest_path or local_scan_path.parent / "manifest.json")

    positive_payload = _load_json_object(positive_control_path, label="positive control")

    snapshot_artifact = bind_snapshot(
        snapshot_path,
        expected_seed=gate.calibration_seed,
        expected_config=experiment,
    )
    if snapshot_artifact["checks"]["seed"] != "matched":
        raise ValueError("calibration snapshot must carry matched seed metadata")
    if snapshot_artifact["checks"]["config"] not in {"matched", "unavailable"}:
        raise ValueError("calibration snapshot configuration was not checked")
    snapshot_input = {
        "artifact": snapshot_artifact,
        "source_manifest": bind_nearest_manifest(snapshot_path),
    }

    positive_control_manifest_binding = _validate_positive_control_manifest(
        positive_control_manifest_path,
        result_path=positive_control_path,
        protocol_version=protocol_version,
    )
    calibration_key = CellKey(
        gate.calibration_seed,
        gate.calibration_batch_size,
        Direction.ERASURE,
    )
    erasure_artifact = validate_scan_artifact(
        erasure_scan_path,
        key=calibration_key,
        role="empirical",
        expected_rates=tuple(gate.learning_rates),
        expected_protocol_version=protocol_version,
        expected_gate0_config=_json_normalized(asdict(gate)),
    )
    local_artifact = validate_scan_artifact(
        local_scan_path,
        key=calibration_key,
        role="local",
        expected_rates=tuple(gate.learning_rates),
        expected_protocol_version=protocol_version,
        expected_gate0_config=_json_normalized(asdict(gate)),
    )
    if erasure_artifact.snapshot_binding != local_artifact.snapshot_binding:
        raise ValueError("calibration scans do not bind the same snapshot")
    if erasure_artifact.snapshot_binding[0] != snapshot_artifact["sha256"]:
        raise ValueError("calibration scans do not bind the adjudicated snapshot")
    erasure_manifest_binding = bind_manifest(erasure_manifest_path)
    local_manifest_binding = bind_manifest(local_manifest_path)

    positive_passed, positive_failures = _validate_positive_control(positive_payload)
    selected_bracket, eligible_brackets = _validate_erasure_scan(
        erasure_artifact.payload, gate
    )
    local_passed, uncertified_rates = _validate_local_scan(local_artifact.payload, gate)

    reasons = []
    if not positive_passed:
        reasons.extend(f"positive control: {failure}" for failure in positive_failures)
    if selected_bracket is None:
        reasons.append("empirical scan has no strict adjacent retained-to-erased bracket")
    if not local_passed:
        formatted = ", ".join(f"{rate:g}" for rate in uncertified_rates)
        reasons.append(f"local scan has uncertified cells at learning rates: {formatted}")
    passed = positive_passed and selected_bracket is not None and local_passed
    if passed:
        reasons = [
            "positive control recovered the analytic unit-circle boundary",
            "empirical scan contains a strict nondivergent retained-to-erased bracket",
            "all local-stability cells satisfy the frozen eigenpair residual rule",
        ]

    report: Dict[str, Any] = {
        "schema_version": 1,
        "kind": "gate0_calibration_adjudication",
        "protocol_version": protocol_version,
        "status": "pass" if passed else "stop",
        "action": "open_gate0" if passed else "stop_before_gate0",
        "calibration_seed": gate.calibration_seed,
        "frozen_batch_size": gate.calibration_batch_size,
        "frozen_learning_rates": list(gate_rates),
        "gate0": _json_normalized(asdict(gate)),
        "selected_empirical_bracket": selected_bracket,
        "frozen_erasure_bracket": selected_bracket if passed else None,
        "eligible_empirical_brackets": list(eligible_brackets),
        "checks": {
            "positive_control_passed": positive_passed,
            "strict_empirical_bracket_found": selected_bracket is not None,
            "all_local_cells_certified": local_passed,
        },
        "reasons": reasons,
        "inputs": {
            "positive_control": {
                "artifact": bind_file(positive_control_path),
                "manifest": positive_control_manifest_binding,
            },
            "empirical_erasure_scan": {
                "artifact": bind_file(erasure_scan_path),
                "manifest": erasure_manifest_binding,
            },
            "local_stability_scan": {
                "artifact": bind_file(local_scan_path),
                "manifest": local_manifest_binding,
            },
            "calibration_snapshot": snapshot_input,
        },
    }
    if output_path is not None:
        _atomic_write_json(Path(output_path), report)
    return report


def bind_passed_gate0_calibration(
    path: Path,
    *,
    expected_gate: Optional[Gate0Config] = None,
    expected_protocol_version: Optional[str] = None,
) -> Dict[str, Any]:
    """Bind a passed calibration report for an official Gate 0 manifest."""

    path = Path(path)
    payload = _load_json_object(path, label="Gate 0 calibration report")
    if payload.get("schema_version") != 1 or payload.get("kind") != "gate0_calibration_adjudication":
        raise ValueError("invalid Gate 0 calibration report schema")
    if payload.get("status") != "pass" or payload.get("action") != "open_gate0":
        raise ValueError("official Gate 0 scans require a passed calibration")
    protocol_version = payload.get("protocol_version")
    if not isinstance(protocol_version, str) or not protocol_version:
        raise ValueError("calibration report has no valid protocol version")
    if (
        expected_protocol_version is not None
        and protocol_version != expected_protocol_version
    ):
        raise ValueError("calibration report has the wrong protocol version")
    rates_raw = _require_sequence(payload.get("frozen_learning_rates"), label="frozen_learning_rates")
    rates = tuple(
        _finite_number(rate, label=f"frozen learning rate {index}")
        for index, rate in enumerate(rates_raw)
    )
    if not rates or tuple(sorted(rates)) != rates or len(set(rates)) != len(rates):
        raise ValueError("calibration report has an invalid frozen rate grid")
    bracket = _require_mapping(payload.get("frozen_erasure_bracket"), label="frozen_erasure_bracket")
    lower = _finite_number(bracket.get("lower_learning_rate"), label="frozen bracket lower rate")
    upper = _finite_number(bracket.get("upper_learning_rate"), label="frozen bracket upper rate")
    if not 0 < lower < upper:
        raise ValueError("calibration report has an invalid frozen bracket")
    if not any(left == lower and right == upper for left, right in zip(rates, rates[1:])):
        raise ValueError("frozen calibration bracket is not adjacent on the frozen grid")
    selected = payload.get("selected_empirical_bracket")
    if selected != dict(bracket):
        raise ValueError("selected and frozen calibration brackets disagree")
    seed = _integer(payload.get("calibration_seed"), label="calibration_seed")
    batch_size = _integer(payload.get("frozen_batch_size"), label="frozen_batch_size")
    if batch_size <= 0:
        raise ValueError("frozen batch size must be positive")
    if expected_gate is not None:
        if seed != expected_gate.calibration_seed:
            raise ValueError("calibration report has the wrong calibration seed")
        if batch_size != expected_gate.calibration_batch_size:
            raise ValueError("calibration report has the wrong frozen batch size")
        if rates != tuple(expected_gate.learning_rates):
            raise ValueError("calibration report has the wrong frozen learning-rate grid")
        if payload.get("gate0") != _json_normalized(asdict(expected_gate)):
            raise ValueError("calibration report has the wrong Gate 0 config")
    checks = _require_mapping(payload.get("checks"), label="calibration checks")
    expected_checks = {
        "positive_control_passed": True,
        "strict_empirical_bracket_found": True,
        "all_local_cells_certified": True,
    }
    if dict(checks) != expected_checks:
        raise ValueError("passed calibration report has inconsistent checks")
    inputs = _require_mapping(payload.get("inputs"), label="calibration inputs")
    positive_input = _require_mapping(
        inputs.get("positive_control"), label="positive-control input"
    )
    _require_mapping(
        positive_input.get("manifest"), label="positive-control manifest binding"
    )
    snapshot_input = _require_mapping(
        inputs.get("calibration_snapshot"), label="calibration snapshot input"
    )
    snapshot_artifact = _require_mapping(
        snapshot_input.get("artifact"), label="calibration snapshot artifact"
    )
    snapshot_sha256 = _sha256(
        snapshot_artifact.get("sha256"), label="calibration snapshot digest"
    )

    manifest_path = path.parent / "manifest.json"
    manifest_binding = bind_manifest(manifest_path)
    manifest = _load_json_object(manifest_path, label="calibration adjudication manifest")
    manifest_config = _require_mapping(
        manifest.get("config"), label="calibration adjudication manifest config"
    )
    if manifest_config.get("kind") != "gate0_calibration_adjudication":
        raise ValueError("calibration report has the wrong sibling manifest kind")
    if manifest_config.get("protocol_version") != protocol_version:
        raise ValueError("calibration report and sibling manifest disagree on protocol")
    derived_gate = _gate_from_json(payload.get("gate0"))
    if expected_gate is not None and derived_gate != expected_gate:
        raise ValueError("calibration report has the wrong Gate 0 config")
    experiment_raw = _require_mapping(
        manifest_config.get("experiment"), label="calibration experiment config"
    )
    experiment = MBCExperimentConfig(**dict(experiment_raw))
    if experiment.seed != seed or experiment.batch_size != batch_size:
        raise ValueError("calibration manifest has the wrong seed or batch size")
    if manifest_config.get("gate0") != _json_normalized(asdict(derived_gate)):
        raise ValueError("calibration manifest and report disagree on Gate 0 controls")

    positive_artifact = _require_mapping(
        positive_input.get("artifact"), label="positive-control artifact"
    )
    positive_manifest = _require_mapping(
        positive_input.get("manifest"), label="positive-control manifest"
    )
    erasure_input = _require_mapping(
        inputs.get("empirical_erasure_scan"), label="empirical erasure input"
    )
    local_input = _require_mapping(
        inputs.get("local_stability_scan"), label="local-stability input"
    )

    def current_path(binding: Mapping[str, Any], *, label: str) -> Path:
        value = binding.get("path")
        if not isinstance(value, str):
            raise ValueError(f"{label} has no path")
        return Path(value)

    positive_path = current_path(positive_artifact, label="positive-control artifact")
    erasure_artifact = _require_mapping(
        erasure_input.get("artifact"), label="empirical erasure artifact"
    )
    local_artifact = _require_mapping(
        local_input.get("artifact"), label="local-stability artifact"
    )
    erasure_path = current_path(erasure_artifact, label="empirical erasure artifact")
    local_path = current_path(local_artifact, label="local-stability artifact")
    snapshot_path = current_path(snapshot_artifact, label="calibration snapshot artifact")

    expected_manifest_inputs = {
        "positive_control_input": {
            "artifact": dict(positive_artifact),
            "source_manifest": dict(positive_manifest),
        },
        "erasure_scan_input": {
            "artifact": dict(erasure_artifact),
            "source_manifest": dict(
                _require_mapping(
                    erasure_input.get("manifest"), label="empirical erasure manifest"
                )
            ),
        },
        "local_scan_input": {
            "artifact": dict(local_artifact),
            "source_manifest": dict(
                _require_mapping(
                    local_input.get("manifest"), label="local-stability manifest"
                )
            ),
        },
    }
    for name, expected in expected_manifest_inputs.items():
        if manifest_config.get(name) != expected:
            raise ValueError(f"calibration sibling manifest has mismatched {name}")
    if manifest_config.get("snapshot_input") != snapshot_input:
        raise ValueError("calibration sibling manifest has a mismatched snapshot input")
    if manifest_config.get("snapshot") != str(snapshot_path):
        raise ValueError("calibration sibling manifest has a mismatched snapshot path")

    rederived = adjudicate_gate0_calibration(
        positive_control_path=positive_path,
        erasure_scan_path=erasure_path,
        local_scan_path=local_path,
        snapshot_path=snapshot_path,
        experiment=experiment,
        gate=derived_gate,
        protocol_version=protocol_version,
    )
    if rederived != payload:
        raise ValueError("calibration report does not equal a fresh adjudication of its inputs")
    return {
        "artifact": bind_reference(path),
        "source_manifest": manifest_binding,
        "status": "pass",
        "protocol_version": protocol_version,
        "calibration_seed": seed,
        "frozen_batch_size": batch_size,
        "frozen_learning_rates": list(rates),
        "frozen_erasure_bracket": dict(bracket),
        "calibration_snapshot_sha256": snapshot_sha256,
    }
