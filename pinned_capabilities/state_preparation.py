"""Construct validated expressed and suppressed starting checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional

from .config import MBCExperimentConfig, MetricConfig
from .experiment import JSONLWriter, MBCExperiment
from .manifest import content_hash
from .provenance import sha256_file
from .snapshot import load_snapshot, save_snapshot
from .state import ReferenceBands, StateThresholds, is_expressed, is_jointly_suppressed


@dataclass(frozen=True)
class StatePreparationResult:
    seed: int
    learning_rate: float
    outcome: str
    step: int
    c_int: float
    exact_match: float
    delta_z: float
    full_vocab_ce: float
    snapshot_path: Optional[str]
    provenance: Optional[Dict[str, object]] = None
    snapshot_sha256: Optional[str] = None
    result_sha256: Optional[str] = None


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


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


def _preparation_provenance(
    kind: str,
    experiment_config: MBCExperimentConfig,
    metric: MetricConfig,
    reference: ReferenceBands,
    thresholds: StateThresholds,
    controls: Dict[str, object],
) -> Dict[str, object]:
    identity: Dict[str, object] = {
        "schema_version": 1,
        "kind": kind,
        "inputs": {
            "experiment": _control_value(experiment_config),
            "metric": _control_value(metric),
            "reference": _control_value(reference),
            "thresholds": _control_value(thresholds),
            "controls": _control_value(controls),
        },
    }
    return {**identity, "preparation_sha256": content_hash(identity)}


def _validate_provenance(
    observed: object, expected: Dict[str, object], *, location: Path
) -> None:
    if not isinstance(observed, dict):
        raise ValueError(f"state-preparation artifact at {location} has no provenance binding")
    declared = observed.get("preparation_sha256")
    unsigned = {key: value for key, value in observed.items() if key != "preparation_sha256"}
    if declared != content_hash(unsigned):
        raise ValueError(f"state-preparation artifact at {location} has tampered provenance")
    if observed != expected:
        raise ValueError(f"state-preparation artifact at {location} has mismatched provenance")


def _seal_result(result: StatePreparationResult) -> StatePreparationResult:
    payload = asdict(result)
    payload.pop("result_sha256", None)
    return replace(result, result_sha256=content_hash(payload))


def _validate_result(
    result: StatePreparationResult,
    *,
    expected_provenance: Dict[str, object],
    expected_snapshot_path: Path,
    location: Path,
) -> None:
    _validate_provenance(result.provenance, expected_provenance, location=location)
    payload = asdict(result)
    declared = payload.pop("result_sha256", None)
    if declared != content_hash(payload):
        raise ValueError("completed state-preparation result has a tampered result payload")
    if result.snapshot_path is None:
        if result.snapshot_sha256 is not None:
            raise ValueError("completed state-preparation result has a digest without a snapshot")
        return
    snapshot_path = Path(result.snapshot_path)
    if snapshot_path.resolve() != expected_snapshot_path.resolve():
        raise ValueError("completed state-preparation result names the wrong final snapshot")
    if result.snapshot_sha256 is None:
        raise ValueError("completed state-preparation result has no final snapshot digest")
    if not snapshot_path.exists():
        raise FileNotFoundError("completed state-preparation result references a missing snapshot")
    if sha256_file(snapshot_path) != result.snapshot_sha256:
        raise ValueError("completed state-preparation final snapshot digest mismatch")


def _completed_result(
    path: Path,
    *,
    expected_provenance: Dict[str, object],
    expected_snapshot_path: Path,
) -> Optional[StatePreparationResult]:
    if not path.exists():
        return None
    result = StatePreparationResult(**json.loads(path.read_text()))
    _validate_result(
        result,
        expected_provenance=expected_provenance,
        expected_snapshot_path=expected_snapshot_path,
        location=path,
    )
    return result


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validated_metrics_prefix(metrics_path: Path, progress: dict) -> bytes:
    if not metrics_path.exists():
        raise FileNotFoundError("state-preparation progress has no metrics log")
    size = progress.get("metrics_prefix_size_bytes")
    digest = progress.get("metrics_prefix_sha256")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("state-preparation progress has no valid metrics-prefix size")
    if not isinstance(digest, str):
        raise ValueError("state-preparation progress has no metrics-prefix digest")
    current = metrics_path.read_bytes()
    if len(current) < size:
        raise ValueError("state-preparation metrics are shorter than the checkpoint prefix")
    prefix = current[:size]
    if _sha256_bytes(prefix) != digest:
        raise ValueError("state-preparation metrics prefix digest mismatch")
    return prefix


def _resume_or_start(
    experiment: MBCExperiment,
    output_dir: Path,
    *,
    kind: str,
    controls: Dict[str, object],
    provenance: Dict[str, object],
) -> tuple[list[dict], dict, int, JSONLWriter, bool]:
    progress_path = output_dir / "progress.json"
    metrics_path = output_dir / "metrics.jsonl"
    rows: list[dict] = []
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        _validate_provenance(progress.get("provenance"), provenance, location=progress_path)
        for key, value in controls.items():
            if progress.get(key) != value:
                raise ValueError(f"state-preparation progress has mismatched {key}")
        checkpoint_path = output_dir / progress["checkpoint_path"]
        if checkpoint_path.resolve().parent != (output_dir / "checkpoints").resolve():
            raise ValueError("state-preparation progress names a checkpoint outside its directory")
        if not checkpoint_path.exists():
            raise FileNotFoundError("state-preparation progress references a missing checkpoint")
        checkpoint_sha256 = progress.get("checkpoint_sha256")
        if not isinstance(checkpoint_sha256, str):
            raise ValueError("state-preparation progress has no checkpoint digest")
        if sha256_file(checkpoint_path) != checkpoint_sha256:
            raise ValueError("state-preparation progress checkpoint digest mismatch")
        prefix = _validated_metrics_prefix(metrics_path, progress)
        restored = load_snapshot(
            checkpoint_path,
            model=experiment.model,
            optimizer=experiment.optimizer,
            stream=experiment.stream,
            map_location=experiment.device,
        )
        experiment.step = restored["step"]
        completed_step = int(progress["completed_step"])
        if experiment.step != completed_step:
            raise ValueError("state-preparation checkpoint and progress disagree")
        try:
            logged = [json.loads(line) for line in prefix.decode("utf-8").splitlines()]
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("state-preparation metrics prefix is not valid JSONL") from error
        if not logged or logged[0].get("kind") != f"{kind}_start":
            raise ValueError("state-preparation metrics prefix has no matching start row")
        if any(row.get("kind") not in {f"{kind}_start", kind} for row in logged):
            raise ValueError("state-preparation metrics prefix contains an unexpected row kind")
        rows = [row for row in logged if row["kind"] == kind]
        if not rows or int(rows[-1]["step"]) != completed_step:
            raise ValueError("state-preparation metrics do not reach the progress checkpoint")
        steps = [int(row["step"]) for row in rows]
        if steps != sorted(set(steps)) or any(step > completed_step for step in steps):
            raise ValueError("state-preparation metrics prefix has an invalid step history")
        metrics_path.write_bytes(prefix)
        latest = rows[-1]
        active_slot = int(progress["active_slot"])
        fresh = False
    else:
        if metrics_path.exists():
            metrics_path.write_text("")
        latest = experiment.evaluate()
        active_slot = -1
        fresh = True
    return rows, latest, active_slot, JSONLWriter(metrics_path), fresh


def _checkpoint_progress(
    experiment: MBCExperiment,
    output_dir: Path,
    *,
    kind: str,
    controls: Dict[str, object],
    provenance: Dict[str, object],
    active_slot: int,
) -> int:
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
            "kind": f"{kind}_progress",
            **controls,
            "preparation_sha256": provenance["preparation_sha256"],
        },
    )
    temporary.replace(checkpoint_path)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    metrics_path = output_dir / "metrics.jsonl"
    metrics_prefix = metrics_path.read_bytes()
    _atomic_json(
        output_dir / "progress.json",
        {
            **controls,
            "completed_step": experiment.step,
            "active_slot": active_slot,
            "checkpoint_path": str(checkpoint_path.relative_to(output_dir)),
            "checkpoint_sha256": checkpoint_sha256,
            "metrics_prefix_size_bytes": len(metrics_prefix),
            "metrics_prefix_sha256": _sha256_bytes(metrics_prefix),
            "provenance": provenance,
        },
    )
    return active_slot


def prepare_suppressed_checkpoint(
    experiment_config: MBCExperimentConfig,
    metric: MetricConfig,
    reference: ReferenceBands,
    thresholds: StateThresholds,
    *,
    learning_rate: float,
    maximum_steps: int,
    output_dir: Path,
    save_at_step: Optional[int] = None,
) -> StatePreparationResult:
    if maximum_steps < thresholds.suppressed_duration:
        raise ValueError("state-preparation budget is shorter than suppressed duration")
    if save_at_step is not None and not thresholds.suppressed_duration <= save_at_step <= maximum_steps:
        raise ValueError("save_at_step must fit after the suppressed hold and inside the budget")
    if save_at_step is not None and save_at_step % metric.eval_every:
        raise ValueError("save_at_step must align with the metric evaluation cadence")
    output_dir = Path(output_dir)
    result_path = output_dir / "result.json"
    controls: Dict[str, object] = {
        "seed": experiment_config.seed,
        "learning_rate": learning_rate,
        "maximum_steps": maximum_steps,
        "save_at_step": save_at_step,
    }
    provenance = _preparation_provenance(
        "suppressed_preparation",
        experiment_config,
        metric,
        reference,
        thresholds,
        controls,
    )
    expected_snapshot_path = output_dir / "suppressed_snapshot.pt"
    completed = _completed_result(
        result_path,
        expected_provenance=provenance,
        expected_snapshot_path=expected_snapshot_path,
    )
    if completed is not None:
        return completed
    experiment = MBCExperiment(replace(experiment_config, learning_rate=learning_rate), metric)
    rows, latest, active_slot, writer, fresh = _resume_or_start(
        experiment,
        output_dir,
        kind="suppressed_preparation",
        controls=controls,
        provenance=provenance,
    )
    if fresh:
        writer.write({"kind": "suppressed_preparation_start", **latest})
    outcome = "censored"
    snapshot_path: Optional[Path] = None
    while experiment.step < maximum_steps:
        chunk = min(metric.eval_every, maximum_steps - experiment.step)
        training = experiment.advance(chunk)
        latest = {**training, **experiment.evaluate(), "learning_rate": learning_rate}
        rows.append(latest)
        writer.write({"kind": "suppressed_preparation", **latest})
        if experiment.step % max(1_000, metric.eval_every) == 0:
            active_slot = _checkpoint_progress(
                experiment,
                output_dir,
                kind="suppressed_preparation",
                controls=controls,
                provenance=provenance,
                active_slot=active_slot,
            )
        if is_jointly_suppressed(rows, reference, thresholds) and (
            save_at_step is None or experiment.step >= save_at_step
        ):
            outcome = "suppressed"
            snapshot_path = expected_snapshot_path
            save_snapshot(
                snapshot_path,
                model=experiment.model,
                optimizer=experiment.optimizer,
                stream=experiment.stream,
                step=experiment.step,
                metadata={
                    "kind": "suppressed_start",
                    "seed": experiment_config.seed,
                    "learning_rate": learning_rate,
                    "experiment_config": asdict(experiment_config),
                    "preparation_sha256": provenance["preparation_sha256"],
                },
            )
            break
        if is_expressed(
            latest["c_int"], latest["exact_match"], latest["delta_z"], reference, thresholds
        ):
            outcome = "expressed_before_suppressed"
            break
        if (
            not math.isfinite(latest["full_vocab_ce"])
            or latest["full_vocab_ce"] > 5.0 * reference.q_star_loss_mean
        ):
            outcome = "diverged"
            break
    snapshot_sha256 = sha256_file(snapshot_path) if snapshot_path is not None else None
    result = _seal_result(StatePreparationResult(
        seed=experiment_config.seed,
        learning_rate=learning_rate,
        outcome=outcome,
        step=experiment.step,
        c_int=latest["c_int"],
        exact_match=latest["exact_match"],
        delta_z=latest["delta_z"],
        full_vocab_ce=latest["full_vocab_ce"],
        snapshot_path=str(snapshot_path) if snapshot_path is not None else None,
        provenance=provenance,
        snapshot_sha256=snapshot_sha256,
    ))
    _validate_result(
        result,
        expected_provenance=provenance,
        expected_snapshot_path=expected_snapshot_path,
        location=result_path,
    )
    _atomic_json(result_path, asdict(result))
    return result


def prepare_expressed_checkpoint(
    experiment_config: MBCExperimentConfig,
    metric: MetricConfig,
    reference: ReferenceBands,
    thresholds: StateThresholds,
    *,
    learning_rate: float,
    save_at_step: int,
    output_dir: Path,
) -> StatePreparationResult:
    if save_at_step < metric.solved_hold_steps:
        raise ValueError("expressed save step must accommodate the solved hold")
    if save_at_step % metric.eval_every or metric.solved_hold_steps % metric.eval_every:
        raise ValueError("expressed save and hold steps must align with evaluation cadence")
    output_dir = Path(output_dir)
    result_path = output_dir / "result.json"
    controls: Dict[str, object] = {
        "seed": experiment_config.seed,
        "learning_rate": learning_rate,
        "save_at_step": save_at_step,
    }
    provenance = _preparation_provenance(
        "expressed_preparation",
        experiment_config,
        metric,
        reference,
        thresholds,
        controls,
    )
    expected_snapshot_path = output_dir / "expressed_snapshot.pt"
    completed = _completed_result(
        result_path,
        expected_provenance=provenance,
        expected_snapshot_path=expected_snapshot_path,
    )
    if completed is not None:
        return completed
    experiment = MBCExperiment(replace(experiment_config, learning_rate=learning_rate), metric)
    rows, latest, active_slot, writer, fresh = _resume_or_start(
        experiment,
        output_dir,
        kind="expressed_preparation",
        controls=controls,
        provenance=provenance,
    )
    if fresh:
        writer.write({"kind": "expressed_preparation_start", **latest})
    while experiment.step < save_at_step:
        chunk = min(metric.eval_every, save_at_step - experiment.step)
        training = experiment.advance(chunk)
        latest = {**training, **experiment.evaluate(), "learning_rate": learning_rate}
        rows.append(latest)
        writer.write({"kind": "expressed_preparation", **latest})
        if (
            experiment.step % max(1_000, metric.eval_every) == 0
            or experiment.step == save_at_step
        ):
            active_slot = _checkpoint_progress(
                experiment,
                output_dir,
                kind="expressed_preparation",
                controls=controls,
                provenance=provenance,
                active_slot=active_slot,
            )
        if (
            not math.isfinite(latest["full_vocab_ce"])
            or latest["full_vocab_ce"] > 5.0 * reference.q_star_loss_mean
        ):
            break
    hold_start = save_at_step - metric.solved_hold_steps
    hold = [row for row in rows if row["step"] >= hold_start]
    stable_expression = bool(hold) and hold[0]["step"] <= hold_start and all(
        is_expressed(
            row["c_int"], row["exact_match"], row["delta_z"], reference, thresholds
        )
        for row in hold
    )
    snapshot_path: Optional[Path] = None
    if experiment.step == save_at_step and stable_expression:
        outcome = "expressed"
        snapshot_path = expected_snapshot_path
        save_snapshot(
            snapshot_path,
            model=experiment.model,
            optimizer=experiment.optimizer,
            stream=experiment.stream,
            step=experiment.step,
            metadata={
                "kind": "expressed_start",
                "seed": experiment_config.seed,
                "learning_rate": learning_rate,
                "experiment_config": asdict(experiment_config),
                "preparation_sha256": provenance["preparation_sha256"],
            },
        )
    elif experiment.step < save_at_step:
        outcome = "diverged"
    else:
        outcome = "not_stably_expressed"
    snapshot_sha256 = sha256_file(snapshot_path) if snapshot_path is not None else None
    result = _seal_result(StatePreparationResult(
        seed=experiment_config.seed,
        learning_rate=learning_rate,
        outcome=outcome,
        step=experiment.step,
        c_int=latest["c_int"],
        exact_match=latest["exact_match"],
        delta_z=latest["delta_z"],
        full_vocab_ce=latest["full_vocab_ce"],
        snapshot_path=str(snapshot_path) if snapshot_path is not None else None,
        provenance=provenance,
        snapshot_sha256=snapshot_sha256,
    ))
    _validate_result(
        result,
        expected_provenance=provenance,
        expected_snapshot_path=expected_snapshot_path,
        location=result_path,
    )
    _atomic_json(result_path, asdict(result))
    return result
