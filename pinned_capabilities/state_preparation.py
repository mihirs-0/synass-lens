"""Construct validated expressed and suppressed starting checkpoints."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Optional

from .config import MBCExperimentConfig, MetricConfig
from .experiment import JSONLWriter, MBCExperiment
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


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _completed_result(
    path: Path, *, seed: int, learning_rate: float
) -> Optional[StatePreparationResult]:
    if not path.exists():
        return None
    result = StatePreparationResult(**json.loads(path.read_text()))
    if result.seed != seed or result.learning_rate != learning_rate:
        raise ValueError("completed state-preparation result has mismatched controls")
    return result


def _resume_or_start(
    experiment: MBCExperiment,
    output_dir: Path,
    *,
    kind: str,
    controls: Dict[str, object],
) -> tuple[list[dict], dict, int, JSONLWriter, bool]:
    progress_path = output_dir / "progress.json"
    metrics_path = output_dir / "metrics.jsonl"
    rows: list[dict] = []
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        for key, value in controls.items():
            if progress.get(key) != value:
                raise ValueError(f"state-preparation progress has mismatched {key}")
        checkpoint_path = output_dir / progress["checkpoint_path"]
        if not checkpoint_path.exists():
            raise FileNotFoundError("state-preparation progress references a missing checkpoint")
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
        if not metrics_path.exists():
            raise FileNotFoundError("state-preparation progress has no metrics log")
        logged = [json.loads(line) for line in metrics_path.read_text().splitlines()]
        logged = [
            row
            for row in logged
            if row["kind"] == f"{kind}_start" or int(row["step"]) <= completed_step
        ]
        metrics_path.write_text(
            "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in logged)
        )
        rows = [row for row in logged if row["kind"] == kind]
        if not rows or int(rows[-1]["step"]) != completed_step:
            raise ValueError("state-preparation metrics do not reach the progress checkpoint")
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
        metadata={"kind": f"{kind}_progress", **controls},
    )
    temporary.replace(checkpoint_path)
    _atomic_json(
        output_dir / "progress.json",
        {
            **controls,
            "completed_step": experiment.step,
            "active_slot": active_slot,
            "checkpoint_path": str(checkpoint_path.relative_to(output_dir)),
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
    completed = _completed_result(
        result_path, seed=experiment_config.seed, learning_rate=learning_rate
    )
    if completed is not None:
        return completed
    experiment = MBCExperiment(replace(experiment_config, learning_rate=learning_rate), metric)
    controls: Dict[str, object] = {
        "seed": experiment_config.seed,
        "learning_rate": learning_rate,
        "maximum_steps": maximum_steps,
        "save_at_step": save_at_step,
    }
    rows, latest, active_slot, writer, fresh = _resume_or_start(
        experiment, output_dir, kind="suppressed_preparation", controls=controls
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
                active_slot=active_slot,
            )
        if is_jointly_suppressed(rows, reference, thresholds) and (
            save_at_step is None or experiment.step >= save_at_step
        ):
            outcome = "suppressed"
            snapshot_path = output_dir / "suppressed_snapshot.pt"
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
                },
            )
            break
        if is_expressed(latest["c_int"], latest["exact_match"], reference, thresholds):
            outcome = "expressed_before_suppressed"
            break
        if (
            not math.isfinite(latest["full_vocab_ce"])
            or latest["full_vocab_ce"] > 5.0 * reference.q_star_loss_mean
        ):
            outcome = "diverged"
            break
    result = StatePreparationResult(
        seed=experiment_config.seed,
        learning_rate=learning_rate,
        outcome=outcome,
        step=experiment.step,
        c_int=latest["c_int"],
        exact_match=latest["exact_match"],
        delta_z=latest["delta_z"],
        full_vocab_ce=latest["full_vocab_ce"],
        snapshot_path=str(snapshot_path) if snapshot_path is not None else None,
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
    completed = _completed_result(
        result_path, seed=experiment_config.seed, learning_rate=learning_rate
    )
    if completed is not None:
        return completed
    experiment = MBCExperiment(replace(experiment_config, learning_rate=learning_rate), metric)
    controls = {
        "seed": experiment_config.seed,
        "learning_rate": learning_rate,
        "save_at_step": save_at_step,
    }
    rows, latest, active_slot, writer, fresh = _resume_or_start(
        experiment, output_dir, kind="expressed_preparation", controls=controls
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
        is_expressed(row["c_int"], row["exact_match"], reference, thresholds)
        for row in hold
    )
    snapshot_path: Optional[Path] = None
    if experiment.step == save_at_step and stable_expression:
        outcome = "expressed"
        snapshot_path = output_dir / "expressed_snapshot.pt"
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
            },
        )
    elif experiment.step < save_at_step:
        outcome = "diverged"
    else:
        outcome = "not_stably_expressed"
    result = StatePreparationResult(
        seed=experiment_config.seed,
        learning_rate=learning_rate,
        outcome=outcome,
        step=experiment.step,
        c_int=latest["c_int"],
        exact_match=latest["exact_match"],
        delta_z=latest["delta_z"],
        full_vocab_ce=latest["full_vocab_ce"],
        snapshot_path=str(snapshot_path) if snapshot_path is not None else None,
    )
    _atomic_json(result_path, asdict(result))
    return result
