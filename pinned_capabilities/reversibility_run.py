"""Common timing harness for acquisition and strong reversibility arms."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Optional

from .config import MBCExperimentConfig, MetricConfig
from .experiment import JSONLWriter, MBCExperiment
from .gate1 import reset_optimizer_state
from .parameter_groups import set_learning_rates
from .snapshot import load_snapshot, save_snapshot
from .state import ReferenceBands, StateThresholds, first_transition_step, is_expressed


@dataclass(frozen=True)
class TransitionTimingResult:
    arm: str
    learning_rate: float
    maximum_steps: int
    optimizer_reset: bool
    outcome: str
    transition_step: Optional[int]
    expressed_step: Optional[int]
    final_step: int
    final_c_int: float
    final_exact_match: float
    final_delta_z: float
    final_full_vocab_ce: float


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run_transition_timing(
    experiment_config: MBCExperimentConfig,
    metric: MetricConfig,
    reference: ReferenceBands,
    thresholds: StateThresholds,
    *,
    arm: str,
    learning_rate: float,
    maximum_steps: int,
    output_dir: Path,
    start_snapshot: Optional[Path] = None,
    optimizer_reset: bool = False,
) -> TransitionTimingResult:
    if maximum_steps <= thresholds.no_return_duration:
        raise ValueError("transition timing budget must exceed no-return duration")
    output_dir = Path(output_dir)
    result_path = output_dir / "result.json"
    if result_path.exists():
        result = TransitionTimingResult(**json.loads(result_path.read_text()))
        if (
            result.arm != arm
            or result.learning_rate != learning_rate
            or result.maximum_steps != maximum_steps
            or result.optimizer_reset != optimizer_reset
        ):
            raise ValueError("completed transition-timing result has mismatched controls")
        return result
    experiment = MBCExperiment(replace(experiment_config, learning_rate=learning_rate), metric)
    progress_path = output_dir / "progress.json"
    metrics_path = output_dir / "metrics.jsonl"
    rows = []
    expressed_step = None
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        expected = {
            "arm": arm,
            "learning_rate": learning_rate,
            "maximum_steps": maximum_steps,
            "optimizer_reset": optimizer_reset,
        }
        for key, value in expected.items():
            if progress.get(key) != value:
                raise ValueError(f"transition-timing progress has mismatched {key}")
        checkpoint_path = output_dir / progress["checkpoint_path"]
        if not checkpoint_path.exists():
            raise FileNotFoundError("transition-timing progress references a missing checkpoint")
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
        expressed_step = progress.get("expressed_step")
        active_slot = int(progress["active_slot"])
        if experiment.step - branch_start != completed_steps:
            raise ValueError("transition-timing checkpoint and progress disagree")
        if not metrics_path.exists():
            raise FileNotFoundError("transition-timing progress has no metrics log")
        logged = [json.loads(line) for line in metrics_path.read_text().splitlines()]
        logged = [
            row
            for row in logged
            if row["kind"] == "transition_timing_start"
            or int(row["branch_step"]) <= completed_steps
        ]
        metrics_path.write_text(
            "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in logged)
        )
        timed = [row for row in logged if row["kind"] == "transition_timing"]
        if not timed or int(timed[-1]["branch_step"]) != completed_steps:
            raise ValueError("transition-timing metrics do not reach the progress checkpoint")
        rows = [{**row, "step": float(row["branch_step"])} for row in timed]
        latest = timed[-1]
    else:
        active_slot = -1
        if metrics_path.exists():
            metrics_path.write_text("")
        if start_snapshot is not None:
            restored = load_snapshot(
                start_snapshot,
                model=experiment.model,
                optimizer=experiment.optimizer,
                stream=experiment.stream,
                map_location=experiment.device,
            )
            experiment.step = restored["step"]
        if optimizer_reset:
            reset_optimizer_state(experiment.optimizer)
        branch_start = experiment.step
        latest = experiment.evaluate()
    set_learning_rates(experiment.optimizer, learning_rate)
    writer = JSONLWriter(metrics_path)
    if not rows:
        writer.write(
            {
                "kind": "transition_timing_start",
                **latest,
                "branch_step": 0.0,
                "arm": arm,
                "optimizer_reset": optimizer_reset,
            }
        )
    outcome = "censored"
    while experiment.step - branch_start < maximum_steps:
        chunk = min(metric.eval_every, maximum_steps - (experiment.step - branch_start))
        training = experiment.advance(chunk)
        latest = {**training, **experiment.evaluate()}
        branch_step = experiment.step - branch_start
        row = {**latest, "step": float(branch_step)}
        rows.append(row)
        writer.write(
            {
                "kind": "transition_timing",
                **latest,
                "branch_step": float(branch_step),
                "arm": arm,
                "optimizer_reset": optimizer_reset,
                "learning_rate": learning_rate,
            }
        )
        expressed_now = is_expressed(
            latest["c_int"],
            latest["exact_match"],
            latest["delta_z"],
            reference,
            thresholds,
        )
        if expressed_now and expressed_step is None:
            expressed_step = branch_step
        elif not expressed_now:
            expressed_step = None
        if expressed_step is not None and branch_step >= expressed_step + thresholds.no_return_duration:
            outcome = "expressed"
        if (
            not math.isfinite(latest["full_vocab_ce"])
            or latest["full_vocab_ce"] > 5.0 * reference.q_star_loss_mean
        ):
            outcome = "diverged"
        checkpoint_every = max(1_000, metric.eval_every)
        if (
            branch_step % checkpoint_every == 0
            or branch_step == maximum_steps
            or outcome != "censored"
        ):
            active_slot = 1 if active_slot != 1 else 0
            checkpoint_path = (
                output_dir
                / "checkpoints"
                / f"slot_{active_slot}.pt"
            )
            temporary_checkpoint = checkpoint_path.with_suffix(".pt.tmp")
            save_snapshot(
                temporary_checkpoint,
                model=experiment.model,
                optimizer=experiment.optimizer,
                stream=experiment.stream,
                step=experiment.step,
                metadata={
                    "kind": "transition_timing_progress",
                    "arm": arm,
                    "branch_start": branch_start,
                },
            )
            temporary_checkpoint.replace(checkpoint_path)
            _atomic_json(
                progress_path,
                {
                    "arm": arm,
                    "learning_rate": learning_rate,
                    "maximum_steps": maximum_steps,
                    "optimizer_reset": optimizer_reset,
                    "branch_start": branch_start,
                    "completed_branch_steps": branch_step,
                    "expressed_step": expressed_step,
                    "checkpoint_path": str(checkpoint_path.relative_to(output_dir)),
                    "active_slot": active_slot,
                },
            )
        if outcome != "censored":
            break
    transition = first_transition_step(rows, reference, thresholds)
    result = TransitionTimingResult(
        arm=arm,
        learning_rate=learning_rate,
        maximum_steps=maximum_steps,
        optimizer_reset=optimizer_reset,
        outcome=outcome,
        transition_step=transition,
        expressed_step=expressed_step,
        final_step=int(rows[-1]["step"]) if rows else 0,
        final_c_int=latest["c_int"],
        final_exact_match=latest["exact_match"],
        final_delta_z=latest["delta_z"],
        final_full_vocab_ce=latest["full_vocab_ce"],
    )
    _atomic_json(result_path, asdict(result))
    return result
