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
from .snapshot import load_snapshot
from .state import ReferenceBands, StateThresholds, first_transition_step, is_expressed


@dataclass(frozen=True)
class TransitionTimingResult:
    arm: str
    learning_rate: float
    optimizer_reset: bool
    outcome: str
    transition_step: Optional[int]
    expressed_step: Optional[int]
    final_step: int
    final_c_int: float
    final_exact_match: float
    final_delta_z: float
    final_full_vocab_ce: float


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
    experiment = MBCExperiment(replace(experiment_config, learning_rate=learning_rate), metric)
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
    set_learning_rates(experiment.optimizer, learning_rate)
    output_dir = Path(output_dir)
    writer = JSONLWriter(output_dir / "metrics.jsonl")
    branch_start = experiment.step
    rows = []
    expressed_step = None
    latest = experiment.evaluate()
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
            latest["c_int"], latest["exact_match"], reference, thresholds
        )
        if expressed_now and expressed_step is None:
            expressed_step = branch_step
        elif not expressed_now:
            expressed_step = None
        if expressed_step is not None and branch_step >= expressed_step + thresholds.no_return_duration:
            outcome = "expressed"
            break
        if (
            not math.isfinite(latest["full_vocab_ce"])
            or latest["full_vocab_ce"] > 5.0 * reference.q_star_loss_mean
        ):
            outcome = "diverged"
            break
    transition = first_transition_step(rows, reference, thresholds)
    result = TransitionTimingResult(
        arm=arm,
        learning_rate=learning_rate,
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
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True) + "\n"
    )
    return result
