"""Construct validated expressed and suppressed starting checkpoints."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Optional

from .config import MBCExperimentConfig, MetricConfig
from .experiment import JSONLWriter, MBCExperiment
from .snapshot import save_snapshot
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


def prepare_suppressed_checkpoint(
    experiment_config: MBCExperimentConfig,
    metric: MetricConfig,
    reference: ReferenceBands,
    thresholds: StateThresholds,
    *,
    learning_rate: float,
    maximum_steps: int,
    output_dir: Path,
) -> StatePreparationResult:
    if maximum_steps < thresholds.suppressed_duration:
        raise ValueError("state-preparation budget is shorter than suppressed duration")
    output_dir = Path(output_dir)
    experiment = MBCExperiment(replace(experiment_config, learning_rate=learning_rate), metric)
    writer = JSONLWriter(output_dir / "metrics.jsonl")
    rows = []
    latest = experiment.evaluate()
    writer.write({"kind": "suppressed_preparation_start", **latest})
    outcome = "censored"
    snapshot_path: Optional[Path] = None
    while experiment.step < maximum_steps:
        chunk = min(metric.eval_every, maximum_steps - experiment.step)
        training = experiment.advance(chunk)
        latest = {**training, **experiment.evaluate(), "learning_rate": learning_rate}
        rows.append(latest)
        writer.write({"kind": "suppressed_preparation", **latest})
        if is_jointly_suppressed(rows, reference, thresholds):
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
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True) + "\n"
    )
    return result
