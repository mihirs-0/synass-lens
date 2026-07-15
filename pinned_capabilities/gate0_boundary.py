"""Directional empirical boundaries for Gate 0."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Optional, Sequence

from .config import Gate0Config, MBCExperimentConfig, MetricConfig
from .experiment import JSONLWriter, MBCExperiment
from .parameter_groups import set_learning_rates
from .snapshot import load_snapshot
from .state import (
    ReferenceBands,
    StateThresholds,
    first_transition_step,
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


@dataclass(frozen=True)
class AcquisitionResult:
    learning_rate: float
    outcome: str
    transition_step: Optional[int]
    final_c_int: float
    final_exact_match: float
    final_delta_z: float
    final_full_vocab_ce: float


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
) -> ErasureResult:
    thresholds = thresholds or StateThresholds()
    experiment = MBCExperiment(replace(experiment_config, learning_rate=learning_rate), metric)
    restored = load_snapshot(
        snapshot_path,
        model=experiment.model,
        optimizer=experiment.optimizer,
        stream=experiment.stream,
        map_location=experiment.device,
    )
    experiment.step = restored["step"]
    set_learning_rates(experiment.optimizer, learning_rate)
    output_dir = Path(output_dir)
    writer = JSONLWriter(output_dir / "metrics.jsonl")
    branch_start = experiment.step
    rows = []
    while experiment.step - branch_start < gate.erase_hold_steps:
        chunk = min(metric.eval_every, gate.erase_hold_steps - (experiment.step - branch_start))
        training = experiment.advance(chunk)
        row = {**training, **experiment.evaluate()}
        row["branch_step"] = float(experiment.step - branch_start)
        row["learning_rate"] = learning_rate
        rows.append(row)
        writer.write({"kind": "gate0_erasure", **row})
    entry = sustained_band_entry(
        rows, reference, branch_end_step=gate.erase_hold_steps, thresholds=thresholds
    )
    latest = rows[-1]
    erased = entry is not None and entry <= gate.erase_horizon
    divergent = (
        not math.isfinite(latest["full_vocab_ce"])
        or latest["full_vocab_ce"] > 5.0 * reference.q_star_loss_mean
    )
    result = ErasureResult(
        learning_rate=learning_rate,
        erased=erased,
        outcome="erased" if erased else ("diverged" if divergent else "retained_or_unresolved"),
        sustained_entry_step=entry,
        final_c_int=latest["c_int"],
        final_exact_match=latest["exact_match"],
        final_delta_z=latest["delta_z"],
        final_full_vocab_ce=latest["full_vocab_ce"],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(asdict(result), indent=2) + "\n")
    return result


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
    cache: Dict[float, ErasureResult] = {}

    def evaluate(learning_rate: float) -> ErasureResult:
        if learning_rate not in cache:
            label = f"eta_{learning_rate:.12g}".replace(".", "p")
            cache[learning_rate] = run_erasure_branch(
                experiment_config,
                metric,
                gate,
                reference,
                snapshot_path,
                learning_rate,
                output_dir / label,
                thresholds=thresholds,
            )
        return cache[learning_rate]

    lower = evaluate(lower_learning_rate)
    upper = evaluate(upper_learning_rate)
    if lower.erased or lower.outcome == "diverged" or not upper.erased:
        raise ValueError(
            "invalid erasure bracket: lower must retain expression and upper must erase"
        )
    for _ in range(gate.boundary_bisection_steps):
        midpoint = math.sqrt(lower.learning_rate * upper.learning_rate)
        result = evaluate(midpoint)
        if result.erased:
            upper = result
        else:
            lower = result
    summary = {
        "largest_non_erasing_learning_rate": lower.learning_rate,
        "smallest_erasing_learning_rate": upper.learning_rate,
        "multiplicative_interval": upper.learning_rate / lower.learning_rate,
        "branches": [asdict(cache[key]) for key in sorted(cache)],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "boundary.json").write_text(json.dumps(summary, indent=2) + "\n")
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
) -> AcquisitionResult:
    """Run a censored fixed-rule acquisition branch from one suppressed state."""
    thresholds = thresholds or StateThresholds()
    experiment = MBCExperiment(replace(experiment_config, learning_rate=learning_rate), metric)
    restored = load_snapshot(
        snapshot_path,
        model=experiment.model,
        optimizer=experiment.optimizer,
        stream=experiment.stream,
        map_location=experiment.device,
    )
    experiment.step = restored["step"]
    set_learning_rates(experiment.optimizer, learning_rate)
    output_dir = Path(output_dir)
    writer = JSONLWriter(output_dir / "metrics.jsonl")
    branch_start = experiment.step
    analysis_rows = []
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
        raise ValueError("acquisition branch must start from a suppressed flat-loss snapshot")
    writer.write(
        {
            "kind": "gate0_acquisition_start",
            **latest,
            "branch_step": 0.0,
            "learning_rate": learning_rate,
        }
    )
    while experiment.step - branch_start < gate.acquire_horizon:
        chunk = min(metric.eval_every, gate.acquire_horizon - (experiment.step - branch_start))
        training = experiment.advance(chunk)
        latest = {**training, **experiment.evaluate()}
        branch_step = experiment.step - branch_start
        logged = {**latest, "branch_step": float(branch_step), "learning_rate": learning_rate}
        writer.write({"kind": "gate0_acquisition", **logged})
        analysis_rows.append({**latest, "step": float(branch_step)})
    transition = first_transition_step(analysis_rows, reference, thresholds)
    divergent = (
        not math.isfinite(latest["full_vocab_ce"])
        or latest["full_vocab_ce"] > 5.0 * reference.q_star_loss_mean
    )
    outcome = "transitioned" if transition is not None else ("diverged" if divergent else "censored")
    result = AcquisitionResult(
        learning_rate=learning_rate,
        outcome=outcome,
        transition_step=transition,
        final_c_int=latest["c_int"],
        final_exact_match=latest["exact_match"],
        final_delta_z=latest["delta_z"],
        final_full_vocab_ce=latest["full_vocab_ce"],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(asdict(result), indent=2) + "\n")
    return result
