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


@dataclass(frozen=True)
class AcquisitionResult:
    learning_rate: float
    outcome: str
    transition_step: Optional[int]
    final_c_int: float
    final_exact_match: float
    final_delta_z: float
    final_full_vocab_ce: float


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _branch_label(learning_rate: float) -> str:
    return f"eta_{learning_rate:.12g}".replace(".", "p")


def _load_erasure_result(path: Path, learning_rate: float) -> Optional[ErasureResult]:
    if not path.exists():
        return None
    result = ErasureResult(**json.loads(path.read_text()))
    if not math.isclose(result.learning_rate, learning_rate, rel_tol=1e-12, abs_tol=0.0):
        raise ValueError(f"completed branch at {path} has the wrong learning rate")
    return result


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
    if entry is not None and entry <= gate.erase_horizon:
        outcome = "erased"
    elif divergent:
        outcome = "diverged"
    elif is_expressed(
        latest["c_int"], latest["exact_match"], reference, thresholds
    ):
        outcome = "retained"
    else:
        outcome = "unresolved"
    result = ErasureResult(
        learning_rate=learning_rate,
        erased=erased,
        outcome=outcome,
        sustained_entry_step=entry,
        final_c_int=latest["c_int"],
        final_exact_match=latest["exact_match"],
        final_delta_z=latest["delta_z"],
        final_full_vocab_ce=latest["full_vocab_ce"],
    )
    _atomic_write_json(output_dir / "result.json", asdict(result))
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
) -> dict:
    """Run a resumable coarse scan and expose only strict adjacent brackets."""
    rates = sorted(float(rate) for rate in learning_rates)
    if not rates or any(rate <= 0 for rate in rates):
        raise ValueError("erasure scan learning rates must be positive")
    if len(set(rates)) != len(rates):
        raise ValueError("erasure scan learning rates must be unique")
    output_dir = Path(output_dir)
    results = []
    for learning_rate in rates:
        branch_dir = output_dir / _branch_label(learning_rate)
        result_path = branch_dir / "result.json"
        result = _load_erasure_result(result_path, learning_rate)
        if result is None:
            result = run_erasure_branch(
                experiment_config,
                metric,
                gate,
                reference,
                snapshot_path,
                learning_rate,
                branch_dir,
                thresholds=thresholds,
            )
            _atomic_write_json(result_path, asdict(result))
        results.append(result)
        partial = {
            "complete": len(results) == len(rates),
            "requested_learning_rates": rates,
            "branches": [asdict(branch) for branch in results],
        }
        _atomic_write_json(output_dir / "scan.json", partial)
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
    cache: Dict[float, ErasureResult] = {}

    def evaluate(learning_rate: float) -> ErasureResult:
        if learning_rate not in cache:
            branch_dir = output_dir / _branch_label(learning_rate)
            existing = _load_erasure_result(branch_dir / "result.json", learning_rate)
            if existing is not None:
                cache[learning_rate] = existing
            else:
                cache[learning_rate] = run_erasure_branch(
                    experiment_config,
                    metric,
                    gate,
                    reference,
                    snapshot_path,
                    learning_rate,
                    branch_dir,
                    thresholds=thresholds,
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
    _atomic_write_json(output_dir / "result.json", asdict(result))
    return result
