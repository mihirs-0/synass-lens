"""Matched-step weights x optimizer-state memory factorial."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import torch

from .config import MBCExperimentConfig, MetricConfig
from .experiment import JSONLWriter, MBCExperiment
from .gate1 import add_norm_matched_gaussian_noise
from .parameter_groups import set_learning_rates
from .snapshot import load_crossed_snapshot
from .state import (
    ReferenceBands,
    StateThresholds,
    first_transition_step,
    is_expressed,
    is_jointly_suppressed,
)


@dataclass(frozen=True)
class MemoryArmResult:
    weights_source: str
    optimizer_source: str
    outcome: str
    transition_step: Optional[int]
    final_c_int: float
    final_exact_match: float
    final_delta_z: float
    final_full_vocab_ce: float


def _load_payload(path: Path) -> dict:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1:
        raise ValueError("memory surgery requires snapshot schema version 1")
    return payload


def _weight_distance(left: dict, right: dict) -> float:
    squared = 0.0
    for name, value in left["model"].items():
        other = right["model"][name]
        if torch.is_floating_point(value):
            squared += float((value.double() - other.double()).square().sum().item())
    return math.sqrt(squared)


def _run_arm(
    experiment_config: MBCExperimentConfig,
    metric: MetricConfig,
    reference: ReferenceBands,
    thresholds: StateThresholds,
    *,
    weights_path: Path,
    optimizer_path: Path,
    weights_source: str,
    optimizer_source: str,
    learning_rate: float,
    challenge_steps: int,
    output_dir: Path,
    gaussian_norm: Optional[float] = None,
    gaussian_seed: int = 0,
) -> MemoryArmResult:
    experiment = MBCExperiment(experiment_config, metric)
    restored = load_crossed_snapshot(
        weights_path=weights_path,
        optimizer_path=optimizer_path,
        model=experiment.model,
        optimizer=experiment.optimizer,
        stream=experiment.stream,
        map_location=experiment.device,
    )
    experiment.step = restored["step"]
    if gaussian_norm is not None:
        add_norm_matched_gaussian_noise(experiment.model, gaussian_norm, gaussian_seed)
    set_learning_rates(experiment.optimizer, learning_rate)
    writer = JSONLWriter(output_dir / "metrics.jsonl")
    branch_start = experiment.step
    rows = []
    while experiment.step - branch_start < challenge_steps:
        chunk = min(metric.eval_every, challenge_steps - (experiment.step - branch_start))
        training = experiment.advance(chunk)
        latest = {**training, **experiment.evaluate()}
        branch_step = experiment.step - branch_start
        writer.write(
            {
                "kind": "memory_surgery",
                **latest,
                "branch_step": float(branch_step),
                "weights_source": weights_source,
                "optimizer_source": optimizer_source,
                "learning_rate": learning_rate,
            }
        )
        rows.append({**latest, "step": float(branch_step)})
    latest = rows[-1]
    if is_expressed(latest["c_int"], latest["exact_match"], reference, thresholds):
        outcome = "expressed"
    elif is_jointly_suppressed(rows, reference, thresholds):
        outcome = "suppressed"
    elif (
        not math.isfinite(latest["full_vocab_ce"])
        or latest["full_vocab_ce"] > 5.0 * reference.q_star_loss_mean
    ):
        outcome = "diverged"
    else:
        outcome = "intermediate"
    result = MemoryArmResult(
        weights_source=weights_source,
        optimizer_source=optimizer_source,
        outcome=outcome,
        transition_step=first_transition_step(rows, reference, thresholds),
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


def run_memory_factorial(
    experiment_config: MBCExperimentConfig,
    metric: MetricConfig,
    reference: ReferenceBands,
    thresholds: StateThresholds,
    *,
    expressed_snapshot: Path,
    suppressed_snapshot: Path,
    learning_rate: float,
    challenge_steps: int,
    output_dir: Path,
) -> dict:
    expressed = _load_payload(expressed_snapshot)
    suppressed = _load_payload(suppressed_snapshot)
    if int(expressed["step"]) != int(suppressed["step"]):
        raise ValueError("memory-source snapshots must be matched at the same training step")
    expected_seed = experiment_config.seed
    for label, payload in (("expressed", expressed), ("suppressed", suppressed)):
        seed = payload.get("metadata", {}).get("seed")
        if seed is not None and int(seed) != expected_seed:
            raise ValueError(f"{label} snapshot seed does not match experiment seed")
    distance = _weight_distance(expressed, suppressed)
    output_dir = Path(output_dir)
    sources: Dict[str, Path] = {
        "expressed": Path(expressed_snapshot),
        "suppressed": Path(suppressed_snapshot),
    }
    results = []
    for weights_source in ("expressed", "suppressed"):
        for optimizer_source in ("expressed", "suppressed"):
            result = _run_arm(
                experiment_config,
                metric,
                reference,
                thresholds,
                weights_path=sources[weights_source],
                optimizer_path=sources[optimizer_source],
                weights_source=weights_source,
                optimizer_source=optimizer_source,
                learning_rate=learning_rate,
                challenge_steps=challenge_steps,
                output_dir=output_dir / f"weights_{weights_source}__optimizer_{optimizer_source}",
            )
            results.append(asdict(result))
    gaussian = _run_arm(
        experiment_config,
        metric,
        reference,
        thresholds,
        weights_path=sources["suppressed"],
        optimizer_path=sources["suppressed"],
        weights_source="suppressed_gaussian",
        optimizer_source="suppressed",
        learning_rate=learning_rate,
        challenge_steps=challenge_steps,
        output_dir=output_dir / "weights_suppressed_gaussian__optimizer_suppressed",
        gaussian_norm=distance,
        gaussian_seed=91_003 + experiment_config.seed,
    )
    results.append(asdict(gaussian))
    summary = {
        "seed": experiment_config.seed,
        "matched_step": int(expressed["step"]),
        "learning_rate": learning_rate,
        "challenge_steps": challenge_steps,
        "expressed_suppressed_weight_distance": distance,
        "arms": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "memory_factorial.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary
