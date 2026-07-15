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
from .snapshot import load_crossed_snapshot, load_snapshot, save_snapshot
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


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_completed_arm(
    path: Path, *, weights_source: str, optimizer_source: str
) -> Optional[MemoryArmResult]:
    if not path.exists():
        return None
    result = MemoryArmResult(**json.loads(path.read_text()))
    if (
        result.weights_source != weights_source
        or result.optimizer_source != optimizer_source
    ):
        raise ValueError(f"completed memory arm at {path} has mismatched sources")
    return result


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
    output_dir = Path(output_dir)
    result_path = output_dir / "result.json"
    completed = _load_completed_arm(
        result_path,
        weights_source=weights_source,
        optimizer_source=optimizer_source,
    )
    if completed is not None:
        return completed
    experiment = MBCExperiment(experiment_config, metric)
    progress_path = output_dir / "progress.json"
    metrics_path = output_dir / "metrics.jsonl"
    rows = []
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        expected = {
            "weights_source": weights_source,
            "optimizer_source": optimizer_source,
            "learning_rate": learning_rate,
            "challenge_steps": challenge_steps,
        }
        for key, value in expected.items():
            if progress.get(key) != value:
                raise ValueError(f"memory-arm progress has mismatched {key}")
        checkpoint_path = output_dir / progress["checkpoint_path"]
        if not checkpoint_path.exists():
            raise FileNotFoundError("memory-arm progress references a missing checkpoint")
        restored = load_snapshot(
            checkpoint_path,
            model=experiment.model,
            optimizer=experiment.optimizer,
            stream=experiment.stream,
            map_location=experiment.device,
        )
        branch_start = int(progress["branch_start"])
        completed_steps = int(progress["completed_branch_steps"])
        active_slot = int(progress["active_slot"])
        if int(restored["step"]) - branch_start != completed_steps:
            raise ValueError("memory-arm checkpoint and progress disagree")
        if not metrics_path.exists():
            raise FileNotFoundError("memory-arm progress has no metrics log")
        logged = [json.loads(line) for line in metrics_path.read_text().splitlines()]
        logged = [row for row in logged if int(row["branch_step"]) <= completed_steps]
        if not logged or int(logged[-1]["branch_step"]) != completed_steps:
            raise ValueError("memory-arm metrics do not reach the progress checkpoint")
        metrics_path.write_text(
            "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in logged)
        )
        rows = [{**row, "step": float(row["branch_step"])} for row in logged]
    else:
        active_slot = -1
        if metrics_path.exists():
            metrics_path.write_text("")
        restored = load_crossed_snapshot(
            weights_path=weights_path,
            optimizer_path=optimizer_path,
            model=experiment.model,
            optimizer=experiment.optimizer,
            stream=experiment.stream,
            map_location=experiment.device,
        )
        branch_start = int(restored["step"])
        if gaussian_norm is not None:
            add_norm_matched_gaussian_noise(experiment.model, gaussian_norm, gaussian_seed)
    experiment.step = restored["step"]
    set_learning_rates(experiment.optimizer, learning_rate)
    writer = JSONLWriter(metrics_path)
    checkpoint_every = max(1_000, metric.eval_every)
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
        if branch_step % checkpoint_every == 0 or branch_step == challenge_steps:
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
                    "kind": "memory_surgery_progress",
                    "weights_source": weights_source,
                    "optimizer_source": optimizer_source,
                    "branch_start": branch_start,
                },
            )
            temporary_checkpoint.replace(checkpoint_path)
            _atomic_json(
                progress_path,
                {
                    "weights_source": weights_source,
                    "optimizer_source": optimizer_source,
                    "learning_rate": learning_rate,
                    "challenge_steps": challenge_steps,
                    "branch_start": branch_start,
                    "completed_branch_steps": branch_step,
                    "checkpoint_path": str(checkpoint_path.relative_to(output_dir)),
                    "active_slot": active_slot,
                },
            )
    latest = rows[-1]
    if is_expressed(
        latest["c_int"], latest["exact_match"], latest["delta_z"], reference, thresholds
    ):
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
    _atomic_json(result_path, asdict(result))
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

    def record(result: MemoryArmResult) -> None:
        results.append(asdict(result))
        _atomic_json(
            output_dir / "memory_factorial.progress.json",
            {
                "seed": experiment_config.seed,
                "matched_step": int(expressed["step"]),
                "learning_rate": learning_rate,
                "challenge_steps": challenge_steps,
                "expressed_suppressed_weight_distance": distance,
                "completed_arms": results,
            },
        )
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
            record(result)
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
    record(gaussian)
    summary = {
        "seed": experiment_config.seed,
        "matched_step": int(expressed["step"]),
        "learning_rate": learning_rate,
        "challenge_steps": challenge_steps,
        "expressed_suppressed_weight_distance": distance,
        "arms": results,
    }
    _atomic_json(output_dir / "memory_factorial.json", summary)
    return summary
