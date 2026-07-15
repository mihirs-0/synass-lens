"""Checkpoint-level Gate 0 measurements on fixed data and capability probes."""

from __future__ import annotations

import copy
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch

from src.training.trainer import compute_loss

from .config import Gate0Config, MBCExperimentConfig, MetricConfig
from .experiment import MBCExperiment
from .local_stability import (
    AugmentedAdamWLinearization,
    capability_preconditioned_curvature,
    largest_preconditioned_curvature,
)
from .manifest import content_hash
from .mbc import build_mbc_probes, differentiable_mbc_c_int
from .metrics import sample_quartets
from .parameter_groups import set_learning_rates
from .provenance import sha256_file
from .snapshot import load_snapshot
from .training import next_batch


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _rate_label(learning_rate: float) -> str:
    return f"eta_{learning_rate:.12g}".replace(".", "p")


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


def _local_provenance(
    experiment_config: MBCExperimentConfig,
    metric: MetricConfig,
    gate: Gate0Config,
    snapshot_path: Path,
    learning_rate: float,
    *,
    augmented_tolerance: float,
    augmented_max_iterations: int,
    snapshot_sha256: Optional[str] = None,
) -> Dict[str, object]:
    identity: Dict[str, object] = {
        "schema_version": 1,
        "kind": "gate0_local_stability",
        "controls": {
            "experiment": _control_value(experiment_config),
            "metric": _control_value(metric),
            "gate": _control_value(gate),
            "learning_rate": float(learning_rate),
            "augmented_tolerance": float(augmented_tolerance),
            "augmented_max_iterations": int(augmented_max_iterations),
        },
        "sources": {
            "snapshot_sha256": snapshot_sha256 or sha256_file(snapshot_path),
        },
    }
    return {**identity, "cell_sha256": content_hash(identity)}


def _validate_local_provenance(
    observed: object, expected: Dict[str, object], *, location: Path
) -> None:
    if not isinstance(observed, dict):
        raise ValueError(f"cached local-stability cell at {location} has no provenance binding")
    declared = observed.get("cell_sha256")
    unsigned = {key: value for key, value in observed.items() if key != "cell_sha256"}
    if declared != content_hash(unsigned):
        raise ValueError(f"cached local-stability cell at {location} has tampered provenance")
    if observed != expected:
        raise ValueError(f"cached local-stability cell at {location} has mismatched provenance")


def _verify_snapshot(snapshot_path: Path, expected_sha256: str) -> None:
    if sha256_file(snapshot_path) != expected_sha256:
        raise ValueError(f"snapshot changed before worker load: {snapshot_path}")


def _seal_local_result(result: Dict[str, object]) -> Dict[str, object]:
    payload = {key: value for key, value in result.items() if key != "result_sha256"}
    return {**payload, "result_sha256": content_hash(payload)}


def _validate_local_result_digest(result: Dict[str, object], *, location: Path) -> None:
    payload = {key: value for key, value in result.items() if key != "result_sha256"}
    if result.get("result_sha256") != content_hash(payload):
        raise ValueError(
            f"completed local-stability cell at {location} has a tampered result payload"
        )


def _validate_expected_local_provenance(
    provenance: Dict[str, object],
    experiment_config: MBCExperimentConfig,
    metric: MetricConfig,
    gate: Gate0Config,
    snapshot_path: Path,
    learning_rate: float,
    *,
    augmented_tolerance: float,
    augmented_max_iterations: int,
    location: Path,
) -> None:
    sources = provenance.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("expected local-stability provenance has no source binding")
    expected = _local_provenance(
        experiment_config,
        metric,
        gate,
        snapshot_path,
        learning_rate,
        augmented_tolerance=augmented_tolerance,
        augmented_max_iterations=augmented_max_iterations,
        snapshot_sha256=str(sources.get("snapshot_sha256")),
    )
    _validate_local_provenance(provenance, expected, location=location)


def _measure_local_stability_scan_cell(
    experiment_config: MBCExperimentConfig,
    metric: MetricConfig,
    gate: Gate0Config,
    snapshot_path: Path,
    learning_rate: float,
    result_path: Path,
    expected_provenance: Dict[str, object],
) -> Dict[str, object]:
    """Spawn-safe entry point for one independent local-stability rate cell."""
    expected_snapshot_sha256 = str(
        expected_provenance["sources"]["snapshot_sha256"]
    )
    _verify_snapshot(snapshot_path, expected_snapshot_sha256)
    result = measure_checkpoint_local_stability(
        experiment_config,
        metric,
        gate,
        snapshot_path,
        result_path,
        learning_rate=learning_rate,
        _expected_provenance=expected_provenance,
    )
    if not isinstance(result.get("provenance"), dict):
        result = {**result, "provenance": expected_provenance}
    result = _seal_local_result(result)
    _atomic_json(result_path, result)
    _validate_local_provenance(
        result["provenance"], expected_provenance, location=result_path
    )
    _validate_local_result_digest(result, location=result_path)
    return result


def measure_checkpoint_local_stability(
    experiment_config: MBCExperimentConfig,
    metric: MetricConfig,
    gate: Gate0Config,
    snapshot_path: Path,
    output_path: Path,
    *,
    learning_rate: float | None = None,
    augmented_tolerance: float = 1e-3,
    augmented_max_iterations: int = 100,
    _expected_provenance: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Measure all registered local predictors without advancing training state."""
    if experiment_config.n_unique_b < 2 * gate.local_probe_b_count:
        raise ValueError("dataset is too small for the local counterfactual probe")
    effective_learning_rate = (
        float(learning_rate)
        if learning_rate is not None
        else float(experiment_config.learning_rate)
    )
    provenance = _expected_provenance or _local_provenance(
        experiment_config,
        metric,
        gate,
        snapshot_path,
        effective_learning_rate,
        augmented_tolerance=augmented_tolerance,
        augmented_max_iterations=augmented_max_iterations,
    )
    _validate_expected_local_provenance(
        provenance,
        experiment_config,
        metric,
        gate,
        snapshot_path,
        effective_learning_rate,
        augmented_tolerance=augmented_tolerance,
        augmented_max_iterations=augmented_max_iterations,
        location=Path(output_path),
    )
    experiment = MBCExperiment(experiment_config, metric)
    _verify_snapshot(
        snapshot_path, str(provenance["sources"]["snapshot_sha256"])
    )
    restored = load_snapshot(
        snapshot_path,
        model=experiment.model,
        optimizer=experiment.optimizer,
        stream=experiment.stream,
        map_location=experiment.device,
    )
    experiment.step = restored["step"]
    if learning_rate is not None:
        if learning_rate <= 0:
            raise ValueError("local-measurement learning rate must be positive")
        set_learning_rates(experiment.optimizer, learning_rate)
    stream_state = copy.deepcopy(experiment.stream.state_dict())
    training_batch = next_batch(experiment.dataset, experiment.stream, experiment.device)
    experiment.stream.load_state_dict(stream_state)

    local_probe = build_mbc_probes(
        experiment.mapping,
        experiment.tokenizer,
        n_b=gate.local_probe_b_count,
        seeds=metric.probe_seeds,
    )[0].to(experiment.device)
    local_quartets = sample_quartets(
        gate.local_probe_b_count,
        experiment_config.k,
        gate.local_quartet_count,
        seed=metric.probe_seeds[0],
        device=experiment.device,
    )
    experiment.model.eval()
    training_loss, _, _ = compute_loss(experiment.model, training_batch)
    capability_score = differentiable_mbc_c_int(
        experiment.model, local_probe, local_quartets
    )
    capability_curvature = capability_preconditioned_curvature(
        training_loss,
        capability_score,
        experiment.model.parameters(),
        experiment.optimizer,
    )

    def loss_closure():
        return compute_loss(experiment.model, training_batch)[0]

    largest_curvature = largest_preconditioned_curvature(
        loss_closure,
        experiment.model.parameters(),
        experiment.optimizer,
        iterations=gate.curvature_power_iterations,
        seed=experiment_config.seed,
    )
    augmented_loss = loss_closure()
    augmented = AugmentedAdamWLinearization(
        augmented_loss, experiment.model.parameters(), experiment.optimizer
    )
    eigenvalues, eigenvectors = augmented.dominant_eigenpairs(
        count=gate.augmented_eigenvalue_count,
        tolerance=augmented_tolerance,
        max_iterations=augmented_max_iterations,
        seed=experiment_config.seed,
    )
    eigen_residuals = augmented.eigenpair_residuals(eigenvalues, eigenvectors)
    result: Dict[str, object] = {
        "step": experiment.step,
        "learning_rate": float(experiment.optimizer.param_groups[0]["lr"]),
        "training_loss": float(training_loss.item()),
        "capability_score": float(capability_score.item()),
        "capability_preconditioned_curvature": float(capability_curvature.item()),
        "largest_preconditioned_curvature": float(largest_curvature.item()),
        "augmented_eigenvalues": [
            {
                "real": float(value.real),
                "imag": float(value.imag),
                "magnitude": float(abs(value)),
                "relative_residual": float(residual),
            }
            for value, residual in zip(eigenvalues, eigen_residuals)
        ],
        "augmented_spectral_radius": float(np.max(np.abs(eigenvalues))),
        "augmented_max_relative_residual": float(np.max(eigen_residuals)),
        "augmented_certified": bool(
            np.max(eigen_residuals) <= gate.augmented_max_relative_residual
        ),
        "augmented_balance_block_scales": list(augmented.balance_block_scales),
        "local_probe_b_count": gate.local_probe_b_count,
        "local_quartet_count": gate.local_quartet_count,
        "stream_unchanged": (
            experiment.stream.epoch == stream_state["epoch"]
            and experiment.stream.cursor == stream_state["cursor"]
            and torch.equal(experiment.stream.order, stream_state["order"])
            and torch.equal(
                experiment.stream.generator.get_state(), stream_state["generator_state"]
            )
        ),
        "provenance": provenance,
    }
    output_path = Path(output_path)
    result = _seal_local_result(result)
    _atomic_json(output_path, result)
    return result


def measure_checkpoint_local_stability_scan(
    experiment_config: MBCExperimentConfig,
    metric: MetricConfig,
    gate: Gate0Config,
    snapshot_path: Path,
    learning_rates: Sequence[float],
    output_dir: Path,
    *,
    workers: int = 1,
) -> dict:
    """Measure a manifest-frozen rate grid, resuming at completed rate cells.

    Spawned workers own disjoint rate directories; only the parent writes the
    ordered aggregate. Worker count is therefore an execution detail that may be
    changed safely when resuming the same frozen scan.
    """
    rates = sorted(float(rate) for rate in learning_rates)
    if not rates or any(rate <= 0 for rate in rates):
        raise ValueError("local-stability scan rates must be positive")
    if len(set(rates)) != len(rates):
        raise ValueError("local-stability scan rates must be unique")
    if workers < 1:
        raise ValueError("local-stability scan workers must be at least one")
    output_dir = Path(output_dir)
    snapshot_sha256 = sha256_file(snapshot_path)
    provenances = {
        learning_rate: _local_provenance(
            experiment_config,
            metric,
            gate,
            snapshot_path,
            learning_rate,
            augmented_tolerance=1e-3,
            augmented_max_iterations=100,
            snapshot_sha256=snapshot_sha256,
        )
        for learning_rate in rates
    }
    cached: Dict[float, Dict[str, object]] = {}
    missing = []
    for learning_rate in rates:
        result_path = output_dir / _rate_label(learning_rate) / "local_stability.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if not np.isclose(
                float(result["learning_rate"]), learning_rate, rtol=1e-12, atol=0.0
            ):
                raise ValueError(f"completed local-stability cell at {result_path} has wrong rate")
            _validate_local_provenance(
                result.get("provenance"),
                provenances[learning_rate],
                location=result_path,
            )
            _validate_local_result_digest(result, location=result_path)
            cached[learning_rate] = result
        else:
            missing.append((learning_rate, result_path))

    futures = {}
    executor = None
    if workers > 1 and len(missing) > 1:
        executor = ProcessPoolExecutor(
            max_workers=min(workers, len(missing)),
            mp_context=multiprocessing.get_context("spawn"),
        )
        futures = {
            learning_rate: executor.submit(
                _measure_local_stability_scan_cell,
                experiment_config,
                metric,
                gate,
                snapshot_path,
                learning_rate,
                result_path,
                provenances[learning_rate],
            )
            for learning_rate, result_path in missing
        }

    results = []
    try:
        for learning_rate in rates:
            result = cached.get(learning_rate)
            if result is None:
                result_path = output_dir / _rate_label(learning_rate) / "local_stability.json"
                if executor is None:
                    result = _measure_local_stability_scan_cell(
                        experiment_config,
                        metric,
                        gate,
                        snapshot_path,
                        learning_rate,
                        result_path,
                        provenances[learning_rate],
                    )
                else:
                    result = futures[learning_rate].result()
            results.append(result)
            _atomic_json(
                output_dir / "scan.json",
                {
                    "complete": len(results) == len(rates),
                    "requested_learning_rates": rates,
                    "measurements": results,
                },
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
    summary = {
        "complete": True,
        "requested_learning_rates": rates,
        "all_augmented_eigenpairs_certified": all(
            bool(result["augmented_certified"]) for result in results
        ),
        "uncertified_learning_rates": [
            float(result["learning_rate"])
            for result in results
            if not bool(result["augmented_certified"])
        ],
        "measurements": results,
    }
    _atomic_json(output_dir / "scan.json", summary)
    return summary
