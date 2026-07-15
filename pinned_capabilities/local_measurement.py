"""Checkpoint-level Gate 0 measurements on fixed data and capability probes."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict, Sequence

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
from .mbc import build_mbc_probes, differentiable_mbc_c_int
from .metrics import sample_quartets
from .parameter_groups import set_learning_rates
from .snapshot import load_snapshot
from .training import next_batch


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _rate_label(learning_rate: float) -> str:
    return f"eta_{learning_rate:.12g}".replace(".", "p")


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
) -> Dict[str, object]:
    """Measure all registered local predictors without advancing training state."""
    if experiment_config.n_unique_b < 2 * gate.local_probe_b_count:
        raise ValueError("dataset is too small for the local counterfactual probe")
    experiment = MBCExperiment(experiment_config, metric)
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
    }
    output_path = Path(output_path)
    _atomic_json(output_path, result)
    return result


def measure_checkpoint_local_stability_scan(
    experiment_config: MBCExperimentConfig,
    metric: MetricConfig,
    gate: Gate0Config,
    snapshot_path: Path,
    learning_rates: Sequence[float],
    output_dir: Path,
) -> dict:
    """Measure a manifest-frozen rate grid, resuming at completed rate cells."""
    rates = sorted(float(rate) for rate in learning_rates)
    if not rates or any(rate <= 0 for rate in rates):
        raise ValueError("local-stability scan rates must be positive")
    if len(set(rates)) != len(rates):
        raise ValueError("local-stability scan rates must be unique")
    output_dir = Path(output_dir)
    results = []
    for learning_rate in rates:
        result_path = output_dir / _rate_label(learning_rate) / "local_stability.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if not np.isclose(
                float(result["learning_rate"]), learning_rate, rtol=1e-12, atol=0.0
            ):
                raise ValueError(f"completed local-stability cell at {result_path} has wrong rate")
        else:
            result = measure_checkpoint_local_stability(
                experiment_config,
                metric,
                gate,
                snapshot_path,
                result_path,
                learning_rate=learning_rate,
            )
        results.append(result)
        _atomic_json(
            output_dir / "scan.json",
            {
                "complete": len(results) == len(rates),
                "requested_learning_rates": rates,
                "measurements": results,
            },
        )
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
