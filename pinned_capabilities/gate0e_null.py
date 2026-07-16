"""Gate 0-E null predictor: v-conditioned augmented stability against c*.

Amendment v1.5 section 2.3 gives the local theory one dev-fit constant. The
implementation is frozen by design note 4: every condition (seed, batch size)
receives a uniform **moment refresh** — exactly ``REFRESH_STEPS`` steps at the
preparation rate and the condition's batch size from the condition's starting
snapshot, with a deterministically derived fresh data order — and the
certified augmented-AdamW spectral radius is then measured at every challenge
rate from the refreshed state.

Crossing convention (frozen before any dev number was read): the predicted
eta50 is the smallest rate at which the certified radius reaches c*, found by
log-log interpolation on the first upward crossing between adjacent certified
grid rates. A radius already above c* at the lowest rate is a left-censored
prediction; one that never reaches c* is right-censored. Censored predictions
cannot satisfy the null-wins criterion. Segments touching an uncertified cell
never produce a crossing. Non-monotone radius tables are reported in full so
multiple crossings are visible in the artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import torch

from .batch_stream import DeterministicBatchStream
from .config import Gate0Config, MBCExperimentConfig, MetricConfig
from .experiment import MBCExperiment
from .gate0_boundary import _atomic_write_json
from .gate0e_escape import load_snapshot_fresh_stream
from .local_measurement import measure_checkpoint_local_stability_scan
from .manifest import content_hash
from .parameter_groups import set_learning_rates
from .provenance import bind_file, sha256_file
from .snapshot import save_snapshot

REFRESH_STEPS = 1_000
REFRESH_RATE = 1e-3


def refresh_data_seed(seed: int, batch_size: int) -> int:
    key = f"gate0e_refresh|seed={seed}|batch={batch_size}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")


def _flatten_parameters(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()])


def refresh_moments(
    experiment_config: MBCExperimentConfig,
    metric: MetricConfig,
    snapshot_path: Path,
    output_dir: Path,
    *,
    refresh_steps: int = REFRESH_STEPS,
    refresh_rate: float = REFRESH_RATE,
) -> Dict[str, object]:
    """Produce the refreshed snapshot for one (seed, batch) condition."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    record_path = output_dir / "refresh.json"
    snapshot_out = output_dir / "refreshed_snapshot.pt"
    base_sha256 = sha256_file(snapshot_path)
    controls = {
        "kind": "gate0e_moment_refresh",
        "seed": experiment_config.seed,
        "batch_size": experiment_config.batch_size,
        "refresh_steps": refresh_steps,
        "refresh_rate": refresh_rate,
        "refresh_data_seed": refresh_data_seed(
            experiment_config.seed, experiment_config.batch_size
        ),
        "base_snapshot_sha256": base_sha256,
    }
    if record_path.exists() and snapshot_out.exists():
        record = json.loads(record_path.read_text())
        if record.get("controls") != controls:
            raise ValueError("existing moment refresh has mismatched controls")
        if sha256_file(snapshot_out) != record.get("refreshed_snapshot_sha256"):
            raise ValueError("existing refreshed snapshot digest mismatch")
        return record
    experiment = MBCExperiment(experiment_config, metric)
    restored = load_snapshot_fresh_stream(
        snapshot_path,
        model=experiment.model,
        optimizer=experiment.optimizer,
        map_location=experiment.device,
    )
    experiment.step = restored["step"]
    experiment.stream = DeterministicBatchStream(
        len(experiment.dataset),
        experiment_config.batch_size,
        seed=controls["refresh_data_seed"],
    )
    set_learning_rates(experiment.optimizer, refresh_rate)
    theta_before = _flatten_parameters(experiment.model)
    previous = theta_before.clone()
    step_sum = torch.zeros_like(previous)
    squared_power = 0.0
    first = last = None
    for step_index in range(refresh_steps):
        last = experiment.advance(1)
        if step_index == 0:
            first = last
        current = _flatten_parameters(experiment.model)
        step = current - previous
        step_sum += step
        squared_power += float(step.pow(2).sum())
        previous = current
    theta_after = previous
    drift = float(
        torch.linalg.vector_norm(theta_after - theta_before)
        / torch.linalg.vector_norm(theta_before)
    )
    mean_step_power = squared_power / refresh_steps
    drift_power = float((step_sum / refresh_steps).pow(2).sum())
    update_diffusion = {
        "steps": refresh_steps,
        "refresh_rate": refresh_rate,
        "batch_size": experiment_config.batch_size,
        "mean_step_power": mean_step_power,
        "drift_power": drift_power,
        "diffusion_power": max(mean_step_power - drift_power, 0.0),
    }
    save_snapshot(
        snapshot_out,
        model=experiment.model,
        optimizer=experiment.optimizer,
        stream=experiment.stream,
        step=experiment.step,
        metadata={
            "kind": "gate0e_refreshed",
            "seed": experiment_config.seed,
            "controls": controls,
        },
    )
    record = {
        "controls": controls,
        "step_after": experiment.step,
        "first_step_loss": float(first["train_loss"]),
        "last_step_loss": float(last["train_loss"]),
        "theta_relative_drift": drift,
        "update_diffusion": update_diffusion,
        "refreshed_snapshot_sha256": sha256_file(snapshot_out),
    }
    _atomic_write_json(record_path, record)
    return record


def run_null_scan(
    experiment_config: MBCExperimentConfig,
    metric: MetricConfig,
    gate: Gate0Config,
    snapshot_path: Path,
    learning_rates: Sequence[float],
    output_dir: Path,
    *,
    workers: int = 1,
) -> dict:
    """Moment refresh followed by the certified local-stability scan."""
    output_dir = Path(output_dir)
    record = refresh_moments(experiment_config, metric, snapshot_path, output_dir)
    scan = measure_checkpoint_local_stability_scan(
        experiment_config,
        metric,
        gate,
        output_dir / "refreshed_snapshot.pt",
        learning_rates,
        output_dir / "local",
        workers=workers,
    )
    summary = {
        "schema_version": 1,
        "kind": "gate0e_null_scan",
        "refresh": record,
        "scan": scan,
    }
    _atomic_write_json(output_dir / "null_scan.json", summary)
    return summary


def radius_table(scan: Mapping[str, object]) -> List[dict]:
    measurements = scan["scan"]["measurements"] if "scan" in scan else scan["measurements"]
    table = [
        {
            "learning_rate": float(row["learning_rate"]),
            "radius": float(row["augmented_spectral_radius"]),
            "certified": bool(row["augmented_certified"]),
        }
        for row in measurements
    ]
    return sorted(table, key=lambda row: row["learning_rate"])


def radius_at(table: Sequence[Mapping[str, object]], eta: float) -> float:
    """Log-log interpolated certified radius at eta (eta inside the grid)."""
    certified = [row for row in table if row["certified"]]
    if len(certified) < 2:
        raise ValueError("radius interpolation needs at least two certified cells")
    for lower, upper in zip(certified, certified[1:]):
        if lower["learning_rate"] <= eta <= upper["learning_rate"]:
            x0, x1 = math.log(lower["learning_rate"]), math.log(upper["learning_rate"])
            y0, y1 = math.log(lower["radius"]), math.log(upper["radius"])
            weight = 0.0 if x1 == x0 else (math.log(eta) - x0) / (x1 - x0)
            return math.exp(y0 + weight * (y1 - y0))
    raise ValueError("eta lies outside the certified grid")


def crossing_rate(table: Sequence[Mapping[str, object]], c_star: float) -> dict:
    """First upward crossing of c_star on adjacent certified cells."""
    certified = [row for row in table if row["certified"]]
    if len(certified) < 2:
        return {"kind": "uncertified", "predicted_eta50": None}
    crossings = []
    for lower, upper in zip(certified, certified[1:]):
        below = lower["radius"] < c_star
        above = upper["radius"] >= c_star
        if below and above:
            x0, x1 = math.log(lower["learning_rate"]), math.log(upper["learning_rate"])
            y0, y1 = math.log(lower["radius"]), math.log(upper["radius"])
            weight = 0.0 if y1 == y0 else (math.log(c_star) - y0) / (y1 - y0)
            crossings.append(math.exp(x0 + weight * (x1 - x0)))
    if crossings:
        return {"kind": "crossing", "predicted_eta50": crossings[0], "all_crossings": crossings}
    if certified[0]["radius"] >= c_star:
        return {
            "kind": "left_censored",
            "predicted_eta50": None,
            "bound": certified[0]["learning_rate"],
        }
    return {
        "kind": "right_censored",
        "predicted_eta50": None,
        "bound": certified[-1]["learning_rate"],
    }


FLAT_SLOPE_THRESHOLD = 0.15
DISCRIMINATOR_SEEDS = (0, 1)
DISCRIMINATOR_BATCHES = (32, 128, 512)
BASE_BATCH = 128


def diffusion_slope(powers_by_batch: Mapping[int, float]) -> float:
    """OLS slope of ln(diffusion power) on ln(batch size)."""
    if len(powers_by_batch) < 2:
        raise ValueError("diffusion slope requires at least two batch sizes")
    x = [math.log(batch) for batch in sorted(powers_by_batch)]
    y = [math.log(powers_by_batch[batch]) for batch in sorted(powers_by_batch)]
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    sxx = sum((value - mean_x) ** 2 for value in x)
    sxy = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    return sxy / sxx


def batch_discriminator(
    predicted_eta50: Mapping[str, Optional[float]],
    diffusion_powers: Mapping[str, float],
    *,
    seeds: Sequence[int] = DISCRIMINATOR_SEEDS,
    batches: Sequence[int] = DISCRIMINATOR_BATCHES,
    flat_threshold: float = FLAT_SLOPE_THRESHOLD,
) -> dict:
    """Amendment v1.5.2 section 2: decisive only when the measured-diffusion
    escape direction exists, is not flat, and opposes the v-conditioned
    direction. Any other configuration is non-discriminating, which makes
    null_wins unsatisfiable at verdict time."""
    slopes = []
    for seed in seeds:
        powers = {}
        for batch in batches:
            value = diffusion_powers.get(f"{seed}:{batch}")
            if value is not None and value > 0:
                powers[batch] = value
        if len(powers) == len(batches):
            slopes.append(diffusion_slope(powers))
    pooled_slope = sum(slopes) / len(slopes) if slopes else None
    if pooled_slope is None:
        noise_direction = None
    elif abs(pooled_slope) < flat_threshold:
        noise_direction = "flat"
    else:
        noise_direction = "up" if pooled_slope < 0 else "down"

    def shift_sign(seed: int, batch: int) -> Optional[int]:
        base = predicted_eta50.get(f"{seed}:{BASE_BATCH}")
        contrast = predicted_eta50.get(f"{seed}:{batch}")
        if base is None or contrast is None or base <= 0 or contrast <= 0:
            return None
        difference = math.log(contrast / base)
        if difference == 0.0:
            return None
        return 1 if difference > 0 else -1

    v_direction = None
    low_signs = {shift_sign(seed, 32) for seed in seeds}
    high_signs = {shift_sign(seed, 512) for seed in seeds}
    if (
        len(low_signs) == 1
        and len(high_signs) == 1
        and None not in low_signs
        and None not in high_signs
    ):
        low = low_signs.pop()
        high = high_signs.pop()
        if low == -high:
            v_direction = "up" if high > 0 else "down"
    decisive = (
        v_direction is not None
        and noise_direction in ("up", "down")
        and noise_direction != v_direction
    )
    return {
        "per_seed_slopes": slopes,
        "pooled_dlnD_dlnB": pooled_slope,
        "flat_threshold": flat_threshold,
        "noise_predicted_direction": noise_direction,
        "v_conditioned_direction": v_direction,
        "status": "decisive" if decisive else "non_discriminating",
    }


def freeze_null_predictions(
    dev_curve_path: Path,
    dev_null_path: Path,
    gate_null_paths: Mapping[str, Path],
    output_path: Path,
) -> dict:
    """Compute c* from the dev pair and commit out-of-sample predictions.

    ``gate_null_paths`` is keyed by condition label ``"<seed>:<batch>"`` so the
    same artifact carries the primary-curve predictions (batch 128) and the
    batch-cell predictions used by the directional discriminator.
    """
    dev_curve = json.loads(Path(dev_curve_path).read_text())
    if dev_curve.get("kind") != "gate0e_escape_curve":
        raise ValueError("dev curve artifact has the wrong kind")
    dev_eta50 = dev_curve["statistics"]["eta50"]
    if dev_eta50 is None:
        raise ValueError(
            "dev curve has no eta50; the null constant cannot be calibrated"
        )
    dev_null = json.loads(Path(dev_null_path).read_text())
    dev_table = radius_table(dev_null)
    c_star = radius_at(dev_table, float(dev_eta50))
    predictions = {}
    predicted_eta50: Dict[str, Optional[float]] = {}
    diffusion_powers: Dict[str, float] = {}
    for label, path in sorted(gate_null_paths.items()):
        seed_text, _, batch_text = str(label).partition(":")
        int(seed_text), int(batch_text)  # labels must parse as seed:batch
        gate_null = json.loads(Path(path).read_text())
        table = radius_table(gate_null)
        prediction = crossing_rate(table, c_star)
        predictions[str(label)] = {
            "prediction": prediction,
            "radius_table": table,
            "input": bind_file(Path(path)),
        }
        predicted_eta50[str(label)] = prediction.get("predicted_eta50")
        refresh = gate_null.get("refresh", {})
        diffusion = refresh.get("update_diffusion")
        if isinstance(diffusion, dict):
            diffusion_powers[str(label)] = float(diffusion["diffusion_power"])
    payload = {
        "schema_version": 1,
        "kind": "gate0e_null_predictions",
        "dev_eta50": float(dev_eta50),
        "c_star": c_star,
        "dev_radius_table": dev_table,
        "predictions": predictions,
        "batch_discriminator": batch_discriminator(predicted_eta50, diffusion_powers),
        "inputs": {
            "dev_curve": bind_file(Path(dev_curve_path)),
            "dev_null": bind_file(Path(dev_null_path)),
        },
    }
    sealed = {**payload, "result_sha256": content_hash(payload)}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(output_path, sealed)
    return sealed
