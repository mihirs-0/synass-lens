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
    block_partial = {length: torch.zeros_like(previous) for length in BLOCK_LENGTHS}
    block_fill = {length: 0 for length in BLOCK_LENGTHS}
    block_powers = {length: [] for length in BLOCK_LENGTHS}
    block_sums = {length: torch.zeros_like(previous) for length in BLOCK_LENGTHS}
    decision_vectors: List[torch.Tensor] = []
    first = last = None
    for step_index in range(refresh_steps):
        last = experiment.advance(1)
        if step_index == 0:
            first = last
        current = _flatten_parameters(experiment.model)
        step = current - previous
        step_sum += step
        squared_power += float(step.pow(2).sum())
        for length in BLOCK_LENGTHS:
            block_partial[length] += step
            block_fill[length] += 1
            if block_fill[length] == length:
                block_powers[length].append(float(block_partial[length].pow(2).sum()))
                block_sums[length] += block_partial[length]
                if length == DECISION_BLOCK_LENGTH:
                    decision_vectors.append(block_partial[length].clone())
                block_partial[length].zero_()
                block_fill[length] = 0
        previous = current
    theta_after = previous
    drift = float(
        torch.linalg.vector_norm(theta_after - theta_before)
        / torch.linalg.vector_norm(theta_before)
    )
    mean_step_power = squared_power / refresh_steps
    drift_power = float((step_sum / refresh_steps).pow(2).sum())
    block_diffusion: Dict[str, Optional[float]] = {}
    for length in BLOCK_LENGTHS:
        count = len(block_powers[length])
        if count < 2:
            block_diffusion[str(length)] = None
            continue
        mean_power = sum(block_powers[length]) / count
        mean_vector = block_sums[length] / count
        block_diffusion[str(length)] = max(
            mean_power - float(mean_vector.pow(2).sum()), 0.0
        ) / length
    decision_block = None
    if len(decision_vectors) >= 2:
        count = len(decision_vectors)
        mean_vector = block_sums[DECISION_BLOCK_LENGTH] / count
        decision_block = {
            "length": DECISION_BLOCK_LENGTH,
            "count": count,
            "block_powers": block_powers[DECISION_BLOCK_LENGTH],
            "block_mean_dots": [
                float(torch.dot(mean_vector, vector)) for vector in decision_vectors
            ],
            "mean_power": float(mean_vector.pow(2).sum()),
        }
    update_diffusion = {
        "steps": refresh_steps,
        "refresh_rate": refresh_rate,
        "batch_size": experiment_config.batch_size,
        "mean_step_power": mean_step_power,
        "drift_power": drift_power,
        "update_variance_power": max(mean_step_power - drift_power, 0.0),
        "block_lengths": list(BLOCK_LENGTHS),
        "block_diffusion": block_diffusion,
        "decision_block": decision_block,
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
BLOCK_LENGTHS = (1, 4, 16, 64)
DECISION_BLOCK_LENGTH = 64
MINIMUM_DECISION_BLOCKS = 8
V_DIRECTION_MAGNITUDE_FLOOR = math.log(1.10)
DISCRIMINATOR_RESAMPLES = 2_000
DISCRIMINATOR_SEED = 20_260_715


def diffusion_slope(powers_by_batch: Mapping[int, float]) -> float:
    """OLS slope of ln(diffusion) on ln(batch size)."""
    if len(powers_by_batch) < 2:
        raise ValueError("diffusion slope requires at least two batch sizes")
    x = [math.log(batch) for batch in sorted(powers_by_batch)]
    y = [math.log(powers_by_batch[batch]) for batch in sorted(powers_by_batch)]
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    sxx = sum((value - mean_x) ** 2 for value in x)
    sxy = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    return sxy / sxx


def decision_block_diffusion(decision_block: Optional[Mapping[str, object]]) -> Optional[dict]:
    """Point estimate and leave-one-block-out jackknife SE of ln D_64.

    Returns None whenever the estimate cannot support the uncertainty gate:
    too few blocks, non-positive point estimate, or a non-positive
    leave-one-out estimate (v1.5.3: uncertainty destroys decisiveness)."""
    if not decision_block:
        return None
    count = int(decision_block["count"])
    length = int(decision_block["length"])
    if count < MINIMUM_DECISION_BLOCKS:
        return None
    powers = [float(value) for value in decision_block["block_powers"]]
    dots = [float(value) for value in decision_block["block_mean_dots"]]
    mean_power = float(decision_block["mean_power"])
    if len(powers) != count or len(dots) != count:
        raise ValueError("decision block scalars are inconsistent")
    total_power = sum(powers)
    point = (total_power / count - mean_power) / length
    if point <= 0:
        return None
    leave_one_out = []
    for index in range(count):
        mean_power_without = (total_power - powers[index]) / (count - 1)
        norm_without = (
            count * count * mean_power - 2 * count * dots[index] + powers[index]
        ) / ((count - 1) ** 2)
        estimate = (mean_power_without - norm_without) / length
        if estimate <= 0:
            return None
        leave_one_out.append(math.log(estimate))
    mean_loo = sum(leave_one_out) / count
    variance = sum((value - mean_loo) ** 2 for value in leave_one_out)
    return {
        "ln_d": math.log(point),
        "jackknife_se": math.sqrt((count - 1) / count * variance),
        "blocks": count,
    }


def batch_discriminator(
    predicted_eta50: Mapping[str, Optional[float]],
    diffusion_cells: Mapping[str, Optional[Mapping[str, float]]],
    *,
    seeds: Sequence[int] = DISCRIMINATOR_SEEDS,
    batches: Sequence[int] = DISCRIMINATOR_BATCHES,
    flat_threshold: float = FLAT_SLOPE_THRESHOLD,
    magnitude_floor: float = V_DIRECTION_MAGNITUDE_FLOOR,
    resamples: int = DISCRIMINATOR_RESAMPLES,
    seed: int = DISCRIMINATOR_SEED,
) -> dict:
    """v1.5.3 discriminator: decisive only when the resampled 95% interval of
    the pooled ln D_64 batch slope lies wholly beyond the flat threshold, the
    v-conditioned direction clears its magnitude floor, and the two
    directions oppose. Uncertainty can only destroy decisiveness."""
    import numpy as np

    cells: Dict[str, Mapping[str, float]] = {}
    complete = True
    for seed_value in seeds:
        for batch in batches:
            entry = diffusion_cells.get(f"{seed_value}:{batch}")
            if entry is None:
                complete = False
            else:
                cells[f"{seed_value}:{batch}"] = entry
    point_slopes = []
    pooled_point = None
    slope_interval = None
    noise_direction = None
    if complete:
        for seed_value in seeds:
            point_slopes.append(
                diffusion_slope(
                    {
                        batch: math.exp(cells[f"{seed_value}:{batch}"]["ln_d"])
                        for batch in batches
                    }
                )
            )
        pooled_point = sum(point_slopes) / len(point_slopes)
        rng = np.random.default_rng(seed)
        draws = []
        for _ in range(resamples):
            seed_slopes = []
            for seed_value in seeds:
                sampled = {}
                for batch in batches:
                    cell = cells[f"{seed_value}:{batch}"]
                    sampled[batch] = math.exp(
                        rng.normal(cell["ln_d"], cell["jackknife_se"])
                    )
                seed_slopes.append(diffusion_slope(sampled))
            draws.append(sum(seed_slopes) / len(seed_slopes))
        low, high = np.percentile(draws, [2.5, 97.5])
        slope_interval = [float(low), float(high)]
        if low > flat_threshold:
            noise_direction = "down"
        elif high < -flat_threshold:
            noise_direction = "up"
        else:
            noise_direction = "flat_or_uncertain"

    def shift_sign(seed_value: int, batch: int) -> Optional[int]:
        base = predicted_eta50.get(f"{seed_value}:{BASE_BATCH}")
        contrast = predicted_eta50.get(f"{seed_value}:{batch}")
        if base is None or contrast is None or base <= 0 or contrast <= 0:
            return None
        difference = math.log(contrast / base)
        if abs(difference) < magnitude_floor:
            return None
        return 1 if difference > 0 else -1

    v_direction = None
    low_signs = {shift_sign(seed_value, 32) for seed_value in seeds}
    high_signs = {shift_sign(seed_value, 512) for seed_value in seeds}
    if (
        len(low_signs) == 1
        and len(high_signs) == 1
        and None not in low_signs
        and None not in high_signs
    ):
        low_sign = low_signs.pop()
        high_sign = high_signs.pop()
        if low_sign == -high_sign:
            v_direction = "up" if high_sign > 0 else "down"
    decisive = (
        v_direction is not None
        and noise_direction in ("up", "down")
        and noise_direction != v_direction
    )
    return {
        "per_seed_point_slopes": point_slopes,
        "pooled_dlnD_dlnB": pooled_point,
        "slope_interval_95": slope_interval,
        "flat_threshold": flat_threshold,
        "v_magnitude_floor": magnitude_floor,
        "cells_complete": complete,
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
    diffusion_cells: Dict[str, Optional[dict]] = {}
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
            diffusion_cells[str(label)] = decision_block_diffusion(
                diffusion.get("decision_block")
            )
    payload = {
        "schema_version": 1,
        "kind": "gate0e_null_predictions",
        "dev_eta50": float(dev_eta50),
        "c_star": c_star,
        "dev_radius_table": dev_table,
        "predictions": predictions,
        "batch_discriminator": batch_discriminator(predicted_eta50, diffusion_cells),
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
