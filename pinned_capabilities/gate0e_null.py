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
    first = experiment.advance(1)
    remainder = experiment.advance(refresh_steps - 1) if refresh_steps > 1 else first
    theta_after = _flatten_parameters(experiment.model)
    drift = float(
        torch.linalg.vector_norm(theta_after - theta_before)
        / torch.linalg.vector_norm(theta_before)
    )
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
        "last_step_loss": float(remainder["train_loss"]),
        "theta_relative_drift": drift,
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


def freeze_null_predictions(
    dev_curve_path: Path,
    dev_null_path: Path,
    gate_null_paths: Mapping[int, Path],
    output_path: Path,
) -> dict:
    """Compute c* from the dev pair and commit out-of-sample predictions."""
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
    for seed, path in sorted(gate_null_paths.items()):
        gate_null = json.loads(Path(path).read_text())
        table = radius_table(gate_null)
        prediction = crossing_rate(table, c_star)
        predictions[str(seed)] = {
            "prediction": prediction,
            "radius_table": table,
            "input": bind_file(Path(path)),
        }
    payload = {
        "schema_version": 1,
        "kind": "gate0e_null_predictions",
        "dev_eta50": float(dev_eta50),
        "c_star": c_star,
        "dev_radius_table": dev_table,
        "predictions": predictions,
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
