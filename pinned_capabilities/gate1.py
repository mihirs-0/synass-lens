"""Registered hysteresis dwells, reversibility, and memory surgery."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Mapping, Optional, Sequence

import torch

from .config import Gate1Config
from .experiment import JSONLWriter, MBCExperiment
from .parameter_groups import set_learning_rates
from .state import ReferenceBands, StateThresholds, is_expressed, is_jointly_suppressed


@dataclass(frozen=True)
class DwellResult:
    learning_rate: float
    start_step: int
    end_step: int
    equilibrated: bool
    right_censored: bool
    state: str
    probe_0_slope_per_1000: Optional[float]
    probe_1_slope_per_1000: Optional[float]


@dataclass(frozen=True)
class SweepResult:
    direction: str
    eta_on: Optional[float]
    eta_off: Optional[float]
    dwells: tuple[DwellResult, ...]


def geometric_levels(high: float, low: float, levels_per_decade: int) -> tuple[float, ...]:
    if not 0 < low <= high or levels_per_decade <= 0:
        raise ValueError("require 0 < low <= high and positive levels_per_decade")
    n_intervals = max(1, math.ceil(math.log10(high / low) * levels_per_decade))
    ratio = (low / high) ** (1.0 / n_intervals)
    levels = [high * ratio**index for index in range(n_intervals + 1)]
    levels[-1] = low
    return tuple(levels)


def _slope_per_1000(rows: Sequence[Mapping[str, float]], key: str, window: int = 2_000) -> Optional[float]:
    if len(rows) < 2:
        return None
    end = float(rows[-1]["step"])
    selected = [row for row in rows if float(row["step"]) >= end - window]
    if len(selected) < 2 or float(selected[0]["step"]) > end - window:
        return None
    x = torch.tensor([float(row["step"]) for row in selected], dtype=torch.float64)
    y = torch.tensor([float(row[key]) for row in selected], dtype=torch.float64)
    centered = x - x.mean()
    denominator = centered.square().sum()
    if denominator == 0:
        return None
    return float(((centered * (y - y.mean())).sum() / denominator * 1_000).item())


def equilibration_slopes(
    rows: Sequence[Mapping[str, float]], window: int = 2_000
) -> tuple[Optional[float], Optional[float]]:
    return (
        _slope_per_1000(rows, "probe_0_c_int", window),
        _slope_per_1000(rows, "probe_1_c_int", window),
    )


def is_equilibrated(rows: Sequence[Mapping[str, float]], plateau_sd: float) -> bool:
    slopes = equilibration_slopes(rows)
    return all(slope is not None and abs(slope) < plateau_sd for slope in slopes)


def classify_latest_state(
    rows: Sequence[Mapping[str, float]],
    reference: ReferenceBands,
    thresholds: StateThresholds,
) -> str:
    if not rows:
        return "unresolved"
    latest = rows[-1]
    if is_expressed(float(latest["c_int"]), float(latest["exact_match"]), reference, thresholds):
        return "expressed"
    if is_jointly_suppressed(rows, reference, thresholds):
        return "suppressed"
    return "intermediate"


def run_dwell(
    experiment: MBCExperiment,
    learning_rate: float,
    reference: ReferenceBands,
    thresholds: StateThresholds,
    config: Gate1Config,
    *,
    writer: Optional[JSONLWriter] = None,
    dwell_multiplier: int = 1,
) -> DwellResult:
    set_learning_rates(experiment.optimizer, learning_rate)
    start = experiment.step
    minimum = config.base_dwell_steps * dwell_multiplier
    maximum = config.maximum_dwell_steps * dwell_multiplier
    rows: List[Dict[str, float]] = []
    while experiment.step - start < maximum:
        chunk = min(experiment.metric.eval_every, maximum - (experiment.step - start))
        training = experiment.advance(chunk)
        metrics = experiment.evaluate()
        row = {**training, **metrics, "learning_rate": learning_rate}
        rows.append(row)
        if writer is not None:
            writer.write({"kind": "dwell", **row})
        if experiment.step - start >= minimum and is_equilibrated(rows, reference.plateau_sd):
            break
    equilibrated = is_equilibrated(rows, reference.plateau_sd)
    slopes = equilibration_slopes(rows)
    return DwellResult(
        learning_rate=learning_rate,
        start_step=start,
        end_step=experiment.step,
        equilibrated=equilibrated,
        right_censored=not equilibrated,
        state=classify_latest_state(rows, reference, thresholds),
        probe_0_slope_per_1000=slopes[0],
        probe_1_slope_per_1000=slopes[1],
    )


def run_sweep(
    experiment: MBCExperiment,
    levels: Sequence[float],
    direction: str,
    reference: ReferenceBands,
    thresholds: StateThresholds,
    config: Gate1Config,
    *,
    writer: Optional[JSONLWriter] = None,
    dwell_multiplier: int = 1,
) -> SweepResult:
    if direction not in {"down", "up"}:
        raise ValueError("direction must be down or up")
    ordered = tuple(levels if direction == "down" else reversed(levels))
    dwells = []
    boundary = None
    saw_expressed = direction == "down"
    for learning_rate in ordered:
        result = run_dwell(
            experiment,
            learning_rate,
            reference,
            thresholds,
            config,
            writer=writer,
            dwell_multiplier=dwell_multiplier,
        )
        dwells.append(result)
        if not result.equilibrated:
            continue
        if direction == "down" and boundary is None and result.state == "expressed":
            boundary = learning_rate
        if direction == "up" and result.state == "expressed":
            saw_expressed = True
        if (
            direction == "up"
            and saw_expressed
            and boundary is None
            and result.state == "suppressed"
        ):
            boundary = learning_rate
    return SweepResult(
        direction=direction,
        eta_on=boundary if direction == "down" else None,
        eta_off=boundary if direction == "up" else None,
        dwells=tuple(dwells),
    )


@torch.no_grad()
def add_norm_matched_gaussian_noise(
    model: torch.nn.Module, target_norm: float, seed: int
) -> None:
    if target_norm < 0:
        raise ValueError("target norm cannot be negative")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = [
        torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype).to(parameter.device)
        for parameter in parameters
    ]
    norm = sum(value.square().sum() for value in noise).sqrt()
    if norm == 0 and target_norm > 0:
        raise RuntimeError("sampled zero Gaussian perturbation")
    scale = target_norm / float(norm.item()) if target_norm else 0.0
    for parameter, value in zip(parameters, noise):
        parameter.add_(value, alpha=scale)


def reset_optimizer_state(optimizer: torch.optim.Optimizer) -> None:
    optimizer.state.clear()


def sweep_to_dict(result: SweepResult) -> dict:
    return asdict(result)
