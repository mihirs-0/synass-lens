"""Frozen state labels derived from reference distributions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence


@dataclass(frozen=True)
class ReferenceBands:
    plateau_mean: float
    plateau_sd: float
    solved_c_int: float
    chance_em_mean: float
    chance_em_sd: float


@dataclass(frozen=True)
class StateThresholds:
    plateau_sd_multiplier: float = 3.0
    suppressed_duration: int = 2_000
    expressed_fraction: float = 0.5
    expressed_em_sd: float = 5.0
    transition_fraction: float = 0.2
    no_return_duration: int = 1_000


def plateau_bounds(reference: ReferenceBands, thresholds: StateThresholds) -> tuple[float, float]:
    radius = thresholds.plateau_sd_multiplier * reference.plateau_sd
    return reference.plateau_mean - radius, reference.plateau_mean + radius


def is_expressed(
    c_int: float,
    exact_match: float,
    reference: ReferenceBands,
    thresholds: StateThresholds,
) -> bool:
    return (
        c_int >= thresholds.expressed_fraction * reference.solved_c_int
        and exact_match >= reference.chance_em_mean + thresholds.expressed_em_sd * reference.chance_em_sd
    )


def is_suppressed(
    rows: Sequence[Mapping[str, float]],
    reference: ReferenceBands,
    thresholds: StateThresholds,
) -> bool:
    if not rows:
        return False
    end_step = int(rows[-1]["step"])
    start_step = end_step - thresholds.suppressed_duration
    window = [row for row in rows if int(row["step"]) >= start_step]
    if not window or int(window[0]["step"]) > start_step:
        return False
    low, high = plateau_bounds(reference, thresholds)
    return all(low <= float(row["c_int"]) <= high for row in window)


def first_transition_step(
    rows: Sequence[Mapping[str, float]],
    reference: ReferenceBands,
    thresholds: StateThresholds,
) -> Optional[int]:
    low, high = plateau_bounds(reference, thresholds)
    crossing = thresholds.transition_fraction * reference.solved_c_int
    for index, row in enumerate(rows):
        step = int(row["step"])
        if float(row["c_int"]) < crossing:
            continue
        end = step + thresholds.no_return_duration
        future = [candidate for candidate in rows[index:] if int(candidate["step"]) <= end]
        if not future or int(future[-1]["step"]) < end:
            continue
        if all(not (low <= float(candidate["c_int"]) <= high) for candidate in future):
            return step
    return None
