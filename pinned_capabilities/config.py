"""Frozen configuration objects shared by all pinned-capability gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class MetricConfig:
    eval_every: int = 50
    quartets_per_probe: int = 1024
    probe_seeds: Tuple[int, int] = (17_071, 91_237)

    def validate(self) -> None:
        if self.eval_every <= 0:
            raise ValueError("eval_every must be positive")
        if self.quartets_per_probe <= 0:
            raise ValueError("quartets_per_probe must be positive")
        if len(set(self.probe_seeds)) != 2:
            raise ValueError("the two fixed probe batches need distinct seeds")


@dataclass(frozen=True)
class StateConfig:
    plateau_sd_multiplier: float = 3.0
    suppressed_duration: int = 2_000
    expressed_fraction: float = 0.5
    expressed_em_sd: float = 5.0
    transition_fraction: float = 0.2
    no_return_duration: int = 1_000

    def validate(self) -> None:
        if self.plateau_sd_multiplier <= 0:
            raise ValueError("plateau_sd_multiplier must be positive")
        if self.suppressed_duration <= 0 or self.no_return_duration <= 0:
            raise ValueError("state durations must be positive")
        if not 0 < self.transition_fraction < self.expressed_fraction <= 1:
            raise ValueError("require 0 < transition_fraction < expressed_fraction <= 1")


@dataclass(frozen=True)
class Gate0Config:
    erase_horizon: int = 2_000
    acquire_horizon: int = 40_000
    boundary_bisection_steps: int = 12
    batch_sizes: Tuple[int, ...] = (32, 128, 512, 2_048)
    residual_batch_shift_fraction: float = 0.25
    reduction_match_factor: float = 1.5
    reduction_miss_factor: float = 2.0


@dataclass(frozen=True)
class Gate1Config:
    levels_per_decade: int = 12
    base_dwell_steps: int = 5_000
    extended_dwell_multiplier: int = 3
    hold_steps: int = 10_000
    cycles: int = 2
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    minimum_band_ratio: float = 1.25
    maximum_dwell_shrinkage: float = 0.25
    maximum_cycle_shrinkage: float = 0.30


@dataclass(frozen=True)
class ProtocolConfig:
    protocol_version: str = "1.1.0"
    output_root: str = "pinned_capabilities/results"
    metric: MetricConfig = field(default_factory=MetricConfig)
    state: StateConfig = field(default_factory=StateConfig)
    gate0: Gate0Config = field(default_factory=Gate0Config)
    gate1: Gate1Config = field(default_factory=Gate1Config)

    def validate(self) -> None:
        self.metric.validate()
        self.state.validate()
        if self.gate0.erase_horizon >= self.gate0.acquire_horizon:
            raise ValueError("erase_horizon must be shorter than acquire_horizon")
        if self.gate1.extended_dwell_multiplier <= 1:
            raise ValueError("extended dwell must exceed base dwell")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)

    @property
    def output_path(self) -> Path:
        return Path(self.output_root)
