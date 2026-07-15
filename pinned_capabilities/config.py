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
    solved_exact_match: float = 0.99
    solved_full_vocab_ce: float = 0.05
    solved_hold_steps: int = 2_000
    solved_summary_window: int = 1_000

    def validate(self) -> None:
        if self.eval_every <= 0:
            raise ValueError("eval_every must be positive")
        if self.quartets_per_probe <= 0:
            raise ValueError("quartets_per_probe must be positive")
        if len(set(self.probe_seeds)) != 2:
            raise ValueError("the two fixed probe batches need distinct seeds")
        if not 0 < self.solved_exact_match <= 1 or self.solved_full_vocab_ce <= 0:
            raise ValueError("invalid solved behavioral endpoint")
        if self.solved_summary_window <= 0 or self.solved_hold_steps < self.solved_summary_window:
            raise ValueError("solved summary window must fit inside the solved hold")


@dataclass(frozen=True)
class StateConfig:
    plateau_sd_multiplier: float = 3.0
    suppressed_duration: int = 2_000
    expressed_fraction: float = 0.5
    expressed_em_sd: float = 5.0
    expressed_exact_match_floor: float = 0.90
    transition_fraction: float = 0.2
    no_return_duration: int = 1_000

    def validate(self) -> None:
        if self.plateau_sd_multiplier <= 0:
            raise ValueError("plateau_sd_multiplier must be positive")
        if self.suppressed_duration <= 0 or self.no_return_duration <= 0:
            raise ValueError("state durations must be positive")
        if not 0 < self.transition_fraction < self.expressed_fraction <= 1:
            raise ValueError("require 0 < transition_fraction < expressed_fraction <= 1")
        if not 0 < self.expressed_exact_match_floor <= 1:
            raise ValueError("expressed exact-match floor must be in (0,1]")


@dataclass(frozen=True)
class Gate0Config:
    erase_horizon: int = 2_000
    erase_hold_steps: int = 8_000
    acquire_horizon: int = 40_000
    boundary_bisection_steps: int = 12
    batch_sizes: Tuple[int, ...] = (32, 128, 512, 2_048)
    residual_batch_shift_fraction: float = 0.25
    reduction_match_factor: float = 1.5
    reduction_miss_factor: float = 2.0
    local_probe_b_count: int = 32
    local_quartet_count: int = 256
    curvature_power_iterations: int = 30
    augmented_eigenvalue_count: int = 3


@dataclass(frozen=True)
class Gate1Config:
    levels_per_decade: int = 12
    base_dwell_steps: int = 5_000
    maximum_dwell_steps: int = 20_000
    extended_dwell_multiplier: int = 3
    hold_steps: int = 10_000
    cycles: int = 2
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    minimum_band_ratio: float = 1.25
    maximum_dwell_shrinkage: float = 0.25
    maximum_cycle_shrinkage: float = 0.30


@dataclass(frozen=True)
class MBCExperimentConfig:
    seed: int = 0
    n_unique_b: int = 1_000
    k: int = 10
    b_length: int = 6
    a_length: int = 4
    z_length: int = 2
    vocab_chars: str = "abcdefghijklmnopqrstuvwxyz0123456789"
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 128
    d_head: int = 32
    d_mlp: int = 512
    act_fn: str = "gelu"
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    probe_b_count: int = 128
    device: str = "auto"

    def validate(self) -> None:
        if self.n_unique_b < 2 * self.probe_b_count:
            raise ValueError("two disjoint probes require n_unique_b >= 2 * probe_b_count")
        if self.k < 2 or self.batch_size <= 0:
            raise ValueError("k must be at least two and batch_size must be positive")
        if self.d_model != self.n_heads * self.d_head:
            raise ValueError("d_model must equal n_heads * d_head")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer hyperparameters")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError(f"unsupported device: {self.device}")


@dataclass(frozen=True)
class ProtocolConfig:
    protocol_version: str = "1.2.4"
    output_root: str = "pinned_capabilities/results"
    metric: MetricConfig = field(default_factory=MetricConfig)
    state: StateConfig = field(default_factory=StateConfig)
    gate0: Gate0Config = field(default_factory=Gate0Config)
    gate1: Gate1Config = field(default_factory=Gate1Config)
    experiment: MBCExperimentConfig = field(default_factory=MBCExperimentConfig)

    def validate(self) -> None:
        self.metric.validate()
        self.state.validate()
        self.experiment.validate()
        if self.gate0.erase_horizon >= self.gate0.acquire_horizon:
            raise ValueError("erase_horizon must be shorter than acquire_horizon")
        if self.gate0.erase_horizon >= self.gate0.erase_hold_steps:
            raise ValueError("erasure entry horizon must be shorter than the complete hold")
        if min(
            self.gate0.local_probe_b_count,
            self.gate0.local_quartet_count,
            self.gate0.curvature_power_iterations,
            self.gate0.augmented_eigenvalue_count,
        ) <= 0:
            raise ValueError("Gate 0 local-measurement sizes must be positive")
        if self.gate1.extended_dwell_multiplier <= 1:
            raise ValueError("extended dwell must exceed base dwell")
        if self.gate1.maximum_dwell_steps < self.gate1.base_dwell_steps:
            raise ValueError("maximum dwell must not be shorter than base dwell")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)

    @property
    def output_path(self) -> Path:
        return Path(self.output_root)
