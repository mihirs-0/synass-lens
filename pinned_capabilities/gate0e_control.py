"""Gate 0-E positive control: noise-driven escape in a deep-linear system.

Amendment v1.5 section 2.4 requires the existing deep-linear pipeline
"extended with SGD noise in a regime where escape theory is semi-analytic".
The system is the weight-tied depth-3 scalar deep-linear network with L2
regularization — the Ersoy-Wiesner mechanism reduced to one dimension:

    L(x) = 0.5 * (x^3 - t)^2 + (3 * lam / 2) * x^2

The origin is always a local minimum (curvature 3 * lam), and for small
enough lam a distant solution basin exists past a finite barrier. SGD with
Gaussian gradient noise escapes the origin at Kramers rate with effective
temperature eta * sigma^2 / 2, so ln(median tau) is linear in 1/eta with
semi-analytic slope 2 * DeltaE / sigma^2.

Frozen pass rule (section 2.4): monotone escape-probability curve and an
Arrhenius fit with R^2 >= 0.9. The control's stream outcomes are aggregated,
fitted, and flagged by the same ``gate0e_stats`` functions used for the
neural escape curves, which is the point: the survival-statistics path is
validated before any neural verdict is trusted. The registered median
estimator, here and for the neural curves, is the median of observed escape
times within a cell (conditional median).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .gate0e_stats import (
    ArrheniusFit,
    RateOutcomes,
    arrhenius_fit,
    curve_valid,
    logistic_fit,
    monotonicity_flags,
)


@dataclass(frozen=True)
class ControlConfig:
    target: float = 1.0
    weight_decay: float = 0.3
    noise_sd: float = 1.0
    horizon_steps: int = 50_000
    streams: int = 32
    learning_rates: Tuple[float, ...] = (0.006, 0.007, 0.008, 0.009, 0.0105, 0.012)
    divergence_bound: float = 100.0
    seed: int = 20_260_715

    def validate(self) -> None:
        if self.target <= 0 or self.weight_decay <= 0 or self.noise_sd <= 0:
            raise ValueError("control system parameters must be positive")
        if self.horizon_steps <= 0 or self.streams < 2:
            raise ValueError("control needs a positive horizon and >= 2 streams")
        if len(self.learning_rates) < 3:
            raise ValueError("control needs at least three rates")


def _potential(x: np.ndarray, target: float, lam: float) -> np.ndarray:
    return 0.5 * (x**3 - target) ** 2 + 1.5 * lam * x**2


def _gradient(x: np.ndarray, target: float, lam: float) -> np.ndarray:
    return 3.0 * x**2 * (x**3 - target) + 3.0 * lam * x


def _bisect(f, low: float, high: float, iterations: int = 200) -> float:
    f_low = f(low)
    if f_low == 0.0:
        return low
    if f_low * f(high) > 0:
        raise ValueError("bisection bracket does not straddle a root")
    for _ in range(iterations):
        mid = 0.5 * (low + high)
        value = f(mid)
        if value == 0.0:
            return mid
        if f_low * value < 0:
            high = mid
        else:
            low = mid
            f_low = value
    return 0.5 * (low + high)


def critical_points(target: float, lam: float) -> Dict[str, float]:
    """Barrier and distant-well roots of x^4 - target * x + lam."""

    def poly(x: float) -> float:
        return x**4 - target * x + lam

    stationary = (target / 4.0) ** (1.0 / 3.0)
    if poly(stationary) >= 0:
        raise ValueError("no barrier: weight decay too large for this target")
    barrier = _bisect(poly, 0.0, stationary)
    well = _bisect(poly, stationary, max(2.0, 2.0 * target ** (1.0 / 3.0)))
    barrier_height = float(
        _potential(np.asarray(barrier), target, lam)
        - _potential(np.asarray(0.0), target, lam)
    )
    return {
        "barrier_x": float(barrier),
        "well_x": float(well),
        "barrier_height": barrier_height,
        "escape_threshold": float(0.5 * (barrier + well)),
    }


def _rate_seed(base_seed: int, learning_rate: float) -> int:
    key = f"gate0e_control|seed={base_seed}|rate={learning_rate:.10g}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")


def simulate_rate(
    config: ControlConfig, learning_rate: float
) -> Tuple[RateOutcomes, List[int], List[int]]:
    """Vectorized SGD streams; returns outcomes, escape taus, stream labels."""
    geometry = critical_points(config.target, config.weight_decay)
    threshold = geometry["escape_threshold"]
    rng = np.random.default_rng(_rate_seed(config.seed, learning_rate))
    x = np.zeros(config.streams)
    tau = np.full(config.streams, -1, dtype=np.int64)
    diverged = np.zeros(config.streams, dtype=bool)
    active = np.ones(config.streams, dtype=bool)
    for step in range(1, config.horizon_steps + 1):
        if not active.any():
            break
        noise = rng.normal(0.0, config.noise_sd, size=config.streams)
        gradient = _gradient(x, config.target, config.weight_decay)
        x = np.where(active, x - learning_rate * (gradient + noise), x)
        newly_diverged = active & (np.abs(x) > config.divergence_bound)
        diverged |= newly_diverged
        escaped = active & ~newly_diverged & (x >= threshold)
        tau[escaped] = step
        active &= ~(escaped | newly_diverged)
    escaped_mask = tau > 0
    outcomes = RateOutcomes(
        learning_rate=learning_rate,
        erased=int(escaped_mask.sum()),
        retained=int((~escaped_mask & ~diverged).sum()),
        unresolved=0,
        diverged=int(diverged.sum()),
    )
    labels = [int(value) for value in escaped_mask[~diverged]]
    taus = [int(value) for value in tau[escaped_mask]]
    return outcomes, taus, labels


def run_positive_control(config: Optional[ControlConfig] = None) -> dict:
    config = config or ControlConfig()
    config.validate()
    geometry = critical_points(config.target, config.weight_decay)
    cells: List[RateOutcomes] = []
    median_taus: Dict[float, float] = {}
    per_rate: Dict[str, dict] = {}
    for rate in config.learning_rates:
        outcomes, taus, labels = simulate_rate(config, rate)
        cells.append(outcomes)
        if taus and 0.0 < (outcomes.primary_fraction() or 0.0) < 1.0:
            median_taus[rate] = float(np.median(taus))
        per_rate[f"{rate:g}"] = {
            "outcomes": asdict(outcomes),
            "primary_fraction": outcomes.primary_fraction(),
            "observed_taus": taus,
            "labels": labels,
        }
    flags = monotonicity_flags(cells)
    fit = arrhenius_fit(cells, median_taus)
    predicted_slope = 2.0 * geometry["barrier_height"] / config.noise_sd**2
    logistic = None
    partial = [
        cell
        for cell in cells
        if cell.primary_fraction() is not None and cell.classified > 0
    ]
    if len(partial) >= 2:
        result = logistic_fit(partial)
        logistic = {
            "intercept": result.intercept,
            "slope": result.slope,
            "eta50": result.eta50,
            "converged": result.converged,
            "separation": result.separation,
        }
    passed = (
        not flags
        and fit is not None
        and fit.points >= 3
        and fit.r_squared >= 0.9
        and curve_valid(cells, minimum_rates=3, minimum_classified=config.streams // 2)
    )
    return {
        "schema_version": 1,
        "kind": "gate0e_positive_control",
        "config": asdict(config),
        "geometry": geometry,
        "predicted_arrhenius_slope": predicted_slope,
        "cells": per_rate,
        "monotonicity_flags": flags,
        "arrhenius": None if fit is None else asdict(fit),
        "arrhenius_slope_ratio": None
        if fit is None or predicted_slope == 0
        else fit.slope / predicted_slope,
        "logistic": logistic,
        "passed": passed,
    }


def write_positive_control(output_dir: Path, config: Optional[ControlConfig] = None) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = run_positive_control(config)
    path = output_dir / "control.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=1, sort_keys=True, allow_nan=False))
    temporary.replace(path)
    return report
