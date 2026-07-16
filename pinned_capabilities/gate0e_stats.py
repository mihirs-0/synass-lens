"""Survival and dose-response statistics for Gate 0-E escape curves.

Frozen by amendment v1.5 section 2.6: eta50 comes from a binomial-likelihood
logistic fit in log learning rate, confidence intervals from stream-level
bootstrap, monotonicity is checked but never enforced, and divergence-marked
rates never enter fits. The positive control routes its outcomes through these
same functions so the statistics path is validated before any neural verdict.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 20_260_715
SLOPE_CAP = 60.0


@dataclass(frozen=True)
class RateOutcomes:
    """Per-rate stream outcomes after divergence exclusion."""

    learning_rate: float
    erased: int
    retained: int
    unresolved: int
    diverged: int

    @property
    def classified(self) -> int:
        return self.erased + self.retained + self.unresolved

    @property
    def unstable(self) -> bool:
        total = self.classified + self.diverged
        return total > 0 and self.diverged * 2 >= total

    def primary_fraction(self) -> Optional[float]:
        if self.classified == 0:
            return None
        return self.erased / self.classified

    def sensitivity_fraction(self) -> Optional[float]:
        """Opposite convention: unresolved counts as erased."""
        if self.classified == 0:
            return None
        return (self.erased + self.unresolved) / self.classified


def fit_cells(cells: Sequence[RateOutcomes]) -> List[RateOutcomes]:
    """Cells eligible for curve fitting: non-unstable with classified streams."""
    return [cell for cell in cells if not cell.unstable and cell.classified > 0]


def curve_valid(
    cells: Sequence[RateOutcomes],
    *,
    minimum_rates: int = 4,
    minimum_classified: int = 6,
) -> bool:
    eligible = [
        cell
        for cell in cells
        if not cell.unstable and cell.classified >= minimum_classified
    ]
    return len(eligible) >= minimum_rates


@dataclass(frozen=True)
class LogisticFit:
    intercept: float
    slope: float
    eta50: Optional[float]
    converged: bool
    separation: bool
    log_likelihood: float


def _binomial_arrays(
    cells: Sequence[RateOutcomes],
    counts: Optional[Mapping[float, Tuple[int, int]]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, k, n = [], [], []
    for cell in cells:
        if counts is not None:
            erased, classified = counts[cell.learning_rate]
        else:
            erased, classified = cell.erased, cell.classified
        if classified == 0:
            continue
        x.append(math.log(cell.learning_rate))
        k.append(float(erased))
        n.append(float(classified))
    return np.asarray(x), np.asarray(k), np.asarray(n)


def logistic_fit(
    cells: Sequence[RateOutcomes],
    counts: Optional[Mapping[float, Tuple[int, int]]] = None,
) -> LogisticFit:
    """Binomial MLE for P(erase) = sigmoid(a + b * ln eta)."""
    x, k, n = _binomial_arrays(cells, counts)
    if x.size < 2:
        raise ValueError("logistic fit requires at least two rates with streams")
    fractions = np.clip(k / n, 1.0 / (2.0 * n.max() + 2.0), 1.0 - 1.0 / (2.0 * n.max() + 2.0))
    logits = np.log(fractions / (1.0 - fractions))
    slope, intercept = np.polyfit(x, logits, 1)
    theta = np.array([intercept, slope], dtype=float)

    def log_likelihood(params: np.ndarray) -> float:
        eta = params[0] + params[1] * x
        return float(np.sum(k * eta - n * np.logaddexp(0.0, eta)))

    current = log_likelihood(theta)
    converged = False
    for _ in range(200):
        eta = theta[0] + theta[1] * x
        p = 1.0 / (1.0 + np.exp(-eta))
        gradient = np.array([np.sum(k - n * p), np.sum((k - n * p) * x)])
        w = n * p * (1.0 - p)
        hessian = -np.array([[np.sum(w), np.sum(w * x)], [np.sum(w * x), np.sum(w * x * x)]])
        if abs(np.linalg.det(hessian)) < 1e-12:
            break
        step = np.linalg.solve(hessian, gradient)
        candidate = theta - step
        candidate_ll = log_likelihood(candidate)
        halvings = 0
        while candidate_ll < current - 1e-12 and halvings < 30:
            step *= 0.5
            candidate = theta - step
            candidate_ll = log_likelihood(candidate)
            halvings += 1
        theta, previous, current = candidate, current, candidate_ll
        if abs(current - previous) < 1e-10 and float(np.max(np.abs(step))) < 1e-8:
            converged = True
            break
        if abs(theta[1]) > SLOPE_CAP:
            theta[1] = math.copysign(SLOPE_CAP, theta[1])
            theta[0] = _profile_intercept(x, k, n, theta[1])
            current = log_likelihood(theta)
            break
    separation = bool(abs(theta[1]) >= SLOPE_CAP)
    eta50 = None
    if theta[1] > 0:
        eta50 = float(math.exp(-theta[0] / theta[1]))
    return LogisticFit(
        intercept=float(theta[0]),
        slope=float(theta[1]),
        eta50=eta50,
        converged=converged,
        separation=separation,
        log_likelihood=current,
    )


def _profile_intercept(x: np.ndarray, k: np.ndarray, n: np.ndarray, slope: float) -> float:
    intercept = 0.0
    for _ in range(100):
        eta = intercept + slope * x
        p = 1.0 / (1.0 + np.exp(-eta))
        gradient = float(np.sum(k - n * p))
        curvature = float(-np.sum(n * p * (1.0 - p)))
        if abs(curvature) < 1e-12:
            break
        step = gradient / curvature
        intercept -= step
        if abs(step) < 1e-10:
            break
    return intercept


@dataclass(frozen=True)
class BootstrapCI:
    lower: Optional[float]
    upper: Optional[float]
    replicates: int
    valid_replicates: int


def bootstrap_eta50_ci(
    cells: Sequence[RateOutcomes],
    stream_labels: Mapping[float, Sequence[int]],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> BootstrapCI:
    """Percentile CI for eta50 by resampling stream labels within each rate.

    ``stream_labels[rate]`` holds one 0/1 entry per classified (non-diverged)
    stream: 1 for erased, 0 for not-erased under the primary convention.
    """
    rng = np.random.default_rng(seed)
    estimates: List[float] = []
    for _ in range(replicates):
        counts: Dict[float, Tuple[int, int]] = {}
        for cell in cells:
            labels = np.asarray(stream_labels[cell.learning_rate])
            if labels.size == 0:
                counts[cell.learning_rate] = (0, 0)
                continue
            resample = rng.choice(labels, size=labels.size, replace=True)
            counts[cell.learning_rate] = (int(resample.sum()), int(resample.size))
        try:
            fit = logistic_fit(cells, counts)
        except (ValueError, FloatingPointError):
            continue
        if fit.eta50 is not None and 1e-6 < fit.eta50 < 1.0:
            estimates.append(fit.eta50)
    if len(estimates) < replicates // 2:
        return BootstrapCI(None, None, replicates, len(estimates))
    lower, upper = np.percentile(estimates, [2.5, 97.5])
    return BootstrapCI(float(lower), float(upper), replicates, len(estimates))


@dataclass(frozen=True)
class ArrheniusFit:
    slope: float
    intercept: float
    r_squared: float
    points: int


def ols_r2(x: Sequence[float], y: Sequence[float]) -> ArrheniusFit:
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("OLS requires at least two matched points")
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    slope, intercept = np.polyfit(x_array, y_array, 1)
    predicted = intercept + slope * x_array
    residual = float(np.sum((y_array - predicted) ** 2))
    total = float(np.sum((y_array - np.mean(y_array)) ** 2))
    r_squared = 1.0 if total == 0.0 else 1.0 - residual / total
    return ArrheniusFit(float(slope), float(intercept), r_squared, len(x))


def arrhenius_fit(
    cells: Sequence[RateOutcomes],
    median_taus: Mapping[float, float],
    *,
    noise_scale_numerator: float = 1.0,
) -> Optional[ArrheniusFit]:
    """ln(median tau) against numerator/eta over strictly partial cells."""
    points = [
        (noise_scale_numerator / cell.learning_rate, math.log(median_taus[cell.learning_rate]))
        for cell in cells
        if cell.learning_rate in median_taus
        and cell.primary_fraction() is not None
        and 0.0 < cell.primary_fraction() < 1.0
    ]
    if len(points) < 2:
        return None
    return ols_r2([p[0] for p in points], [p[1] for p in points])


def monotonicity_flags(cells: Sequence[RateOutcomes]) -> List[dict]:
    """Adjacent decreases beyond twice the pooled binomial SE."""
    ordered = sorted(
        (cell for cell in cells if cell.primary_fraction() is not None),
        key=lambda cell: cell.learning_rate,
    )
    flags = []
    for lower, upper in zip(ordered, ordered[1:]):
        p_low = lower.primary_fraction()
        p_high = upper.primary_fraction()
        pooled = (lower.erased + upper.erased) / (lower.classified + upper.classified)
        se = math.sqrt(
            max(pooled * (1.0 - pooled), 1e-12)
            * (1.0 / lower.classified + 1.0 / upper.classified)
        )
        if p_high < p_low - 2.0 * se:
            flags.append(
                {
                    "lower_rate": lower.learning_rate,
                    "upper_rate": upper.learning_rate,
                    "lower_fraction": p_low,
                    "upper_fraction": p_high,
                    "pooled_se": se,
                }
            )
    return flags


def shifted_grid(grid: Sequence[float]) -> List[float]:
    """The single permitted down-shift: every rate halved exactly once."""
    return [rate * 0.5 for rate in grid]
