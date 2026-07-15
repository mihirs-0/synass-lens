"""Finite, interval-aware decision analysis for the registered Gate 0.

The empirical and predicted critical learning rates are intervals, not points.
This module therefore distinguishes evidence *for* reduction, evidence *against*
reduction, and an inconclusive region.  Missing, censored, and uncertified data
are represented explicitly and can never silently contribute to a kill.

Residual batch effects are expressed on the log-critical-rate scale after
conditioning on the registered local predictor.  For each exact 16x contrast,
the high-minus-low residual is paired within seed and its percentile interval
is obtained by resampling whole seed clusters.  With the five registered seeds
the bootstrap is enumerated exactly (5**5 resamples), so the result is
deterministic.
"""

from __future__ import annotations

import itertools
import json
import math
import numbers
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .manifest import content_hash
from .provenance import bind_file, bind_manifest


_FROZEN_GATE0_LEARNING_RATES = (0.003, 0.006, 0.012, 0.025, 0.05)


class Direction(str, Enum):
    ERASURE = "erasure"
    ACQUISITION = "acquisition"


class IntervalKind(str, Enum):
    OBSERVED = "observed"
    LEFT_CENSORED = "left_censored"
    RIGHT_CENSORED = "right_censored"
    MISSING = "missing"


class Gate0Decision(str, Enum):
    CONTINUE = "continue"
    KILL = "kill"
    INCONCLUSIVE = "inconclusive"


class Gate0Action(str, Enum):
    CONTINUE = "continue"
    COLLECT = "collect"
    STOP = "stop"


class CellConclusion(str, Enum):
    REDUCTION = "reduction"
    NON_REDUCTION = "non_reduction"
    AMBIGUOUS = "ambiguous"
    MISSING_BOUNDARY = "missing_boundary"
    MISSING_PREDICTOR = "missing_predictor"
    UNCERTIFIED_PREDICTOR = "uncertified_predictor"


class ContrastConclusion(str, Enum):
    REDUCTION = "reduction"
    NON_REDUCTION = "non_reduction"
    AMBIGUOUS = "ambiguous"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True, order=True)
class CellKey:
    """One registered directional configuration."""

    seed: int
    batch_size: int
    direction: Direction
    configuration: str = "default"

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not isinstance(self.direction, Direction):
            raise TypeError("direction must be a Direction")
        if not self.configuration:
            raise ValueError("configuration must be non-empty")


@dataclass(frozen=True)
class ThresholdInterval:
    """A positive threshold interval with explicit censoring semantics.

    ``LEFT_CENSORED`` means the threshold is in ``(0, upper]``;
    ``RIGHT_CENSORED`` means it is in ``[lower, infinity)``.  A missing
    interval has neither bound.  Observed intervals must have both bounds.
    """

    lower: Optional[float]
    upper: Optional[float]
    kind: IntervalKind = IntervalKind.OBSERVED

    def __post_init__(self) -> None:
        if not isinstance(self.kind, IntervalKind):
            raise TypeError("kind must be an IntervalKind")
        for name, value in (("lower", self.lower), ("upper", self.upper)):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive when present")
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError("interval lower bound exceeds upper bound")

        expected = {
            IntervalKind.OBSERVED: (True, True),
            IntervalKind.LEFT_CENSORED: (False, True),
            IntervalKind.RIGHT_CENSORED: (True, False),
            IntervalKind.MISSING: (False, False),
        }[self.kind]
        actual = (self.lower is not None, self.upper is not None)
        if actual != expected:
            raise ValueError(f"{self.kind.value} interval requires bounds {expected}")

    @classmethod
    def point(cls, value: float) -> "ThresholdInterval":
        return cls(value, value, IntervalKind.OBSERVED)

    @classmethod
    def observed(cls, lower: float, upper: float) -> "ThresholdInterval":
        return cls(lower, upper, IntervalKind.OBSERVED)

    @classmethod
    def left_censored(cls, upper: float) -> "ThresholdInterval":
        return cls(None, upper, IntervalKind.LEFT_CENSORED)

    @classmethod
    def right_censored(cls, lower: float) -> "ThresholdInterval":
        return cls(lower, None, IntervalKind.RIGHT_CENSORED)

    @classmethod
    def missing(cls) -> "ThresholdInterval":
        return cls(None, None, IntervalKind.MISSING)

    @property
    def is_missing(self) -> bool:
        return self.kind is IntervalKind.MISSING

    @property
    def is_bounded(self) -> bool:
        return self.kind is IntervalKind.OBSERVED


@dataclass(frozen=True)
class CellEvidence:
    key: CellKey
    empirical_boundary: ThresholdInterval
    predicted_boundary: ThresholdInterval
    predictor_certified: bool


@dataclass(frozen=True)
class CellAssessment:
    key: CellKey
    conclusion: CellConclusion
    minimum_mismatch_factor: Optional[float]
    maximum_mismatch_factor: Optional[float]


@dataclass(frozen=True, order=True)
class BatchContrast:
    """A preregistered exact 16x comparison within one direction/configuration."""

    lower_batch_size: int
    upper_batch_size: int
    direction: Direction
    configuration: str = "default"

    def __post_init__(self) -> None:
        if self.lower_batch_size <= 0:
            raise ValueError("lower_batch_size must be positive")
        if self.upper_batch_size != 16 * self.lower_batch_size:
            raise ValueError("a registered Gate 0 batch contrast must be exactly 16x")
        if not isinstance(self.direction, Direction):
            raise TypeError("direction must be a Direction")
        if not self.configuration:
            raise ValueError("configuration must be non-empty")


@dataclass(frozen=True)
class ResidualObservation:
    """Predictor-conditioned log-boundary residual for one seed/batch cell."""

    seed: int
    batch_size: int
    direction: Direction
    adjusted_log_boundary_residual: float
    configuration: str = "default"

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not isinstance(self.direction, Direction):
            raise TypeError("direction must be a Direction")
        if not math.isfinite(self.adjusted_log_boundary_residual):
            raise ValueError("adjusted log residual must be finite")
        if not self.configuration:
            raise ValueError("configuration must be non-empty")


@dataclass(frozen=True)
class ContrastAssessment:
    contrast: BatchContrast
    conclusion: ContrastConclusion
    paired_seed_count: int
    missing_seeds: Tuple[int, ...]
    mean_log_effect: Optional[float]
    confidence_interval: Optional[Tuple[float, float]]
    multiplicative_effect: Optional[float]


@dataclass(frozen=True)
class Gate0Analysis:
    decision: Gate0Decision
    action: Gate0Action
    cell_assessments: Tuple[CellAssessment, ...]
    contrast_assessments: Tuple[ContrastAssessment, ...]
    reasons: Tuple[str, ...]


@dataclass(frozen=True)
class ValidatedScanArtifact:
    """A scan whose manifest, aggregate, and child cells agree exactly."""

    path: Path
    payload: dict
    manifest_config: dict
    snapshot_binding: Tuple[str, str]
    reference_binding: Optional[Tuple[str, str]]


def _mismatch_factor_bounds(
    empirical: ThresholdInterval,
    predicted: ThresholdInterval,
) -> Tuple[float, float]:
    """Return the sharp min/max fold mismatch over two positive intervals."""

    if empirical.is_missing or predicted.is_missing:
        raise ValueError("mismatch bounds are undefined for a missing interval")

    empirical_lower = empirical.lower if empirical.lower is not None else 0.0
    empirical_upper = empirical.upper if empirical.upper is not None else math.inf
    predicted_lower = predicted.lower if predicted.lower is not None else 0.0
    predicted_upper = predicted.upper if predicted.upper is not None else math.inf

    if empirical_upper < predicted_lower:
        minimum = predicted_lower / empirical_upper
    elif predicted_upper < empirical_lower:
        minimum = empirical_lower / predicted_upper
    else:
        minimum = 1.0

    if (
        empirical_lower == 0.0
        or predicted_lower == 0.0
        or math.isinf(empirical_upper)
        or math.isinf(predicted_upper)
    ):
        maximum = math.inf
    else:
        maximum = max(
            empirical_upper / predicted_lower,
            predicted_upper / empirical_lower,
        )
    return minimum, maximum


def assess_cell(
    evidence: CellEvidence,
    *,
    reduction_match_factor: float = 1.5,
    reduction_miss_factor: float = 2.0,
) -> CellAssessment:
    """Classify one cell without treating uncertainty as agreement."""

    if not 1.0 <= reduction_match_factor < reduction_miss_factor:
        raise ValueError("require 1 <= match factor < miss factor")
    if evidence.empirical_boundary.is_missing:
        return CellAssessment(evidence.key, CellConclusion.MISSING_BOUNDARY, None, None)
    if evidence.predicted_boundary.is_missing:
        return CellAssessment(evidence.key, CellConclusion.MISSING_PREDICTOR, None, None)
    if not evidence.predictor_certified:
        return CellAssessment(
            evidence.key, CellConclusion.UNCERTIFIED_PREDICTOR, None, None
        )

    minimum, maximum = _mismatch_factor_bounds(
        evidence.empirical_boundary, evidence.predicted_boundary
    )
    if evidence.empirical_boundary.is_bounded and evidence.predicted_boundary.is_bounded:
        empirical_center = math.sqrt(
            float(evidence.empirical_boundary.lower)
            * float(evidence.empirical_boundary.upper)
        )
        predicted_center = math.sqrt(
            float(evidence.predicted_boundary.lower)
            * float(evidence.predicted_boundary.upper)
        )
        center_mismatch = max(
            empirical_center / predicted_center,
            predicted_center / empirical_center,
        )
        if center_mismatch > reduction_miss_factor:
            conclusion = CellConclusion.NON_REDUCTION
        elif center_mismatch <= reduction_match_factor:
            conclusion = CellConclusion.REDUCTION
        else:
            conclusion = CellConclusion.AMBIGUOUS
    else:
        # The registered estimator is the geometric center of a strict adjacent
        # bracket.  A censored interval has no eligible center and therefore
        # cannot open Gate 1 even when its nearest endpoint looks far away.
        conclusion = CellConclusion.AMBIGUOUS
    return CellAssessment(evidence.key, conclusion, minimum, maximum)


def _linear_quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot take a quantile of an empty sequence")
    position = (len(sorted_values) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _exact_cluster_bootstrap_interval(
    cluster_values: Sequence[float], confidence_level: float
) -> Tuple[float, float]:
    if not cluster_values:
        raise ValueError("cluster bootstrap requires at least one cluster")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    count = len(cluster_values)
    if count > 7:
        raise ValueError("exact cluster bootstrap is limited to at most seven clusters")
    resampled_means = sorted(
        sum(cluster_values[index] for index in resample) / count
        for resample in itertools.product(range(count), repeat=count)
    )
    alpha = 1.0 - confidence_level
    return (
        _linear_quantile(resampled_means, alpha / 2.0),
        _linear_quantile(resampled_means, 1.0 - alpha / 2.0),
    )


def assess_batch_contrast(
    contrast: BatchContrast,
    observations: Sequence[ResidualObservation],
    *,
    required_seeds: Sequence[int] = (0, 1, 2, 3, 4),
    shift_fraction: float = 0.25,
    confidence_level: float = 0.95,
) -> ContrastAssessment:
    """Assess a paired 16x residual effect by resampling seed clusters.

    A continuation result follows the registered alternative: the point effect
    exceeds 25% multiplicatively and its confidence interval excludes zero.
    A reduction result is deliberately stronger: the entire confidence interval
    lies in the +/-25% equivalence region.  Everything else is inconclusive.
    """

    seeds = tuple(required_seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("required_seeds must be a non-empty unique sequence")
    if not 0.0 < shift_fraction < 1.0:
        raise ValueError("shift_fraction must lie in (0, 1)")

    relevant: Dict[Tuple[int, int], float] = {}
    for observation in observations:
        if (
            observation.direction is not contrast.direction
            or observation.configuration != contrast.configuration
            or observation.seed not in seeds
            or observation.batch_size
            not in (contrast.lower_batch_size, contrast.upper_batch_size)
        ):
            continue
        key = (observation.seed, observation.batch_size)
        if key in relevant:
            raise ValueError(f"duplicate residual observation for seed/batch {key}")
        relevant[key] = observation.adjusted_log_boundary_residual

    paired_effects = []
    missing = []
    for seed in seeds:
        low_key = (seed, contrast.lower_batch_size)
        high_key = (seed, contrast.upper_batch_size)
        if low_key not in relevant or high_key not in relevant:
            missing.append(seed)
            continue
        paired_effects.append(relevant[high_key] - relevant[low_key])

    if missing:
        return ContrastAssessment(
            contrast=contrast,
            conclusion=ContrastConclusion.INCOMPLETE,
            paired_seed_count=len(paired_effects),
            missing_seeds=tuple(missing),
            mean_log_effect=None,
            confidence_interval=None,
            multiplicative_effect=None,
        )

    mean_effect = sum(paired_effects) / len(paired_effects)
    confidence_interval = _exact_cluster_bootstrap_interval(
        paired_effects, confidence_level
    )
    threshold = math.log1p(shift_fraction)
    excludes_zero = confidence_interval[0] > 0.0 or confidence_interval[1] < 0.0
    if abs(mean_effect) > threshold and excludes_zero:
        conclusion = ContrastConclusion.NON_REDUCTION
    elif confidence_interval[0] >= -threshold and confidence_interval[1] <= threshold:
        conclusion = ContrastConclusion.REDUCTION
    else:
        conclusion = ContrastConclusion.AMBIGUOUS

    return ContrastAssessment(
        contrast=contrast,
        conclusion=conclusion,
        paired_seed_count=len(paired_effects),
        missing_seeds=(),
        mean_log_effect=mean_effect,
        confidence_interval=confidence_interval,
        multiplicative_effect=math.exp(mean_effect),
    )


def analyze_gate0(
    *,
    registered_cells: Sequence[CellKey],
    cell_evidence: Sequence[CellEvidence],
    registered_contrasts: Sequence[BatchContrast],
    residual_observations: Sequence[ResidualObservation],
    required_seeds: Sequence[int] = (0, 1, 2, 3, 4),
    reduction_match_factor: float = 1.5,
    reduction_miss_factor: float = 2.0,
    residual_shift_fraction: float = 0.25,
    confidence_level: float = 0.95,
) -> Gate0Analysis:
    """Return a total three-way Gate 0 decision over preregistered evidence.

    Any certified non-reduction result is sufficient to continue immediately.
    Killing is possible only when the finite registry is complete, every cell
    supports reduction, both directional boundaries are represented, and every
    registered batch contrast supplies equivalence evidence.  Completion is
    derived from supplied cell evidence rather than a discretionary CLI flag.
    """

    registered = tuple(registered_cells)
    if len(set(registered)) != len(registered):
        raise ValueError("registered_cells contains duplicates")
    contrasts = tuple(registered_contrasts)
    if len(set(contrasts)) != len(contrasts):
        raise ValueError("registered_contrasts contains duplicates")

    evidence_by_key: Dict[CellKey, CellEvidence] = {}
    for evidence in cell_evidence:
        if evidence.key not in registered:
            raise ValueError(f"cell evidence is not registered: {evidence.key}")
        if evidence.key in evidence_by_key:
            raise ValueError(f"duplicate cell evidence for {evidence.key}")
        evidence_by_key[evidence.key] = evidence
    evidence_complete = set(evidence_by_key) == set(registered)

    assessments = []
    for key in registered:
        evidence = evidence_by_key.get(key)
        if evidence is None:
            evidence = CellEvidence(
                key=key,
                empirical_boundary=ThresholdInterval.missing(),
                predicted_boundary=ThresholdInterval.missing(),
                predictor_certified=False,
            )
        assessments.append(
            assess_cell(
                evidence,
                reduction_match_factor=reduction_match_factor,
                reduction_miss_factor=reduction_miss_factor,
            )
        )

    contrast_assessments = tuple(
        assess_batch_contrast(
            contrast,
            residual_observations,
            required_seeds=required_seeds,
            shift_fraction=residual_shift_fraction,
            confidence_level=confidence_level,
        )
        for contrast in contrasts
    )
    cell_assessments = tuple(assessments)

    non_reducing_cells = [
        item for item in cell_assessments if item.conclusion is CellConclusion.NON_REDUCTION
    ]
    non_reducing_contrasts = [
        item
        for item in contrast_assessments
        if item.conclusion is ContrastConclusion.NON_REDUCTION
    ]
    if non_reducing_cells or non_reducing_contrasts:
        reasons = []
        if non_reducing_cells:
            reasons.append(
                f"{len(non_reducing_cells)} registered cell(s) exceed the "
                f"{reduction_miss_factor:g}x non-reduction threshold"
            )
        if non_reducing_contrasts:
            reasons.append(
                f"{len(non_reducing_contrasts)} registered 16x contrast(s) show a "
                "predictor-conditioned residual shift"
            )
        return Gate0Analysis(
            Gate0Decision.CONTINUE,
            Gate0Action.CONTINUE,
            cell_assessments,
            contrast_assessments,
            tuple(reasons),
        )

    directions = {key.direction for key in registered}
    direction_complete = directions == {Direction.ERASURE, Direction.ACQUISITION}
    required_seed_set = set(required_seeds)
    seed_families: Dict[Tuple[int, Direction, str], set[int]] = {}
    for key in registered:
        family = (key.batch_size, key.direction, key.configuration)
        seed_families.setdefault(family, set()).add(key.seed)
    seed_registry_complete = bool(seed_families) and all(
        seeds == required_seed_set for seeds in seed_families.values()
    )

    expected_contrasts = set()
    direction_configurations = {
        (key.direction, key.configuration) for key in registered
    }
    for direction, configuration in direction_configurations:
        batch_sizes = {
            key.batch_size
            for key in registered
            if key.direction is direction and key.configuration == configuration
        }
        expected_contrasts.update(
            BatchContrast(low, high, direction, configuration)
            for low in batch_sizes
            for high in batch_sizes
            if high == 16 * low
        )
    contrast_registry_complete = bool(expected_contrasts) and set(contrasts) == expected_contrasts
    cells_reduce = bool(cell_assessments) and all(
        item.conclusion is CellConclusion.REDUCTION for item in cell_assessments
    )
    contrasts_reduce = bool(contrast_assessments) and all(
        item.conclusion is ContrastConclusion.REDUCTION
        for item in contrast_assessments
    )
    if (
        evidence_complete
        and direction_complete
        and seed_registry_complete
        and contrast_registry_complete
        and cells_reduce
        and contrasts_reduce
    ):
        return Gate0Analysis(
            Gate0Decision.KILL,
            Gate0Action.STOP,
            cell_assessments,
            contrast_assessments,
            (
                "both directional boundaries reduce in every registered cell",
                "every registered 16x residual contrast lies inside the equivalence region",
            ),
        )

    reasons = []
    if not registered:
        reasons.append("no registered boundary cells")
    elif not direction_complete:
        reasons.append("both directional boundary families are not registered")
    if registered and not seed_registry_complete:
        reasons.append("not every registered boundary family contains every required seed")
    unresolved_cells = [
        item for item in cell_assessments if item.conclusion is not CellConclusion.REDUCTION
    ]
    if unresolved_cells:
        labels = sorted({item.conclusion.value for item in unresolved_cells})
        reasons.append("boundary evidence unresolved: " + ", ".join(labels))
    if not contrasts:
        reasons.append("no registered exact 16x residual contrasts")
    elif not contrast_registry_complete:
        reasons.append("registered contrasts do not exactly cover the registered 16x ladder")
    unresolved_contrasts = [
        item
        for item in contrast_assessments
        if item.conclusion is not ContrastConclusion.REDUCTION
    ]
    if unresolved_contrasts:
        labels = sorted({item.conclusion.value for item in unresolved_contrasts})
        reasons.append("batch-effect evidence unresolved: " + ", ".join(labels))
    if not evidence_complete:
        reasons.append("the finite registered cell set is not yet complete")
    return Gate0Analysis(
        Gate0Decision.INCONCLUSIVE,
        Gate0Action.STOP if evidence_complete else Gate0Action.COLLECT,
        cell_assessments,
        contrast_assessments,
        tuple(reasons),
    )


def registered_exact_16x_contrasts(
    *,
    batch_sizes: Sequence[int] = (32, 128, 512, 2_048),
    directions: Iterable[Direction] = (Direction.ERASURE, Direction.ACQUISITION),
    configuration: str = "default",
) -> Tuple[BatchContrast, ...]:
    """Construct all and only exact 16x contrasts from a registered batch ladder."""

    sizes = tuple(sorted(set(batch_sizes)))
    return tuple(
        BatchContrast(low, high, direction, configuration)
        for direction in directions
        for low in sizes
        for high in sizes
        if high == 16 * low
    )


def _threshold_from_payload(payload: dict) -> ThresholdInterval:
    return ThresholdInterval(
        payload.get("lower"),
        payload.get("upper"),
        IntervalKind(payload.get("kind", IntervalKind.OBSERVED.value)),
    )


def _strict_learning_rate_grid(value: object, *, label: str) -> Tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty JSON list")
    rates = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, numbers.Real):
            raise ValueError(f"{label} contains a non-numeric learning rate")
        rate = float(item)
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError(f"{label} contains a non-positive or non-finite learning rate")
        rates.append(rate)
    if len(set(rates)) != len(rates):
        raise ValueError(f"{label} must contain unique learning rates")
    return tuple(rates)


def _validate_scan_grid(
    scan: dict,
    *,
    rows_key: str,
    expected_rates: Tuple[float, ...],
    label: str,
) -> Tuple[float, ...]:
    requested = _strict_learning_rate_grid(
        scan.get("requested_learning_rates"),
        label=f"{label} requested_learning_rates",
    )
    if requested != expected_rates:
        raise ValueError(f"{label} does not request the exact frozen Gate 0 grid")
    rows = scan.get(rows_key)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{label} must contain non-empty {rows_key}")
    observed = _strict_learning_rate_grid(
        [row.get("learning_rate") if isinstance(row, dict) else None for row in rows],
        label=f"{label} observed learning rates",
    )
    if tuple(sorted(observed)) != tuple(sorted(requested)):
        raise ValueError(
            f"{label} requested learning rates do not equal its unique observed rates"
        )
    return requested


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _binding_sha(config: dict, name: str, *, label: str) -> Tuple[str, str]:
    binding = config.get(name)
    if not isinstance(binding, dict):
        raise ValueError(f"{label} manifest is missing {name}")
    artifact = binding.get("artifact")
    source_manifest = binding.get("source_manifest")
    if not isinstance(artifact, dict) or not isinstance(source_manifest, dict):
        raise ValueError(f"{label} manifest has an incomplete {name} binding")
    return (
        _require_sha256(artifact.get("sha256"), label=f"{label} artifact binding"),
        _require_sha256(
            source_manifest.get("sha256"), label=f"{label} source-manifest binding"
        ),
    )


def _input_binding_identity(value: object, *, label: str) -> Tuple[str, str]:
    binding = _validated_mapping(value, label=label)
    artifact = _validated_mapping(binding.get("artifact"), label=f"{label} artifact")
    source_manifest = _validated_mapping(
        binding.get("source_manifest"), label=f"{label} source manifest"
    )
    return (
        _require_sha256(artifact.get("sha256"), label=f"{label} artifact"),
        _require_sha256(
            source_manifest.get("sha256"), label=f"{label} source manifest"
        ),
    )


def _input_binding_paths(value: object, *, label: str) -> Tuple[Path, Path]:
    binding = _validated_mapping(value, label=label)
    artifact = _validated_mapping(binding.get("artifact"), label=f"{label} artifact")
    source_manifest = _validated_mapping(
        binding.get("source_manifest"), label=f"{label} source manifest"
    )
    artifact_path = artifact.get("path")
    manifest_path = source_manifest.get("path")
    if not isinstance(artifact_path, str) or not isinstance(manifest_path, str):
        raise ValueError(f"{label} does not contain bound file paths")
    return Path(artifact_path), Path(manifest_path)


def _require_bound_input_matches_files(
    value: object,
    *,
    artifact_path: Path,
    manifest_path: Path,
    label: str,
) -> Tuple[str, str]:
    identity = _input_binding_identity(value, label=label)
    current = (
        bind_file(artifact_path)["sha256"],
        bind_manifest(manifest_path)["sha256"],
    )
    if identity != current:
        raise ValueError(f"{label} does not bind the current prerequisite files")
    return identity


def _validate_scan_manifest(
    scan_path: Path,
    *,
    key: CellKey,
    role: str,
    expected_rates: Tuple[float, ...],
    expected_protocol_version: Optional[str],
    expected_gate0_config: Optional[dict],
) -> Tuple[dict, Tuple[str, str], Optional[Tuple[str, str]]]:
    manifest_path = scan_path.parent / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except FileNotFoundError as error:
        raise ValueError(f"{role} scan has no sibling manifest: {scan_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{role} scan sibling manifest is invalid JSON: {manifest_path}"
        ) from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError(f"{role} scan sibling manifest has an invalid schema")
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"{role} scan sibling manifest has no configuration")
    if manifest.get("config_sha256") != content_hash(config):
        raise ValueError(f"{role} scan sibling manifest configuration digest mismatch")

    expected_kind = (
        "gate0_local_stability_scan"
        if role == "local"
        else (
            "gate0_erasure_scan"
            if key.direction is Direction.ERASURE
            else "gate0_acquisition_scan"
        )
    )
    if config.get("kind") != expected_kind:
        raise ValueError(
            f"{role} scan manifest kind does not match the {key.direction.value} evidence row"
        )
    if (
        expected_protocol_version is not None
        and config.get("protocol_version") != expected_protocol_version
    ):
        raise ValueError(f"{role} scan manifest has the wrong protocol version")
    if expected_gate0_config is not None and config.get("gate0") != expected_gate0_config:
        raise ValueError(f"{role} scan manifest has the wrong Gate 0 configuration")
    experiment = config.get("experiment")
    if not isinstance(experiment, dict):
        raise ValueError(f"{role} scan manifest has no experiment configuration")
    manifest_seed = experiment.get("seed")
    manifest_batch = experiment.get("batch_size")
    if (
        isinstance(manifest_seed, bool)
        or not isinstance(manifest_seed, numbers.Integral)
        or int(manifest_seed) != key.seed
    ):
        raise ValueError(f"{role} scan manifest seed does not match its evidence row")
    if (
        isinstance(manifest_batch, bool)
        or not isinstance(manifest_batch, numbers.Integral)
        or int(manifest_batch) != key.batch_size
    ):
        raise ValueError(f"{role} scan manifest batch size does not match its evidence row")
    manifest_rates = _strict_learning_rate_grid(
        config.get("learning_rates"), label=f"{role} scan manifest learning_rates"
    )
    if manifest_rates != expected_rates:
        raise ValueError(f"{role} scan manifest does not freeze the exact Gate 0 grid")

    snapshot_binding = _binding_sha(config, "snapshot_input", label=f"{role} scan")
    reference_binding = (
        None
        if role == "local"
        else _binding_sha(config, "reference_input", label=f"{role} scan")
    )
    return config, snapshot_binding, reference_binding


def _rate_cell_path(scan_path: Path, role: str, learning_rate: float) -> Path:
    label = f"eta_{learning_rate:.12g}".replace(".", "p")
    filename = "local_stability.json" if role == "local" else "result.json"
    return scan_path.parent / label / filename


def _validated_mapping(value: object, *, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _finite_real(value: object, *, label: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise ValueError(f"{label} must be finite" + (" and nonnegative" if nonnegative else ""))
    return result


def _validate_self_digest(payload: dict, digest_key: str, *, label: str) -> None:
    declared = _require_sha256(payload.get(digest_key), label=f"{label} {digest_key}")
    unsigned = {key: value for key, value in payload.items() if key != digest_key}
    if declared != content_hash(unsigned):
        raise ValueError(f"{label} has a tampered {digest_key}")


def _validate_scan_row(
    row: dict,
    *,
    expected_rate: float,
    role: str,
    manifest_config: dict,
    snapshot_sha256: str,
    reference_sha256: Optional[str],
    label: str,
) -> None:
    _validate_self_digest(row, "result_sha256", label=label)
    provenance = _validated_mapping(row.get("provenance"), label=f"{label} provenance")
    _validate_self_digest(provenance, "cell_sha256", label=f"{label} provenance")
    expected_kind = (
        "gate0_local_stability"
        if role == "local"
        else (
            "gate0_erasure"
            if manifest_config.get("kind") == "gate0_erasure_scan"
            else "gate0_acquisition"
        )
    )
    if provenance.get("schema_version") != 1 or provenance.get("kind") != expected_kind:
        raise ValueError(f"{label} has the wrong cell provenance kind")
    controls = _validated_mapping(
        provenance.get("controls"), label=f"{label} provenance controls"
    )
    for name, manifest_name in (
        ("experiment", "experiment"),
        ("metric", "metric"),
        ("gate", "gate0"),
    ):
        if controls.get(name) != manifest_config.get(manifest_name):
            raise ValueError(f"{label} provenance has mismatched {name} controls")
    observed_rate = controls.get("learning_rate")
    if (
        isinstance(observed_rate, bool)
        or not isinstance(observed_rate, numbers.Real)
        or float(observed_rate) != expected_rate
    ):
        raise ValueError(f"{label} provenance has the wrong learning rate")
    if role == "local":
        if controls.get("augmented_tolerance") != 1e-3:
            raise ValueError(f"{label} has the wrong frozen eigensolver tolerance")
        if controls.get("augmented_max_iterations") != 100:
            raise ValueError(f"{label} has the wrong frozen eigensolver iteration limit")
    elif controls.get("thresholds") != manifest_config.get("state"):
        raise ValueError(f"{label} provenance has mismatched state thresholds")
    sources = _validated_mapping(
        provenance.get("sources"), label=f"{label} provenance sources"
    )
    if sources.get("snapshot_sha256") != snapshot_sha256:
        raise ValueError(f"{label} provenance does not bind the manifest snapshot")
    if role != "local" and sources.get("reference_sha256") != reference_sha256:
        raise ValueError(f"{label} provenance does not bind the manifest reference")

    if role == "local":
        if row.get("stream_unchanged") is not True:
            raise ValueError(f"{label} advanced the registered data stream")
        eigenvalues = row.get("augmented_eigenvalues")
        gate = manifest_config.get("gate0", {})
        expected_count = gate.get("augmented_eigenvalue_count")
        if not isinstance(eigenvalues, list) or len(eigenvalues) != expected_count:
            raise ValueError(f"{label} has the wrong number of augmented eigenpairs")
        residuals = []
        magnitudes = []
        for index, raw_eigenvalue in enumerate(eigenvalues):
            eigenvalue = _validated_mapping(
                raw_eigenvalue, label=f"{label} eigenvalue {index}"
            )
            residual = eigenvalue.get("relative_residual")
            magnitude = eigenvalue.get("magnitude")
            residuals.append(
                _finite_real(
                    residual,
                    label=f"{label} eigenpair residual",
                    nonnegative=True,
                )
            )
            magnitudes.append(
                _finite_real(
                    magnitude,
                    label=f"{label} eigenvalue magnitude",
                    nonnegative=True,
                )
            )
        max_residual = max(residuals)
        spectral_radius = max(magnitudes)
        if not math.isclose(
            _finite_real(
                row.get("augmented_max_relative_residual"),
                label=f"{label} maximum residual",
                nonnegative=True,
            ),
            max_residual,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError(f"{label} maximum residual is inconsistent")
        if not math.isclose(
            _finite_real(
                row.get("augmented_spectral_radius"),
                label=f"{label} spectral radius",
                nonnegative=True,
            ),
            spectral_radius,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError(f"{label} spectral radius is inconsistent")
        expected_certified = max_residual <= float(
            gate.get("augmented_max_relative_residual")
        )
        if row.get("augmented_certified") is not expected_certified:
            raise ValueError(f"{label} certification label is inconsistent")
    elif manifest_config.get("kind") == "gate0_erasure_scan":
        outcome = row.get("outcome")
        if outcome not in {"retained", "erased", "diverged", "unresolved"}:
            raise ValueError(f"{label} has an invalid erasure outcome")
        if row.get("erased") is not (outcome == "erased"):
            raise ValueError(f"{label} has inconsistent erasure fields")
        entry = row.get("sustained_entry_step")
        if outcome == "erased":
            entry_value = _finite_real(entry, label=f"{label} erasure entry")
            if entry_value < 0.0 or entry_value > float(
                manifest_config["gate0"]["erase_horizon"]
            ):
                raise ValueError(f"{label} misses the frozen erasure-entry horizon")
        elif entry is not None:
            raise ValueError(f"{label} has an entry step without erasure")
    else:
        outcome = row.get("outcome")
        if outcome not in {"transitioned", "censored", "diverged"}:
            raise ValueError(f"{label} has an invalid acquisition outcome")
        transition_step = row.get("transition_step")
        if (outcome == "transitioned") != (transition_step is not None):
            raise ValueError(f"{label} has inconsistent acquisition-transition fields")


def validate_scan_artifact(
    scan_path: Path,
    *,
    key: CellKey,
    role: str,
    expected_rates: Tuple[float, ...],
    expected_protocol_version: Optional[str],
    expected_gate0_config: Optional[dict],
) -> ValidatedScanArtifact:
    """Validate an official scan from manifest through aggregate to child cells."""

    if role not in {"local", "empirical"}:
        raise ValueError("scan role must be local or empirical")
    scan_path = Path(scan_path)
    try:
        payload = json.loads(scan_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{role} scan is not readable JSON: {scan_path}") from error
    if not isinstance(payload, dict) or payload.get("complete") is not True:
        raise ValueError(f"{role} scan is incomplete or malformed")
    rows_key = "measurements" if role == "local" else "branches"
    requested = _validate_scan_grid(
        payload,
        rows_key=rows_key,
        expected_rates=expected_rates,
        label=f"{role} scan",
    )
    manifest_config, snapshot_binding, reference_binding = _validate_scan_manifest(
        scan_path,
        key=key,
        role=role,
        expected_rates=expected_rates,
        expected_protocol_version=expected_protocol_version,
        expected_gate0_config=expected_gate0_config,
    )
    rows = payload[rows_key]
    observed_rates = tuple(float(row["learning_rate"]) for row in rows)
    if observed_rates != requested:
        raise ValueError(f"{role} scan rows are not in frozen learning-rate order")
    for index, (row, rate) in enumerate(zip(rows, requested)):
        row = _validated_mapping(row, label=f"{role} scan row {index}")
        child_path = _rate_cell_path(scan_path, role, rate)
        try:
            child = json.loads(child_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"{role} scan has no readable child cell: {child_path}") from error
        if child != row:
            raise ValueError(f"{role} scan aggregate disagrees with child cell at {rate:g}")
        _validate_scan_row(
            row,
            expected_rate=rate,
            role=role,
            manifest_config=manifest_config,
            snapshot_sha256=snapshot_binding[0],
            reference_sha256=None if reference_binding is None else reference_binding[0],
            label=f"{role} scan row {index}",
        )
    if role == "local":
        uncertified = [
            float(row["learning_rate"])
            for row in rows
            if row.get("augmented_certified") is not True
        ]
        if payload.get("all_augmented_eigenpairs_certified") is not (not uncertified):
            raise ValueError("local scan aggregate certification is inconsistent")
        if payload.get("uncertified_learning_rates") != uncertified:
            raise ValueError("local scan aggregate uncertified-rate list is inconsistent")
    elif manifest_config.get("kind") == "gate0_erasure_scan":
        derived_brackets = [
            {
                "lower_learning_rate": float(lower["learning_rate"]),
                "upper_learning_rate": float(upper["learning_rate"]),
            }
            for lower, upper in zip(rows, rows[1:])
            if lower.get("outcome") == "retained" and upper.get("outcome") == "erased"
        ]
        if payload.get("strict_adjacent_brackets") != derived_brackets:
            raise ValueError("erasure scan aggregate brackets are inconsistent")
    return ValidatedScanArtifact(
        scan_path,
        payload,
        manifest_config,
        snapshot_binding,
        reference_binding,
    )


def analyze_gate0_payload(
    payload: dict,
    *,
    registered_seeds: Sequence[int] = (0, 1, 2, 3, 4),
    registered_batch_sizes: Sequence[int] = (32, 128, 512, 2_048),
    reduction_match_factor: float = 1.5,
    reduction_miss_factor: float = 2.0,
    residual_shift_fraction: float = 0.25,
    confidence_level: float = 0.95,
    require_scan_sources: bool = False,
    expected_learning_rates: Sequence[float] = _FROZEN_GATE0_LEARNING_RATES,
    expected_protocol_version: Optional[str] = None,
    expected_gate0_config: Optional[dict] = None,
) -> Gate0Analysis:
    """Parse the frozen JSON evidence schema and execute the registered gate."""
    expected_rates = tuple(float(rate) for rate in expected_learning_rates)
    if (
        expected_rates != tuple(sorted(expected_rates))
        or len(expected_rates) != len(set(expected_rates))
        or any(not math.isfinite(rate) or rate <= 0.0 for rate in expected_rates)
    ):
        raise ValueError("expected_learning_rates must be a sorted unique positive grid")
    if "finalize" in payload:
        raise ValueError("Gate 0 completion is derived; finalize is not a valid field")
    cells = tuple(
        CellKey(seed, batch_size, direction)
        for direction in (Direction.ERASURE, Direction.ACQUISITION)
        for batch_size in registered_batch_sizes
        for seed in registered_seeds
    )
    registered_set = set(cells)
    evidence = []
    derived_residuals = []
    assigned_sources = set()
    common_reference_binding: Optional[Tuple[str, str]] = None
    common_calibration_binding: Optional[Tuple[str, str]] = None
    common_calibration_input: Optional[dict] = None
    common_first_cell_binding: Optional[Tuple[str, str]] = None
    common_first_cell_input: Optional[dict] = None
    verify_prerequisite_chain = (
        require_scan_sources
        and expected_protocol_version is not None
        and expected_gate0_config is not None
    )
    for row in payload.get("cells", []):
        key = CellKey(
            seed=int(row["seed"]),
            batch_size=int(row["batch_size"]),
            direction=Direction(row["direction"]),
            configuration=str(row.get("configuration", "default")),
        )
        if key not in registered_set:
            raise ValueError(f"evidence contains an unregistered cell: {key}")
        has_sources = "empirical_scan" in row and "local_scan" in row
        if require_scan_sources and not has_sources:
            raise ValueError("official Gate 0 evidence must name raw empirical and local scans")
        if has_sources:
            empirical_path = Path(row["empirical_scan"])
            local_path = Path(row["local_scan"])
            for source in (empirical_path, local_path):
                identity = source.resolve()
                if identity in assigned_sources:
                    raise ValueError(f"raw scan source is assigned more than once: {source}")
                assigned_sources.add(identity)
            empirical_artifact = validate_scan_artifact(
                empirical_path,
                key=key,
                role="empirical",
                expected_rates=expected_rates,
                expected_protocol_version=expected_protocol_version,
                expected_gate0_config=expected_gate0_config,
            )
            local_artifact = validate_scan_artifact(
                local_path,
                key=key,
                role="local",
                expected_rates=expected_rates,
                expected_protocol_version=expected_protocol_version,
                expected_gate0_config=expected_gate0_config,
            )
            if empirical_artifact.snapshot_binding != local_artifact.snapshot_binding:
                raise ValueError(
                    "empirical and local scan manifests do not bind the same snapshot artifact"
                )
            reference_binding = empirical_artifact.reference_binding
            if common_reference_binding is None:
                common_reference_binding = reference_binding
            elif reference_binding != common_reference_binding:
                raise ValueError("empirical scan manifests do not bind the same reference")

            empirical_config = empirical_artifact.manifest_config
            local_config = local_artifact.manifest_config
            if verify_prerequisite_chain:
                empirical_calibration_input = _validated_mapping(
                    empirical_config.get("calibration_input"),
                    label="empirical calibration input",
                )
                empirical_calibration = _input_binding_identity(
                    empirical_calibration_input,
                    label="empirical calibration input",
                )
                local_calibration = _input_binding_identity(
                    local_config.get("calibration_input"),
                    label="local calibration input",
                )
                if empirical_calibration != local_calibration:
                    raise ValueError("empirical and local scans do not bind the same calibration")
                if common_calibration_binding is None:
                    common_calibration_binding = empirical_calibration
                    common_calibration_input = empirical_calibration_input
                elif empirical_calibration != common_calibration_binding:
                    raise ValueError("Gate 0 cells do not bind one common calibration")

                _require_bound_input_matches_files(
                    empirical_config.get("local_prerequisite"),
                    artifact_path=local_path,
                    manifest_path=local_path.parent / "manifest.json",
                    label="empirical local prerequisite",
                )
                first_empirical = empirical_config.get("first_cell_analysis")
                first_local = local_config.get("first_cell_analysis")
                if first_empirical != first_local:
                    raise ValueError(
                        "empirical and local scans disagree on first-cell authorization"
                    )
                is_first = (
                    key.seed == 0
                    and key.batch_size == 128
                    and key.direction is Direction.ERASURE
                )
                if is_first:
                    if first_empirical is not None:
                        raise ValueError("the first registered cell must not depend on itself")
                else:
                    first_identity = _input_binding_identity(
                        first_empirical, label="first-cell analysis input"
                    )
                    if common_first_cell_binding is None:
                        common_first_cell_binding = first_identity
                        common_first_cell_input = _validated_mapping(
                            first_empirical, label="first-cell analysis input"
                        )
                    elif first_identity != common_first_cell_binding:
                        raise ValueError("Gate 0 cells do not share first-cell authorization")

            empirical_scan = empirical_artifact.payload
            local_scan = local_artifact.payload
            empirical_boundary = (
                empirical_erasure_boundary(empirical_scan)
                if key.direction is Direction.ERASURE
                else empirical_acquisition_boundary(empirical_scan)
            )
            predicted_boundary, predictor_certified = predicted_local_boundary(
                local_scan, key.direction
            )
        else:
            empirical_boundary = _threshold_from_payload(row["empirical_boundary"])
            predicted_boundary = _threshold_from_payload(row["predicted_boundary"])
            predictor_certified = bool(row["predictor_certified"])
        evidence.append(
            CellEvidence(key, empirical_boundary, predicted_boundary, predictor_certified)
        )
        if empirical_boundary.is_bounded and predicted_boundary.is_bounded:
            empirical_center = math.sqrt(
                float(empirical_boundary.lower) * float(empirical_boundary.upper)
            )
            predicted_center = math.sqrt(
                float(predicted_boundary.lower) * float(predicted_boundary.upper)
            )
            derived_residuals.append(
                ResidualObservation(
                    key.seed,
                    key.batch_size,
                    key.direction,
                    math.log(empirical_center / predicted_center),
                    key.configuration,
                )
            )
    if verify_prerequisite_chain and evidence:
        if common_calibration_input is None:
            raise ValueError("official Gate 0 evidence has no common calibration")
        calibration_path, calibration_manifest_path = _input_binding_paths(
            common_calibration_input, label="common calibration input"
        )
        _require_bound_input_matches_files(
            common_calibration_input,
            artifact_path=calibration_path,
            manifest_path=calibration_manifest_path,
            label="common calibration input",
        )
        from .gate0_calibration import bind_passed_gate0_calibration

        calibrated = bind_passed_gate0_calibration(
            calibration_path,
            expected_protocol_version=expected_protocol_version,
        )
        if calibrated["artifact"]["sha256"] != common_calibration_binding[0]:
            raise ValueError("common calibration binding changed after scan creation")
        if common_first_cell_input is not None:
            first_path, first_manifest_path = _input_binding_paths(
                common_first_cell_input, label="common first-cell analysis input"
            )
            _require_bound_input_matches_files(
                common_first_cell_input,
                artifact_path=first_path,
                manifest_path=first_manifest_path,
                label="common first-cell analysis input",
            )
            if expected_protocol_version is None or expected_gate0_config is None:
                raise ValueError("official first-cell authorization needs frozen controls")
            first_bound = bind_gate0_analysis_report(
                first_path,
                expected_protocol_version=expected_protocol_version,
                expected_gate0_config=expected_gate0_config,
                registered_seeds=registered_seeds,
                registered_batch_sizes=registered_batch_sizes,
                required_action=Gate0Action.COLLECT,
            )
            if first_bound["artifact"]["sha256"] != common_first_cell_binding[0]:
                raise ValueError("first-cell authorization changed after scan creation")
    contrasts = registered_exact_16x_contrasts(
        batch_sizes=registered_batch_sizes,
    )
    declared_residuals = tuple(
        ResidualObservation(
            int(row["seed"]),
            int(row["batch_size"]),
            Direction(row["direction"]),
            float(row["adjusted_log_boundary_residual"]),
            str(row.get("configuration", "default")),
        )
        for row in payload.get("residual_observations", [])
    )
    residuals = tuple(derived_residuals) if require_scan_sources else declared_residuals
    return analyze_gate0(
        registered_cells=cells,
        cell_evidence=tuple(evidence),
        registered_contrasts=contrasts,
        residual_observations=residuals,
        required_seeds=registered_seeds,
        reduction_match_factor=reduction_match_factor,
        reduction_miss_factor=reduction_miss_factor,
        residual_shift_fraction=residual_shift_fraction,
        confidence_level=confidence_level,
    )


def gate0_analysis_to_dict(analysis: Gate0Analysis) -> dict:
    """Return a JSON-compatible, deterministic gate report."""
    return json.loads(json.dumps(asdict(analysis)))


def gate0_payload_source_paths(payload: dict) -> Tuple[Path, ...]:
    """Return raw scan paths after proving one-to-one source assignment."""
    paths = []
    identities = set()
    for row in payload.get("cells", []):
        for key in ("empirical_scan", "local_scan"):
            if key not in row:
                raise ValueError("official Gate 0 evidence must name both raw scan paths")
            path = Path(row[key])
            identity = path.resolve()
            if identity in identities:
                raise ValueError(f"raw scan source is assigned more than once: {path}")
            identities.add(identity)
            paths.append(path)
    return tuple(paths)


def gate0_payload_registry_complete(
    payload: dict,
    *,
    registered_seeds: Sequence[int] = (0, 1, 2, 3, 4),
    registered_batch_sizes: Sequence[int] = (32, 128, 512, 2_048),
) -> bool:
    rows = payload.get("cells")
    if not isinstance(rows, list):
        raise ValueError("Gate 0 evidence cells must be a JSON list")
    observed = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Gate 0 evidence cell must be a JSON object")
        key = (
            int(row["seed"]),
            int(row["batch_size"]),
            Direction(row["direction"]),
            str(row.get("configuration", "default")),
        )
        if key in observed:
            raise ValueError(f"Gate 0 evidence contains a duplicate cell: {key}")
        observed.add(key)
    expected = {
        (seed, batch, direction, "default")
        for direction in (Direction.ERASURE, Direction.ACQUISITION)
        for batch in registered_batch_sizes
        for seed in registered_seeds
    }
    if not observed.issubset(expected):
        raise ValueError("Gate 0 evidence contains cells outside the frozen registry")
    return observed == expected


def bind_gate0_analysis_report(
    path: Path,
    *,
    expected_protocol_version: str,
    expected_gate0_config: dict,
    registered_seeds: Sequence[int] = (0, 1, 2, 3, 4),
    registered_batch_sizes: Sequence[int] = (32, 128, 512, 2_048),
    required_action: Optional[Gate0Action] = None,
) -> dict:
    """Re-derive a Gate 0 report from its bound evidence and raw scans."""

    path = Path(path)
    try:
        report = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Gate 0 report is not readable JSON: {path}") from error
    if not isinstance(report, dict):
        raise ValueError("Gate 0 report must be a JSON object")
    manifest_path = path.parent / "manifest.json"
    manifest_binding = bind_manifest(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    config = _validated_mapping(manifest.get("config"), label="Gate 0 report manifest config")
    if config.get("kind") != "gate0_decision_analysis":
        raise ValueError("Gate 0 report has the wrong sibling manifest kind")
    if config.get("protocol_version") != expected_protocol_version:
        raise ValueError("Gate 0 report has the wrong protocol version")
    if config.get("gate0") != expected_gate0_config:
        raise ValueError("Gate 0 report has the wrong frozen Gate 0 config")
    evidence_input = _validated_mapping(
        config.get("evidence_input"), label="Gate 0 evidence input"
    )
    evidence_path_value = evidence_input.get("path")
    if not isinstance(evidence_path_value, str):
        raise ValueError("Gate 0 evidence binding has no path")
    evidence_path = Path(evidence_path_value)
    if bind_file(evidence_path) != evidence_input:
        raise ValueError("Gate 0 evidence file changed after analysis")
    try:
        evidence = json.loads(evidence_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Gate 0 evidence input is not readable JSON") from error
    if not isinstance(evidence, dict) or "finalize" in evidence:
        raise ValueError("Gate 0 evidence has an invalid structure or control field")
    source_paths = gate0_payload_source_paths(evidence)
    current_scan_inputs = [
        {
            "artifact": bind_file(source),
            "source_manifest": bind_manifest(source.parent / "manifest.json"),
        }
        for source in source_paths
    ]
    if config.get("scan_inputs") != current_scan_inputs:
        raise ValueError("Gate 0 raw scan inputs changed after analysis")
    registry_complete = gate0_payload_registry_complete(
        evidence,
        registered_seeds=registered_seeds,
        registered_batch_sizes=registered_batch_sizes,
    )
    if config.get("registry_complete") is not registry_complete:
        raise ValueError("Gate 0 manifest has a false registry-completion claim")
    analysis = analyze_gate0_payload(
        evidence,
        registered_seeds=registered_seeds,
        registered_batch_sizes=registered_batch_sizes,
        reduction_match_factor=float(expected_gate0_config["reduction_match_factor"]),
        reduction_miss_factor=float(expected_gate0_config["reduction_miss_factor"]),
        residual_shift_fraction=float(
            expected_gate0_config["residual_batch_shift_fraction"]
        ),
        require_scan_sources=True,
        expected_learning_rates=tuple(expected_gate0_config["learning_rates"]),
        expected_protocol_version=expected_protocol_version,
        expected_gate0_config=expected_gate0_config,
    )
    expected_report = {
        "schema_version": 1,
        "kind": "gate0_decision_analysis",
        "protocol_version": expected_protocol_version,
        "registry_complete": registry_complete,
        **gate0_analysis_to_dict(analysis),
    }
    if report != expected_report:
        raise ValueError("Gate 0 report does not equal a fresh analysis of its inputs")
    if required_action is not None and report.get("action") != required_action.value:
        raise ValueError(f"Gate 0 report does not authorize {required_action.value}")
    return {
        "artifact": bind_file(path),
        "source_manifest": manifest_binding,
        "decision": report["decision"],
        "action": report["action"],
        "registry_complete": registry_complete,
    }


def empirical_erasure_boundary(scan: dict) -> ThresholdInterval:
    """Convert a complete strict erasure scan into one threshold interval."""
    if not scan.get("complete"):
        return ThresholdInterval.missing()
    branches = scan.get("branches")
    if not isinstance(branches, list) or not branches:
        return ThresholdInterval.missing()
    rows = sorted(branches, key=lambda row: float(row["learning_rate"]))
    rates = [float(row["learning_rate"]) for row in rows]
    outcomes = [row.get("outcome") for row in rows]
    if any(outcome not in {"retained", "erased"} for outcome in outcomes):
        return ThresholdInterval.missing()
    crossings = [
        (rates[index], rates[index + 1])
        for index in range(len(rates) - 1)
        if outcomes[index] == "retained" and outcomes[index + 1] == "erased"
    ]
    if len(crossings) == 1 and outcomes == sorted(
        outcomes, key=lambda value: {"retained": 0, "erased": 1}[value]
    ):
        return ThresholdInterval.observed(*crossings[0])
    if all(outcome == "retained" for outcome in outcomes):
        return ThresholdInterval.right_censored(max(rates))
    if all(outcome == "erased" for outcome in outcomes):
        return ThresholdInterval.left_censored(min(rates))
    return ThresholdInterval.missing()


def empirical_acquisition_boundary(scan: dict) -> ThresholdInterval:
    """Convert a complete acquisition scan into the largest acquiring rate interval."""
    if not scan.get("complete"):
        return ThresholdInterval.missing()
    branches = scan.get("branches")
    if not isinstance(branches, list) or not branches:
        return ThresholdInterval.missing()
    rows = sorted(branches, key=lambda row: float(row["learning_rate"]))
    rates = [float(row["learning_rate"]) for row in rows]
    outcomes = [row.get("outcome") for row in rows]
    if any(outcome not in {"transitioned", "censored"} for outcome in outcomes):
        return ThresholdInterval.missing()
    crossings = [
        (rates[index], rates[index + 1])
        for index in range(len(rates) - 1)
        if outcomes[index] == "transitioned" and outcomes[index + 1] == "censored"
    ]
    if len(crossings) == 1 and outcomes == sorted(
        outcomes, key=lambda value: {"transitioned": 0, "censored": 1}[value]
    ):
        return ThresholdInterval.observed(*crossings[0])
    if all(outcome == "transitioned" for outcome in outcomes):
        return ThresholdInterval.right_censored(max(rates))
    if all(outcome == "censored" for outcome in outcomes):
        return ThresholdInterval.left_censored(min(rates))
    return ThresholdInterval.missing()


def predicted_local_boundary(scan: dict, direction: Direction) -> tuple[ThresholdInterval, bool]:
    """Apply the frozen unit-circle rule to a complete local-stability scan."""
    if not scan.get("complete"):
        return ThresholdInterval.missing(), False
    measurements = scan.get("measurements")
    if not isinstance(measurements, list) or not measurements:
        return ThresholdInterval.missing(), False
    rows = sorted(measurements, key=lambda row: float(row["learning_rate"]))
    certified = all(bool(row.get("augmented_certified")) for row in rows)
    if not certified:
        return ThresholdInterval.missing(), False
    rates = [float(row["learning_rate"]) for row in rows]
    radii = [float(row["augmented_spectral_radius"]) for row in rows]
    if any(not math.isfinite(value) or value == 1.0 for value in radii):
        return ThresholdInterval.missing(), False
    states = ["unstable" if value > 1.0 else "stable" for value in radii]
    if direction is Direction.ERASURE:
        low_state, high_state = "stable", "unstable"
    elif direction is Direction.ACQUISITION:
        low_state, high_state = "unstable", "stable"
    else:
        raise TypeError("direction must be a Direction")
    crossings = [
        (rates[index], rates[index + 1])
        for index in range(len(rates) - 1)
        if states[index] == low_state and states[index + 1] == high_state
    ]
    ordered = states == sorted(
        states, key=lambda value: {low_state: 0, high_state: 1}[value]
    )
    if len(crossings) == 1 and ordered:
        return ThresholdInterval.observed(*crossings[0]), True
    if all(state == low_state for state in states):
        return ThresholdInterval.right_censored(max(rates)), True
    if all(state == high_state for state in states):
        return ThresholdInterval.left_censored(min(rates)), True
    return ThresholdInterval.missing(), True
