"""Amendment v1.5 section 1 autopsy classifier for sealed Gate 0 cells.

The five labels, their precedence, and the branch mapping are frozen in
``AMENDMENT_v1_5.md``, which was committed before this module existed. The
amendment references registered v1.4.1 quantities by description rather than
value; the resolutions below were fixed while writing this module and before
any sealed cell was classified:

1. **Divergence multiplier.** The v1.4.1 registered erasure divergence rule is
   ``full_vocab_ce > 5.0 * q_star_loss_mean`` or a non-finite loss
   (``gate0_boundary`` erasure classification). The autopsy applies it to
   every logged row, per probe, as the amendment's "at any point" demands.
2. **Both probes.** A rule fires for a cell only when its complete condition
   holds on probe 0 and on probe 1, evaluated independently on the per-probe
   series (``probe_i_c_int``, ``probe_i_full_vocab_ce``). For "at any point"
   conditions the firing rows may differ between probes.
3. **Bands.** The suppressed C_int band is ``plateau_mean +/- 3 * plateau_sd``
   and the q* CE band radius is ``max(3 * q_star_loss_sd,
   0.02 * |q_star_loss_mean|)``, exactly the registered ``plateau_bounds`` and
   ``is_flat_loss`` conventions.
4. **Final 2,000 logged steps.** Mirrors ``is_suppressed``: the window is every
   row with ``branch_step >= hold - 2_000`` and coverage requires the first
   window row to sit at or before that start.
5. **Quarters.** Quarter ``i`` is the half-open interval
   ``((i - 1) * hold / 4, i * hold / 4]`` over ``branch_step``.
6. **"Final quarter still above the band"** (rule 3) means the final-quarter
   mean exceeds the suppressed band's upper edge.
7. **Equilibration tolerance** (rule 4) is the registered Gate 1 rule: the
   absolute OLS slope per 1,000 steps over the final quarter must be below one
   plateau SD. "Final level" is the final-quarter mean; "CE inside the q*
   band" tests the final-quarter mean CE.
8. **Branch mapping completion.** The amendment enumerates: any
   ``stochastic`` selects Branch A; all three in
   {``slow_erasure``, ``deadline_artifact``} selects Branch B; all three
   ``intermediate`` runs C then A; ``near_divergent`` selects no branch by
   itself. The unenumerated mixtures (e.g. ``intermediate`` alongside
   ``slow_erasure``) are completed deterministically toward the more general
   instrument: Branch B activates only when every branch-selecting label is in
   {``slow_erasure``, ``deadline_artifact``} and at least one such label
   exists; every other combination activates Branch A. Branch C attaches
   whenever any label is ``intermediate``.
9. **Hold escalation.** For each ``slow_erasure`` cell the final-quarter OLS
   line (per probe) is extrapolated to the suppressed band's upper edge; a
   non-negative slope never reaches the band and counts as "later than step
   14,000". The cell's crossing is the later of its two probes. Any
   ``slow_erasure`` cell crossing beyond step 14,000 escalates T_hold from
   16,000 to 24,000.

These resolutions were committed together with this classifier before it was
run on any sealed cell; the autopsy artifact embeds them for audit.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .provenance import bind_file
from .references import load_reference_bands
from .state import ReferenceBands, StateThresholds, plateau_bounds

AUTOPSY_LABELS = (
    "near_divergent",
    "deadline_artifact",
    "slow_erasure",
    "intermediate",
    "stochastic",
)
DIVERGENCE_MULTIPLIER = 5.0
TARGET_RATES = (0.006, 0.012, 0.05)
BASE_HOLD_STEPS = 16_000
ESCALATED_HOLD_STEPS = 24_000
ESCALATION_CROSSING_STEP = 14_000
GRID_CAP_ON_DIVERGENT_TOP = 0.032
PROBES = (0, 1)


@dataclass(frozen=True)
class ProbeSeries:
    steps: Tuple[float, ...]
    c_int: Tuple[float, ...]
    full_vocab_ce: Tuple[float, ...]


@dataclass(frozen=True)
class CellAutopsy:
    learning_rate: float
    label: str
    fired_rule: str
    diagnostics: Dict[str, object]
    excerpts: Dict[str, object]


@dataclass(frozen=True)
class BranchActivation:
    primary_branch: str
    include_branch_c: bool
    grid_cap: Optional[float]
    t_hold_steps: int
    labels: Dict[str, str]
    reasons: Tuple[str, ...]


def q_star_bounds(
    reference: ReferenceBands, thresholds: StateThresholds
) -> Tuple[float, float]:
    radius = max(
        3.0 * reference.q_star_loss_sd,
        thresholds.flat_loss_relative_tolerance * abs(reference.q_star_loss_mean),
    )
    return reference.q_star_loss_mean - radius, reference.q_star_loss_mean + radius


def _probe_series(rows: Sequence[Mapping[str, float]], probe: int) -> ProbeSeries:
    steps, c_int, ce = [], [], []
    for row in rows:
        steps.append(float(row["branch_step"]))
        c_int.append(float(row[f"probe_{probe}_c_int"]))
        ce.append(float(row[f"probe_{probe}_full_vocab_ce"]))
    return ProbeSeries(tuple(steps), tuple(c_int), tuple(ce))


def _quarter_indices(steps: Sequence[float], hold_steps: int, quarter: int) -> List[int]:
    low = (quarter - 1) * hold_steps / 4.0
    high = quarter * hold_steps / 4.0
    return [index for index, step in enumerate(steps) if low < step <= high]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _ols_line(steps: Sequence[float], values: Sequence[float]) -> Tuple[float, float]:
    """Return (slope per step, intercept) of the least-squares line."""
    if len(steps) < 2:
        raise ValueError("OLS line requires at least two points")
    n = float(len(steps))
    mean_x = _mean(steps)
    mean_y = _mean(values)
    sxx = sum((x - mean_x) ** 2 for x in steps)
    if sxx == 0.0:
        raise ValueError("OLS line requires distinct steps")
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(steps, values))
    slope = sxy / sxx
    return slope, mean_y - slope * mean_x


def _validate_rows(rows: Sequence[Mapping[str, float]], hold_steps: int) -> None:
    if not rows:
        raise ValueError("autopsy requires a nonempty sealed metrics log")
    steps = [float(row["branch_step"]) for row in rows]
    if any(later <= earlier for earlier, later in zip(steps, steps[1:])):
        raise ValueError("sealed metrics log must have strictly increasing branch steps")
    if steps[-1] != float(hold_steps):
        raise ValueError("sealed metrics log does not reach the registered hold")


def _rule_near_divergent(
    series: Mapping[int, ProbeSeries],
    reference: ReferenceBands,
) -> Tuple[bool, Dict[str, object]]:
    threshold = DIVERGENCE_MULTIPLIER * reference.q_star_loss_mean
    diagnostics: Dict[str, object] = {"threshold": threshold}
    fired_probes = {}
    for probe, data in series.items():
        first = next(
            (
                {"branch_step": step, "full_vocab_ce": ce}
                for step, ce in zip(data.steps, data.full_vocab_ce)
                if not math.isfinite(ce) or ce > threshold
            ),
            None,
        )
        fired_probes[probe] = first
        diagnostics[f"probe_{probe}_first_divergent_row"] = first
    return all(value is not None for value in fired_probes.values()), diagnostics


def _final_window(
    data: ProbeSeries, hold_steps: int, duration: int
) -> Optional[List[int]]:
    start = hold_steps - duration
    indices = [index for index, step in enumerate(data.steps) if step >= start]
    if not indices or data.steps[indices[0]] > start:
        return None
    return indices


def _rule_deadline_artifact(
    series: Mapping[int, ProbeSeries],
    reference: ReferenceBands,
    thresholds: StateThresholds,
    hold_steps: int,
) -> Tuple[bool, Dict[str, object]]:
    low_c, high_c = plateau_bounds(reference, thresholds)
    low_q, high_q = q_star_bounds(reference, thresholds)
    diagnostics: Dict[str, object] = {
        "suppressed_band": [low_c, high_c],
        "q_star_band": [low_q, high_q],
        "window_start": hold_steps - thresholds.suppressed_duration,
    }
    verdicts = {}
    for probe, data in series.items():
        indices = _final_window(data, hold_steps, thresholds.suppressed_duration)
        if indices is None:
            verdicts[probe] = False
            diagnostics[f"probe_{probe}_window"] = "uncovered"
            continue
        inside = all(
            low_c <= data.c_int[index] <= high_c and low_q <= data.full_vocab_ce[index] <= high_q
            for index in indices
        )
        verdicts[probe] = inside
        diagnostics[f"probe_{probe}_window"] = {
            "rows": len(indices),
            "jointly_inside": inside,
            "first_c_int": data.c_int[indices[0]],
            "last_c_int": data.c_int[indices[-1]],
            "last_full_vocab_ce": data.full_vocab_ce[indices[-1]],
        }
    return all(verdicts.values()), diagnostics


def _rule_slow_erasure(
    series: Mapping[int, ProbeSeries],
    reference: ReferenceBands,
    thresholds: StateThresholds,
    hold_steps: int,
) -> Tuple[bool, Dict[str, object]]:
    _, high_c = plateau_bounds(reference, thresholds)
    diagnostics: Dict[str, object] = {"suppressed_upper_edge": high_c}
    verdicts = {}
    for probe, data in series.items():
        first = _quarter_indices(data.steps, hold_steps, 1)
        final = _quarter_indices(data.steps, hold_steps, 4)
        if not first or not final:
            raise ValueError("quarter windows are empty; sealed log is malformed")
        first_mean = _mean([data.c_int[index] for index in first])
        final_mean = _mean([data.c_int[index] for index in final])
        halved = final_mean < 0.5 * first_mean
        above = final_mean > high_c
        verdicts[probe] = halved and above
        diagnostics[f"probe_{probe}"] = {
            "first_quarter_mean": first_mean,
            "final_quarter_mean": final_mean,
            "halved": halved,
            "above_band": above,
        }
    return all(verdicts.values()), diagnostics


def _rule_intermediate(
    series: Mapping[int, ProbeSeries],
    reference: ReferenceBands,
    thresholds: StateThresholds,
    hold_steps: int,
) -> Tuple[bool, Dict[str, object]]:
    _, high_c = plateau_bounds(reference, thresholds)
    low_q, high_q = q_star_bounds(reference, thresholds)
    tolerance = reference.plateau_sd
    ceiling = thresholds.expressed_fraction * reference.solved_c_int
    diagnostics: Dict[str, object] = {
        "slope_tolerance_per_1000": tolerance,
        "level_floor": high_c,
        "level_ceiling": ceiling,
        "q_star_band": [low_q, high_q],
    }
    verdicts = {}
    for probe, data in series.items():
        final = _quarter_indices(data.steps, hold_steps, 4)
        steps = [data.steps[index] for index in final]
        values = [data.c_int[index] for index in final]
        ce_values = [data.full_vocab_ce[index] for index in final]
        slope_per_step, _ = _ols_line(steps, values)
        slope_per_1000 = slope_per_step * 1_000.0
        level = _mean(values)
        ce_level = _mean(ce_values)
        flat = abs(slope_per_1000) < tolerance
        between = high_c < level < ceiling
        ce_inside = low_q <= ce_level <= high_q
        verdicts[probe] = flat and between and ce_inside
        diagnostics[f"probe_{probe}"] = {
            "slope_per_1000": slope_per_1000,
            "final_level": level,
            "final_ce_level": ce_level,
            "flat": flat,
            "between_bands": between,
            "ce_inside": ce_inside,
        }
    return all(verdicts.values()), diagnostics


def _excerpt(rows: Sequence[Mapping[str, float]], steps: Sequence[float]) -> List[dict]:
    wanted = set(steps)
    keys = (
        "branch_step",
        "probe_0_c_int",
        "probe_1_c_int",
        "probe_0_full_vocab_ce",
        "probe_1_full_vocab_ce",
    )
    return [
        {key: float(row[key]) for key in keys}
        for row in rows
        if float(row["branch_step"]) in wanted
    ]


def classify_cell(
    learning_rate: float,
    rows: Sequence[Mapping[str, float]],
    reference: ReferenceBands,
    thresholds: StateThresholds,
    hold_steps: int,
) -> CellAutopsy:
    _validate_rows(rows, hold_steps)
    series = {probe: _probe_series(rows, probe) for probe in PROBES}
    rules = (
        ("near_divergent", lambda: _rule_near_divergent(series, reference)),
        (
            "deadline_artifact",
            lambda: _rule_deadline_artifact(series, reference, thresholds, hold_steps),
        ),
        (
            "slow_erasure",
            lambda: _rule_slow_erasure(series, reference, thresholds, hold_steps),
        ),
        (
            "intermediate",
            lambda: _rule_intermediate(series, reference, thresholds, hold_steps),
        ),
    )
    diagnostics: Dict[str, object] = {}
    label = "stochastic"
    fired_rule = "none_of_the_above"
    for name, evaluate in rules:
        fired, rule_diagnostics = evaluate()
        diagnostics[name] = {"fired": fired, **rule_diagnostics}
        if fired:
            label = name
            fired_rule = name
            break
    excerpt_steps: List[float] = []
    if label == "near_divergent":
        for probe in PROBES:
            row = diagnostics["near_divergent"][f"probe_{probe}_first_divergent_row"]
            if row is not None:
                excerpt_steps.append(row["branch_step"])
    else:
        first = _quarter_indices(series[0].steps, hold_steps, 1)
        final = _quarter_indices(series[0].steps, hold_steps, 4)
        excerpt_steps = [
            series[0].steps[first[0]],
            series[0].steps[first[-1]],
            series[0].steps[final[0]],
            series[0].steps[final[-1]],
        ]
    return CellAutopsy(
        learning_rate=learning_rate,
        label=label,
        fired_rule=fired_rule,
        diagnostics=diagnostics,
        excerpts={"rows": _excerpt(rows, excerpt_steps)},
    )


def slow_erasure_crossing_step(
    rows: Sequence[Mapping[str, float]],
    reference: ReferenceBands,
    thresholds: StateThresholds,
    hold_steps: int,
) -> float:
    """Later of the two probes' extrapolated band crossings (may be inf)."""
    _, high_c = plateau_bounds(reference, thresholds)
    crossings = []
    for probe in PROBES:
        data = _probe_series(rows, probe)
        final = _quarter_indices(data.steps, hold_steps, 4)
        steps = [data.steps[index] for index in final]
        values = [data.c_int[index] for index in final]
        slope, intercept = _ols_line(steps, values)
        if slope >= 0.0:
            crossings.append(math.inf)
            continue
        crossing = (high_c - intercept) / slope
        crossings.append(crossing if crossing > steps[-1] else steps[-1])
    return max(crossings)


def escalated_hold_steps(crossing_steps: Sequence[float]) -> int:
    if any(step > ESCALATION_CROSSING_STEP for step in crossing_steps):
        return ESCALATED_HOLD_STEPS
    return BASE_HOLD_STEPS


def select_branches(
    labels: Mapping[float, str],
    crossing_steps: Sequence[float] = (),
) -> BranchActivation:
    for rate, label in labels.items():
        if label not in AUTOPSY_LABELS:
            raise ValueError(f"unknown autopsy label {label!r} at rate {rate}")
    reasons: List[str] = []
    selecting = {
        rate: label for rate, label in labels.items() if label != "near_divergent"
    }
    if any(label == "stochastic" for label in selecting.values()):
        primary = "A"
        reasons.append("at least one cell is stochastic")
    elif selecting and all(
        label in {"slow_erasure", "deadline_artifact"} for label in selecting.values()
    ):
        primary = "B"
        reasons.append(
            "every branch-selecting cell is slow_erasure or deadline_artifact"
        )
    else:
        primary = "A"
        reasons.append(
            "completion rule: no stochastic cell, but branch-selecting labels are "
            "not all in {slow_erasure, deadline_artifact}"
        )
    include_c = any(label == "intermediate" for label in labels.values())
    if include_c:
        reasons.append("at least one cell is intermediate: Branch C attaches")
    grid_cap = None
    top_rate = max(labels)
    if labels[top_rate] == "near_divergent":
        grid_cap = GRID_CAP_ON_DIVERGENT_TOP
        reasons.append(
            f"top rate {top_rate} is near_divergent: future grids capped at "
            f"{GRID_CAP_ON_DIVERGENT_TOP}"
        )
    return BranchActivation(
        primary_branch=primary,
        include_branch_c=include_c,
        grid_cap=grid_cap,
        t_hold_steps=escalated_hold_steps(crossing_steps),
        labels={f"{rate:g}": label for rate, label in sorted(labels.items())},
        reasons=tuple(reasons),
    )


def _branch_label(learning_rate: float) -> str:
    return "eta_" + f"{learning_rate:g}".replace(".", "p").replace("-", "m")


def _load_cell_rows(cell_dir: Path) -> List[dict]:
    rows = []
    with (cell_dir / "metrics.jsonl").open() as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("kind") == "gate0_erasure":
                rows.append(row)
    return rows


def run_autopsy(
    scan_dir: Path,
    reference_path: Path,
    output_path: Path,
    *,
    hold_steps: int = 8_000,
    target_rates: Sequence[float] = TARGET_RATES,
    thresholds: Optional[StateThresholds] = None,
) -> dict:
    """Classify the sealed unresolved cells and write the bound artifact."""
    scan_dir = Path(scan_dir)
    reference_path = Path(reference_path)
    effective_thresholds = thresholds or StateThresholds()
    reference = load_reference_bands(reference_path)
    scan = json.loads((scan_dir / "scan.json").read_text())
    by_rate = {branch["learning_rate"]: branch for branch in scan["branches"]}
    bindings = {
        "scan": bind_file(scan_dir / "scan.json"),
        "reference": bind_file(reference_path),
        "cells": {},
    }
    cells: Dict[float, CellAutopsy] = {}
    crossing_steps: List[float] = []
    for rate in sorted(target_rates):
        sealed = by_rate.get(rate)
        if sealed is None:
            raise ValueError(f"sealed scan has no branch at rate {rate}")
        if sealed["outcome"] != "unresolved":
            raise ValueError(
                f"amendment targets unresolved cells only; rate {rate} is "
                f"{sealed['outcome']!r}"
            )
        cell_dir = scan_dir / _branch_label(rate)
        bindings["cells"][f"{rate:g}"] = {
            "metrics": bind_file(cell_dir / "metrics.jsonl"),
            "result": bind_file(cell_dir / "result.json"),
        }
        rows = _load_cell_rows(cell_dir)
        cell = classify_cell(rate, rows, reference, effective_thresholds, hold_steps)
        cells[rate] = cell
        if cell.label == "slow_erasure":
            crossing_steps.append(
                slow_erasure_crossing_step(
                    rows, reference, effective_thresholds, hold_steps
                )
            )
    activation = select_branches(
        {rate: cell.label for rate, cell in cells.items()}, crossing_steps
    )
    report = {
        "schema_version": 1,
        "kind": "gate0_autopsy_v1_5",
        "amendment": "AMENDMENT_v1_5.md",
        "divergence_multiplier": DIVERGENCE_MULTIPLIER,
        "hold_steps": hold_steps,
        "thresholds": asdict(effective_thresholds),
        "reference_bands": asdict(reference),
        "slow_erasure_crossing_steps": [
            None if math.isinf(step) else step for step in crossing_steps
        ],
        "cells": {f"{rate:g}": asdict(cell) for rate, cell in sorted(cells.items())},
        "activation": asdict(activation),
        "inputs": bindings,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=1, sort_keys=True, allow_nan=False))
    temporary.replace(output_path)
    return report
