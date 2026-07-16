"""Dwell/recrossing audit: is "basin" honest language for these trajectories?

NON-DECISION-BEARING. This module registers no gate rule and alters none
(v1.5.3 freeze intact). It is the interpretation audit recommended by the
2026-07-16 external triage: from existing metric logs it measures whether
the trajectories show the timescale separation that justifies basin
language — long dwells in the expressed or suppressed region, short
transitions between them, few threshold-grazing recrossings.

Region labels per logged row:

- ``S`` (suppressed): jointly inside the suppressed C_int band and the
  empirical q* CE band on BOTH probes (the escape-runner convention).
- ``E`` (expressed): the registered expressed criteria on the aggregate row.
- ``B`` (between): neither.

Episodes are maximal same-label runs. A ``B`` run bounded by different
regions is a transition; bounded by the same region it is a graze. Runs
touching the trajectory ends are censored and reported separately. The
sustained-window sensitivity reports first sustained-suppression entries at
window lengths {1,000, 2,000, 4,000} without altering the registered 2,000.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from .gate0_boundary import _atomic_write_json
from .gate0e_escape import _row_inside, sustained_suppression_entry
from .gate0_autopsy import q_star_bounds
from .references import load_reference_bands
from .provenance import bind_file
from .state import ReferenceBands, StateThresholds, is_expressed, plateau_bounds

WINDOW_LADDER = (1_000, 2_000, 4_000)


def label_row(
    row: Mapping[str, float],
    reference: ReferenceBands,
    thresholds: StateThresholds,
) -> str:
    c_bounds = plateau_bounds(reference, thresholds)
    q_bounds = q_star_bounds(reference, thresholds)
    if _row_inside(row, c_bounds, q_bounds):
        return "S"
    if is_expressed(
        float(row["c_int"]),
        float(row["exact_match"]),
        float(row["delta_z"]),
        reference,
        thresholds,
    ):
        return "E"
    return "B"


def episodes(
    rows: Sequence[Mapping[str, float]],
    reference: ReferenceBands,
    thresholds: StateThresholds,
) -> List[dict]:
    """Maximal same-label runs with spans in steps."""
    result: List[dict] = []
    for row in rows:
        label = label_row(row, reference, thresholds)
        step = float(row["branch_step"])
        if result and result[-1]["label"] == label:
            result[-1]["end"] = step
        else:
            result.append({"label": label, "start": step, "end": step})
    for index, episode in enumerate(result):
        episode["span"] = episode["end"] - episode["start"]
        episode["censored"] = index == 0 or index == len(result) - 1
    return result


def audit_trajectory(
    rows: Sequence[Mapping[str, float]],
    reference: ReferenceBands,
    thresholds: StateThresholds,
) -> dict:
    if not rows:
        raise ValueError("dwell audit requires a nonempty trajectory")
    runs = episodes(rows, reference, thresholds)
    dwells = {"E": [], "S": []}
    censored_dwells = {"E": [], "S": []}
    transitions: List[float] = []
    grazes: List[float] = []
    for index, run in enumerate(runs):
        if run["label"] in ("E", "S"):
            target = censored_dwells if run["censored"] else dwells
            target[run["label"]].append(run["span"])
            continue
        if run["censored"]:
            continue
        before = runs[index - 1]["label"] if index > 0 else None
        after = runs[index + 1]["label"] if index + 1 < len(runs) else None
        if before in ("E", "S") and after in ("E", "S"):
            if before == after:
                grazes.append(run["span"])
            else:
                transitions.append(run["span"])
    region_sequence = [run["label"] for run in runs if run["label"] in ("E", "S")]
    recrossings = sum(
        1 for a, b in zip(region_sequence, region_sequence[1:]) if a != b
    )
    window_entries = {}
    for window in WINDOW_LADDER:
        entry = sustained_suppression_entry(
            rows, reference, replace(thresholds, suppressed_duration=window)
        )
        window_entries[str(window)] = entry
    return {
        "rows": len(rows),
        "final_step": float(rows[-1]["branch_step"]),
        "episodes": runs,
        "completed_dwells": dwells,
        "censored_dwells": censored_dwells,
        "transition_spans": transitions,
        "graze_spans": grazes,
        "recrossings": recrossings,
        "sustained_entry_by_window": window_entries,
    }


def _summary(values: Sequence[float]) -> Optional[dict]:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "median": statistics.median(ordered),
        "min": ordered[0],
        "max": ordered[-1],
    }


def audit_cells(
    cell_logs: Mapping[str, Sequence[Mapping[str, float]]],
    reference: ReferenceBands,
    thresholds: StateThresholds,
) -> dict:
    per_cell = {}
    all_dwells: List[float] = []
    all_censored: List[float] = []
    all_transitions: List[float] = []
    all_grazes: List[float] = []
    total_recrossings = 0
    for name, rows in sorted(cell_logs.items()):
        report = audit_trajectory(rows, reference, thresholds)
        per_cell[name] = report
        all_dwells.extend(report["completed_dwells"]["E"])
        all_dwells.extend(report["completed_dwells"]["S"])
        all_censored.extend(report["censored_dwells"]["E"])
        all_censored.extend(report["censored_dwells"]["S"])
        all_transitions.extend(report["transition_spans"])
        all_grazes.extend(report["graze_spans"])
        total_recrossings += report["recrossings"]
    dwell_summary = _summary(all_dwells)
    censored_summary = _summary(all_censored)
    transition_summary = _summary(all_transitions)
    ratio = None
    if dwell_summary and transition_summary and transition_summary["median"] > 0:
        ratio = dwell_summary["median"] / transition_summary["median"]
    ratio_lower_bound = None
    if censored_summary and transition_summary and transition_summary["median"] > 0:
        ratio_lower_bound = censored_summary["median"] / transition_summary["median"]
    crossings = len(all_transitions)
    graze_fraction = (
        len(all_grazes) / (len(all_grazes) + crossings)
        if (all_grazes or crossings)
        else None
    )
    return {
        "schema_version": 1,
        "kind": "gate0e_dwell_audit",
        "decision_bearing": False,
        "trajectories": len(per_cell),
        "completed_dwell_summary": dwell_summary,
        "censored_dwell_summary": censored_summary,
        "transition_span_summary": transition_summary,
        "dwell_to_transition_ratio": ratio,
        "censored_dwell_ratio_lower_bound": ratio_lower_bound,
        "graze_fraction": graze_fraction,
        "total_recrossings": total_recrossings,
        "per_cell": per_cell,
    }


def load_rows(path: Path, kinds: Sequence[str]) -> List[dict]:
    rows = []
    with Path(path).open() as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("kind") in kinds:
                rows.append(row)
    return rows


def run_dwell_audit(
    metrics_paths: Mapping[str, Path],
    reference_path: Path,
    output_path: Path,
    *,
    kinds: Sequence[str] = ("gate0_erasure", "gate0e_escape"),
    thresholds: Optional[StateThresholds] = None,
) -> dict:
    thresholds = thresholds or StateThresholds()
    reference = load_reference_bands(reference_path)
    cell_logs = {
        name: load_rows(path, kinds) for name, path in metrics_paths.items()
    }
    report = audit_cells(cell_logs, reference, thresholds)
    report["inputs"] = {
        "reference": bind_file(Path(reference_path)),
        "metrics": {name: bind_file(Path(path)) for name, path in metrics_paths.items()},
    }
    _atomic_write_json(Path(output_path), report)
    return report
