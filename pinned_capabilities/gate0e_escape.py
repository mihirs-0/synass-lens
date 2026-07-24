"""Gate 0-E escape curves: multi-stream erasure holds under Branch A.

Amendment v1.5 section 2 replaces the boundary rate with an escape curve.
From one frozen expressed snapshot, each (rate, stream) cell reloads the
identical weights, optimizer state, and RNG, replaces only the data-order
generator with a fresh deterministically derived seed, and holds for
``t_hold_steps`` at the challenge rate.

Frozen conventions carried from the autopsy activation record:

- **erased** fires on the first window of at least ``suppressed_duration``
  consecutive logged steps that is jointly inside the suppressed C_int band
  and the empirical q* CE band on BOTH probes; tau is the window's first
  logged step; the label is irreversible (re-expression later in the hold
  does not unmark it).
- **diverged** uses the registered v1.4.1 rule on the aggregate answer CE
  (non-finite or above 5.0x the q* mean) at any logged row strictly before
  the sustained window's entry step; a rate with >= 50% diverged streams is
  ``unstable`` and never enters fits.
- **retained** requires the expressed criteria on the final row and no
  sustained suppression anywhere.
- **unresolved** is everything else and counts as not-erased in the primary
  fraction; the sensitivity table reports the opposite convention.

The old 2,000-step onset deadline does not exist here. Streams are
embarrassingly parallel and every stream checkpoint is transactional, so a
killed scan resumes without rerunning completed work.
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from .config import MBCExperimentConfig, MetricConfig
from .experiment import JSONLWriter, MBCExperiment
from .batch_stream import DeterministicBatchStream
from .gate0_boundary import (
    _atomic_write_json,
    _atomic_write_jsonl,
    _branch_label,
    _load_bound_metrics_prefix,
    _metrics_prefix_binding,
    _reference_sha256,
)
from .gate0e_stats import (
    RateOutcomes,
    arrhenius_fit,
    bootstrap_eta50_ci,
    curve_valid,
    fit_cells,
    logistic_fit,
    monotonicity_flags,
)
from .parameter_groups import set_learning_rates
from .provenance import sha256_file
from .snapshot import restore_rng_state, save_snapshot
from .state import (
    ReferenceBands,
    StateThresholds,
    is_expressed,
    plateau_bounds,
)
from .gate0_autopsy import q_star_bounds

DIVERGENCE_MULTIPLIER = 5.0
GATE0E_GRID = (0.003, 0.005, 0.008, 0.0125, 0.02, 0.032, 0.05)
GATE0E_STREAMS = 8
GATE0E_T_HOLD = 16_000
CHECKPOINT_EVERY = 1_000


def stream_data_seed(
    seed: int, learning_rate: float, stream_index: int, batch_size: int
) -> int:
    key = (
        f"gate0e_stream|seed={seed}|rate={learning_rate:.10g}"
        f"|stream={stream_index}|batch={batch_size}"
    )
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")


def load_snapshot_fresh_stream(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    map_location: str | torch.device = "cpu",
) -> Dict[str, object]:
    """Identical weights, optimizer state, and RNG; the data stream is NOT
    restored — Branch A gives every stream a fresh data-order generator."""
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported snapshot schema: {payload.get('schema_version')}")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    restore_rng_state(payload["rng"])
    return {"step": int(payload["step"]), "metadata": payload.get("metadata", {})}


def set_weight_decay(optimizer: torch.optim.Optimizer, weight_decay: float) -> None:
    if weight_decay < 0:
        raise ValueError("weight decay must be nonnegative")
    for group in optimizer.param_groups:
        group["weight_decay"] = weight_decay


@dataclass(frozen=True)
class StreamResult:
    learning_rate: float
    stream_index: int
    batch_size: int
    weight_decay: float
    outcome: str
    tau: Optional[int]
    first_divergence_step: Optional[int]
    final_c_int: float
    final_exact_match: float
    final_delta_z: float
    final_full_vocab_ce: float
    hold_steps: int
    stream_seed: int


def _row_inside(
    row: Mapping[str, float],
    c_bounds: Tuple[float, float],
    q_bounds: Tuple[float, float],
) -> bool:
    for probe in (0, 1):
        c_int = float(row[f"probe_{probe}_c_int"])
        ce = float(row[f"probe_{probe}_full_vocab_ce"])
        if not (c_bounds[0] <= c_int <= c_bounds[1] and q_bounds[0] <= ce <= q_bounds[1]):
            return False
    return True


def sustained_suppression_entry(
    rows: Sequence[Mapping[str, float]],
    reference: ReferenceBands,
    thresholds: StateThresholds,
) -> Optional[int]:
    """First step of the earliest joint per-probe window spanning the
    registered suppressed duration; None when no window ever completes."""
    c_bounds = plateau_bounds(reference, thresholds)
    q_bounds = q_star_bounds(reference, thresholds)
    run_start: Optional[int] = None
    for index, row in enumerate(rows):
        if _row_inside(row, c_bounds, q_bounds):
            if run_start is None:
                run_start = index
            span = float(row["branch_step"]) - float(rows[run_start]["branch_step"])
            if span >= thresholds.suppressed_duration:
                return int(float(rows[run_start]["branch_step"]))
        else:
            run_start = None
    return None


def first_divergence_step(
    rows: Sequence[Mapping[str, float]], reference: ReferenceBands
) -> Optional[int]:
    threshold = DIVERGENCE_MULTIPLIER * reference.q_star_loss_mean
    for row in rows:
        ce = float(row["full_vocab_ce"])
        if not math.isfinite(ce) or ce > threshold:
            return int(float(row["branch_step"]))
    return None


def classify_stream(
    rows: Sequence[Mapping[str, float]],
    reference: ReferenceBands,
    thresholds: StateThresholds,
    hold_steps: int,
) -> Dict[str, object]:
    if not rows or int(float(rows[-1]["branch_step"])) != hold_steps:
        raise ValueError("stream log does not reach the registered hold")
    tau = sustained_suppression_entry(rows, reference, thresholds)
    divergence = first_divergence_step(rows, reference)
    latest = rows[-1]
    if tau is not None and (divergence is None or divergence >= tau):
        outcome = "erased"
    elif divergence is not None:
        outcome = "diverged"
    elif is_expressed(
        float(latest["c_int"]),
        float(latest["exact_match"]),
        float(latest["delta_z"]),
        reference,
        thresholds,
    ):
        outcome = "retained"
    else:
        outcome = "unresolved"
    return {
        "outcome": outcome,
        "tau": tau if outcome == "erased" else None,
        "first_divergence_step": divergence,
    }


def run_escape_stream(
    experiment_config: MBCExperimentConfig,
    metric: MetricConfig,
    reference: ReferenceBands,
    snapshot_path: Path,
    learning_rate: float,
    stream_index: int,
    output_dir: Path,
    *,
    hold_steps: int = GATE0E_T_HOLD,
    weight_decay_override: Optional[float] = None,
    thresholds: Optional[StateThresholds] = None,
    torch_threads: Optional[int] = None,
) -> StreamResult:
    if torch_threads is not None:
        torch.set_num_threads(int(torch_threads))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = thresholds or StateThresholds()
    result_path = output_dir / "result.json"
    weight_decay = (
        experiment_config.weight_decay
        if weight_decay_override is None
        else weight_decay_override
    )
    stream_seed = stream_data_seed(
        experiment_config.seed,
        learning_rate,
        stream_index,
        experiment_config.batch_size,
    )
    expected_controls = {
        "learning_rate": learning_rate,
        "stream_index": stream_index,
        "batch_size": experiment_config.batch_size,
        "weight_decay": weight_decay,
        "hold_steps": hold_steps,
        "stream_seed": stream_seed,
        "snapshot_sha256": sha256_file(snapshot_path),
        "reference_sha256": _reference_sha256(reference),
    }
    if result_path.exists():
        completed = json.loads(result_path.read_text())
        for key, value in expected_controls.items():
            if completed.get("controls", {}).get(key) != value:
                raise ValueError(f"completed stream has mismatched control {key}")
        payload = dict(completed)
        payload.pop("controls", None)
        return StreamResult(**payload)

    run_config = replace(
        experiment_config,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    experiment = MBCExperiment(run_config, metric)
    progress_path = output_dir / "progress.json"
    metrics_path = output_dir / "metrics.jsonl"
    rows: List[dict] = []
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        for key, value in expected_controls.items():
            if progress.get("controls", {}).get(key) != value:
                raise ValueError(f"stream progress has mismatched control {key}")
        checkpoint_path = output_dir / progress["checkpoint_path"]
        if sha256_file(checkpoint_path) != progress["checkpoint_sha256"]:
            raise ValueError("stream progress checkpoint digest mismatch")
        payload = torch.load(checkpoint_path, map_location=experiment.device, weights_only=False)
        experiment.model.load_state_dict(payload["model"])
        experiment.optimizer.load_state_dict(payload["optimizer"])
        experiment.stream = DeterministicBatchStream(
            len(experiment.dataset), run_config.batch_size, seed=stream_seed
        )
        experiment.stream.load_state_dict(payload["stream"])
        restore_rng_state(payload["rng"])
        experiment.step = int(payload["step"])
        branch_start = int(progress["branch_start"])
        completed_steps = int(progress["completed_branch_steps"])
        active_slot = int(progress["active_slot"])
        if experiment.step - branch_start != completed_steps:
            raise ValueError("stream checkpoint and progress disagree")
        logged = _load_bound_metrics_prefix(
            metrics_path, progress.get("metrics_prefix"), label="gate0e"
        )
        logged = [row for row in logged if int(float(row["branch_step"])) <= completed_steps]
        if not logged or int(float(logged[-1]["branch_step"])) != completed_steps:
            raise ValueError("stream metrics do not reach the progress checkpoint")
        _atomic_write_jsonl(metrics_path, logged)
        rows = logged
    else:
        if metrics_path.exists():
            _atomic_write_jsonl(metrics_path, [])
        restored = load_snapshot_fresh_stream(
            snapshot_path,
            model=experiment.model,
            optimizer=experiment.optimizer,
            map_location=experiment.device,
        )
        experiment.step = restored["step"]
        experiment.stream = DeterministicBatchStream(
            len(experiment.dataset), run_config.batch_size, seed=stream_seed
        )
        branch_start = experiment.step
        active_slot = -1
    set_learning_rates(experiment.optimizer, learning_rate)
    set_weight_decay(experiment.optimizer, weight_decay)
    writer = JSONLWriter(metrics_path)
    while experiment.step - branch_start < hold_steps:
        chunk = min(metric.eval_every, hold_steps - (experiment.step - branch_start))
        training = experiment.advance(chunk)
        row = {**training, **experiment.evaluate()}
        row["branch_step"] = float(experiment.step - branch_start)
        row["learning_rate"] = learning_rate
        rows.append(row)
        writer.write({"kind": "gate0e_escape", **row})
        branch_step = int(row["branch_step"])
        if branch_step % CHECKPOINT_EVERY == 0 or branch_step == hold_steps:
            active_slot = 1 if active_slot != 1 else 0
            checkpoint_path = output_dir / "checkpoints" / f"slot_{active_slot}.pt"
            temporary = checkpoint_path.with_suffix(".pt.tmp")
            save_snapshot(
                temporary,
                model=experiment.model,
                optimizer=experiment.optimizer,
                stream=experiment.stream,
                step=experiment.step,
                metadata={
                    "kind": "gate0e_stream_progress",
                    "learning_rate": learning_rate,
                    "stream_index": stream_index,
                    "branch_start": branch_start,
                },
            )
            temporary.replace(checkpoint_path)
            _atomic_write_json(
                progress_path,
                {
                    "controls": expected_controls,
                    "branch_start": branch_start,
                    "completed_branch_steps": branch_step,
                    "active_slot": active_slot,
                    "checkpoint_path": str(checkpoint_path.relative_to(output_dir)),
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                    "metrics_prefix": _metrics_prefix_binding(metrics_path),
                },
            )
    verdict = classify_stream(rows, reference, thresholds, hold_steps)
    latest = rows[-1]
    result = StreamResult(
        learning_rate=learning_rate,
        stream_index=stream_index,
        batch_size=run_config.batch_size,
        weight_decay=weight_decay,
        outcome=str(verdict["outcome"]),
        tau=verdict["tau"],
        first_divergence_step=verdict["first_divergence_step"],
        final_c_int=float(latest["c_int"]),
        final_exact_match=float(latest["exact_match"]),
        final_delta_z=float(latest["delta_z"]),
        final_full_vocab_ce=float(latest["full_vocab_ce"]),
        hold_steps=hold_steps,
        stream_seed=stream_seed,
    )
    _atomic_write_json(
        result_path, {**asdict(result), "controls": expected_controls}
    )
    return result


def _stream_worker(
    experiment_config: MBCExperimentConfig,
    metric: MetricConfig,
    reference: ReferenceBands,
    snapshot_path: str,
    learning_rate: float,
    stream_index: int,
    output_dir: str,
    hold_steps: int,
    weight_decay_override: Optional[float],
    thresholds: StateThresholds,
    torch_threads: int,
) -> dict:
    result = run_escape_stream(
        experiment_config,
        metric,
        reference,
        Path(snapshot_path),
        learning_rate,
        stream_index,
        Path(output_dir),
        hold_steps=hold_steps,
        weight_decay_override=weight_decay_override,
        thresholds=thresholds,
        torch_threads=torch_threads,
    )
    return asdict(result)


def aggregate_cell(streams: Sequence[StreamResult]) -> dict:
    if not streams:
        raise ValueError("cell aggregation requires stream results")
    rates = {stream.learning_rate for stream in streams}
    if len(rates) != 1:
        raise ValueError("cell aggregation mixes learning rates")
    counts = {"erased": 0, "retained": 0, "unresolved": 0, "diverged": 0}
    for stream in streams:
        counts[stream.outcome] += 1
    outcomes = RateOutcomes(
        learning_rate=streams[0].learning_rate,
        erased=counts["erased"],
        retained=counts["retained"],
        unresolved=counts["unresolved"],
        diverged=counts["diverged"],
    )
    taus = sorted(
        stream.tau for stream in streams if stream.outcome == "erased" and stream.tau
    )
    labels = [
        1 if stream.outcome == "erased" else 0
        for stream in streams
        if stream.outcome != "diverged"
    ]
    return {
        "learning_rate": streams[0].learning_rate,
        "counts": counts,
        "unstable": outcomes.unstable,
        "primary_fraction": outcomes.primary_fraction(),
        "sensitivity_fraction": outcomes.sensitivity_fraction(),
        "observed_taus": taus,
        "median_observed_tau": float(taus[len(taus) // 2]) if taus else None,
        "stream_labels": labels,
        "streams": [asdict(stream) for stream in streams],
    }


def curve_statistics(cells: Sequence[dict]) -> dict:
    outcome_cells = [
        RateOutcomes(
            learning_rate=cell["learning_rate"],
            erased=cell["counts"]["erased"],
            retained=cell["counts"]["retained"],
            unresolved=cell["counts"]["unresolved"],
            diverged=cell["counts"]["diverged"],
        )
        for cell in cells
    ]
    eligible = fit_cells(outcome_cells)
    valid = curve_valid(outcome_cells)
    statistics: dict = {
        "curve_valid": valid,
        "monotonicity_flags": monotonicity_flags(outcome_cells),
        "logistic": None,
        "eta50": None,
        "eta50_ci": None,
        "arrhenius": None,
    }
    fractions = [cell.primary_fraction() for cell in eligible]
    varied = any(f not in (0.0, None) for f in fractions) and any(
        f not in (1.0, None) for f in fractions
    )
    if len(eligible) >= 2 and varied:
        fit = logistic_fit(eligible)
        statistics["logistic"] = {
            "intercept": fit.intercept,
            "slope": fit.slope,
            "converged": fit.converged,
            "separation": fit.separation,
        }
        statistics["eta50"] = fit.eta50
        labels = {
            cell["learning_rate"]: cell["stream_labels"]
            for cell in cells
            if not cell["unstable"]
        }
        ci = bootstrap_eta50_ci(eligible, labels)
        statistics["eta50_ci"] = asdict(ci)
    median_taus = {
        cell["learning_rate"]: cell["median_observed_tau"]
        for cell in cells
        if cell["median_observed_tau"] is not None
    }
    fit = arrhenius_fit(eligible, median_taus)
    if fit is not None:
        statistics["arrhenius"] = asdict(fit)
    return statistics


def run_escape_curve(
    experiment_config: MBCExperimentConfig,
    metric: MetricConfig,
    reference: ReferenceBands,
    snapshot_path: Path,
    output_dir: Path,
    *,
    learning_rates: Sequence[float] = GATE0E_GRID,
    streams: int = GATE0E_STREAMS,
    hold_steps: int = GATE0E_T_HOLD,
    weight_decay_override: Optional[float] = None,
    thresholds: Optional[StateThresholds] = None,
    workers: int = 1,
    torch_threads: int = 2,
) -> dict:
    rates = sorted(float(rate) for rate in learning_rates)
    if len(set(rates)) != len(rates) or any(rate <= 0 for rate in rates):
        raise ValueError("escape grid must be positive and unique")
    if streams < 1 or workers < 1:
        raise ValueError("streams and workers must be positive")
    output_dir = Path(output_dir)
    thresholds = thresholds or StateThresholds()
    jobs = []
    for rate in rates:
        for stream_index in range(streams):
            stream_dir = output_dir / _branch_label(rate) / f"stream_{stream_index:02d}"
            jobs.append((rate, stream_index, stream_dir))
    pending = [job for job in jobs if not (job[2] / "result.json").exists()]
    if workers > 1 and len(pending) > 1:
        executor = ProcessPoolExecutor(
            max_workers=min(workers, len(pending)),
            mp_context=multiprocessing.get_context("spawn"),
        )
        try:
            futures = [
                executor.submit(
                    _stream_worker,
                    experiment_config,
                    metric,
                    reference,
                    str(snapshot_path),
                    rate,
                    stream_index,
                    str(stream_dir),
                    hold_steps,
                    weight_decay_override,
                    thresholds,
                    torch_threads,
                )
                for rate, stream_index, stream_dir in pending
            ]
            for future in as_completed(futures):
                future.result()
        finally:
            executor.shutdown(wait=True, cancel_futures=False)
    else:
        for rate, stream_index, stream_dir in pending:
            run_escape_stream(
                experiment_config,
                metric,
                reference,
                snapshot_path,
                rate,
                stream_index,
                stream_dir,
                hold_steps=hold_steps,
                weight_decay_override=weight_decay_override,
                thresholds=thresholds,
                torch_threads=None,
            )
    snapshot_sha256 = sha256_file(snapshot_path)
    reference_sha256 = _reference_sha256(reference)
    cells = []
    for rate in rates:
        stream_results = []
        for stream_index in range(streams):
            stream_dir = output_dir / _branch_label(rate) / f"stream_{stream_index:02d}"
            payload = json.loads((stream_dir / "result.json").read_text())
            controls = payload.pop("controls", {})
            expected_seed = stream_data_seed(
                experiment_config.seed, rate, stream_index, experiment_config.batch_size
            )
            if (
                controls.get("snapshot_sha256") != snapshot_sha256
                or controls.get("reference_sha256") != reference_sha256
                or controls.get("stream_seed") != expected_seed
                or controls.get("hold_steps") != hold_steps
            ):
                raise ValueError(
                    f"stream {stream_dir} result does not match this curve's controls"
                )
            stream_results.append(StreamResult(**payload))
        cells.append(aggregate_cell(stream_results))
    summary = {
        "schema_version": 1,
        "kind": "gate0e_escape_curve",
        "seed": experiment_config.seed,
        "batch_size": experiment_config.batch_size,
        "weight_decay": (
            experiment_config.weight_decay
            if weight_decay_override is None
            else weight_decay_override
        ),
        "hold_steps": hold_steps,
        "learning_rates": rates,
        "streams_per_rate": streams,
        "snapshot_sha256": sha256_file(snapshot_path),
        "reference_sha256": _reference_sha256(reference),
        "cells": cells,
        "statistics": curve_statistics(cells),
    }
    _atomic_write_json(output_dir / "curve.json", summary)
    return summary
