"""Resumable two-cycle Gate 1 hysteresis experiment."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from .config import Gate1Config, MBCExperimentConfig, MetricConfig
from .experiment import JSONLWriter, MBCExperiment
from .gate1 import DwellResult, geometric_levels, run_dwell
from .snapshot import load_snapshot, save_snapshot
from .state import ReferenceBands, StateThresholds


def _schedule(levels: tuple[float, ...], cycles: int) -> list[dict]:
    schedule = []
    for cycle in range(1, cycles + 1):
        schedule.extend(
            {"cycle": cycle, "direction": "down", "learning_rate": value}
            for value in levels
        )
        schedule.extend(
            {"cycle": cycle, "direction": "up", "learning_rate": value}
            for value in reversed(levels)
        )
    return schedule


def _cycle_summary(completed: List[Dict[str, object]], cycle: int) -> dict:
    rows = [row for row in completed if row["cycle"] == cycle]
    eta_on = next(
        (
            float(row["learning_rate"])
            for row in rows
            if row["direction"] == "down"
            and row["equilibrated"]
            and row["state"] == "expressed"
        ),
        None,
    )
    saw_expressed = eta_on is not None
    eta_off = None
    for row in rows:
        if row["direction"] != "up" or not row["equilibrated"]:
            continue
        if row["state"] == "expressed":
            saw_expressed = True
        elif saw_expressed and row["state"] == "suppressed":
            eta_off = float(row["learning_rate"])
            break
    return {
        "cycle": cycle,
        "eta_on": eta_on,
        "eta_off": eta_off,
        "rho": eta_off / eta_on if eta_on is not None and eta_off is not None else None,
    }


def run_hysteresis_cycles(
    experiment_config: MBCExperimentConfig,
    metric: MetricConfig,
    gate: Gate1Config,
    reference: ReferenceBands,
    thresholds: StateThresholds,
    suppressed_snapshot: Path,
    *,
    high_learning_rate: float,
    low_learning_rate: float,
    output_dir: Path,
    dwell_multiplier: int = 1,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    levels = geometric_levels(high_learning_rate, low_learning_rate, gate.levels_per_decade)
    schedule = _schedule(levels, gate.cycles)
    progress_path = output_dir / "progress.json"
    experiment = MBCExperiment(experiment_config, metric)
    completed: List[Dict[str, object]] = []
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        completed = progress["completed"]
        progress_snapshot = output_dir / progress["snapshot_path"]
        if not progress_snapshot.exists():
            raise FileNotFoundError("hysteresis progress references a missing snapshot")
        restored = load_snapshot(
            progress_snapshot,
            model=experiment.model,
            optimizer=experiment.optimizer,
            stream=experiment.stream,
            map_location=experiment.device,
        )
    else:
        restored = load_snapshot(
            suppressed_snapshot,
            model=experiment.model,
            optimizer=experiment.optimizer,
            stream=experiment.stream,
            map_location=experiment.device,
        )
    experiment.step = restored["step"]
    metrics_path = output_dir / "metrics.jsonl"
    if progress_path.exists() and metrics_path.exists():
        rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
        rows = [row for row in rows if int(row["step"]) <= experiment.step]
        metrics_path.write_text(
            "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows)
        )
    writer = JSONLWriter(metrics_path)
    for item in schedule[len(completed) :]:
        dwell = run_dwell(
            experiment,
            float(item["learning_rate"]),
            reference,
            thresholds,
            gate,
            writer=writer,
            dwell_multiplier=dwell_multiplier,
        )
        row = {**item, **asdict(dwell)}
        completed.append(row)
        progress_snapshot = output_dir / "checkpoints" / f"dwell_{len(completed):03d}.pt"
        save_snapshot(
            progress_snapshot,
            model=experiment.model,
            optimizer=experiment.optimizer,
            stream=experiment.stream,
            step=experiment.step,
            metadata={"kind": "hysteresis_progress", **item},
        )
        progress_path.write_text(
            json.dumps(
                {
                    "completed": completed,
                    "snapshot_path": str(progress_snapshot.relative_to(output_dir)),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(
            f"[hysteresis seed={experiment_config.seed}] cycle={item['cycle']} "
            f"{item['direction']} eta={item['learning_rate']:.6g} "
            f"state={dwell.state} equilibrated={dwell.equilibrated}",
            flush=True,
        )
    summary = {
        "seed": experiment_config.seed,
        "high_learning_rate": high_learning_rate,
        "low_learning_rate": low_learning_rate,
        "dwell_multiplier": dwell_multiplier,
        "levels": levels,
        "cycles": [_cycle_summary(completed, cycle) for cycle in range(1, gate.cycles + 1)],
        "dwells": completed,
    }
    (output_dir / "hysteresis.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary
