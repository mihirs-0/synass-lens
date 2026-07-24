"""Generate frozen order-zero and solved behavioral reference ensembles."""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, Optional

from .config import MBCExperimentConfig, MetricConfig
from .experiment import JSONLWriter, MBCExperiment
from .references import build_reference_bands, empirical_constant_machine
from .snapshot import save_snapshot
from .snapshot import load_snapshot


@dataclass(frozen=True)
class ReferenceSeedResult:
    seed: int
    success: bool
    endpoint_step: Optional[int]
    final_step: int
    order_zero: Dict[str, float]
    q_star_answer_loss: float
    solved_c_int: Optional[float]


def _behaviorally_solved(row: Dict[str, float], metric: MetricConfig) -> bool:
    return all(
        row[f"probe_{index}_exact_match"] >= metric.solved_exact_match
        and row[f"probe_{index}_full_vocab_ce"] <= metric.solved_full_vocab_ce
        for index in (0, 1)
    )


def run_reference_seed(
    base_config: MBCExperimentConfig,
    metric: MetricConfig,
    *,
    seed: int,
    acquisition_budget: int,
    output_dir: Path,
) -> ReferenceSeedResult:
    if acquisition_budget <= 0:
        raise ValueError("acquisition budget must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    if result_path.exists():
        return ReferenceSeedResult(**json.loads(result_path.read_text()))
    experiment = MBCExperiment(replace(base_config, seed=seed), metric)
    metrics_path = output_dir / "metrics.jsonl"
    progress_path = output_dir / "progress_snapshot.pt"
    existing_rows = []
    if progress_path.exists():
        restored = load_snapshot(
            progress_path,
            model=experiment.model,
            optimizer=experiment.optimizer,
            stream=experiment.stream,
            map_location=experiment.device,
        )
        experiment.step = restored["step"]
        existing_rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
        existing_rows = [
            row
            for row in existing_rows
            if row.get("kind") == "order_zero" or int(row["step"]) <= experiment.step
        ]
        metrics_path.write_text(
            "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in existing_rows)
        )
        order_zero = {
            key: value
            for key, value in next(
                row for row in existing_rows if row.get("kind") == "order_zero"
            ).items()
            if key != "kind"
        }
    else:
        if metrics_path.exists():
            metrics_path.unlink()
        order_zero = experiment.evaluate()
    writer = JSONLWriter(metrics_path)
    if not existing_rows:
        writer.write({"kind": "order_zero", **order_zero})
    q_star = empirical_constant_machine(experiment.mapping, experiment.tokenizer)
    training_rows = [row for row in existing_rows if row.get("kind") == "reference_training"]
    solved_existing = [row for row in training_rows if _behaviorally_solved(row, metric)]
    endpoint_step: Optional[int] = (
        int(solved_existing[0]["step"])
        if solved_existing
        else (0 if _behaviorally_solved(order_zero, metric) else None)
    )
    solved_rows = [
        row for row in training_rows if endpoint_step is not None and row["step"] >= endpoint_step
    ]
    while True:
        if endpoint_step is None and experiment.step >= acquisition_budget:
            break
        if endpoint_step is not None and experiment.step >= endpoint_step + metric.solved_hold_steps:
            break
        experiment.advance(metric.eval_every)
        row = experiment.evaluate()
        writer.write({"kind": "reference_training", **row})
        if endpoint_step is None and _behaviorally_solved(row, metric):
            endpoint_step = experiment.step
        if endpoint_step is not None:
            solved_rows.append(row)
        if experiment.step % 500 == 0 or experiment.step == endpoint_step:
            print(
                f"[reference seed={seed}] step={experiment.step} "
                f"c_int={row['c_int']:.4f} em={row['exact_match']:.4f} "
                f"ce={row['full_vocab_ce']:.4f}",
                flush=True,
            )
        if experiment.step % 1_000 == 0 or experiment.step == endpoint_step:
            save_snapshot(
                progress_path,
                model=experiment.model,
                optimizer=experiment.optimizer,
                stream=experiment.stream,
                step=experiment.step,
                metadata={"kind": "reference_progress", "seed": seed},
            )
    success = endpoint_step is not None and experiment.step >= endpoint_step + metric.solved_hold_steps
    solved_c_int = None
    if success:
        window_start = experiment.step - metric.solved_summary_window
        window = [row["c_int"] for row in solved_rows if row["step"] >= window_start]
        if not window:
            raise RuntimeError("solved summary window contains no evaluations")
        solved_c_int = statistics.median(window)
        save_snapshot(
            output_dir / "solved_snapshot.pt",
            model=experiment.model,
            optimizer=experiment.optimizer,
            stream=experiment.stream,
            step=experiment.step,
            metadata={"kind": "solved_reference", "seed": seed, "endpoint_step": endpoint_step},
        )
    result = ReferenceSeedResult(
        seed=seed,
        success=success,
        endpoint_step=endpoint_step,
        final_step=experiment.step,
        order_zero=order_zero,
        q_star_answer_loss=float(q_star["answer_token_loss"]),
        solved_c_int=solved_c_int,
    )
    result_path.write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n")
    if progress_path.exists():
        progress_path.unlink()
    return result


def run_reference_ensemble(
    base_config: MBCExperimentConfig,
    metric: MetricConfig,
    *,
    seeds: Iterable[int],
    acquisition_budget: int,
    output_dir: Path,
) -> dict:
    output_dir = Path(output_dir)
    results = [
        run_reference_seed(
            base_config,
            metric,
            seed=seed,
            acquisition_budget=acquisition_budget,
            output_dir=output_dir / f"seed_{seed}",
        )
        for seed in seeds
    ]
    failures = [result.seed for result in results if not result.success]
    summary = {
        "all_requested_succeeded": not failures,
        "bands_ready": not failures and len(results) >= 2,
        "failed_seeds": failures,
        "seeds": [asdict(r) for r in results],
    }
    if summary["bands_ready"]:
        bands = build_reference_bands(
            [result.order_zero for result in results],
            [float(result.solved_c_int) for result in results if result.solved_c_int is not None],
            [result.q_star_answer_loss for result in results],
        )
        summary["bands"] = asdict(bands)
    (output_dir / "reference_ensemble.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary
