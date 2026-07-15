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
    experiment = MBCExperiment(replace(base_config, seed=seed), metric)
    writer = JSONLWriter(output_dir / "metrics.jsonl")
    order_zero = experiment.evaluate()
    writer.write({"kind": "order_zero", **order_zero})
    q_star = empirical_constant_machine(experiment.mapping, experiment.tokenizer)
    endpoint_step: Optional[int] = 0 if _behaviorally_solved(order_zero, metric) else None
    solved_rows = []
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
    (output_dir / "result.json").write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n")
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
