"""Gate 0-E verdict: the frozen pass/kill rules of amendment v1.5 section 2.5.

Everything decision-bearing here restates the amendment or the committed
design notes; nothing is tunable at verdict time:

- **Null wins (program stops):** every gate seed's observed eta50 is within
  1.5x of its committed prediction, AND the observed batch shift matches the
  v-conditioned direction on both contrasts with each bootstrap CI excluding
  zero on the opposite side.
- **Null loses (Gate 1 opens, claim earned):** at least two of three gate
  seeds off by more than 2x — for censored predictions, the observation must
  sit at least 2x inside the uncrossed side — OR the batch shift carries the
  opposite sign on both contrasts with CIs excluding agreement.
- **Ambiguous (Gate 1 opens, claim demoted):** anything else. Stopping
  requires the null to affirmatively win.

The weight-decay flag (`erasure_is_wd_mediated`) is frozen as: all sixteen
lambda=0 streams labeled retained AND the matched lambda>0 rates on the same
seed's primary curve pooled to a primary fraction of at least one half. The
Arrhenius secondary pools (gate seed, rate) cells at the production batch
size with strictly partial fractions and defined conditional medians; R^2 of
at least 0.8 over at least five cells registers `noise_activated_regime`.
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .gate0e_stats import (
    BOOTSTRAP_SEED,
    RateOutcomes,
    logistic_fit,
    ols_r2,
)

RATIO_MATCH = 1.5
RATIO_MISS = 2.0
BATCH_CONTRASTS = (32, 512)
BASE_BATCH = 128
GATE_SEEDS = (0, 1, 2)
BATCH_SEEDS = (0, 1)
ARRHENIUS_MINIMUM_CELLS = 5
ARRHENIUS_R2 = 0.8


def _curve_outcomes(curve: Mapping[str, object]) -> List[RateOutcomes]:
    return [
        RateOutcomes(
            learning_rate=float(cell["learning_rate"]),
            erased=int(cell["counts"]["erased"]),
            retained=int(cell["counts"]["retained"]),
            unresolved=int(cell["counts"]["unresolved"]),
            diverged=int(cell["counts"]["diverged"]),
        )
        for cell in curve["cells"]
    ]


def _restricted(curve: Mapping[str, object], rates: Sequence[float]) -> List[dict]:
    cells = [
        cell
        for cell in curve["cells"]
        if any(math.isclose(float(cell["learning_rate"]), rate, rel_tol=1e-9) for rate in rates)
    ]
    if len(cells) != len(rates):
        raise ValueError("curve does not contain every requested contrast rate")
    return cells


def restricted_eta50(
    curve: Mapping[str, object],
    rates: Sequence[float],
    counts: Optional[Mapping[float, Tuple[int, int]]] = None,
) -> Optional[float]:
    cells = _restricted(curve, rates)
    outcomes = [
        RateOutcomes(
            learning_rate=float(cell["learning_rate"]),
            erased=int(cell["counts"]["erased"]),
            retained=int(cell["counts"]["retained"]),
            unresolved=int(cell["counts"]["unresolved"]),
            diverged=int(cell["counts"]["diverged"]),
        )
        for cell in cells
        if not cell["unstable"]
    ]
    if len(outcomes) < 2:
        return None
    try:
        fit = logistic_fit(outcomes, counts)
    except (ValueError, FloatingPointError):
        return None
    if fit.eta50 is None or not (1e-6 < fit.eta50 < 1.0):
        return None
    return fit.eta50


def seed_ratio_check(
    observed_eta50: Optional[float], prediction: Mapping[str, object]
) -> dict:
    kind = prediction.get("kind")
    result = {
        "observed_eta50": observed_eta50,
        "prediction_kind": kind,
        "log_ratio": None,
        "within_match": False,
        "off_miss": False,
    }
    if observed_eta50 is None:
        result["note"] = "observed eta50 undefined; cannot support any rule"
        return result
    if kind == "crossing":
        predicted = float(prediction["predicted_eta50"])
        log_ratio = math.log(observed_eta50 / predicted)
        result["log_ratio"] = log_ratio
        result["within_match"] = abs(log_ratio) <= math.log(RATIO_MATCH)
        result["off_miss"] = abs(log_ratio) > math.log(RATIO_MISS)
    elif kind == "right_censored":
        bound = float(prediction["bound"])
        result["off_miss"] = observed_eta50 <= bound / RATIO_MISS
    elif kind == "left_censored":
        bound = float(prediction["bound"])
        result["off_miss"] = observed_eta50 >= bound * RATIO_MISS
    return result


def _predicted_shift_sign(
    predictions: Mapping[str, Mapping[str, object]],
    seeds: Sequence[int],
    batch: int,
) -> Optional[int]:
    signs = set()
    for seed in seeds:
        base = predictions.get(f"{seed}:{BASE_BATCH}", {}).get("prediction", {})
        contrast = predictions.get(f"{seed}:{batch}", {}).get("prediction", {})
        if base.get("kind") != "crossing" or contrast.get("kind") != "crossing":
            return None
        difference = math.log(
            float(contrast["predicted_eta50"]) / float(base["predicted_eta50"])
        )
        if difference == 0.0:
            return None
        signs.add(1 if difference > 0 else -1)
    if len(signs) != 1:
        return None
    return signs.pop()


def _pooled_logodds(counts: Sequence[Tuple[int, int]]) -> float:
    """Haldane-Anscombe pooled log-odds over the contrast rates."""
    logits = []
    for erased, classified in counts:
        fraction = (erased + 0.5) / (classified + 1.0)
        logits.append(math.log(fraction / (1.0 - fraction)))
    return float(np.mean(logits))


def batch_shift_analysis(
    batch_curves: Mapping[str, Mapping[str, object]],
    contrast_rates: Sequence[float],
    predictions: Mapping[str, Mapping[str, object]],
    *,
    seeds: Sequence[int] = BATCH_SEEDS,
    batches: Sequence[int] = BATCH_CONTRASTS,
    replicates: int = 2_000,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Directional batch-shift discriminator (v1.5.1 statistic).

    The registered sign statistic is the pooled Haldane-Anscombe log-odds
    shift of the erasure fraction at the fixed contrast rates,
    ``mean over seeds of [pooled_logodds(B) - pooled_logodds(base)]``.
    A boundary moving up in learning rate lowers erasure fractions at fixed
    rates, so the predicted log-odds sign is the NEGATIVE of the predicted
    ``log eta50`` shift sign. This statistic is defined even when a saturated
    batch cell leaves eta50 unbracketed, and every bootstrap resample is
    valid, which the eta50-difference statistic could not guarantee at six
    streams per cell. Point-estimate eta50 shifts are reported descriptively.

    ``batch_curves`` is keyed ``"<seed>:<batch>"`` and must include the base
    batch, whose cells come from the primary curves at the contrast rates.
    """
    rng = np.random.default_rng(seed)
    label_pools: Dict[str, List[List[int]]] = {}
    for seed_value in seeds:
        for batch in (BASE_BATCH, *batches):
            key = f"{seed_value}:{batch}"
            curve = batch_curves.get(key)
            if curve is None:
                raise ValueError(f"batch analysis is missing condition {key}")
            label_pools[key] = [
                list(cell["stream_labels"])
                for cell in _restricted(curve, contrast_rates)
            ]

    def condition_logodds(key: str, resample: bool) -> float:
        counts = []
        for labels in label_pools[key]:
            values = np.asarray(labels)
            if resample and values.size:
                values = rng.choice(values, size=values.size, replace=True)
            counts.append((int(values.sum()), int(values.size)))
        return _pooled_logodds(counts)

    def condition_eta50_point(key: str) -> Optional[float]:
        counts = {}
        for rate, labels in zip(contrast_rates, label_pools[key]):
            counts[float(rate)] = (int(sum(labels)), len(labels))
        return restricted_eta50(batch_curves[key], contrast_rates, counts)

    observed: Dict[str, dict] = {}
    for batch in batches:
        deltas = [
            condition_logodds(f"{seed_value}:{batch}", resample=False)
            - condition_logodds(f"{seed_value}:{BASE_BATCH}", resample=False)
            for seed_value in seeds
        ]
        replicated = []
        for _ in range(replicates):
            replicated.append(
                float(
                    np.mean(
                        [
                            condition_logodds(f"{seed_value}:{batch}", resample=True)
                            - condition_logodds(
                                f"{seed_value}:{BASE_BATCH}", resample=True
                            )
                            for seed_value in seeds
                        ]
                    )
                )
            )
        ci = [float(value) for value in np.percentile(replicated, [2.5, 97.5])]
        eta50_shifts = {}
        for seed_value in seeds:
            base = condition_eta50_point(f"{seed_value}:{BASE_BATCH}")
            contrast = condition_eta50_point(f"{seed_value}:{batch}")
            eta50_shifts[str(seed_value)] = (
                None
                if base is None or contrast is None
                else math.log(contrast / base)
            )
        observed[str(batch)] = {
            "mean_logodds_shift": float(np.mean(deltas)),
            "ci": ci,
            "descriptive_eta50_log_shifts": eta50_shifts,
        }

    predicted_eta50_sign = {
        str(batch): _predicted_shift_sign(predictions, seeds, batch)
        for batch in batches
    }

    def _direction(batch: str, eta50_sign: int) -> bool:
        entry = observed[batch]
        logodds_sign = -eta50_sign
        low, high = entry["ci"]
        if logodds_sign > 0:
            return entry["mean_logodds_shift"] > 0 and low > 0
        return entry["mean_logodds_shift"] < 0 and high < 0

    matches: Optional[bool] = True
    opposite: Optional[bool] = True
    for batch in map(str, batches):
        sign = predicted_eta50_sign[batch]
        if sign is None:
            matches = None
            opposite = None
            break
        matches = matches and _direction(batch, sign)
        opposite = opposite and _direction(batch, -sign)
    return {
        "contrast_rates": [float(rate) for rate in contrast_rates],
        "statistic": "pooled_logodds_shift_v1_5_1",
        "observed": observed,
        "predicted_eta50_sign": predicted_eta50_sign,
        "matches_prediction": matches,
        "opposite_of_prediction": opposite,
    }


def wd_mediation_flag(
    wd_curve: Mapping[str, object], primary_curve: Mapping[str, object]
) -> dict:
    wd_cells = wd_curve["cells"]
    wd_rates = [float(cell["learning_rate"]) for cell in wd_cells]
    total_streams = sum(
        sum(cell["counts"].values()) for cell in wd_cells
    )
    all_retained = all(
        cell["counts"]["retained"] == sum(cell["counts"].values())
        for cell in wd_cells
    )
    matched = _restricted(primary_curve, wd_rates)
    erased = sum(cell["counts"]["erased"] for cell in matched)
    classified = sum(
        cell["counts"]["erased"] + cell["counts"]["retained"] + cell["counts"]["unresolved"]
        for cell in matched
    )
    pooled_fraction = None if classified == 0 else erased / classified
    flagged = bool(
        all_retained and pooled_fraction is not None and pooled_fraction >= 0.5
    )
    return {
        "erasure_is_wd_mediated": flagged,
        "lambda0_streams": total_streams,
        "lambda0_all_retained": all_retained,
        "matched_rates": wd_rates,
        "matched_pooled_fraction": pooled_fraction,
    }


def arrhenius_secondary(gate_curves: Mapping[int, Mapping[str, object]]) -> dict:
    points = []
    for seed, curve in sorted(gate_curves.items()):
        if int(curve.get("batch_size", BASE_BATCH)) != BASE_BATCH:
            raise ValueError("Arrhenius secondary uses production-batch curves only")
        for cell in curve["cells"]:
            fraction = cell["primary_fraction"]
            median = cell["median_observed_tau"]
            if (
                fraction is not None
                and 0.0 < fraction < 1.0
                and median is not None
                and not cell["unstable"]
            ):
                points.append(
                    (BASE_BATCH / float(cell["learning_rate"]), math.log(median))
                )
    if len(points) < ARRHENIUS_MINIMUM_CELLS:
        return {
            "noise_activated_regime": False,
            "cells": len(points),
            "fit": None,
            "note": "fewer than the required strictly partial cells",
        }
    fit = ols_r2([p[0] for p in points], [p[1] for p in points])
    return {
        "noise_activated_regime": bool(fit.r_squared >= ARRHENIUS_R2),
        "cells": len(points),
        "fit": {
            "slope": fit.slope,
            "intercept": fit.intercept,
            "r_squared": fit.r_squared,
        },
    }


def gate0e_verdict(
    predictions_artifact: Mapping[str, object],
    gate_curves: Mapping[int, Mapping[str, object]],
    batch_curves: Mapping[str, Mapping[str, object]],
    contrast_rates: Sequence[float],
    wd_curve: Optional[Mapping[str, object]] = None,
    *,
    replicates: int = 2_000,
) -> dict:
    if predictions_artifact.get("kind") != "gate0e_null_predictions":
        raise ValueError("verdict requires the committed prediction artifact")
    predictions = predictions_artifact["predictions"]
    seed_checks = {}
    for seed in GATE_SEEDS:
        curve = gate_curves.get(seed)
        if curve is None:
            raise ValueError(f"verdict is missing the gate curve for seed {seed}")
        entry = predictions.get(f"{seed}:{BASE_BATCH}")
        if entry is None:
            raise ValueError(f"prediction artifact has no entry for seed {seed}")
        seed_checks[str(seed)] = seed_ratio_check(
            curve["statistics"]["eta50"], entry["prediction"]
        )
    batch_analysis = batch_shift_analysis(
        batch_curves, contrast_rates, predictions, replicates=replicates
    )
    all_within = all(check["within_match"] for check in seed_checks.values())
    miss_count = sum(1 for check in seed_checks.values() if check["off_miss"])
    if all_within and batch_analysis["matches_prediction"] is True:
        outcome = "null_wins"
        action = "stop_program"
        claim = None
    elif miss_count >= 2 or batch_analysis["opposite_of_prediction"] is True:
        outcome = "null_loses"
        action = "open_gate1"
        claim = "earned"
    else:
        outcome = "ambiguous"
        action = "open_gate1"
        claim = "demoted"
    report = {
        "schema_version": 1,
        "kind": "gate0e_verdict",
        "outcome": outcome,
        "action": action,
        "claim_status": claim,
        "seed_checks": seed_checks,
        "miss_count": miss_count,
        "batch_shift": batch_analysis,
        "arrhenius_secondary": arrhenius_secondary(gate_curves),
        "c_star": predictions_artifact["c_star"],
        "dev_eta50": predictions_artifact["dev_eta50"],
    }
    if wd_curve is not None:
        report["wd_mediation"] = wd_mediation_flag(wd_curve, gate_curves[GATE_SEEDS[0]])
    return report
