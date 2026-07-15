"""Reference quantities that freeze behavioral state thresholds."""

from __future__ import annotations

import math
import statistics
from collections import Counter
from typing import Dict, Mapping, Sequence

from .state import ReferenceBands

from src.data import CharTokenizer, MappingData


def empirical_constant_machine(mapping: MappingData, tokenizer: CharTokenizer) -> Dict[str, object]:
    """Return the maximum-likelihood input-blind distribution and its CE.

    The reported loss matches the training objective: all target characters and
    the deterministic EOS position receive equal weight.
    """
    answers = [example["a"] for example in mapping.examples]
    if not answers or len({len(answer) for answer in answers}) != 1:
        raise ValueError("constant-machine reference requires fixed nonempty answer length")
    counts = []
    entropies = []
    for position in range(len(answers[0])):
        position_counts = Counter(tokenizer.token_to_id[answer[position]] for answer in answers)
        total = sum(position_counts.values())
        probabilities = {token: count / total for token, count in position_counts.items()}
        entropy = -sum(probability * math.log(probability) for probability in probabilities.values())
        counts.append({str(token): count for token, count in sorted(position_counts.items())})
        entropies.append(entropy)
    counts.append({str(tokenizer.eos_token_id): len(answers)})
    entropies.append(0.0)
    return {
        "position_counts": counts,
        "position_entropies": entropies,
        "training_loss": sum(entropies) / len(entropies),
        "answer_token_loss": sum(entropies[:-1]) / len(entropies[:-1]),
        "candidate_first_token_loss": math.log(mapping.k),
    }


def build_reference_bands(
    order_zero_rows: Sequence[Mapping[str, float]],
    solved_c_int_values: Sequence[float],
    q_star_answer_losses: Sequence[float],
) -> ReferenceBands:
    if len(order_zero_rows) < 2 or len(solved_c_int_values) < 2 or len(q_star_answer_losses) < 2:
        raise ValueError("reference bands require at least two independent seeds per ensemble")
    return ReferenceBands(
        plateau_mean=statistics.mean(float(row["c_int"]) for row in order_zero_rows),
        plateau_sd=statistics.stdev(float(row["c_int"]) for row in order_zero_rows),
        solved_c_int=statistics.median(float(value) for value in solved_c_int_values),
        chance_em_mean=statistics.mean(float(row["exact_match"]) for row in order_zero_rows),
        chance_em_sd=statistics.stdev(float(row["exact_match"]) for row in order_zero_rows),
        q_star_loss_mean=statistics.mean(float(value) for value in q_star_answer_losses),
        q_star_loss_sd=statistics.stdev(float(value) for value in q_star_answer_losses),
    )
