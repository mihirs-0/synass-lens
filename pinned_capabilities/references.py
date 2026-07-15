"""Reference quantities that freeze behavioral state thresholds."""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict

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
        "candidate_first_token_loss": math.log(mapping.k),
    }
