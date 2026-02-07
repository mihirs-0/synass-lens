"""
Candidate-set normalized evaluation for Late Disambiguation Lag.
"""

from typing import Any, Dict, List, Optional
import math
import random

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer

from ..data import CharTokenizer, MappingData


def score_candidate_sequences(
    model: HookedTransformer,
    tokenizer: CharTokenizer,
    base_string: str,
    z_string: str,
    candidate_a_strings: List[str],
    correct_index: int,
    task: str = "bz_to_a",
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Score all candidate A strings for a single (B, z) context.
    """
    if correct_index < 0 or correct_index >= len(candidate_a_strings):
        raise ValueError("correct_index is out of range for candidate_a_strings.")

    encoded_candidates = []
    expected_len: Optional[int] = None
    for a_string in candidate_a_strings:
        encoded = tokenizer.encode_sequence(base_string, z_string, a_string, task=task)
        seq_len = len(encoded["input_ids"])
        if expected_len is None:
            expected_len = seq_len
        elif seq_len != expected_len:
            raise ValueError("Candidate sequences have different lengths; no padding is allowed.")
        encoded_candidates.append(encoded)

    input_ids = torch.stack([e["input_ids"] for e in encoded_candidates]).to(device)
    labels = torch.stack([e["labels"] for e in encoded_candidates]).to(device)
    target_starts = [int(e["target_start_position"]) for e in encoded_candidates]
    target_ends = [int(e["target_end_position"]) for e in encoded_candidates]

    model.eval()
    with torch.no_grad():
        logits = model(input_ids)

    sequence_log_probs: List[float] = []
    for k in range(len(candidate_a_strings)):
        total_log_prob = torch.zeros((), dtype=logits.dtype, device=logits.device)
        labels_k = labels[k]
        start = target_starts[k]
        end = target_ends[k]
        for pos in range(start, end + 1):
            target_token = labels_k[pos].item()
            if target_token == -100:
                continue
            log_probs = F.log_softmax(logits[k, pos - 1], dim=-1)
            total_log_prob += log_probs[target_token]
        sequence_log_probs.append(float(total_log_prob.item()))

    log_probs = torch.tensor(sequence_log_probs, dtype=torch.float64)
    log_normalizer = torch.logsumexp(log_probs, dim=0)
    normalized = log_probs - log_normalizer

    correct_log_prob = float(log_probs[correct_index].item())
    if len(candidate_a_strings) > 1:
        incorrect_sum = float((log_probs.sum() - log_probs[correct_index]).item())
        incorrect_mean = incorrect_sum / (len(candidate_a_strings) - 1)
    else:
        incorrect_mean = float("nan")

    predicted_index = int(torch.argmax(normalized).item())
    topk_indices = torch.topk(normalized, k=min(3, len(candidate_a_strings))).indices.tolist()

    return {
        "candidate_loss": -float(normalized[correct_index].item()),
        "candidate_correct": predicted_index == correct_index,
        "candidate_top3": correct_index in topk_indices,
        "sequence_log_probs": [float(v) for v in sequence_log_probs],
        "normalized_log_probs": [float(v) for v in normalized.detach().cpu().tolist()],
        "correct_log_prob": correct_log_prob,
        "incorrect_mean_log_prob": incorrect_mean,
    }


def run_candidate_eval(
    model: HookedTransformer,
    tokenizer: CharTokenizer,
    mapping_data: MappingData,
    n_examples: int = 32,
    task: str = "bz_to_a",
    device: str = "cuda",
    seed: int = 42,
) -> Dict[str, float]:
    """
    Evaluate candidate-normalized loss over a sample of B strings.
    """
    rng = random.Random(seed)
    base_strings = list(mapping_data.mappings.keys())
    if not base_strings:
        raise ValueError("mapping_data.mappings is empty.")
    n_examples = min(n_examples, len(base_strings))
    sampled_bases = rng.sample(base_strings, n_examples)

    candidate_losses: List[float] = []
    candidate_correct: List[float] = []
    candidate_top3: List[float] = []
    correct_log_probs: List[float] = []
    incorrect_log_probs: List[float] = []

    for base_string in sampled_bases:
        mappings = mapping_data.mappings[base_string]
        if len(mappings) != mapping_data.k:
            raise ValueError("Candidate set size does not match mapping_data.k.")
        correct_index = rng.randrange(len(mappings))
        z_string = mappings[correct_index][0]
        candidate_a_strings = [entry[1] for entry in mappings]

        result = score_candidate_sequences(
            model=model,
            tokenizer=tokenizer,
            base_string=base_string,
            z_string=z_string,
            candidate_a_strings=candidate_a_strings,
            correct_index=correct_index,
            task=task,
            device=device,
        )
        candidate_losses.append(result["candidate_loss"])
        candidate_correct.append(1.0 if result["candidate_correct"] else 0.0)
        candidate_top3.append(1.0 if result["candidate_top3"] else 0.0)
        correct_log_probs.append(result["correct_log_prob"])
        incorrect_log_probs.append(result["incorrect_mean_log_prob"])

    mean_loss = float(sum(candidate_losses) / len(candidate_losses))
    mean_acc = float(sum(candidate_correct) / len(candidate_correct))
    mean_top3 = float(sum(candidate_top3) / len(candidate_top3))
    mean_correct_log_prob = float(sum(correct_log_probs) / len(correct_log_probs))
    mean_incorrect_log_prob = float(sum(incorrect_log_probs) / len(incorrect_log_probs))

    return {
        "candidate_loss": mean_loss,
        "candidate_accuracy": mean_acc,
        "candidate_top3_accuracy": mean_top3,
        "mean_correct_log_prob": mean_correct_log_prob,
        "mean_incorrect_log_prob": mean_incorrect_log_prob,
        "n_examples": int(n_examples),
        "k": int(mapping_data.k),
        "log_k": float(math.log(mapping_data.k)),
    }


def compute_z_usage_metrics(
    model: HookedTransformer,
    batch: Dict[str, torch.Tensor],
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Compute z-usage gap on a fixed batch using existing trainer utilities.
    """
    from ..training.trainer import compute_loss, shuffle_z_in_batch

    model.eval()
    with torch.no_grad():
        batch_on_device = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }
        _, _, loss_clean = compute_loss(model, batch_on_device)
        shuffled_batch = shuffle_z_in_batch(batch_on_device)
        _, _, loss_z_shuffled = compute_loss(model, shuffled_batch)

    return {
        "loss_clean": float(loss_clean),
        "loss_z_shuffled": float(loss_z_shuffled),
        "z_gap": float(loss_z_shuffled - loss_clean),
    }


def detect_binding_onset(
    z_gaps: List[float],
    steps: List[int],
    gap_threshold: float = 0.5,
    consecutive_required: int = 3,
) -> Optional[int]:
    """
    Detect the first step where z-gap exceeds a threshold for consecutive steps.
    """
    if len(z_gaps) != len(steps):
        raise ValueError("z_gaps and steps must have the same length.")

    consecutive = 0
    start_index: Optional[int] = None
    for idx, gap in enumerate(z_gaps):
        if gap > gap_threshold:
            if consecutive == 0:
                start_index = idx
            consecutive += 1
            if consecutive >= consecutive_required and start_index is not None:
                return steps[start_index]
        else:
            consecutive = 0
            start_index = None
    return None
