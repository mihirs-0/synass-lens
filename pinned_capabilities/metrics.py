"""Behavior-only metrics for conditional capabilities.

The primary interaction contrast scores the same target under a 2x2 grid of
inputs. Logits are centered over the de-duplicated union of the quartet's
targets. Unlike log-softmax normalization, this linear centering guarantees
that additive B-only and z-only score functions cancel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class Quartets:
    b: torch.Tensor
    z: torch.Tensor
    b_prime: torch.Tensor
    z_prime: torch.Tensor

    def __post_init__(self) -> None:
        sizes = {self.b.numel(), self.z.numel(), self.b_prime.numel(), self.z_prime.numel()}
        if len(sizes) != 1:
            raise ValueError("quartet index tensors must have equal length")
        if torch.any(self.b == self.b_prime):
            raise ValueError("quartets require B != B_prime")
        if torch.any(self.z == self.z_prime):
            raise ValueError("quartets require z != z_prime")

    def to(self, device: torch.device | str) -> "Quartets":
        return Quartets(*(tensor.to(device) for tensor in (self.b, self.z, self.b_prime, self.z_prime)))

    @property
    def n(self) -> int:
        return self.b.numel()


@dataclass(frozen=True)
class CounterfactualMetrics:
    c_int: torch.Tensor
    delta_z: torch.Tensor
    candidate_exact_match: torch.Tensor

    def means(self) -> dict[str, float]:
        return {
            "c_int": float(self.c_int.mean().item()),
            "delta_z": float(self.delta_z.mean().item()),
            "candidate_exact_match": float(self.candidate_exact_match.mean().item()),
        }


def sample_quartets(
    n_b: int,
    k: int,
    n: int,
    seed: int,
    device: torch.device | str = "cpu",
) -> Quartets:
    if n_b < 2 or k < 2:
        raise ValueError("interaction contrast requires at least two B values and two z values")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    b = torch.randint(n_b, (n,), generator=generator)
    z = torch.randint(k, (n,), generator=generator)
    b_prime = torch.randint(n_b - 1, (n,), generator=generator)
    z_prime = torch.randint(k - 1, (n,), generator=generator)
    b_prime += (b_prime >= b).long()
    z_prime += (z_prime >= z).long()
    return Quartets(b, z, b_prime, z_prime).to(device)


def _unique_union_mask(tokens: torch.Tensor) -> torch.Tensor:
    """Keep the first occurrence of each token in every row."""
    keep = torch.ones_like(tokens, dtype=torch.bool)
    for column in range(1, tokens.shape[1]):
        keep[:, column] = (tokens[:, column, None] != tokens[:, :column]).all(dim=1)
    return keep


def interaction_contrast(
    logits: torch.Tensor,
    answer_token_ids: torch.Tensor,
    quartets: Quartets,
) -> torch.Tensor:
    """Return one union-centered 2x2 interaction contrast per quartet.

    Args:
        logits: ``[n_B, K, vocab]`` read-position logits.
        answer_token_ids: ``[n_B, K]`` correct target token for each input.
        quartets: counterfactual indices with unequal B and z values.
    """
    if logits.ndim != 3 or answer_token_ids.ndim != 2:
        raise ValueError("expected logits [n_B,K,V] and answer ids [n_B,K]")
    if tuple(logits.shape[:2]) != tuple(answer_token_ids.shape):
        raise ValueError("logit grid and answer grid disagree")
    q = quartets.to(logits.device)
    b_grid = torch.stack((q.b, q.b, q.b_prime, q.b_prime), dim=1)
    z_grid = torch.stack((q.z, q.z_prime, q.z, q.z_prime), dim=1)
    condition_logits = logits[b_grid, z_grid]
    union_tokens = torch.stack(
        (
            answer_token_ids[q.b, q.z],
            answer_token_ids[q.b, q.z_prime],
            answer_token_ids[q.b_prime, q.z],
            answer_token_ids[q.b_prime, q.z_prime],
        ),
        dim=1,
    ).to(logits.device)
    keep = _unique_union_mask(union_tokens)
    union_scores = condition_logits.gather(
        2, union_tokens[:, None, :].expand(-1, condition_logits.shape[1], -1)
    )
    union_scores = union_scores.masked_fill(~keep[:, None, :], float("-inf"))
    union_count = keep.sum(dim=1).to(logits.dtype)
    union_center = union_scores.masked_fill(~keep[:, None, :], 0.0).sum(dim=2)
    union_center = union_center / union_count[:, None]
    target = answer_token_ids[q.b, q.z].to(logits.device)
    target_scores = condition_logits.gather(
        2, target[:, None, None].expand(-1, condition_logits.shape[1], 1)
    ).squeeze(2)
    normalized = target_scores - union_center
    signs = logits.new_tensor((1.0, -1.0, -1.0, 1.0))
    return (normalized * signs).sum(dim=1)


def z_sensitivity(
    logits: torch.Tensor,
    answer_token_ids: torch.Tensor,
    quartets: Quartets,
) -> torch.Tensor:
    """Candidate-normalized score drop after swapping z at fixed B."""
    q = quartets.to(logits.device)
    candidate_tokens = answer_token_ids[q.b].to(logits.device)
    correct_logits = logits[q.b, q.z]
    swapped_logits = logits[q.b, q.z_prime]
    correct_candidates = correct_logits.gather(1, candidate_tokens)
    swapped_candidates = swapped_logits.gather(1, candidate_tokens)
    target = answer_token_ids[q.b, q.z].to(logits.device)
    correct_target = correct_logits.gather(1, target[:, None]).squeeze(1)
    swapped_target = swapped_logits.gather(1, target[:, None]).squeeze(1)
    return (correct_target - torch.logsumexp(correct_candidates, dim=1)) - (
        swapped_target - torch.logsumexp(swapped_candidates, dim=1)
    )


def candidate_exact_match(logits: torch.Tensor, answer_token_ids: torch.Tensor) -> torch.Tensor:
    candidates = answer_token_ids[:, None, :].expand(-1, answer_token_ids.shape[1], -1)
    candidate_logits = logits.gather(2, candidates)
    predicted_slot = candidate_logits.argmax(dim=2)
    expected_slot = torch.arange(answer_token_ids.shape[1], device=logits.device)[None, :]
    return (predicted_slot == expected_slot).float().reshape(-1)


def evaluate_counterfactual_metrics(
    logits: torch.Tensor,
    answer_token_ids: torch.Tensor,
    quartets: Optional[Quartets] = None,
    *,
    quartet_count: int = 1024,
    seed: int = 0,
) -> CounterfactualMetrics:
    if quartets is None:
        quartets = sample_quartets(
            logits.shape[0], logits.shape[1], quartet_count, seed, device=logits.device
        )
    return CounterfactualMetrics(
        c_int=interaction_contrast(logits, answer_token_ids, quartets),
        delta_z=z_sensitivity(logits, answer_token_ids, quartets),
        candidate_exact_match=candidate_exact_match(logits, answer_token_ids),
    )
