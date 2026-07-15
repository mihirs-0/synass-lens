"""MBC-specific adapters for the generic behavioral metrics."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F

from src.data import MappingData

from .metrics import Quartets, interaction_contrast, z_sensitivity


@dataclass
class MBCProbe:
    input_ids: torch.Tensor
    answer_token_ids: torch.Tensor
    read_positions: Tuple[int, ...]
    n_b: int
    k: int
    b_strings: Tuple[str, ...]
    z_strings: Tuple[str, ...]

    def to(self, device: str | torch.device) -> "MBCProbe":
        return MBCProbe(
            input_ids=self.input_ids.to(device),
            answer_token_ids=self.answer_token_ids.to(device),
            read_positions=self.read_positions,
            n_b=self.n_b,
            k=self.k,
            b_strings=self.b_strings,
            z_strings=self.z_strings,
        )


def _ordered_answers(mapping_data: MappingData, b_string: str, z_strings: Sequence[str]) -> List[str]:
    by_z = dict(mapping_data.mappings[b_string])
    if set(by_z) != set(z_strings):
        raise ValueError("C_int requires shared z semantics across selected B values")
    return [by_z[z] for z in z_strings]


def build_mbc_probes(
    mapping_data: MappingData,
    tokenizer,
    *,
    n_b: int,
    seeds: Tuple[int, int],
    task: str = "bz_to_a",
) -> Tuple[MBCProbe, MBCProbe]:
    if task != "bz_to_a":
        raise ValueError("the MBC interaction assay is defined for bz_to_a")
    all_b = sorted(mapping_data.mappings)
    if len(all_b) < 2 * n_b:
        raise ValueError(f"need at least {2*n_b} B values for disjoint probes")
    # Independent seeded ranks, with explicit exclusion to guarantee disjointness.
    first = random.Random(seeds[0]).sample(all_b, n_b)
    remaining = [b for b in all_b if b not in set(first)]
    second = random.Random(seeds[1]).sample(remaining, n_b)

    def build(selected: Sequence[str]) -> MBCProbe:
        z_strings = tuple(z for z, _ in mapping_data.mappings[selected[0]])
        rows: List[torch.Tensor] = []
        answers: List[List[List[int]]] = []
        read_positions: Tuple[int, ...] | None = None
        for b_string in selected:
            b_answers: List[List[int]] = []
            for z_string, answer in zip(z_strings, _ordered_answers(mapping_data, b_string, z_strings)):
                encoded = tokenizer.encode_sequence(b_string, z_string, answer, task=task)
                rows.append(encoded["input_ids"])
                token_ids = [tokenizer.token_to_id[char] for char in answer]
                b_answers.append(token_ids)
                positions = tuple(range(encoded["target_start_position"] - 1, encoded["target_end_position"] - 1))
                if read_positions is None:
                    read_positions = positions
                elif positions != read_positions:
                    raise ValueError("probe sequences do not share target read positions")
            answers.append(b_answers)
        assert read_positions is not None
        return MBCProbe(
            input_ids=torch.stack(rows),
            answer_token_ids=torch.tensor(answers, dtype=torch.long),
            read_positions=read_positions,
            n_b=len(selected),
            k=len(z_strings),
            b_strings=tuple(selected),
            z_strings=z_strings,
        )

    return build(first), build(second)


@torch.no_grad()
def evaluate_mbc_probe(
    model: torch.nn.Module,
    probe: MBCProbe,
    quartets: Quartets,
    *,
    batch_size: int = 512,
) -> dict[str, float]:
    model.eval()
    outputs = []
    for start in range(0, len(probe.input_ids), batch_size):
        outputs.append(model(probe.input_ids[start : start + batch_size]))
    logits = torch.cat(outputs, dim=0).reshape(probe.n_b, probe.k, -1, outputs[0].shape[-1])
    c_values = []
    dz_values = []
    correct_by_position = []
    ce_by_position = []
    for position_index, read_position in enumerate(probe.read_positions):
        position_logits = logits[:, :, read_position, :]
        position_answers = probe.answer_token_ids[:, :, position_index]
        c_values.append(interaction_contrast(position_logits, position_answers, quartets))
        dz_values.append(z_sensitivity(position_logits, position_answers, quartets))
        correct_by_position.append(position_logits.argmax(dim=-1) == position_answers)
        ce_by_position.append(
            F.cross_entropy(
                position_logits.reshape(-1, position_logits.shape[-1]),
                position_answers.reshape(-1),
            )
        )
    full_exact = torch.stack(correct_by_position, dim=2).all(dim=2).float()
    return {
        "c_int": float(torch.stack(c_values).mean().item()),
        "delta_z": float(torch.stack(dz_values).mean().item()),
        "exact_match": float(full_exact.mean().item()),
        "full_vocab_ce": float(torch.stack(ce_by_position).mean().item()),
    }
