"""Exhaustive, auditable parameter-family assignments."""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, List, Tuple

import torch


GROUP_NAMES = ("embeddings", "attention", "mlp", "layer_norms", "unembedding")


def parameter_family(name: str) -> str:
    if name.startswith(("embed.", "pos_embed.")):
        return "embeddings"
    if ".attn." in name:
        return "attention"
    if ".mlp." in name:
        return "mlp"
    if ".ln" in name or name.startswith("ln_final."):
        return "layer_norms"
    if name.startswith("unembed."):
        return "unembedding"
    raise ValueError(f"unclassified trainable parameter: {name}")


def grouped_named_parameters(model: torch.nn.Module) -> Dict[str, List[Tuple[str, torch.nn.Parameter]]]:
    groups: Dict[str, List[Tuple[str, torch.nn.Parameter]]] = OrderedDict(
        (group, []) for group in GROUP_NAMES
    )
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        identity = id(parameter)
        if identity in seen:
            raise ValueError(f"parameter appears more than once: {name}")
        seen.add(identity)
        groups[parameter_family(name)].append((name, parameter))
    empty = [group for group, values in groups.items() if not values]
    if empty:
        raise ValueError(f"empty parameter families: {empty}")
    return groups


def optimizer_groups(
    model: torch.nn.Module,
    base_lr: float,
    multipliers: Dict[str, float] | None = None,
) -> List[dict]:
    multipliers = multipliers or {}
    unknown = set(multipliers) - set(GROUP_NAMES)
    if unknown:
        raise ValueError(f"unknown parameter families: {sorted(unknown)}")
    result = []
    for group, values in grouped_named_parameters(model).items():
        multiplier = float(multipliers.get(group, 1.0))
        if multiplier < 0:
            raise ValueError(f"negative learning-rate multiplier for {group}")
        result.append(
            {
                "params": [parameter for _, parameter in values],
                "lr": base_lr * multiplier,
                "base_lr": base_lr,
                "lr_multiplier": multiplier,
                "group_name": group,
                "parameter_names": [name for name, _ in values],
            }
        )
    return result


def set_learning_rates(
    optimizer: torch.optim.Optimizer,
    base_lr: float,
    multipliers: Dict[str, float] | None = None,
) -> None:
    multipliers = multipliers or {}
    present = {group.get("group_name") for group in optimizer.param_groups}
    if set(multipliers) - present:
        raise ValueError(f"optimizer lacks groups: {sorted(set(multipliers) - present)}")
    for group in optimizer.param_groups:
        name = group.get("group_name")
        multiplier = float(multipliers.get(name, 1.0))
        group["base_lr"] = base_lr
        group["lr_multiplier"] = multiplier
        group["lr"] = base_lr * multiplier
