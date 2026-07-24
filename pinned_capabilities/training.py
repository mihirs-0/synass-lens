"""Shared training primitives for branched MBC experiments."""

from __future__ import annotations

from typing import Dict

import torch

from src.data import collate_fn
from src.training.trainer import compute_loss

from .batch_stream import DeterministicBatchStream
from .parameter_groups import optimizer_groups


def make_adamw(
    model: torch.nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    multipliers: Dict[str, float] | None = None,
    betas: tuple[float, float] = (0.9, 0.999),
) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        optimizer_groups(model, learning_rate, multipliers),
        lr=learning_rate,
        betas=betas,
        weight_decay=weight_decay,
    )


def next_batch(dataset, stream: DeterministicBatchStream, device: str) -> dict:
    items = [dataset[int(index)] for index in stream.next_indices()]
    batch = collate_fn(items)
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict,
    *,
    scheduler=None,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, accuracy, first_target_loss = compute_loss(model, batch)
    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return {
        "train_loss": float(loss.item()),
        "train_accuracy": float(accuracy) if accuracy is not None else float("nan"),
        "first_target_loss": float(first_target_loss),
    }
