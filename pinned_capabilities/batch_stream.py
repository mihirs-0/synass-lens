"""Serializable minibatch order for exact branch-and-resume experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch


class DeterministicBatchStream:
    def __init__(self, n_items: int, batch_size: int, seed: int, drop_last: bool = False) -> None:
        if n_items <= 0 or batch_size <= 0:
            raise ValueError("n_items and batch_size must be positive")
        self.n_items = int(n_items)
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.epoch = 0
        self.cursor = 0
        self.order = torch.randperm(self.n_items, generator=self.generator)

    def _next_epoch(self) -> None:
        self.epoch += 1
        self.cursor = 0
        self.order = torch.randperm(self.n_items, generator=self.generator)

    def next_indices(self) -> torch.Tensor:
        remaining = self.n_items - self.cursor
        if remaining == 0 or (self.drop_last and remaining < self.batch_size):
            self._next_epoch()
        stop = min(self.cursor + self.batch_size, self.n_items)
        indices = self.order[self.cursor:stop].clone()
        self.cursor = stop
        return indices

    def state_dict(self) -> Dict[str, Any]:
        return {
            "n_items": self.n_items,
            "batch_size": self.batch_size,
            "drop_last": self.drop_last,
            "epoch": self.epoch,
            "cursor": self.cursor,
            "order": self.order.clone(),
            "generator_state": self.generator.get_state().clone(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        invariant = (int(state["n_items"]), int(state["batch_size"]), bool(state["drop_last"]))
        if invariant != (self.n_items, self.batch_size, self.drop_last):
            raise ValueError("batch-stream shape/configuration mismatch")
        self.epoch = int(state["epoch"])
        self.cursor = int(state["cursor"])
        self.order = state["order"].clone().cpu()
        self.generator.set_state(state["generator_state"].clone().cpu())
