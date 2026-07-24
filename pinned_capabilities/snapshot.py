"""Complete snapshots for branching weights, optimizer state, and data order."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from .batch_stream import DeterministicBatchStream


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state and hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])


def save_snapshot(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    stream: DeterministicBatchStream,
    step: int,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "step": int(step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "stream": stream.state_dict(),
        "rng": capture_rng_state(),
        "metadata": metadata or {},
    }
    torch.save(payload, path)
    return path


def load_snapshot(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    stream: DeterministicBatchStream,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    restore_rng: bool = True,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported snapshot schema: {payload.get('schema_version')}")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    stream.load_state_dict(payload["stream"])
    if scheduler is not None:
        if payload["scheduler"] is None:
            raise ValueError("snapshot has no scheduler state")
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None:
        if payload.get("scaler") is None:
            raise ValueError("snapshot has no gradient-scaler state")
        scaler.load_state_dict(payload["scaler"])
    if restore_rng:
        restore_rng_state(payload["rng"])
    return {
        "step": int(payload["step"]),
        "metadata": payload.get("metadata", {}),
    }


def load_crossed_snapshot(
    *,
    weights_path: Path,
    optimizer_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    stream: DeterministicBatchStream,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    restore_rng: bool = True,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    """Cross weights with the complete non-weight training state.

    Data order and RNG belong to the optimizer-state source because the Gate 1
    intervention defines that source as the complete update process state.
    """
    weights_payload = torch.load(Path(weights_path), map_location=map_location, weights_only=False)
    state_payload = torch.load(Path(optimizer_path), map_location=map_location, weights_only=False)
    if weights_payload.get("schema_version") != 1 or state_payload.get("schema_version") != 1:
        raise ValueError("crossed snapshots require schema version 1")
    model.load_state_dict(weights_payload["model"])
    optimizer.load_state_dict(state_payload["optimizer"])
    stream.load_state_dict(state_payload["stream"])
    if scheduler is not None:
        if state_payload.get("scheduler") is None:
            raise ValueError("optimizer-state snapshot has no scheduler")
        scheduler.load_state_dict(state_payload["scheduler"])
    if scaler is not None:
        if state_payload.get("scaler") is None:
            raise ValueError("optimizer-state snapshot has no gradient scaler")
        scaler.load_state_dict(state_payload["scaler"])
    if restore_rng:
        restore_rng_state(state_payload["rng"])
    return {
        "step": int(state_payload["step"]),
        "weights_metadata": weights_payload.get("metadata", {}),
        "optimizer_metadata": state_payload.get("metadata", {}),
    }
