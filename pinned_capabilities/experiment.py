"""Self-contained MBC experiment construction and logging."""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from src.data import CharTokenizer, DisambiguationDataset, generate_mappings

from .batch_stream import DeterministicBatchStream
from .config import MBCExperimentConfig, MetricConfig
from .mbc import MBCProbe, build_mbc_probes, evaluate_mbc_probe
from .metrics import Quartets, sample_quartets
from .training import make_adamw, next_batch, train_step


def select_device(requested: str) -> str:
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(config: MBCExperimentConfig, tokenizer: CharTokenizer, device: str) -> HookedTransformer:
    model_config = HookedTransformerConfig(
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_model=config.d_model,
        d_head=config.d_head,
        d_mlp=config.d_mlp,
        d_vocab=tokenizer.vocab_size,
        n_ctx=32,
        act_fn=config.act_fn,
        positional_embedding_type="standard",
        normalization_type="LN",
        attn_only=False,
        device=device,
        seed=config.seed,
        init_weights=True,
    )
    return HookedTransformer(model_config)


class JSONLWriter:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, row: Dict[str, Any]) -> None:
        with self.path.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


class MBCExperiment:
    def __init__(self, config: MBCExperimentConfig, metric: Optional[MetricConfig] = None) -> None:
        config.validate()
        self.config = config
        self.metric = metric or MetricConfig()
        self.metric.validate()
        seed_everything(config.seed)
        self.device = select_device(config.device)
        self.tokenizer = CharTokenizer(vocab_chars=config.vocab_chars)
        self.mapping = generate_mappings(
            n_unique_b=config.n_unique_b,
            k=config.k,
            b_length=config.b_length,
            a_length=config.a_length,
            z_length=config.z_length,
            vocab_chars=config.vocab_chars,
            seed=1_000_003 * config.seed + config.k,
            task="bz_to_a",
            enforce_unique_a_first_char_per_b=True,
        )
        self.dataset = DisambiguationDataset(
            self.mapping,
            self.tokenizer,
            split="train",
            probe_fraction=0.0,
            seed=config.seed,
            task="bz_to_a",
            split_by_base=True,
        )
        self.model = build_model(config, self.tokenizer, self.device)
        self.optimizer = make_adamw(
            self.model,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.stream = DeterministicBatchStream(
            len(self.dataset), config.batch_size, seed=7_919 + config.seed
        )
        probes = build_mbc_probes(
            self.mapping,
            self.tokenizer,
            n_b=config.probe_b_count,
            seeds=self.metric.probe_seeds,
        )
        self.probes: tuple[MBCProbe, MBCProbe] = tuple(
            probe.to(self.device) for probe in probes
        )  # type: ignore[assignment]
        self.quartets: tuple[Quartets, Quartets] = tuple(
            sample_quartets(
                config.probe_b_count,
                config.k,
                self.metric.quartets_per_probe,
                seed,
                device=self.device,
            )
            for seed in self.metric.probe_seeds
        )  # type: ignore[assignment]
        self.step = 0

    def evaluate(self) -> Dict[str, float]:
        rows = [
            evaluate_mbc_probe(self.model, probe, quartets)
            for probe, quartets in zip(self.probes, self.quartets)
        ]
        result: Dict[str, float] = {"step": float(self.step)}
        for probe_index, row in enumerate(rows):
            for key, value in row.items():
                result[f"probe_{probe_index}_{key}"] = value
        for key in rows[0]:
            result[key] = sum(row[key] for row in rows) / len(rows)
        return result

    def advance(self, steps: int, writer: Optional[JSONLWriter] = None) -> Dict[str, float]:
        latest: Dict[str, float] = {}
        for _ in range(steps):
            batch = next_batch(self.dataset, self.stream, self.device)
            latest = train_step(self.model, self.optimizer, batch)
            self.step += 1
            if writer is not None and self.step % self.metric.eval_every == 0:
                writer.write({**latest, **self.evaluate()})
        return latest

    def config_dict(self) -> Dict[str, Any]:
        return asdict(self.config)
