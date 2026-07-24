#!/usr/bin/env python
"""
Experiment 6 — Natural Language Demonstration.

Trains on the controlled entity-disambiguation dataset for
K ∈ {2, 5, 10, 20} × 3 seeds.

The dataset uses role/clause/action strings mapped through the standard
``<BOS> B <SEP> z <SEP> A <EOS>`` pipeline.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.data.tokenizer import CharTokenizer, create_tokenizer_from_config
from src.data.dataset import DisambiguationDataset, collate_fn
from src.data.natural_lang_dataset import generate_natural_lang_mappings
from src.model import create_model_from_config
from src.training import train

OUTPUT_DIR = Path("outputs")
K_VALUES = [2, 5, 10, 20]
SEEDS = [42, 43, 44]
N_GROUPS = 50


def _base_cfg(k: int, seed: int):
    cfg = OmegaConf.load(Path(__file__).parent.parent / "configs" / "base.yaml")
    cfg.data.n_unique_b = N_GROUPS
    cfg.data.k = k
    cfg.data.task = "bz_to_a"
    cfg.data.b_length = 6
    cfg.data.a_length = 6
    cfg.data.z_length = 6
    # Must include '_' in vocab for padding chars in the natural-lang strings
    cfg.data.vocab_chars = "abcdefghijklmnopqrstuvwxyz0123456789_"
    cfg.data.probe_fraction = 0.0
    cfg.data.split_by_base = True
    cfg.training.max_steps = 30_000
    cfg.training.warmup_steps = 500
    cfg.training.checkpoint_every = 100
    cfg.training.eval_every = 50
    cfg.training.batch_size = 128
    cfg.training.learning_rate = 1e-3
    cfg.training.weight_decay = 0.01
    cfg.training.scheduler = "cosine"
    cfg.experiment.seed = seed
    return cfg


def run_single(k: int, seed: int):
    name = f"exp6_natlang_k{k}_seed{seed}"
    cfg = _base_cfg(k, seed)
    cfg.experiment.name = name

    exp_dir = OUTPUT_DIR / name
    history_path = exp_dir / "training_history.json"
    if history_path.exists():
        print(f"[SKIP] {name}")
        return

    print(f"\n{'='*60}\n  {name}  (K={k}, seed={seed})\n{'='*60}")
    torch.manual_seed(seed)

    # Build the tokenizer with the extended vocab (includes '_')
    tokenizer = CharTokenizer(
        vocab_chars=cfg.data.vocab_chars,
        pad_token=cfg.tokenizer.pad_token,
        bos_token=cfg.tokenizer.bos_token,
        eos_token=cfg.tokenizer.eos_token,
        sep_token=cfg.tokenizer.sep_token,
    )

    mapping_data = generate_natural_lang_mappings(
        n_groups=N_GROUPS, k=k,
        b_length=cfg.data.b_length,
        z_length=cfg.data.z_length,
        a_length=cfg.data.a_length,
        seed=seed,
    )

    train_ds = DisambiguationDataset(
        mapping_data=mapping_data,
        tokenizer=tokenizer,
        split="train",
        probe_fraction=0.0,
        seed=seed,
        task="bz_to_a",
        split_by_base=True,
    )
    probe_ds = DisambiguationDataset(
        mapping_data=mapping_data,
        tokenizer=tokenizer,
        split="probe",
        probe_fraction=0.0,
        seed=seed,
        task="bz_to_a",
        split_by_base=True,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size,
                              shuffle=True, collate_fn=collate_fn, num_workers=0)
    probe_loader = DataLoader(probe_ds, batch_size=cfg.training.batch_size,
                              shuffle=False, collate_fn=collate_fn, num_workers=0)

    model = create_model_from_config(cfg, tokenizer)
    train(model, train_loader, probe_loader, cfg, OUTPUT_DIR)

    # Save metadata
    meta = {"k": k, "seed": seed, "n_groups": N_GROUPS, "dataset": "natural_lang"}
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "exp6_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def main():
    for k in K_VALUES:
        for seed in SEEDS:
            run_single(k, seed)
    print("\n[Exp 6] All runs complete.")


if __name__ == "__main__":
    main()
