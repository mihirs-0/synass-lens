#!/usr/bin/env python
"""
Train alternative architectures (GatedMLP, RNN) on the disambiguation task.

Uses the SAME data pipeline, training loop, loss function, optimizer, and
evaluation diagnostics as the transformer experiments.  The only independent
variable is the model architecture.

Usage:
    python scripts/train_alternative_arch.py --arch gated_mlp --k 20 --name gatedmlp_k20
    python scripts/train_alternative_arch.py --arch rnn --k 20 --name rnn_k20
"""

import sys
import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import create_tokenizer_from_config, create_datasets_from_config, collate_fn
from src.model.gated_mlp import create_gated_mlp_from_config
from src.model.rnn_model import create_rnn_from_config
from src.training.trainer import train


# ------------------------------------------------------------------
# Config helpers
# ------------------------------------------------------------------
def _ns_to_dict(ns):
    """Recursively convert SimpleNamespace tree to plain dicts (for YAML)."""
    d = {}
    for k, v in vars(ns).items():
        if isinstance(v, SimpleNamespace):
            d[k] = _ns_to_dict(v)
        else:
            d[k] = v
    return d


def build_config(args) -> SimpleNamespace:
    """Build a config namespace matching the YAML structure used by all scripts."""
    cfg = SimpleNamespace(
        experiment=SimpleNamespace(
            name=args.name,
            seed=args.seed,
        ),
        data=SimpleNamespace(
            n_unique_b=1000,
            k=args.k,
            task="bz_to_a",
            b_length=6,
            a_length=4,
            z_length=2,
            vocab_chars="abcdefghijklmnopqrstuvwxyz0123456789",
            probe_fraction=0.0,
            split_by_base=True,
            enforce_unique_a_first_char_per_b=True,
            disambiguation_prefix_length=args.disambiguation_prefix_length,
            label_noise_prob=0.0,
        ),
        tokenizer=SimpleNamespace(
            pad_token="<PAD>",
            bos_token="<BOS>",
            eos_token="<EOS>",
            sep_token="<SEP>",
        ),
        model=SimpleNamespace(
            architecture=args.arch,
            d_model=args.d_model,
            d_hidden=args.d_hidden,
            n_rnn_layers=args.n_rnn_layers,
            # Transformer-specific fields (unused by alt archs, but kept
            # so that analysis scripts can read the config without error).
            n_layers=4,
            n_heads=4,
            d_head=32,
            d_mlp=512,
            act_fn="gelu",
        ),
        training=SimpleNamespace(
            batch_size=args.bs,
            learning_rate=args.lr,
            weight_decay=0.01,
            max_steps=args.max_steps,
            warmup_steps=0,
            scheduler=args.scheduler,
            checkpoint_every=args.checkpoint_every,
            eval_every=50,
        ),
        output=SimpleNamespace(
            base_dir="outputs",
        ),
    )
    return cfg


def save_config(cfg: SimpleNamespace, experiment_dir: Path):
    """Persist config as YAML so analysis scripts can reload it."""
    config_path = experiment_dir / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(_ns_to_dict(cfg), f, default_flow_style=False)
    print(f"Saved config to {config_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train alternative architectures for the Landauer architecture ablation"
    )
    parser.add_argument("--arch", type=str, required=True,
                        choices=["gated_mlp", "rnn"],
                        help="Architecture: gated_mlp or rnn")
    parser.add_argument("--k", type=int, required=True,
                        help="K (number of candidate targets per base)")
    parser.add_argument("--name", type=str, required=True,
                        help="Experiment name (used as output subdirectory)")
    parser.add_argument("--max-steps", type=int, default=30000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--scheduler", type=str, default="constant")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d-model", type=int, default=128,
                        help="Embedding / model dimension")
    parser.add_argument("--d-hidden", type=int, default=None,
                        help="Hidden dimension (default: 512 for gated_mlp, 256 for rnn)")
    parser.add_argument("--n-rnn-layers", type=int, default=2,
                        help="Number of LSTM layers (only for --arch rnn)")
    parser.add_argument("--disambiguation-prefix-length", type=int, default=1,
                        help="Prefix length for unique A disambiguation (1=first char, 2=first two chars)")
    parser.add_argument("--checkpoint-every", type=int, default=100,
                        help="Save checkpoint every N steps")
    args = parser.parse_args()

    # Architecture-specific hidden dim defaults
    if args.d_hidden is None:
        args.d_hidden = 512 if args.arch == "gated_mlp" else 256

    cfg = build_config(args)

    # Reproducibility
    torch.manual_seed(cfg.experiment.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.experiment.seed)

    # Data
    tokenizer = create_tokenizer_from_config(cfg)
    train_dataset, probe_dataset, mapping_data = create_datasets_from_config(
        cfg, tokenizer
    )

    print(f"Architecture : {args.arch}")
    print(f"K            : {args.k}")
    print(f"Train size   : {len(train_dataset):,}")
    print(f"Vocab size   : {tokenizer.vocab_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    probe_loader = DataLoader(
        probe_dataset if len(probe_dataset) > 0 else train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Model
    if args.arch == "gated_mlp":
        model = create_gated_mlp_from_config(cfg, tokenizer)
    else:
        model = create_rnn_from_config(cfg, tokenizer)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters   : {n_params:,}")
    print(f"Device       : {model.cfg.device}")

    # Persist config before training starts
    output_dir = Path(cfg.output.base_dir)
    experiment_dir = output_dir / cfg.experiment.name
    save_config(cfg, experiment_dir)

    # Train — reuses the EXACT same loop as the transformer experiments
    history = train(model, train_loader, probe_loader, cfg, output_dir)

    print(f"\nTraining complete.  Results in: {experiment_dir}")


if __name__ == "__main__":
    main()
