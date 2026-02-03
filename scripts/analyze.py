#!/usr/bin/env python
"""
Analysis script: Run probes on checkpoints and generate figures.

Usage:
    python scripts/analyze.py --experiment k10_n1000
    python scripts/analyze.py --experiment k10_n1000 --probes attention_to_z logit_lens
"""

import sys
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.data import create_tokenizer_from_config, create_datasets_from_config, collate_fn
from src.model import create_model_from_config
from src.analysis import run_analysis, generate_all_figures


def main():
    parser = argparse.ArgumentParser(description="Run probes and generate figures")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment name")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Base output directory")
    parser.add_argument("--probes", nargs="+", default=None, help="Probes to run (default: all)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for probing")
    parser.add_argument("--figures-only", action="store_true", help="Only generate figures from existing results")
    args = parser.parse_args()
    
    # Setup paths
    experiment_dir = Path(args.output_dir) / args.experiment
    config_path = experiment_dir / "config.yaml"
    
    if not config_path.exists():
        print(f"Error: Config not found at {config_path}")
        print("Make sure you've run training first.")
        sys.exit(1)
    
    # Load config
    cfg = OmegaConf.load(config_path)
    
    print("=" * 60)
    print(f"Analyzing experiment: {args.experiment}")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if not args.figures_only:
        # Create tokenizer and dataset
        tokenizer = create_tokenizer_from_config(cfg)
        train_dataset, probe_dataset, mapping_data = create_datasets_from_config(cfg, tokenizer)
        
        print(f"Probe examples: {len(probe_dataset)}")
        
        # Model factory (creates fresh model for each checkpoint)
        def model_factory():
            return create_model_from_config(cfg, tokenizer)
        
        # Run analysis
        print("\nRunning probes on checkpoints...")
        results = run_analysis(
            experiment_dir=experiment_dir,
            dataset=probe_dataset,
            tokenizer=tokenizer,
            model_factory=model_factory,
            probe_names=args.probes,
            batch_size=args.batch_size,
            device=device,
        )
    
    # Generate figures
    print("\nGenerating figures...")
    generate_all_figures(experiment_dir)
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Figures saved to: {experiment_dir / 'figures'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
