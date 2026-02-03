#!/usr/bin/env python
"""
RunPod launcher for parallel experiments.

This script helps you run multiple K configurations in parallel on RunPod.

Usage:
    # Generate launch commands for all K values
    python scripts/launch_runpod.py --generate
    
    # Run a specific K value (use this inside RunPod)
    python scripts/launch_runpod.py --k 10
"""

import sys
from pathlib import Path
import argparse
import subprocess

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Default experiments to run
DEFAULT_EXPERIMENTS = [
    {"k": 1, "n_unique_b": 1000, "name": "k1_n1000"},
    {"k": 10, "n_unique_b": 1000, "name": "k10_n1000"},
    {"k": 50, "n_unique_b": 1000, "name": "k50_n1000"},
    {"k": 100, "n_unique_b": 1000, "name": "k100_n1000"},
]


def generate_commands():
    """Generate bash commands for all experiments."""
    print("# Commands to run on separate RunPod instances:")
    print("# Copy each line to a different GPU")
    print()
    
    for exp in DEFAULT_EXPERIMENTS:
        cmd = (
            f"python scripts/train.py "
            f"experiment.name={exp['name']} "
            f"data.k={exp['k']} "
            f"data.n_unique_b={exp['n_unique_b']}"
        )
        print(f"# K={exp['k']}")
        print(cmd)
        print()


def run_experiment(k: int, n_unique_b: int = 1000):
    """Run a single experiment."""
    name = f"k{k}_n{n_unique_b}"
    
    cmd = [
        "python", "scripts/train.py",
        f"experiment.name={name}",
        f"data.k={k}",
        f"data.n_unique_b={n_unique_b}",
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    # Also run analysis
    cmd_analyze = [
        "python", "scripts/analyze.py",
        f"--experiment={name}",
    ]
    print(f"Running: {' '.join(cmd_analyze)}")
    subprocess.run(cmd_analyze, check=True)


def main():
    parser = argparse.ArgumentParser(description="RunPod launcher")
    parser.add_argument("--generate", action="store_true", help="Generate commands for all experiments")
    parser.add_argument("--k", type=int, help="Run experiment for specific K value")
    parser.add_argument("--n-unique-b", type=int, default=1000, help="Number of unique B strings")
    parser.add_argument("--all", action="store_true", help="Run all experiments sequentially (not recommended)")
    args = parser.parse_args()
    
    if args.generate:
        generate_commands()
    elif args.k is not None:
        run_experiment(args.k, args.n_unique_b)
    elif args.all:
        print("Running all experiments sequentially...")
        for exp in DEFAULT_EXPERIMENTS:
            run_experiment(exp["k"], exp["n_unique_b"])
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
