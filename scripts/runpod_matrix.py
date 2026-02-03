#!/usr/bin/env python
"""
Generate RunPod command matrix for Bz->A vs Az->B runs.

Usage:
    python scripts/runpod_matrix.py --ks 1 2 4 8 16 25
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Generate RunPod command matrix")
    parser.add_argument("--ks", nargs="+", type=int, required=True, help="K values to sweep")
    parser.add_argument("--n-unique-b", type=int, default=1000, help="n_unique_b value")
    parser.add_argument("--config-path", type=str, default="../configs", help="Hydra config path")
    parser.add_argument("--config-name", type=str, default="base", help="Hydra config name")
    parser.add_argument("--include-analyze", action="store_true", help="Include analyze commands")
    parser.add_argument("--include-overlay", action="store_true", help="Include overlay commands per K")
    args = parser.parse_args()

    ks = args.ks
    n_unique_b = args.n_unique_b

    print("# Bz->A training commands")
    for k in ks:
        print(
            "python scripts/train.py "
            f"--config-path {args.config_path} --config-name {args.config_name} "
            f"experiment.name=bz_to_a_k{k} data.task=bz_to_a data.k={k} data.n_unique_b={n_unique_b}"
        )

    print("\n# Az->B training commands")
    for k in ks:
        print(
            "python scripts/train.py "
            f"--config-path {args.config_path} --config-name {args.config_name} "
            f"experiment.name=az_to_b_k{k} data.task=az_to_b data.k={k} data.n_unique_b={n_unique_b}"
        )

    if args.include_analyze:
        print("\n# Analyze commands")
        for k in ks:
            print(f"python scripts/analyze.py --experiment bz_to_a_k{k}")
            print(f"python scripts/analyze.py --experiment az_to_b_k{k}")

    if args.include_overlay:
        print("\n# Overlay commands (per K)")
        for k in ks:
            print(f"python scripts/overlay.py --experiments bz_to_a_k{k} az_to_b_k{k} --overlay-dir overlays_k{k}")


if __name__ == "__main__":
    main()
