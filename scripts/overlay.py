#!/usr/bin/env python
"""
Generate overlay figures across multiple experiments.

Usage:
    python scripts/overlay.py --experiments k1_n1000 k10_n1000
"""

import sys
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.visualize import generate_overlay_figures


def main():
    parser = argparse.ArgumentParser(description="Overlay figures across experiments")
    parser.add_argument("--experiments", nargs="+", required=True, help="Experiment names")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Base output directory")
    parser.add_argument("--overlay-dir", type=str, default="overlays", help="Overlay output folder name")
    args = parser.parse_args()
    
    base_dir = Path(args.output_dir)
    experiment_dirs = [base_dir / name for name in args.experiments]
    output_dir = base_dir / args.overlay_dir
    
    generate_overlay_figures(experiment_dirs, output_dir)
    print(f"Overlay figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
