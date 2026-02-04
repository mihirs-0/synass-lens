#!/usr/bin/env python
"""
Generate probe-only figures from saved probe_results.

Useful for incomplete runs where training_history.json is missing.
"""

import sys
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.visualize import (
    load_results,
    plot_attention_to_z_evolution,
    plot_logit_lens_evolution,
    plot_z_dependence_evolution,
    plot_random_z_sensitivity_evolution,
)


def main():
    parser = argparse.ArgumentParser(description="Plot probe-only figures")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment name")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Base output directory")
    args = parser.parse_args()

    experiment_dir = Path(args.output_dir) / args.experiment
    results_path = experiment_dir / "probe_results" / "all_probes.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Probe results not found at {results_path}")

    out_dir = experiment_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(results_path)
    exp_name = experiment_dir.name

    plot_attention_to_z_evolution(
        results,
        save_path=out_dir / "attention_to_z.png",
        title=f"{exp_name}: Attention to z (probe-only)",
    )

    plot_logit_lens_evolution(
        results,
        save_path=out_dir / "logit_lens.png",
        title=f"{exp_name}: Logit Lens (probe-only)",
    )

    plot_z_dependence_evolution(
        results,
        save_path=out_dir / "z_dependence.png",
        title=f"{exp_name}: Causal z-Dependence (probe-only)",
    )

    if "random_z_eval" in results["probe_results"]:
        plot_random_z_sensitivity_evolution(
            results,
            save_path=out_dir / "random_z_sensitivity.png",
            title=f"{exp_name}: Random-z Sensitivity (probe-only)",
        )

    print(f"Saved probe-only figures to {out_dir}")


if __name__ == "__main__":
    main()
