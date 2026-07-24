#!/usr/bin/env python3
"""
Generate Hydra YAML configs for the phase boundary experiment.

Creates one config per (K, η, seed) combination in configs/experiments/.
Reuses existing results where available (K=20/η=1e-3/seed=42, etc.).

Usage:
    python scripts/generate_phase_boundary_configs.py [--dry-run]
"""

import argparse
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs" / "experiments"
OUTPUTS_DIR = ROOT / "outputs"

GRID = {
    5:  ["5e-3", "7e-3", "1e-2", "1.5e-2", "2e-2"],
    10: ["3e-3", "5e-3", "7e-3", "1e-2"],
    20: ["2e-3", "3e-3", "5e-3", "7e-3"],
    36: ["1e-3", "2e-3", "3e-3", "5e-3"],
}

SEEDS = [42, 123, 7]

# Only reuse prior runs that SUCCEEDED (transitioned within their max_steps).
# K=20/η=5e-3 FAILED at 50K steps — must re-run at 200K to be sure.
EXISTING_RUNS = {
    (20, "2e-3", 42): "lr_sweep_eta2e-3",     # τ≈7550, succeeded
    (36, "1e-3", 42): "landauer_dense_k36",    # τ≈6350, succeeded
}

CONFIG_TEMPLATE = """\
# Phase boundary: K={k}, η={lr_str}, seed={seed}
experiment:
  name: "{name}"
  seed: {seed}

data:
  n_unique_b: 1000
  k: {k}
  task: "bz_to_a"
  b_length: 6
  a_length: 4
  z_length: 2
  vocab_chars: "abcdefghijklmnopqrstuvwxyz0123456789"
  probe_fraction: 0.0
  split_by_base: true
  enforce_unique_a_first_char_per_b: true
  disambiguation_prefix_length: 1

tokenizer:
  pad_token: "<PAD>"
  bos_token: "<BOS>"
  eos_token: "<EOS>"
  sep_token: "<SEP>"

model:
  n_layers: 4
  n_heads: 4
  d_model: 128
  d_head: 32
  d_mlp: 512
  act_fn: "gelu"

training:
  batch_size: 128
  learning_rate: {lr_float}
  weight_decay: 0.01
  max_steps: 200000
  warmup_steps: 0
  scheduler: "constant"
  checkpoint_every: 999999
  eval_every: 50
  early_stop_convergence_frac: 0.01

probes:
  enabled: []

output:
  base_dir: "outputs"
"""


def run_name(k, lr_str, seed):
    return f"pb_K{k}_lr{lr_str}_s{seed}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print without writing")
    args = parser.parse_args()

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    new = 0
    reused = 0
    skipped = 0

    reuse_map = {}

    for k, lr_strs in sorted(GRID.items()):
        for lr_str in lr_strs:
            for seed in SEEDS:
                total += 1
                name = run_name(k, lr_str, seed)

                existing_key = (k, lr_str, seed)
                if existing_key in EXISTING_RUNS:
                    prior = EXISTING_RUNS[existing_key]
                    prior_hist = OUTPUTS_DIR / prior / "training_history.json"
                    if prior_hist.exists():
                        reuse_map[name] = prior
                        reused += 1
                        continue

                # Check if already generated and trained
                out_dir = OUTPUTS_DIR / name
                if (out_dir / "training_history.json").exists():
                    skipped += 1
                    continue

                lr_float = float(lr_str)
                config_content = CONFIG_TEMPLATE.format(
                    k=k, lr_str=lr_str, lr_float=lr_float,
                    seed=seed, name=name,
                )

                config_path = CONFIG_DIR / f"{name}.yaml"
                if args.dry_run:
                    print(f"  [DRY] Would write: {config_path}")
                else:
                    with open(config_path, "w") as f:
                        f.write(config_content)
                new += 1

    print(f"\nPhase Boundary Config Generation")
    print(f"  Total grid points: {total}")
    print(f"  New configs written: {new}")
    print(f"  Reusing prior runs: {reused}")
    print(f"  Already trained:    {skipped}")

    if reuse_map:
        print(f"\n  Prior runs to symlink/copy:")
        for name, prior in sorted(reuse_map.items()):
            print(f"    {name} ← {prior}")

    # Write the reuse map so the shell script can handle it
    if not args.dry_run and reuse_map:
        map_path = ROOT / "configs" / "phase_boundary_reuse_map.txt"
        with open(map_path, "w") as f:
            for name, prior in sorted(reuse_map.items()):
                f.write(f"{name}\t{prior}\n")
        print(f"\n  Reuse map written to: {map_path}")

    # Write the full run list (excluding reused/completed)
    if not args.dry_run:
        run_list_path = ROOT / "configs" / "phase_boundary_run_list.txt"
        entries = []
        for k, lr_strs in sorted(GRID.items()):
            for lr_str in lr_strs:
                for seed in SEEDS:
                    name = run_name(k, lr_str, seed)
                    if name in reuse_map:
                        continue
                    out_dir = OUTPUTS_DIR / name
                    if (out_dir / "training_history.json").exists():
                        continue
                    entries.append(name)

        with open(run_list_path, "w") as f:
            for e in entries:
                f.write(e + "\n")
        print(f"  Run list ({len(entries)} runs) written to: {run_list_path}")


if __name__ == "__main__":
    main()
