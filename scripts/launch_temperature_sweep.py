#!/usr/bin/env python
"""
Launch temperature sweep experiments for disambiguation lag analysis.

Generates commands for a fixed-K sweep over (learning rate, batch size).
"""

import argparse

K_VALUE = 10
N_UNIQUE_B = 1000
TASK = "bz_to_a"
SEED = 42

DEFAULT_MAX_STEPS = 10000
DEFAULT_CHECKPOINT_EVERY = 100
DEFAULT_EVAL_EVERY = 50

SWEEP_CONFIGS = [
    {"name": "temp_lr1e3_bs64", "lr": 1e-3, "bs": 64},
    {"name": "temp_lr1e3_bs128", "lr": 1e-3, "bs": 128},
    {"name": "temp_lr1e3_bs256", "lr": 1e-3, "bs": 256},
    {"name": "temp_lr1e3_bs512", "lr": 1e-3, "bs": 512},
    {"name": "temp_lr5e4_bs128", "lr": 5e-4, "bs": 128},
    {"name": "temp_lr2e3_bs128", "lr": 2e-3, "bs": 128},
]


def generate_commands(
    sweep_configs=SWEEP_CONFIGS,
    max_steps=DEFAULT_MAX_STEPS,
    checkpoint_every=DEFAULT_CHECKPOINT_EVERY,
    eval_every=DEFAULT_EVAL_EVERY,
    output_dir="outputs",
):
    """Generate training commands for temperature sweep."""
    commands = []

    for cfg in sweep_configs:
        exp_name = cfg["name"]
        lr = cfg["lr"]
        bs = cfg["bs"]

        cmd = f"""python scripts/train.py --config-path ../configs --config-name base \\
  experiment.name={exp_name} \\
  data.task={TASK} \\
  data.k={K_VALUE} \\
  data.n_unique_b={N_UNIQUE_B} \\
  data.enforce_unique_a_first_char_per_b=true \\
  data.probe_fraction=0.0 \\
  training.learning_rate={lr} \\
  training.batch_size={bs} \\
  training.max_steps={max_steps} \\
  training.checkpoint_every={checkpoint_every} \\
  training.eval_every={eval_every} \\
  experiment.seed={SEED} \\
  output.base_dir={output_dir}"""

        commands.append((exp_name, cmd))

    return commands


def main():
    parser = argparse.ArgumentParser(description="Generate temperature sweep commands")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS,
                        help="Max training steps")
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY,
                        help="Checkpoint frequency")
    parser.add_argument("--eval-every", type=int, default=DEFAULT_EVAL_EVERY,
                        help="Evaluation frequency")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Output directory")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of parallel runs (for multi-GPU)")
    args = parser.parse_args()

    commands = generate_commands(
        max_steps=args.max_steps,
        checkpoint_every=args.checkpoint_every,
        eval_every=args.eval_every,
        output_dir=args.output_dir,
    )

    print("=" * 60)
    print("Temperature Sweep Experiment Commands")
    print("=" * 60)
    print(f"K value: {K_VALUE}")
    print(f"Task: {TASK}")
    print(f"Total runs: {len(commands)}")
    print("=" * 60)

    if args.parallel > 1:
        print(f"\n### For {args.parallel}-GPU parallel execution:")
        print("Run these in separate terminals:\n")

        for i, (name, cmd) in enumerate(commands):
            gpu_id = i % args.parallel
            print(f"# Terminal {gpu_id + 1}: {name}")
            print(f"CUDA_VISIBLE_DEVICES={gpu_id} {cmd}")
            print()
    else:
        print("\n### Sequential execution (copy-paste each):\n")
        for name, cmd in commands:
            print(f"# {name}")
            print(cmd)
            print()

    print("=" * 60)
    print("After training, run analysis:")
    print("=" * 60)
    print(f"python scripts/analyze_temperature_sweep.py --output-dir {args.output_dir}")


if __name__ == "__main__":
    main()
