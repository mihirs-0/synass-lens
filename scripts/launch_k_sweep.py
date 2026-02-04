#!/usr/bin/env python
"""
Launch K-sweep experiments for Late Disambiguation Lag analysis.

Generates commands for running Bz->A and Az->B across multiple K values.
"""

import argparse

K_VALUES = [5, 10, 20]
TASKS = ["bz_to_a", "az_to_b"]

# Shorter runs with more frequent checkpoints to capture dynamics
DEFAULT_MAX_STEPS = 5000
DEFAULT_CHECKPOINT_EVERY = 50
DEFAULT_EVAL_EVERY = 50


def generate_commands(
    k_values=K_VALUES,
    tasks=TASKS,
    max_steps=DEFAULT_MAX_STEPS,
    checkpoint_every=DEFAULT_CHECKPOINT_EVERY,
    eval_every=DEFAULT_EVAL_EVERY,
    output_dir="outputs",
):
    """Generate training commands for K-sweep."""
    commands = []
    
    for k in k_values:
        for task in tasks:
            exp_name = f"{task}_k{k}"
            
            # Only enforce unique first chars for bz_to_a (where it matters)
            enforce_unique = "true" if task == "bz_to_a" else "false"
            
            cmd = f"""python scripts/train.py --config-path ../configs --config-name base \\
  experiment.name={exp_name} \\
  data.task={task} \\
  data.k={k} \\
  data.enforce_unique_a_first_char_per_b={enforce_unique} \\
  data.probe_fraction=0.0 \\
  training.max_steps={max_steps} \\
  training.checkpoint_every={checkpoint_every} \\
  training.eval_every={eval_every} \\
  output.base_dir={output_dir}"""
            
            commands.append((exp_name, cmd))
    
    return commands


def main():
    parser = argparse.ArgumentParser(description="Generate K-sweep commands")
    parser.add_argument("--k-values", nargs="+", type=int, default=K_VALUES,
                        help="K values to sweep")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS,
                        help="Max training steps")
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY,
                        help="Checkpoint frequency")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Output directory")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of parallel runs (for multi-GPU)")
    args = parser.parse_args()
    
    commands = generate_commands(
        k_values=args.k_values,
        max_steps=args.max_steps,
        checkpoint_every=args.checkpoint_every,
        output_dir=args.output_dir,
    )
    
    print("=" * 60)
    print("K-Sweep Experiment Commands")
    print("=" * 60)
    print(f"K values: {args.k_values}")
    print(f"Tasks: {TASKS}")
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
    
    # Also print analysis commands
    print("=" * 60)
    print("After training, run analysis:")
    print("=" * 60)
    for name, _ in commands:
        print(f"python scripts/analyze.py --experiment {name}")
    
    print("\n# Then generate K-sweep overlay:")
    print(f"python scripts/analyze_k_sweep.py --k-values {' '.join(map(str, args.k_values))} --output-dir {args.output_dir}")


if __name__ == "__main__":
    main()
