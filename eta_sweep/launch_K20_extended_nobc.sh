#!/bin/bash
# Re-run the 6 decoupled no-BC K=20 cells with extended budget (max_steps=14000).
# These were underbudget in T1b at max_steps=7000.
# Output goes to a new subdir to avoid clobbering T1b results.

set -e
cd /Users/mihir/synass-lens/synass-lens
mkdir -p eta_sweep/results/T1b_K20_extended
mkdir -p logs/overnight

CONFIGS=(
  "adamw_b1_0_nobc"        # decoupled no-BC: 3/3 inconclusive at 7000
  "rmsprop_decoupled_only" # decoupled no-BC: 2/3 trans, 1 inconclusive at 7000
)

COMMANDS=()
for cfg in "${CONFIGS[@]}"; do
  for seed in 0 1 2; do
    log="logs/overnight/K20_extended_${cfg}_seed${seed}.log"
    cmd="python eta_sweep/run_optimizer_branch.py --branch-config $cfg --k 20 --eta 0.001 --seed $seed --max-steps 14000 --output-subdir T1b_K20_extended > $log 2>&1"
    COMMANDS+=("$cmd")
  done
done

echo "=== K=20 extended budget: 2 configs × 3 seeds = 6 runs (max_steps=14000) ==="
echo "Launching all 6 in parallel..."

pids=()
for cmd in "${COMMANDS[@]}"; do
  eval "$cmd" &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid" || true; done

echo "K20_extended done."
date
