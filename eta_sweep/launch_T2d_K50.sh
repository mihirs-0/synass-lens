#!/bin/bash
# T2d: 4-cell decomposition × 3 seeds at K=50 (12 runs).
# At K=50, τ for AdamW is ~5500-7500 (per phase diagram data).
# Branch step still 1500 (pre-plateau under canonical AdamW).
# After branch, target optimizer runs to budget. Need longer budget than K=20.
# Set max_steps=14000 (= 1500 branch + 12500 post-branch, allows for ~2x slowdown vs K=20).

set -e
cd /Users/mihir/synass-lens/synass-lens
mkdir -p eta_sweep/results/T2d_K50_4cell
mkdir -p logs/overnight

CONFIGS=(
  "adamw_b1_0"           # decoupled+BC: should transition
  "rmsprop_decoupled_bc" # decoupled+BC: should transition
  "adam_coupled_b1_0"    # L2: should be stuck
  "rmsprop"              # L2: should be stuck
)

# 12 cells launched in two batches of 6 each (lower contention than 8).
COMMANDS=()
for cfg in "${CONFIGS[@]}"; do
  for seed in 0 1 2; do
    log="logs/overnight/T2d_K50_${cfg}_seed${seed}.log"
    cmd="python eta_sweep/run_optimizer_branch.py --branch-config $cfg --k 50 --eta 0.0003 --seed $seed --max-steps 14000 --output-subdir T2d_K50_4cell > $log 2>&1"
    COMMANDS+=("$cmd")
  done
done

echo "=== T2d: 4-cell × 3 seeds at K=50 (12 runs) ==="
echo "Launching all 12 in two batches of 6..."

batch_size=6
pids=()
for cmd in "${COMMANDS[@]}"; do
  eval "$cmd" &
  pids+=($!)
  if [ ${#pids[@]} -ge $batch_size ]; then
    for pid in "${pids[@]}"; do wait "$pid" || true; done
    pids=()
  fi
done
for pid in "${pids[@]}"; do wait "$pid" || true; done

echo "T2d done."
date
