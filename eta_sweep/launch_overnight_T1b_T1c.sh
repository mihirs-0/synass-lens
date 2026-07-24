#!/bin/bash
# T1b: Full 9-condition decomposition × 3 seeds at K=20 (27 runs)
# T1c: λ sweep × 3 seeds at K=10 (18 runs total — 6 cells × 3 seeds)
#
# Total: 45 runs. Launches in batches of 8 to avoid overwhelming MPS.
# Each batch waits for its predecessors before starting.
#
# Output dirs:
#   eta_sweep/results/T1b_K20_9cell/{config}_seed{s}/
#   eta_sweep/results/T1c_lambda_sweep/lam{lam}_{config}_seed{s}/

set -e
cd /Users/mihir/synass-lens/synass-lens
mkdir -p eta_sweep/results/T1b_K20_9cell
mkdir -p eta_sweep/results/T1c_lambda_sweep
mkdir -p logs/overnight

CONFIGS_9CELL=(
  "adamw_b1_0"
  "adamw_b1_0_nobc"
  "adamw_b1_0_rmsprop_eps"
  "rmsprop_decoupled_bc"
  "rmsprop_decoupled_only"
  "adam_coupled_b1_0"
  "rmsprop_bc_only"
  "rmsprop"
  "rmsprop_adamw_eps"
)

# Function: launch_batch <config_list_array> ... (8 at a time, wait between batches)
launch_in_batches() {
  local batch_size=8
  local pids=()
  for cmd in "$@"; do
    eval "$cmd" &
    pids+=($!)
    if [ ${#pids[@]} -ge $batch_size ]; then
      for pid in "${pids[@]}"; do wait "$pid" || true; done
      pids=()
    fi
  done
  for pid in "${pids[@]}"; do wait "$pid" || true; done
}

# Build T1b commands (9 configs × 3 seeds = 27)
T1B_COMMANDS=()
for cfg in "${CONFIGS_9CELL[@]}"; do
  for seed in 0 1 2; do
    log="logs/overnight/T1b_K20_${cfg}_seed${seed}.log"
    cmd="python eta_sweep/run_optimizer_branch.py --branch-config $cfg --k 20 --eta 0.001 --seed $seed --max-steps 7000 --output-subdir T1b_K20_9cell > $log 2>&1"
    T1B_COMMANDS+=("$cmd")
  done
done

# Build T1c commands (λ ∈ {0.001, 0.01, 0.1} × {AdamW, Adam-coupled} × 3 seeds = 18)
# Note: λ=0.01 we technically have from existing 9-cell at K=10. But for clean multi-seed
# comparison at K=10 with this exact protocol, re-run.
T1C_COMMANDS=()
for lam in 0.001 0.01 0.1; do
  for cfg in adamw_b1_0 adam_coupled_b1_0; do
    for seed in 0 1 2; do
      log="logs/overnight/T1c_lam${lam}_${cfg}_seed${seed}.log"
      out_subdir="T1c_lambda_sweep/lam${lam}"
      cmd="python eta_sweep/run_optimizer_branch.py --branch-config $cfg --k 10 --eta 0.001 --seed $seed --max-steps 6000 --weight-decay $lam --output-subdir $out_subdir > $log 2>&1"
      T1C_COMMANDS+=("$cmd")
    done
  done
done

echo "=== T1b: 9-cell × 3 seeds at K=20 (27 runs) ==="
echo "First batch (8 configs)..."
launch_in_batches "${T1B_COMMANDS[@]}"
echo "T1b done."

echo "=== T1c: λ sweep × 3 seeds (18 runs) ==="
launch_in_batches "${T1C_COMMANDS[@]}"
echo "T1c done."

echo "=== ALL OVERNIGHT EXPERIMENTS COMPLETE ==="
date
