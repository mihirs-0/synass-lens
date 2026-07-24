#!/bin/bash
# From-scratch verification of the optimizer decomposition.
# Reviewer concern: branch protocol resets m_t and v_t at step 1500.
# Test: --branch-step 0 fires the optimizer swap at step 0 = effectively from-scratch.
#
# Three representative configs × 3 seeds = 9 runs at K=10, eta=10^-3.
# Configs span the key dimensions:
#   - adamw_b1_0: decoupled+BC (expected: transition)
#   - adam_coupled_b1_0: L2 (expected: stuck)
#   - rmsprop_decoupled_bc: decoupled+BC RMSProp (expected: transition, robustness across optimizer family)
#
# Output: eta_sweep/results/T2e_from_scratch/

set -e
cd /Users/mihir/synass-lens/synass-lens
mkdir -p eta_sweep/results/T2e_from_scratch
mkdir -p logs/from_scratch

CONFIGS=(adamw_b1_0 adam_coupled_b1_0 rmsprop_decoupled_bc)
SEEDS=(0 1 2)

PIDS=()
for cfg in "${CONFIGS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    log="/Users/mihir/synass-lens/synass-lens/logs/from_scratch/${cfg}_seed${seed}.log"
    python eta_sweep/run_optimizer_branch.py \
      --branch-config "$cfg" \
      --branch-step 0 \
      --k 10 \
      --eta 0.001 \
      --seed "$seed" \
      --max-steps 6000 \
      --output-subdir T2e_from_scratch > "$log" 2>&1 &
    PIDS+=($!)
  done
done

echo "Launched ${#PIDS[@]} from-scratch runs. PIDs: ${PIDS[@]}"
for pid in "${PIDS[@]}"; do wait "$pid" || true; done
echo "All from-scratch runs complete."
date
