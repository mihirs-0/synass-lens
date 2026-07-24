#!/bin/bash
# ============================================================
# OVERNIGHT MASTER SCRIPT
# Parallel execution: 6 workers share the GPU simultaneously.
# Model is ~600K params — easily fits 6+ copies on any modern GPU.
#
# Usage:
#   cd /workspace/synass-lens
#   bash overnight_master.sh
#
# Expected runtime: ~2-3 hours on RTX 4090 (was 8-10h sequential).
# ============================================================

set -o pipefail

LOG="${OVERNIGHT_LOG:-/workspace/overnight_log.txt}"
cd "$(dirname "$0")"

echo "============================================================" | tee "$LOG"
echo "OVERNIGHT EXPERIMENTS (PARALLEL) - $(date)" | tee -a "$LOG"
echo "Working dir: $(pwd)" | tee -a "$LOG"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

# Suite 1: Reversal Curse (12 parallel + 4 transfer sequential)
echo "" | tee -a "$LOG"
echo "=== SUITE 1: REVERSAL EXPERIMENTS (parallel) ===" | tee -a "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
PYTHONUNBUFFERED=1 python scripts/reversal_experiment.py 2>&1 | tee -a "$LOG"
echo "Suite 1 finished: $(date)" | tee -a "$LOG"

# Suite 2A: Label Noise Sweep (parallel)
echo "" | tee -a "$LOG"
echo "=== SUITE 2A: LABEL NOISE SWEEP (parallel) ===" | tee -a "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
PYTHONUNBUFFERED=1 python scripts/ziyin_label_noise.py 2>&1 | tee -a "$LOG"
echo "Suite 2A finished: $(date)" | tee -a "$LOG"

# Suite 2B: Q Slope (parallel)
echo "" | tee -a "$LOG"
echo "=== SUITE 2B: Q SLOPE (parallel) ===" | tee -a "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
PYTHONUNBUFFERED=1 python scripts/ziyin_q_slope.py 2>&1 | tee -a "$LOG"
echo "Suite 2B finished: $(date)" | tee -a "$LOG"

# Suite 2C: Batch Size Sweep (parallel)
echo "" | tee -a "$LOG"
echo "=== SUITE 2C: BATCH SIZE SWEEP (parallel) ===" | tee -a "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
PYTHONUNBUFFERED=1 python scripts/ziyin_batch_sweep.py 2>&1 | tee -a "$LOG"
echo "Suite 2C finished: $(date)" | tee -a "$LOG"

# Analysis: Summary tables + figures
echo "" | tee -a "$LOG"
echo "=== ANALYSIS ===" | tee -a "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
PYTHONUNBUFFERED=1 python scripts/overnight_analysis.py 2>&1 | tee -a "$LOG"
echo "Analysis finished: $(date)" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
echo "ALL DONE - $(date)" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
echo ""
echo "Results are in:"
echo "  outputs/reversal_*/          (Suite 1)"
echo "  outputs/ziyin_*/             (Suite 2)"
echo "  outputs/paper_figures/fig_*  (Analysis figures)"
echo "  outputs/*_summary.json       (Summary tables)"
echo "  $LOG                         (Full log)"
