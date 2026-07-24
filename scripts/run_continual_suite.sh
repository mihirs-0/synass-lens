#!/bin/bash
#
# Master script for the continual learning experiment suite.
#
# Runs:
#   Phase 1: Ensure base K=20 checkpoint exists
#   Phase 2: Partial reassignment sweep (f=0.0 .. 1.0)
#   Phase 2b: Multiple reassignment seeds for f=0.5 (error bars)
#   Phase 2c: Optimizer state ablation (f=0.5 and f=1.0)
#   Phase 3: Gradient norms and candidate eval for all experiments
#   Phase 4: Compute Landauer cost
#   Phase 5: K-expansion experiment (if K=10 checkpoint exists)
#   Phase 6: Plot everything
#
# Usage:
#   bash scripts/run_continual_suite.sh
#   bash scripts/run_continual_suite.sh --skip-base   # skip Phase 1
#   bash scripts/run_continual_suite.sh --phase 2     # run only Phase 2+

set -euo pipefail

OUTPUT_DIR="outputs"
BASE_EXPERIMENT="landauer_k20"
K10_EXPERIMENT="landauer_k10"

SKIP_BASE=false
START_PHASE=1

for arg in "$@"; do
    case $arg in
        --skip-base) SKIP_BASE=true ;;
        --phase) shift; START_PHASE="${2:-1}" ;;
    esac
done

echo "========================================"
echo " Continual Learning Experiment Suite"
echo "========================================"
echo "Output dir: ${OUTPUT_DIR}"
echo "Base experiment: ${BASE_EXPERIMENT}"
echo ""

# ── Phase 1: Ensure base checkpoint exists ────────────────────────
if [ "$START_PHASE" -le 1 ] && [ "$SKIP_BASE" = false ]; then
    echo "── Phase 1: Checking base checkpoint ──"
    if [ ! -d "${OUTPUT_DIR}/${BASE_EXPERIMENT}/checkpoints" ]; then
        echo "Base experiment not found. Training from scratch..."
        python scripts/train.py --config-path ../configs --config-name base \
            experiment.name=${BASE_EXPERIMENT} \
            data.task=bz_to_a data.k=20 data.n_unique_b=1000 \
            data.enforce_unique_a_first_char_per_b=true \
            training.learning_rate=1e-3 training.batch_size=128 \
            training.max_steps=30000 training.warmup_steps=0 \
            training.scheduler=constant training.checkpoint_every=100 \
            training.eval_every=50 data.probe_fraction=0.0 experiment.seed=42
    else
        echo "Base experiment found at ${OUTPUT_DIR}/${BASE_EXPERIMENT}"
    fi
    echo ""
fi

# ── Phase 2: Partial reassignment sweep ───────────────────────────
if [ "$START_PHASE" -le 2 ]; then
    echo "── Phase 2: Partial reassignment sweep ──"
    for FRAC in 0.0 0.1 0.25 0.5 0.75 1.0; do
        EXP_NAME="continual_reassign_f${FRAC}"
        echo "  Running f=${FRAC} → ${EXP_NAME}"
        python scripts/train_continual.py \
            --base-experiment ${BASE_EXPERIMENT} \
            --variant reassign \
            --fraction ${FRAC} \
            --reassign-seed 137 \
            --name ${EXP_NAME} \
            --max-steps 30000 \
            --lr 1e-3 --bs 128 \
            --scheduler constant \
            --seed 42 \
            --output-dir ${OUTPUT_DIR} \
            --checkpoint-every 100 \
            --eval-every 50
    done
    echo ""
fi

# ── Phase 2b: Multiple reassignment seeds for f=0.5 ──────────────
if [ "$START_PHASE" -le 2 ]; then
    echo "── Phase 2b: Multiple seeds for f=0.5 ──"
    for RSEED in 137 241 389 503 617; do
        EXP_NAME="continual_reassign_f0.5_rs${RSEED}"
        echo "  Running f=0.5, seed=${RSEED} → ${EXP_NAME}"
        python scripts/train_continual.py \
            --base-experiment ${BASE_EXPERIMENT} \
            --variant reassign \
            --fraction 0.5 \
            --reassign-seed ${RSEED} \
            --name ${EXP_NAME} \
            --max-steps 30000 \
            --lr 1e-3 --bs 128 \
            --scheduler constant \
            --seed 42 \
            --output-dir ${OUTPUT_DIR} \
            --checkpoint-every 100 \
            --eval-every 50
    done
    echo ""
fi

# ── Phase 2c: Optimizer state ablation ────────────────────────────
if [ "$START_PHASE" -le 2 ]; then
    echo "── Phase 2c: Optimizer state ablation ──"
    for FRAC in 0.5 1.0; do
        EXP_NAME="continual_reassign_f${FRAC}_warmopt"
        echo "  Running f=${FRAC} with warm optimizer → ${EXP_NAME}"
        python scripts/train_continual.py \
            --base-experiment ${BASE_EXPERIMENT} \
            --variant reassign \
            --fraction ${FRAC} \
            --reassign-seed 137 \
            --name ${EXP_NAME} \
            --max-steps 30000 \
            --lr 1e-3 --bs 128 \
            --scheduler constant \
            --seed 42 \
            --output-dir ${OUTPUT_DIR} \
            --checkpoint-every 100 \
            --eval-every 50 \
            --load-optimizer-state
    done
    echo ""
fi

# ── Phase 3: Gradient norms + candidate eval ──────────────────────
if [ "$START_PHASE" -le 3 ]; then
    echo "── Phase 3: Post-hoc analysis (gradient norms + candidate eval) ──"
    for FRAC in 0.0 0.1 0.25 0.5 0.75 1.0; do
        EXP="continual_reassign_f${FRAC}"
        echo "  Gradient norms for ${EXP}"
        python scripts/compute_gradient_norms.py --experiment ${EXP} --every-n 2 --output-dir ${OUTPUT_DIR}
        echo "  Candidate eval for ${EXP}"
        python scripts/run_candidate_eval.py --experiment ${EXP} --every-n 2 --output-dir ${OUTPUT_DIR}
    done
    echo ""
fi

# ── Phase 4: Compute Landauer cost ────────────────────────────────
if [ "$START_PHASE" -le 4 ]; then
    echo "── Phase 4: Landauer cost computation ──"
    CONTINUAL_EXPERIMENTS=""
    for FRAC in 0.1 0.25 0.5 0.75 1.0; do
        CONTINUAL_EXPERIMENTS="${CONTINUAL_EXPERIMENTS} continual_reassign_f${FRAC}"
    done
    python scripts/compute_landauer_cost.py \
        --experiments ${CONTINUAL_EXPERIMENTS} \
        --output-dir ${OUTPUT_DIR}
    echo ""
fi

# ── Phase 5: K-expansion (if K=10 checkpoint exists) ─────────────
if [ "$START_PHASE" -le 5 ]; then
    if [ -d "${OUTPUT_DIR}/${K10_EXPERIMENT}/checkpoints" ]; then
        echo "── Phase 5: K-expansion experiment ──"
        python scripts/train_continual.py \
            --base-experiment ${K10_EXPERIMENT} \
            --variant expand \
            --target-k 20 \
            --reassign-seed 137 \
            --name continual_expand_k10_to_k20 \
            --max-steps 30000 \
            --lr 1e-3 --bs 128 \
            --scheduler constant \
            --seed 42 \
            --output-dir ${OUTPUT_DIR} \
            --checkpoint-every 100 \
            --eval-every 50

        echo "  Gradient norms for K-expansion"
        python scripts/compute_gradient_norms.py \
            --experiment continual_expand_k10_to_k20 --every-n 2 --output-dir ${OUTPUT_DIR}
        echo "  Candidate eval for K-expansion"
        python scripts/run_candidate_eval.py \
            --experiment continual_expand_k10_to_k20 --every-n 2 --output-dir ${OUTPUT_DIR}
    else
        echo "── Phase 5: Skipping K-expansion (${K10_EXPERIMENT} not found) ──"
    fi
    echo ""
fi

# ── Phase 6: Plot everything ──────────────────────────────────────
if [ "$START_PHASE" -le 6 ]; then
    echo "── Phase 6: Generating plots ──"
    PLOT_EXPERIMENTS=""
    for FRAC in 0.0 0.1 0.25 0.5 0.75 1.0; do
        PLOT_EXPERIMENTS="${PLOT_EXPERIMENTS} continual_reassign_f${FRAC}"
    done
    python scripts/plot_continual_results.py \
        --experiments ${PLOT_EXPERIMENTS} \
        --output-dir ${OUTPUT_DIR} \
        --output outputs/continual_analysis/
    echo ""
fi

echo "========================================"
echo " All phases complete!"
echo "========================================"
