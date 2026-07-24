#!/bin/bash
# Dense K-sweep for Landauer scaling: K ∈ {3, 5, 7, 10, 13, 17, 20, 25, 30, 36}
#
# ALL hyperparameters are locked across K values (constant LR, no warmup,
# no gradient clipping, 50K steps, seed=42). K is the ONLY variable.
#
# Phases:
#   1. Training (sequential — each run ~50K steps)
#   2. Gradient norm computation from checkpoints
#   3. Candidate evaluation from checkpoints
#   4. Landauer cost computation
#   5. Dense analysis + plotting
set -e

cd "$(dirname "$0")/.."

K_VALUES=(3 5 7 10 13 17 20 25 30 36)

echo "============================================================"
echo " Dense K-Sweep for Landauer Scaling"
echo " K values: ${K_VALUES[*]}"
echo "============================================================"
echo ""

# ===================================================================
# Phase 1: Training
# ===================================================================
echo "═══ Phase 1: Training ═══"
for K in "${K_VALUES[@]}"; do
    NAME="landauer_dense_k${K}"
    CONFIG="landauer_dense_k${K}"

    if [ -f "outputs/${NAME}/training_history.json" ]; then
        echo "[SKIP] ${NAME} already trained"
        continue
    fi

    CONFIG_FILE="configs/experiments/${CONFIG}.yaml"
    if [ ! -f "${CONFIG_FILE}" ]; then
        echo "[ERROR] Config not found: ${CONFIG_FILE}"
        continue
    fi

    echo "[START] Training ${NAME} (K=${K}, 50K steps, constant LR=1e-3) ..."
    python scripts/train.py \
        --config-path ../configs/experiments \
        --config-name "${CONFIG}"
    echo "[DONE]  ${NAME}"
    echo ""
done

echo ""
echo "═══ Phase 1 complete. ═══"
echo ""

# ===================================================================
# Phase 2: Gradient norms from checkpoints
# ===================================================================
echo "═══ Phase 2: Gradient Norms ═══"
for K in "${K_VALUES[@]}"; do
    NAME="landauer_dense_k${K}"

    if [ ! -d "outputs/${NAME}/checkpoints" ]; then
        echo "[SKIP] ${NAME} — no checkpoints"
        continue
    fi

    if [ -f "outputs/${NAME}/gradient_norm_results.json" ]; then
        echo "[SKIP] ${NAME} gradient norms already computed"
        continue
    fi

    echo "[GRAD] ${NAME} ..."
    python scripts/compute_gradient_norms.py \
        --experiment "${NAME}" \
        --every-n 1 \
        --n-batches 4
    echo "[DONE] ${NAME}"
done

echo ""
echo "═══ Phase 2 complete. ═══"
echo ""

# ===================================================================
# Phase 3: Candidate evaluation from checkpoints
# ===================================================================
echo "═══ Phase 3: Candidate Evaluation ═══"
for K in "${K_VALUES[@]}"; do
    NAME="landauer_dense_k${K}"

    if [ ! -d "outputs/${NAME}/checkpoints" ]; then
        echo "[SKIP] ${NAME} — no checkpoints"
        continue
    fi

    if [ -f "outputs/${NAME}/candidate_eval_results.json" ]; then
        echo "[SKIP] ${NAME} candidate eval already computed"
        continue
    fi

    echo "[CAND] ${NAME} ..."
    python scripts/run_candidate_eval.py \
        --experiment "${NAME}" \
        --every-n 1 \
        --n-examples 32
    echo "[DONE] ${NAME}"
done

echo ""
echo "═══ Phase 3 complete. ═══"
echo ""

# ===================================================================
# Phase 4: Landauer cost computation
# ===================================================================
echo "═══ Phase 4: Landauer Cost Computation ═══"

EXPERIMENTS=""
for K in "${K_VALUES[@]}"; do
    EXPERIMENTS="${EXPERIMENTS} landauer_dense_k${K}"
done

python scripts/compute_landauer_cost.py \
    --experiments ${EXPERIMENTS} \
    --threshold-robustness

echo ""
echo "═══ Phase 4 complete. ═══"
echo ""

# ===================================================================
# Phase 5: Analysis + Plotting
# ===================================================================
echo "═══ Phase 5: Analysis + Figures ═══"

python scripts/analyze_landauer_dense.py \
    --output-dir outputs \
    --save-dir outputs/paper_figures

echo ""
echo "============================================================"
echo " Dense K-Sweep Complete!"
echo ""
echo " Key outputs:"
echo "   outputs/landauer_dense_results.json      (full Q + fits)"
echo "   outputs/landauer_results.json             (Landauer cost)"
echo "   outputs/threshold_robustness.json         (robustness)"
echo "   outputs/paper_figures/fig_landauer_dense.pdf  (4-panel figure)"
echo "   outputs/paper_figures/fig_landauer_dense.png"
echo "============================================================"
