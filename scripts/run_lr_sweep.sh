#!/bin/bash
# Experiment 2: Learning Rate Sweep at K=20
#
# How does plateau duration τ depend on learning rate η?
#
# Fix K=20. Vary η ∈ {3e-4, 5e-4, 1e-3, 2e-3, 5e-3}.
# η=1e-3 already exists as landauer_dense_k20 — symlink, don't rerun.
#
# Phases:
#   1. Symlink existing η=1e-3 data
#   2. Train 4 new learning rates (sequential)
#   3. Analysis + plotting
set -e

cd "$(dirname "$0")/.."

# LR configs in order (η=1e-3 handled via symlink)
LR_CONFIGS=(
    "lr_sweep_eta3e-4"
    "lr_sweep_eta5e-4"
    "lr_sweep_eta2e-3"
    "lr_sweep_eta5e-3"
)

echo "============================================================"
echo " Experiment 2: Learning Rate Sweep (K=20)"
echo " η values: 3e-4, 5e-4, 1e-3 (existing), 2e-3, 5e-3"
echo "============================================================"
echo ""

# ===================================================================
# Phase 1: Symlink existing η=1e-3 data
# ===================================================================
echo "═══ Phase 1: Link existing η=1e-3 data ═══"

if [ -d "outputs/landauer_dense_k20" ]; then
    if [ ! -e "outputs/lr_sweep_eta1e-3" ]; then
        ln -s "landauer_dense_k20" "outputs/lr_sweep_eta1e-3"
        echo "[LINK] outputs/lr_sweep_eta1e-3 -> landauer_dense_k20"
    else
        echo "[SKIP] outputs/lr_sweep_eta1e-3 already exists"
    fi
else
    echo "[ERROR] outputs/landauer_dense_k20 not found — run the K-sweep first"
    exit 1
fi

echo ""

# ===================================================================
# Phase 2: Training
# ===================================================================
echo "═══ Phase 2: Training ═══"
for CONFIG in "${LR_CONFIGS[@]}"; do
    NAME="${CONFIG}"

    if [ -f "outputs/${NAME}/training_history.json" ]; then
        echo "[SKIP] ${NAME} already trained"
        continue
    fi

    CONFIG_FILE="configs/experiments/${CONFIG}.yaml"
    if [ ! -f "${CONFIG_FILE}" ]; then
        echo "[ERROR] Config not found: ${CONFIG_FILE}"
        continue
    fi

    echo "[START] Training ${NAME} ..."
    python scripts/train.py \
        --config-path ../configs/experiments \
        --config-name "${CONFIG}"
    echo "[DONE]  ${NAME}"
    echo ""
done

echo ""
echo "═══ Phase 2 complete. ═══"
echo ""

# ===================================================================
# Phase 3: Analysis + Plotting
# ===================================================================
echo "═══ Phase 3: Analysis + Figures ═══"

python scripts/analyze_lr_sweep.py

echo ""
echo "============================================================"
echo " LR Sweep Complete!"
echo ""
echo " Key outputs:"
echo "   outputs/lr_sweep_results.json"
echo "   outputs/paper_figures/fig_lr_sweep.pdf"
echo "   outputs/paper_figures/fig_lr_sweep.png"
echo "============================================================"
