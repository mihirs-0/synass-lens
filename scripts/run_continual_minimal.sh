#!/bin/bash
#
# Minimal continual learning test: f=0.5 and f=1.0 only.
# Quick turnaround to see initial Q vs f signal.
#
set -euo pipefail

OUTPUT_DIR="outputs"
BASE="landauer_k20"

echo "════════════════════════════════════════"
echo " Minimal Continual Learning Test"
echo " f=0.5 and f=1.0, 15K steps each"
echo "════════════════════════════════════════"

# ── Train f=0.5 ──────────────────────────────────────────────────
echo ""
echo "── [1/6] Training f=0.5 ──"
python scripts/train_continual.py \
    --base-experiment ${BASE} \
    --variant reassign --fraction 0.5 \
    --reassign-seed 137 \
    --name continual_reassign_f0.5 \
    --max-steps 15000 \
    --lr 1e-3 --bs 128 --scheduler constant --seed 42 \
    --output-dir ${OUTPUT_DIR} \
    --checkpoint-every 100 --eval-every 50

# ── Train f=1.0 ──────────────────────────────────────────────────
echo ""
echo "── [2/6] Training f=1.0 ──"
python scripts/train_continual.py \
    --base-experiment ${BASE} \
    --variant reassign --fraction 1.0 \
    --reassign-seed 137 \
    --name continual_reassign_f1.0 \
    --max-steps 15000 \
    --lr 1e-3 --bs 128 --scheduler constant --seed 42 \
    --output-dir ${OUTPUT_DIR} \
    --checkpoint-every 100 --eval-every 50

# ── Gradient norms ───────────────────────────────────────────────
echo ""
echo "── [3/6] Gradient norms f=0.5 ──"
python scripts/compute_gradient_norms.py \
    --experiment continual_reassign_f0.5 --every-n 5 --output-dir ${OUTPUT_DIR}

echo ""
echo "── [4/6] Gradient norms f=1.0 ──"
python scripts/compute_gradient_norms.py \
    --experiment continual_reassign_f1.0 --every-n 5 --output-dir ${OUTPUT_DIR}

# ── Landauer cost ────────────────────────────────────────────────
echo ""
echo "── [5/6] Landauer cost ──"
python scripts/compute_landauer_cost.py \
    --experiments continual_reassign_f0.5 continual_reassign_f1.0 \
    --output-dir ${OUTPUT_DIR}

# ── Plot ─────────────────────────────────────────────────────────
echo ""
echo "── [6/6] Plotting ──"
python scripts/plot_continual_results.py \
    --experiments continual_reassign_f0.5 continual_reassign_f1.0 \
    --output-dir ${OUTPUT_DIR} \
    --output outputs/continual_analysis/

echo ""
echo "════════════════════════════════════════"
echo " Done! Check outputs/continual_analysis/"
echo "════════════════════════════════════════"
