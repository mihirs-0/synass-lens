#!/bin/bash
# Extended K-range experiments: K ∈ {50, 75, 100, 150} × {Transformer, Gated MLP, RNN}
# Uses 2-char disambiguation prefix to break the K=36 ceiling.
# Existing K ∈ {10, 15, 20, 25, 36} results are preserved and reused.
set -e

cd "$(dirname "$0")/.."

echo "============================================================"
echo " Extended K-Range Experiments (breaking the K=36 ceiling)"
echo "============================================================"
echo ""

# ===================================================================
# Phase 1: Training
# ===================================================================

# ----- Transformer -----
echo "--- Transformer ---"
for K in 50 75 100 150; do
    NAME="landauer_k${K}"
    if [ -f "outputs/${NAME}/training_history.json" ]; then
        echo "[SKIP] ${NAME} already trained"
        continue
    fi

    CONFIG_FILE="configs/experiments/${NAME}.yaml"
    if [ ! -f "${CONFIG_FILE}" ]; then
        echo "[ERROR] Config not found: ${CONFIG_FILE}"
        continue
    fi

    echo "[START] Training ${NAME} ..."
    python scripts/train.py --config-path ../configs/experiments --config-name "${NAME}"
    echo "[DONE]  ${NAME}"
    echo ""
done

# ----- Gated MLP -----
echo "--- Gated MLP ---"
for K in 50 75 100 150; do
    NAME="gatedmlp_k${K}"
    if [ -f "outputs/${NAME}/training_history.json" ]; then
        echo "[SKIP] ${NAME} already trained"
        continue
    fi

    if [ $K -le 75 ]; then STEPS=60000; else STEPS=90000; fi
    if [ $K -eq 150 ]; then STEPS=120000; fi

    echo "[START] Training ${NAME} ..."
    python scripts/train_alternative_arch.py \
        --arch gated_mlp --k ${K} --name ${NAME} \
        --max-steps ${STEPS} --lr 1e-3 --bs 128 \
        --scheduler constant --seed 42 \
        --disambiguation-prefix-length 2 \
        --checkpoint-every 200
    echo "[DONE]  ${NAME}"
    echo ""
done

# ----- RNN (LSTM) -----
echo "--- RNN (LSTM) ---"
for K in 50 75 100 150; do
    NAME="rnn_k${K}"
    if [ -f "outputs/${NAME}/training_history.json" ]; then
        echo "[SKIP] ${NAME} already trained"
        continue
    fi

    if [ $K -le 75 ]; then STEPS=60000; else STEPS=90000; fi
    if [ $K -eq 150 ]; then STEPS=120000; fi

    echo "[START] Training ${NAME} ..."
    python scripts/train_alternative_arch.py \
        --arch rnn --k ${K} --name ${NAME} \
        --max-steps ${STEPS} --lr 1e-3 --bs 128 \
        --scheduler constant --seed 42 \
        --disambiguation-prefix-length 2 \
        --checkpoint-every 200
    echo "[DONE]  ${NAME}"
    echo ""
done

echo "============================================================"
echo " All training complete."
echo "============================================================"

# ===================================================================
# Phase 2: Gradient norms + Candidate eval (new K values only)
# ===================================================================
echo ""
echo "Running gradient norm computation and candidate eval..."

# Transformer
for K in 50 75 100 150; do
    NAME="landauer_k${K}"
    if [ ! -d "outputs/${NAME}/checkpoints" ]; then
        echo "[SKIP] ${NAME} — no checkpoints"
        continue
    fi
    if [ ! -f "outputs/${NAME}/gradient_norm_results.json" ]; then
        echo "[GRAD]  ${NAME} ..."
        python scripts/compute_gradient_norms.py --experiment ${NAME} --every-n 4
    else
        echo "[SKIP] ${NAME} gradient norms already computed"
    fi
    if [ ! -f "outputs/${NAME}/candidate_eval_results.json" ]; then
        echo "[CAND]  ${NAME} ..."
        python scripts/run_candidate_eval.py --experiment ${NAME} --every-n 4 --n-examples 32
    else
        echo "[SKIP] ${NAME} candidate eval already computed"
    fi
done

# Gated MLP + RNN
for ARCH in gatedmlp rnn; do
    for K in 50 75 100 150; do
        NAME="${ARCH}_k${K}"
        if [ ! -d "outputs/${NAME}/checkpoints" ]; then
            echo "[SKIP] ${NAME} — no checkpoints"
            continue
        fi
        if [ ! -f "outputs/${NAME}/gradient_norm_results.json" ]; then
            echo "[GRAD]  ${NAME} ..."
            python scripts/compute_gradient_norms.py --experiment ${NAME} --every-n 4
        else
            echo "[SKIP] ${NAME} gradient norms already computed"
        fi
        if [ ! -f "outputs/${NAME}/candidate_eval_results.json" ]; then
            echo "[CAND]  ${NAME} ..."
            python scripts/run_candidate_eval.py --experiment ${NAME} --every-n 4 --n-examples 32
        else
            echo "[SKIP] ${NAME} candidate eval already computed"
        fi
    done
done

# ===================================================================
# Phase 3: Landauer cost computation (ALL K values)
# ===================================================================
echo ""
echo "============================================================"
echo " Computing Landauer dissipation for all K values..."
echo "============================================================"

# Transformer — all 9 K values
python scripts/compute_landauer_cost.py \
    --experiments landauer_k10 landauer_k15 landauer_k20 landauer_k25 landauer_k36 \
                  landauer_k50 landauer_k75 landauer_k100 landauer_k150 \
    --threshold-robustness

# Gated MLP
python scripts/compute_landauer_cost.py \
    --experiments gatedmlp_k10 gatedmlp_k15 gatedmlp_k20 gatedmlp_k25 gatedmlp_k36 \
                  gatedmlp_k50 gatedmlp_k75 gatedmlp_k100 gatedmlp_k150

# RNN
python scripts/compute_landauer_cost.py \
    --experiments rnn_k10 rnn_k15 rnn_k20 rnn_k25 rnn_k36 \
                  rnn_k50 rnn_k75 rnn_k100 rnn_k150

# ===================================================================
# Phase 4: Model comparison analysis
# ===================================================================
echo ""
echo "============================================================"
echo " Running scaling model comparison (log vs linear vs power vs sqrt)..."
echo "============================================================"
python scripts/compare_scaling_models.py

# ===================================================================
# Phase 5: Updated architecture comparison figure
# ===================================================================
echo ""
echo "============================================================"
echo " Generating updated architecture comparison figure..."
echo "============================================================"
python scripts/plot_architecture_comparison.py

echo ""
echo "============================================================"
echo " All done!"
echo " Key outputs:"
echo "   outputs/landauer_results.json"
echo "   outputs/threshold_robustness.json"
echo "   outputs/scaling_model_comparison.json"
echo "   outputs/figures/architecture_comparison.png"
echo "   outputs/figures/model_comparison.png"
echo "============================================================"
