#!/bin/bash
# Run the full architecture ablation experiment sweep.
# Each architecture × K value is trained for 30K steps.
set -e

cd "$(dirname "$0")/.."

COMMON="--max-steps 30000 --lr 1e-3 --bs 128 --scheduler constant --seed 42"

echo "============================================="
echo " Architecture Ablation — Full Sweep"
echo "============================================="
echo ""

# ----- Gated MLP -----
for K in 10 15 20 25 36; do
    NAME="gatedmlp_k${K}"
    if [ -f "outputs/${NAME}/training_history.json" ]; then
        echo "[SKIP] ${NAME} already trained"
        continue
    fi
    echo "[START] Training ${NAME} ..."
    python scripts/train_alternative_arch.py \
        --arch gated_mlp --k ${K} --name ${NAME} ${COMMON}
    echo "[DONE]  ${NAME}"
    echo ""
done

# ----- RNN (LSTM) -----
for K in 10 15 20 25 36; do
    NAME="rnn_k${K}"
    if [ -f "outputs/${NAME}/training_history.json" ]; then
        echo "[SKIP] ${NAME} already trained"
        continue
    fi
    echo "[START] Training ${NAME} ..."
    python scripts/train_alternative_arch.py \
        --arch rnn --k ${K} --name ${NAME} ${COMMON}
    echo "[DONE]  ${NAME}"
    echo ""
done

echo "============================================="
echo " All training complete."
echo "============================================="

# ----- Gradient norms + Candidate eval -----
echo ""
echo "Running gradient norm computation and candidate eval..."
for ARCH in gatedmlp rnn; do
    for K in 10 15 20 25 36; do
        NAME="${ARCH}_k${K}"
        if [ ! -d "outputs/${NAME}/checkpoints" ]; then
            echo "[SKIP] ${NAME} — no checkpoints"
            continue
        fi
        if [ ! -f "outputs/${NAME}/gradient_norm_results.json" ]; then
            echo "[GRAD]  ${NAME} ..."
            python scripts/compute_gradient_norms.py --experiment ${NAME} --every-n 2
        else
            echo "[SKIP] ${NAME} gradient norms already computed"
        fi
        if [ ! -f "outputs/${NAME}/candidate_eval_results.json" ]; then
            echo "[CAND]  ${NAME} ..."
            python scripts/run_candidate_eval.py --experiment ${NAME} --every-n 2
        else
            echo "[SKIP] ${NAME} candidate eval already computed"
        fi
    done
done

echo ""
echo "============================================="
echo " Generating comparison plot..."
echo "============================================="
python scripts/plot_architecture_comparison.py

echo ""
echo "Done! See outputs/figures/architecture_comparison.png"
