#!/bin/bash
# =============================================================================
# Phase Boundary Experiment — Fire and Forget (Single RTX 3090)
# =============================================================================
#
# Maps the critical learning rate η*(K) at which disambiguation becomes
# unreachable.  4 K values × 4-5 η values × 3 seeds = 49 new runs.
#
# Optimized for a single RTX 3090 (24 GB VRAM) RunPod instance:
#   - Model is tiny (800K params, ~70 MB VRAM per process)
#   - Runs 3 training processes concurrently on the same GPU
#   - No checkpoints saved (only training_history.json needed)
#   - Total disk: ~4 GB (repo+venv+outputs)
#   - Total time: ~20-24 hours with 3-way parallelism
#
# RunPod config:
#   GPU:            1× RTX 3090
#   Container disk: 20 GB
#   Volume disk:    20 GB (mounted at /workspace — persistent across stops)
#
# Usage:
#   1. SSH into RunPod pod
#   2. cd /workspace && git clone -b phase-boundary <repo-url> synass-lens
#   3. cd synass-lens && pip install -r requirements.txt
#   4. python scripts/generate_phase_boundary_configs.py
#   5. nohup bash scripts/run_phase_boundary.sh > /workspace/phase_boundary.log 2>&1 &
#   6. Disconnect.  Come back in ~20-24h.
#
# To monitor:
#   tail -f /workspace/phase_boundary.log
#   cat outputs/phase_boundary_progress.txt
#   ls outputs/pb_K*/training_history.json | wc -l   # count completed
# =============================================================================

set -euo pipefail

cd "$(dirname "$0")/.."
REPO_DIR="$(pwd)"

# ── Configuration ──
N_PARALLEL=4              # Concurrent training processes (4 fits on RTX 4000 Ada: 50 GB RAM, 9 vCPU)
PROGRESS_FILE="outputs/phase_boundary_progress.txt"
REUSE_MAP="configs/phase_boundary_reuse_map.txt"
RUN_LIST="configs/phase_boundary_run_list.txt"

mkdir -p outputs

echo "============================================================"
echo " Phase Boundary Experiment"
echo " Started:    $(date)"
echo " Repo:       ${REPO_DIR}"
echo " Parallel:   ${N_PARALLEL} processes sharing 1 GPU"
echo "============================================================"

# ── Verify environment ──
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')" || {
    echo "ERROR: PyTorch not available or CUDA not detected."
    echo "Install with: pip install -r requirements.txt"
    exit 1
}

echo "STARTED $(date)" > "${PROGRESS_FILE}"
echo "Parallel: ${N_PARALLEL}" >> "${PROGRESS_FILE}"

# ── Step 0: Generate configs if not already done ──
if [ ! -f "${RUN_LIST}" ]; then
    echo "Generating configs..."
    python scripts/generate_phase_boundary_configs.py
fi

# ── Step 1: Symlink reused prior runs ──
if [ -f "${REUSE_MAP}" ]; then
    echo ""
    echo "Linking prior runs..."
    while IFS=$'\t' read -r NAME PRIOR; do
        DEST="outputs/${NAME}"
        SRC="outputs/${PRIOR}"
        if [ -d "${SRC}" ] && [ ! -d "${DEST}" ]; then
            echo "  ${NAME} ← ${PRIOR}"
            mkdir -p "${DEST}"
            cp "${SRC}/training_history.json" "${DEST}/" 2>/dev/null || true
            cp "${SRC}/config.yaml" "${DEST}/" 2>/dev/null || true
        fi
    done < "${REUSE_MAP}"
fi

# ── Step 2: Count runs ──
TOTAL=$(wc -l < "${RUN_LIST}" | tr -d ' ')
echo ""
echo "Runs to execute:  ${TOTAL}"
echo "Runs: ${TOTAL}" >> "${PROGRESS_FILE}"
echo "" >> "${PROGRESS_FILE}"

if [ "${TOTAL}" -eq 0 ]; then
    echo "Nothing to run — all experiments already complete."
    echo "Proceeding to analysis..."
    python scripts/analyze_phase_boundary.py
    exit 0
fi

# ── Disk budget check ──
AVAIL_MB=$(df -m "$(pwd)" | awk 'NR==2{print $4}')
echo "Available disk: ${AVAIL_MB} MB"
# ~6 MB per run (history + log, no checkpoints) + safety
NEEDED_MB=$(( TOTAL * 6 + 500 ))
if [ "${AVAIL_MB}" -lt "${NEEDED_MB}" ]; then
    echo "WARNING: Only ${AVAIL_MB} MB available, need ~${NEEDED_MB} MB"
    echo "Proceeding anyway — checkpoints are disabled so this should be fine."
fi

# ── Step 3: Training with multi-process parallelism ──
#
# All processes share CUDA_VISIBLE_DEVICES=0 (single GPU).
# The model is tiny enough (~70 MB VRAM) that 3-4 instances fit easily
# in 24 GB VRAM.  The CPU/memory overhead is ~800 MB per process.

echo ""
echo "============================================================"
echo " Launching training (${N_PARALLEL} parallel on single GPU)"
echo "============================================================"

COMPLETED=0
FAILED=0

# Active process tracking
declare -a ACTIVE_PIDS=()
declare -a ACTIVE_NAMES=()

reap_finished() {
    # Check for finished processes, update counters
    local new_pids=()
    local new_names=()
    for i in "${!ACTIVE_PIDS[@]}"; do
        local pid="${ACTIVE_PIDS[$i]}"
        local name="${ACTIVE_NAMES[$i]}"
        if kill -0 "$pid" 2>/dev/null; then
            new_pids+=("$pid")
            new_names+=("$name")
        else
            wait "$pid" 2>/dev/null
            local rc=$?
            if [ $rc -eq 0 ]; then
                echo "[$(date '+%H:%M:%S')] DONE  ${name}" | tee -a "${PROGRESS_FILE}"
                ((COMPLETED++)) || true
            else
                echo "[$(date '+%H:%M:%S')] FAIL  ${name} (exit ${rc})" | tee -a "${PROGRESS_FILE}"
                mkdir -p "outputs/${name}"
                echo "${rc}" > "outputs/${name}/FAILED"
                ((FAILED++)) || true
            fi
        fi
    done
    ACTIVE_PIDS=("${new_pids[@]+"${new_pids[@]}"}")
    ACTIVE_NAMES=("${new_names[@]+"${new_names[@]}"}")
}

run_single() {
    local CONFIG_NAME=$1
    local OUT_DIR="outputs/${CONFIG_NAME}"
    local LOG_FILE="${OUT_DIR}/train.log"

    mkdir -p "${OUT_DIR}"

    # Skip if already trained
    if [ -f "${OUT_DIR}/training_history.json" ]; then
        return 0
    fi

    # All processes share the single GPU
    CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
        --config-path ../configs/experiments \
        --config-name "${CONFIG_NAME}" \
        > "${LOG_FILE}" 2>&1
}

# Read all run names
mapfile -t RUNS < "${RUN_LIST}"

START_TIME=$(date +%s)

for RUN_NAME in "${RUNS[@]}"; do
    [ -z "${RUN_NAME}" ] && continue

    # Skip completed
    if [ -f "outputs/${RUN_NAME}/training_history.json" ]; then
        echo "[SKIP] ${RUN_NAME}"
        ((COMPLETED++)) || true
        continue
    fi

    # Skip missing config
    if [ ! -f "configs/experiments/${RUN_NAME}.yaml" ]; then
        echo "[ERROR] Missing config: configs/experiments/${RUN_NAME}.yaml"
        ((FAILED++)) || true
        continue
    fi

    # Wait until we have a free slot
    while [ ${#ACTIVE_PIDS[@]} -ge ${N_PARALLEL} ]; do
        sleep 10
        reap_finished
    done

    echo "[$(date '+%H:%M:%S')] START ${RUN_NAME}" | tee -a "${PROGRESS_FILE}"

    run_single "${RUN_NAME}" &
    ACTIVE_PIDS+=($!)
    ACTIVE_NAMES+=("${RUN_NAME}")

    sleep 2  # Stagger launches to avoid filesystem contention
done

# ── Wait for remaining processes ──
echo ""
echo "All jobs launched.  Waiting for ${#ACTIVE_PIDS[@]} remaining..."
while [ ${#ACTIVE_PIDS[@]} -gt 0 ]; do
    sleep 15
    reap_finished
done

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
HOURS=$(( ELAPSED / 3600 ))
MINS=$(( (ELAPSED % 3600) / 60 ))

echo "" >> "${PROGRESS_FILE}"
echo "ALL TRAINING COMPLETE: $(date)" >> "${PROGRESS_FILE}"
echo "Completed: ${COMPLETED}  Failed: ${FAILED}  Elapsed: ${HOURS}h${MINS}m" >> "${PROGRESS_FILE}"

echo ""
echo "============================================================"
echo " Training complete: $(date)"
echo " Completed: ${COMPLETED}   Failed: ${FAILED}"
echo " Wall time: ${HOURS}h ${MINS}m"
echo "============================================================"

# ── Step 4: Analysis ──
echo ""
echo "Running analysis..."
python scripts/analyze_phase_boundary.py

echo ""
echo "============================================================"
echo " Phase Boundary Experiment DONE: $(date)"
echo " Results: outputs/phase_boundary_summary.json"
echo " Figures: outputs/paper_figures/fig_phase_*.png"
echo "============================================================"
echo "EXPERIMENT COMPLETE: $(date)" >> "${PROGRESS_FILE}"
