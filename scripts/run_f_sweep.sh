#!/bin/bash
#
# Resilient f-sweep runner for continual learning experiments.
#
# Prevents macOS sleep, logs all output, skips completed experiments,
# and survives terminal closure.
#
# Usage:
#   # Run in foreground (terminal must stay open):
#   bash scripts/run_f_sweep.sh
#
#   # Run detached (survives sleep + terminal close):
#   nohup caffeinate -dims bash scripts/run_f_sweep.sh > outputs/f_sweep.log 2>&1 &
#   disown
#   # Monitor with: tail -f outputs/f_sweep.log
#
#   # Force re-run of all experiments (overwrites existing):
#   bash scripts/run_f_sweep.sh --force
#
#   # Skip to plotting only:
#   bash scripts/run_f_sweep.sh --plot-only

set -euo pipefail

OUTPUT_DIR="outputs"
BASE_EXPERIMENT="landauer_k20"
LOG_FILE="${OUTPUT_DIR}/f_sweep.log"

FORCE=false
PLOT_ONLY=false

for arg in "$@"; do
    case $arg in
        --force) FORCE=true ;;
        --plot-only) PLOT_ONLY=true ;;
    esac
done

mkdir -p "${OUTPUT_DIR}"

# Redirect all output to log file AND terminal
exec > >(tee -a "${LOG_FILE}") 2>&1

echo ""
echo "========================================================"
echo " f-Sweep: Measuring Representational Locality"
echo " Started: $(date)"
echo " PID: $$"
echo "========================================================"
echo ""

# On macOS, wrap with caffeinate if not already wrapped
if [[ "$(uname)" == "Darwin" ]] && ! pgrep -f "caffeinate.*$$" > /dev/null 2>&1; then
    if command -v caffeinate &> /dev/null; then
        echo "NOTE: For full sleep prevention, run as:"
        echo "  nohup caffeinate -dims bash scripts/run_f_sweep.sh > outputs/f_sweep.log 2>&1 &"
        echo "  disown"
        echo ""
    fi
fi

is_experiment_done() {
    local exp_name="$1"
    local expected_steps="$2"
    local exp_dir="${OUTPUT_DIR}/${exp_name}"
    local history="${exp_dir}/training_history.json"

    if [ ! -f "${history}" ]; then
        return 1
    fi

    # Check if training_history has enough eval points
    # For 15000 steps at eval_every=25, expect 600 entries
    local n_steps
    n_steps=$(python3 -c "
import json
with open('${history}') as f:
    h = json.load(f)
steps = h.get('steps', [])
print(steps[-1] if steps else 0)
" 2>/dev/null || echo "0")

    if [ "${n_steps}" -ge "${expected_steps}" ]; then
        return 0
    fi
    return 1
}

run_experiment() {
    local frac="$1"
    local max_steps="${2:-15000}"
    local exp_name="continual_reassign_f${frac}"

    echo "────────────────────────────────────────────────────"
    echo " Experiment: f=${frac} (${exp_name})"
    echo " Time: $(date '+%H:%M:%S')"
    echo "────────────────────────────────────────────────────"

    if [ "${FORCE}" = false ] && is_experiment_done "${exp_name}" "${max_steps}"; then
        echo "  SKIP: already complete (${exp_name})"
        echo ""
        return 0
    fi

    local start_time=$SECONDS

    python scripts/train_continual.py \
        --base-experiment "${BASE_EXPERIMENT}" \
        --variant reassign \
        --fraction "${frac}" \
        --reassign-seed 137 \
        --name "${exp_name}" \
        --max-steps "${max_steps}" \
        --lr 1e-3 --bs 128 \
        --scheduler constant \
        --seed 42 \
        --output-dir "${OUTPUT_DIR}" \
        --checkpoint-every 500 \
        --eval-every 25 \
        --n-eval-examples 32

    local elapsed=$(( SECONDS - start_time ))
    echo "  DONE: ${exp_name} in ${elapsed}s"
    echo ""
}

if [ "${PLOT_ONLY}" = false ]; then

    # Verify base checkpoint exists
    if [ ! -d "${OUTPUT_DIR}/${BASE_EXPERIMENT}/checkpoints" ]; then
        echo "ERROR: Base experiment not found at ${OUTPUT_DIR}/${BASE_EXPERIMENT}"
        echo "Run the base K=20 training first."
        exit 1
    fi
    echo "Base checkpoint: ${OUTPUT_DIR}/${BASE_EXPERIMENT} ✓"
    echo ""

    # ── Phase 1: Control (f=0.0) ──────────────────────────────────
    echo "═══ Phase 1/3: Control experiment ═══"
    run_experiment 0.0 15000

    # ── Phase 2: New low-f experiments ─────────────────────────────
    echo "═══ Phase 2/3: Low-f regime (the new experiments) ═══"
    for FRAC in 0.01 0.05 0.1 0.25; do
        run_experiment "${FRAC}" 15000
    done

    # ── Phase 3: Re-run f=0.5, 0.75, 1.0 with fixed reassign ─────
    # (needed because reassign_mappings now uses shuffle+slice for
    # nested subset property; old f=0.5/1.0 data used rng.sample)
    echo "═══ Phase 3/3: High-f experiments (re-run with fixed reassign) ═══"
    for FRAC in 0.5 0.75 1.0; do
        run_experiment "${FRAC}" 15000
    done

fi

# ── Plotting ──────────────────────────────────────────────────────
echo ""
echo "═══ Generating figures ═══"
python scripts/plot_f_sweep.py \
    --output-dir "${OUTPUT_DIR}" \
    --out "${OUTPUT_DIR}/continual_analysis/"

echo ""
echo "========================================================"
echo " f-Sweep Complete!"
echo " Finished: $(date)"
echo " Figures: ${OUTPUT_DIR}/continual_analysis/"
echo " Log: ${LOG_FILE}"
echo "========================================================"
