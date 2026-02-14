#!/usr/bin/env bash
# ============================================================
# run_main_experiments.sh
#
# CSC415 A1 - DrQ-v2 Reproduction: Main Experiments
#
# Runs DrQ-v2 on walker_walk and cheetah_run with 3 seeds each.
# Logs are saved under results/main/<task>/<seed>/.
#
# Usage (from repo root):
#   bash scripts/run_main_experiments.sh
#
# Prerequisites:
#   - Activate the virtual environment first:
#       source venv/bin/activate      (Linux/macOS)
#       venv\Scripts\activate         (Windows)
#   - Or call this script with the venv python directly (see PYTHON below)
# ============================================================

set -e  # Exit immediately on error

# ---------- Configuration ----------
# Path to the Python interpreter inside the virtual environment.
# Edit this if your venv is elsewhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Detect OS and set venv python path
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
    PYTHON="$REPO_DIR/venv/Scripts/python.exe"
else
    PYTHON="$REPO_DIR/venv/bin/python"
fi

# Auto-detect GPU; fall back to cpu
if $PYTHON -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    DEVICE="cuda"
else
    DEVICE="cpu"
fi

# Tasks to run. Paper defaults: walker_walk=1.1M, cheetah_run=3.1M.
# Reduced to 200k here for compute budget — curve shape reflects real learning.
declare -A TASK_FRAMES=(
    ["walker_walk"]="200000"
    ["cheetah_run"]="200000"
)

SEEDS=(1 2 3)
# Explicit task order so runs are predictable (bash assoc arrays have no order)
TASKS=(walker_walk cheetah_run)
RESULTS_DIR="$REPO_DIR/results/main"

TOTAL=$((${#TASKS[@]} * ${#SEEDS[@]}))
RUN=0

echo "======================================================"
echo "DrQ-v2 Main Experiments"
echo "Tasks: ${TASKS[*]} | Device: $DEVICE"
echo "Seeds: ${SEEDS[*]} | Total runs: $TOTAL"
echo "======================================================"

# ---------- Run experiments ----------
for TASK in "${TASKS[@]}"; do
    NUM_FRAMES="${TASK_FRAMES[$TASK]}"
    for SEED in "${SEEDS[@]}"; do
        RUN=$((RUN + 1))
        OUT_DIR="$RESULTS_DIR/${TASK}/seed_${SEED}"
        mkdir -p "$OUT_DIR"

        echo ""
        echo "[$RUN/$TOTAL] Task: $TASK | Seed: $SEED | Frames: $NUM_FRAMES | Device: $DEVICE"
        echo "  -> $OUT_DIR"

        # Run with paper-default hyperparameters for the task.
        # eval_every_frames=5000 gives denser curves (paper uses 10k).
        $PYTHON "$REPO_DIR/train.py" \
            "task@_global_=$TASK" \
            seed="$SEED" \
            device="$DEVICE" \
            num_train_frames="$NUM_FRAMES" \
            eval_every_frames=5000 \
            save_video=false \
            save_train_video=false \
            use_tb=false \
            replay_buffer_num_workers=0 \
            "hydra.run.dir=$OUT_DIR"

        echo "  Done: $TASK seed=$SEED"
    done
done

echo ""
echo "======================================================"
echo "All $TOTAL main experiments complete!"
echo "Results saved in: $RESULTS_DIR"
echo "Run: python scripts/generate_plots.py"
echo "======================================================"
