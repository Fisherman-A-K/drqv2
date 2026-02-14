#!/usr/bin/env bash
# ============================================================
# run_ablation.sh
#
# CSC415 A1 - DrQ-v2 Ablation Study: N-step Return
#
# Hypothesis: Larger n-step returns (n=3) reduce variance and
# improve sample efficiency compared to 1-step TD targets (n=1),
# because they propagate reward information faster. However,
# n=5 may introduce too much bias in long-horizon tasks.
#
# Compares n=1, n=3 (default), n=5 on walker_walk with 3 seeds each.
# All other hyperparameters are identical across conditions.
#
# Usage (from repo root):
#   bash scripts/run_ablation.sh
#
# Prerequisites: same as run_main_experiments.sh
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
    PYTHON="$REPO_DIR/venv/Scripts/python.exe"
else
    PYTHON="$REPO_DIR/venv/bin/python"
fi

if $PYTHON -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    DEVICE="cuda"
else
    DEVICE="cpu"
fi

# Ablation settings
TASK="walker_walk"
NUM_FRAMES="200000"
SEEDS=(1 2 3)
RESULTS_DIR="$REPO_DIR/results/ablation"

# N-step values to compare — the only thing that differs between conditions
NSTEP_VALUES=(1 3 5)

TOTAL=$((${#NSTEP_VALUES[@]} * ${#SEEDS[@]}))
RUN=0

echo "======================================================"
echo "DrQ-v2 N-step Return Ablation Study"
echo "Task: $TASK | Frames: $NUM_FRAMES | Device: $DEVICE"
echo "Conditions: nstep in {${NSTEP_VALUES[*]}}"
echo "Seeds: {${SEEDS[*]}}"
echo "Total runs: $TOTAL"
echo "======================================================"

for NSTEP in "${NSTEP_VALUES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        RUN=$((RUN + 1))
        OUT_DIR="$RESULTS_DIR/nstep_${NSTEP}/seed_${SEED}"
        mkdir -p "$OUT_DIR"

        echo ""
        echo "[$RUN/$TOTAL] nstep=$NSTEP | seed=$SEED | device=$DEVICE"
        echo "  -> $OUT_DIR"

        # All hyperparameters identical except nstep.
        # batch_size=256 is fixed (overrides walker_walk default of 512)
        # so that the only independent variable is nstep.
        $PYTHON "$REPO_DIR/train.py" \
            "task@_global_=$TASK" \
            seed="$SEED" \
            device="$DEVICE" \
            num_train_frames="$NUM_FRAMES" \
            eval_every_frames=5000 \
            nstep="$NSTEP" \
            batch_size=256 \
            save_video=false \
            save_train_video=false \
            use_tb=false \
            replay_buffer_num_workers=0 \
            "hydra.run.dir=$OUT_DIR"

        echo "  Done: nstep=$NSTEP seed=$SEED"
    done
done

echo ""
echo "======================================================"
echo "All $TOTAL ablation runs complete!"
echo "Results in: $RESULTS_DIR"
echo "Run: python scripts/generate_plots.py"
echo "======================================================"
