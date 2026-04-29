#!/bin/bash

# ==============================================================================
# Metrics Runner Script
# Runs:
#   1. OOD evaluation
#   2. MIA evaluation
#   3. OOD plots
#   4. MIA plots
# ==============================================================================

set -e

# 1. Project paths

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

VENV_DIR="$PROJECT_DIR/venv"
PYTHON_BIN="$VENV_DIR/bin/python"

# 2. Metrics configuration


CHECKPOINT="$PROJECT_DIR/fl_logs/master/checkpoints/global_final.pt"
OUTPUT_DIR="$PROJECT_DIR/metrics/results"

BATCH_SIZE=64
NUM_WORKERS=0
MIA_SAMPLES=512
SEED=42

# For faster first test:
# MIA_SAMPLES=128

# 3. Checks

echo "============================================================"
echo " Metrics Runner"
echo "============================================================"
echo "Project directory: $PROJECT_DIR"
echo "Python binary:      $PYTHON_BIN"
echo "Checkpoint:         $CHECKPOINT"
echo "Output directory:   $OUTPUT_DIR"
echo "Batch size:         $BATCH_SIZE"
echo "MIA samples:        $MIA_SAMPLES"
echo "============================================================"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "ERROR: Python venv not found at: $PYTHON_BIN"
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found:"
    echo "  $CHECKPOINT"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/plots"

# 4. Import check

echo ""
echo "Checking Python imports..."

"$PYTHON_BIN" - <<'PY'
import torch
import torchvision
import pandas
import pyarrow
import sklearn
import albumentations
import cv2
import matplotlib

print("Imports OK")
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
PY

# 5. Run OOD evaluation

echo ""
echo "============================================================"
echo " Running OOD Evaluation"
echo "============================================================"

"$PYTHON_BIN" "$PROJECT_DIR/metrics/run_ood.py" \
    --checkpoint "$CHECKPOINT" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS"

# 6. Run MIA evaluation

echo ""
echo "============================================================"
echo " Running Membership Inference Attack"
echo "============================================================"

"$PYTHON_BIN" "$PROJECT_DIR/metrics/run_mia.py" \
    --checkpoint "$CHECKPOINT" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --mia_samples "$MIA_SAMPLES" \
    --seed "$SEED"

# 7. Generate OOD plots

echo ""
echo "============================================================"
echo " Generating OOD Plots"
echo "============================================================"

"$PYTHON_BIN" "$PROJECT_DIR/metrics/plot_ood.py" \
    --results_dir "$OUTPUT_DIR"

# 8. Generate MIA plots

echo ""
echo "============================================================"
echo " Generating MIA Plots"
echo "============================================================"

"$PYTHON_BIN" "$PROJECT_DIR/metrics/plot_mia.py" \
    --results_dir "$OUTPUT_DIR"

# 9. Done

echo ""
echo "============================================================"
echo " Metrics and plots completed successfully."
echo " Results saved in:"
echo "   $OUTPUT_DIR"
echo ""
echo "Generated JSON files:"
find "$OUTPUT_DIR" -maxdepth 1 -type f -name "*.json" | sort
echo ""
echo "Generated plot files:"
find "$OUTPUT_DIR/plots" -maxdepth 1 -type f -name "*.png" | sort
echo "============================================================"
