#!/bin/bash
# RunPod script for Experiment 410 (TTA Weight Optimization) with TIME TRACKING
#
# OBJECTIVE: Find TTA weight that achieves ≥94% accuracy in MINIMUM TIME
# Tracks: Training time + Evaluation time for each configuration

set -e

echo "=========================================="
echo "Experiment 410: TTA Weight Optimization"
echo "WITH TIME TRACKING"
echo "=========================================="

WORK_DIR="/workspace/ese-3060-project"
REPO_URL="https://github.com/PakwhanNK/ese-3060-project.git"

cd /workspace

# Check GPU
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

# Clone/update repository
if [ -d "$WORK_DIR" ]; then
    echo ""
    echo "Cleaning existing directory..."
    rm -rf "$WORK_DIR"
fi

echo ""
echo "Cloning repository..."
git clone "$REPO_URL" "$WORK_DIR"
cd "$WORK_DIR"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Checkout exp410 branch
echo ""
echo "Checking out exp410 branch..."
git checkout experiment/exp410-tta-weight-optimization
git pull origin experiment/exp410-tta-weight-optimization

# Copy the time-tracking version of the sweep script
echo ""
echo "Setting up time-tracking sweep script..."

# Check if the new timing script exists, if not create it inline
if [ ! -f "sweep_tta_weights_with_timing.py" ]; then
    echo "Creating sweep_tta_weights_with_timing.py..."
    # The script should be in your repo, but if not, you'll need to add it
    echo "ERROR: sweep_tta_weights_with_timing.py not found!"
    echo "Please ensure it's committed to the repository."
    echo ""
    echo "Falling back to original sweep script (no detailed timing)..."
    SWEEP_SCRIPT="sweep_tta_weights.py"
else
    SWEEP_SCRIPT="sweep_tta_weights_with_timing.py"
    echo "Using time-tracking version: $SWEEP_SCRIPT"
fi

# Run the experiment
echo ""
echo "=========================================="
echo "Running TTA Weight Sweep with Time Tracking"
echo "=========================================="
echo "Configuration:"
echo "  - Models to train: 25"
echo "  - TTA weights to test: 9 (0.30 to 0.70)"
echo "  - Tracking: Training time + Eval time per config"
echo "  - Objective: Find fastest config with ≥94% accuracy"
echo "=========================================="
echo ""

python "$SWEEP_SCRIPT" --n_models 25

echo ""
echo "✓ Sweep complete!"

# Run time-accuracy tradeoff analysis
echo ""
echo "=========================================="
echo "Analyzing Time-Accuracy Tradeoff"
echo "=========================================="

if [ -f "analyze_tta_time_tradeoff.py" ]; then
    python analyze_tta_time_tradeoff.py
else
    echo "Warning: analyze_tta_time_tradeoff.py not found, using simple analysis..."
    python simple_tta_analysis.py || echo "Analysis failed"
fi

# Package results
echo ""
echo "=========================================="
echo "Packaging Results"
echo "=========================================="

cd /workspace
zip -r exp410-tta-timing-results.zip \
    ese-3060-project/experiments/exp410-tta-weight-optimization* \
    ese-3060-project/experiments/*tta*.csv \
    -x "*.pt" "*/checkpoints/*"

echo ""
echo "=========================================="
echo "EXPERIMENT 410 COMPLETE!"
echo "=========================================="
echo ""
echo "Results Location:"
echo "  - Detailed experiments: $WORK_DIR/experiments/exp410-*/"
echo "  - Time-accuracy CSV: experiments/tta_time_accuracy_tradeoff.csv"
echo "  - Download archive: /workspace/exp410-tta-timing-results.zip"
echo ""
echo "Key Files to Check:"
ls -lh "$WORK_DIR/experiments/"*tta*.csv 2>/dev/null || echo "  (CSVs will be in experiments/)"
echo ""
echo "Next Steps:"
echo "  1. Download: /workspace/exp410-tta-timing-results.zip"
echo "  2. Look for: TTA weight with ≥94% accuracy + minimum time"
echo "  3. Compare time savings vs baseline (weight=0.50)"
echo "=========================================="
