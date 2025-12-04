#!/bin/bash
# RunPod Setup Script for ESE 3060 CIFAR-10 Baseline Experiment
# This script sets up the environment and runs the baseline experiment

set -e  # Exit on error

echo "=========================================="
echo "ESE 3060 RunPod Setup - Baseline Experiment"
echo "=========================================="

# Set working directory
WORK_DIR="/workspace/ese-3060-project"
cd /workspace

# Check GPU availability
echo ""
echo "Checking GPU..."
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

# Clone or update repository
if [ -d "$WORK_DIR" ]; then
    echo ""
    echo "Repository exists. Pulling latest changes..."
    cd "$WORK_DIR"
    git fetch origin
    git checkout experiment/exp000-baseline-100runs
    git pull origin experiment/exp000-baseline-100runs
else
    echo ""
    echo "Cloning repository..."
    git clone https://github.com/PakwhanNK/ese-3060-project.git "$WORK_DIR"
    cd "$WORK_DIR"
    git checkout experiment/exp000-baseline-100runs
fi

# Install dependencies
echo ""
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Download CIFAR-10 dataset (if not already present)
echo ""
echo "Checking CIFAR-10 dataset..."
python -c "
import torch
from torchvision import datasets
import os

cifar_dir = 'cifar10'
if not os.path.exists(cifar_dir):
    print('Downloading CIFAR-10 dataset...')
    datasets.CIFAR10(root='.', train=True, download=True)
    datasets.CIFAR10(root='.', train=False, download=True)
    print('Dataset downloaded successfully!')
else:
    print('CIFAR-10 dataset already exists.')
"

# Run the baseline experiment
echo ""
echo "=========================================="
echo "Starting Baseline Experiment (100 runs)"
echo "This will take approximately 6-7 minutes on an A100"
echo "=========================================="
echo ""

python airbench94.py \
    --exp_name exp000-baseline-100runs \
    --desc "Baseline CIFAR-10 with default hyperparameters - 100 runs for statistical significance" \
    --runs 100

# Display results
echo ""
echo "=========================================="
echo "Experiment Complete!"
echo "=========================================="
echo ""
echo "Results saved to: experiments/exp000-baseline-100runs/"
echo ""
echo "Summary statistics:"
cat experiments/exp000-baseline-100runs/summary.json | python -m json.tool | grep -A 20 '"statistics"'

echo ""
echo "To download results, use:"
echo "  scp -r experiments/exp000-baseline-100runs/ your-local-machine:~/Downloads/"
echo ""
echo "Or zip and download via RunPod file browser:"
echo "  zip -r exp000-baseline-100runs.zip experiments/exp000-baseline-100runs/"
