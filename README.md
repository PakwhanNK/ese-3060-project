# ESE 3060 Final Project Fall 2025

## Project Overview
This project contains two machine learning training benchmarks:
- **airbench94.py**: CIFAR-10 image classification benchmark
- **train_gpt.py**: GPT-2 training on the FineWeb-10B dataset

## Setup and Installation

### Prerequisites
- Python 3.8+
- NVIDIA GPU (A100/H100 recommended)
- CUDA 11.7 or later

### Dependencies
Install all required packages:
```bash
pip install -r requirements.txt
```

## Running airbench94.py

### Overview
CIFAR-10 training benchmark achieving 94.01% average accuracy in 3.83 seconds on an NVIDIA A100. You will want to use a single node of an a100.
- CIFAR-10 dataset automatically downloaded on first run
- Cached to `cifar10/` directory as `.pt` files for faster subsequent runs

### Execution
```bash
python airbench94.py
```

Runs 25 training iterations and reports mean/standard deviation accuracy metrics.

### Output
- Per-epoch training metrics (loss, accuracy)
- Validation and test-time augmentation (TTA) accuracy
- Logs saved to `logs/{uuid}/log.pt`

### Hardware Requirements
- NVIDIA A100 GPU recommended
- CUDA 11.7+
- NVIDIA Driver 515.105.01 or compatible

### Reference
Based on: [cifar10-airbench legacy airbench94.py](https://github.com/KellerJordan/cifar10-airbench/blob/master/legacy/airbench94.py)

## Experiments

### Experiment 410: TTA Weight Optimization

**Branch:** `experiment/exp410-tta-weight-optimization`

**Objective:** Investigate whether modifying the test-time augmentation (TTA) weights can improve classification accuracy beyond the current fixed 50/50 split between untranslated and translated views.

**Background:**
The current TTA Level 2 implementation uses a fixed weight distribution:
- Untranslated views (original + flipped): 0.5 total weight (0.25 each)
- Translated views (4 shifted+flipped versions): 0.5 total weight (0.125 each)

Literature review suggests that optimizing these weights could yield small but significant accuracy improvements.

**Approach:**
1. **Phase 1:** Train n=25 baseline models with standard hyperparameters
2. **Phase 2:** For each trained model, evaluate with 8 different TTA weight configurations:
   - 0.30, 0.35, 0.40, 0.45, 0.50 (baseline), 0.55, 0.60, 0.65, 0.70
3. **Phase 3:** Statistical analysis with paired t-tests and effect size calculations

**Key Advantage:** Since we only re-evaluate (not retrain), this is extremely efficient:
- Total evaluations: 25 models × 8 configs = 200 evaluations (~10 minutes on A100)
- Expected cost: <0.01 A100-hours (negligible)

**Running the Experiment:**

1. Train baseline models and run sweep:
```bash
python sweep_tta_weights.py --n_models 25
```

2. Analyze results:
```bash
python analyze_tta_weights.py
```

3. View outputs in `experiments/exp410-tta-weight-optimization/`:
   - `comparison_table.txt` - Statistical comparison of all weight configs
   - `tta_weight_analysis.png` - Visualization plots
   - `analysis_summary.txt` - Key findings and recommendations

**Expected Deliverables:**
- Modified `airbench94.py` with parameterized TTA weights
- Statistical analysis showing if any weight configuration significantly improves accuracy
- P-values, effect sizes (Cohen's d), and 95% confidence intervals for all configs
- Recommendations for optimal TTA weight setting

## Running train_gpt.py

### Overview
Trains a GPT-2 model on the FineWeb-10B dataset. You will want to use an 8xH100.

### Execution
Download the data with 
```bash
python cached_fineweb10B.py 9
```
and then run the script with 
```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Hardware Requirements
- Tested on 8× NVIDIA H100 80GB GPUs
- PyTorch 2.4.1+ with CUDA 12.1

### Reference
Based on: [modded-nanogpt record number #5](https://github.com/KellerJordan/modded-nanogpt/blob/master/records/track_1_short/2024-10-14_ModernArch/dabaaddd-237c-4ec9-939d-6608a9ed5e27.txt)