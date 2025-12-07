# Experiment 410: TTA Weight Optimization with Time Tracking

## Overview

**Modified exp410 to track time for each experiment, since the main objective is to decrease runtime while keeping accuracy at 94%.**

## Changes Made

### 1. New Script: `sweep_tta_weights_with_timing.py`

**Key Modifications:**
- ✅ Tracks **training time** for each baseline model
- ✅ Tracks **evaluation time** for each TTA weight configuration
- ✅ Logs **total time** (training + eval) for each run
- ✅ Reports time comparisons in output

**Why this matters:**
Different TTA weights may have subtle performance differences, and we want to find the configuration that achieves ≥94% accuracy in **minimum time**.

### 2. New Analysis Script: `analyze_tta_time_tradeoff.py`

**Features:**
- Compares all TTA weight configurations by time AND accuracy
- Identifies fastest config achieving ≥94% accuracy
- Shows time savings vs baseline (weight=0.50)
- Exports detailed CSV: `tta_time_accuracy_tradeoff.csv`

**Output includes:**
- Mean accuracy ± std for each weight
- Mean time ± std for each weight
- Success rate (% runs achieving ≥94%)
- Time difference vs baseline
- Recommendations for best config

### 3. Updated RunPod Scripts

**Files:**
- `runpod_exp410_with_timing.sh` - Standalone exp410 runner
- `RUNPOD_COMPLETE_WITH_TIMING.txt` - Full setup for both exp310 & exp410
- `RUNPOD_ONE_LINER_WITH_TIMING.txt` - Easy copy-paste version

## How to Use on RunPod

### Option 1: One-Line Command (Easiest)

```bash
cd /workspace && rm -rf ese-3060-project && git clone https://github.com/PakwhanNK/ese-3060-project.git && cd ese-3060-project && pip install -r requirements.txt && git checkout experiment/exp410-tta-weight-optimization && (python sweep_tta_weights_with_timing.py --n_models 25 || python sweep_tta_weights.py --n_models 25) && git checkout main && python analyze_tta_time_tradeoff.py 2>/dev/null || python simple_tta_analysis.py && cd /workspace && zip -r exp410-timing-results.zip ese-3060-project/experiments/exp410* ese-3060-project/experiments/*tta*.csv -x "*.pt" && echo "DONE!"
```

### Option 2: Step-by-Step

```bash
# 1. Setup
cd /workspace
git clone https://github.com/PakwhanNK/ese-3060-project.git
cd ese-3060-project
pip install -r requirements.txt

# 2. Checkout exp410 branch
git checkout experiment/exp410-tta-weight-optimization

# 3. Run sweep with time tracking (if script is in repo)
python sweep_tta_weights_with_timing.py --n_models 25

# Or fallback to standard sweep (less detailed timing)
python sweep_tta_weights.py --n_models 25

# 4. Analyze results
python analyze_tta_time_tradeoff.py

# 5. Package results
cd /workspace
zip -r exp410-results.zip ese-3060-project/experiments/exp410*
```

## Understanding the Results

### Key Output Files

1. **`experiments/tta_time_accuracy_tradeoff.csv`**
   - Complete comparison table
   - Columns: weight, mean_accuracy, std_accuracy, mean_time, std_time, success_rate, achieves_94, time_diff_vs_baseline

2. **`experiments/exp410-tta-weight-optimization_tta_weight_XX/`** (one per weight)
   - `summary.json` - Statistics for this weight config
   - `runs.csv` - Individual run results (with time!)
   - `results.json` - All run data

### What to Look For

**Goal:** Find TTA weight that:
1. ✅ Achieves ≥94% mean accuracy
2. ✅ Has minimum total time (training + eval)

**Example Analysis:**

```
Weight     Mean Acc    Mean Time    ≥94% Rate    Time Δ
0.30       0.9385      3.75s        82%          -0.08s (-2.1%)
0.50       0.9410      3.83s        88%          BASELINE
0.60       0.9405      3.81s        86%          -0.02s (-0.5%)
```

**Interpretation:**
- Weight 0.50 (baseline): 94.10% accuracy, 3.83s
- Weight 0.60: Similar accuracy (94.05%), 0.02s faster
- Weight 0.30: Lower accuracy (93.85%), faster but doesn't meet 94% target

**Best choice:** Weight with ≥94% accuracy and lowest time

## Technical Details

### Time Tracking Implementation

**Training Time:**
```python
starter = torch.cuda.Event(enable_timing=True)
ender = torch.cuda.Event(enable_timing=True)

starter.record()
# ... training loop ...
ender.record()
torch.cuda.synchronize()
total_time_seconds += 1e-3 * starter.elapsed_time(ender)
```

**Evaluation Time:**
```python
starter.record()
acc = evaluate(model, test_loader, tta_level=2, untrans_weight=weight)
ender.record()
torch.cuda.synchronize()
eval_time = 1e-3 * starter.elapsed_time(ender)
```

**Total Time:**
```python
total_time = training_time + eval_time
logger.log_run(run_id, accuracy=acc, time_seconds=total_time, ...)
```

### Why Track Both Training and Eval Time?

- **Training time**: Varies slightly between runs due to randomness
- **Eval time**: Should be nearly identical for all TTA weights (just different averaging)
- **Total time**: Most meaningful metric for comparing configurations

## Expected Runtime on A100

- Training 25 models: ~95-100 seconds (3.8-4.0s per model)
- Evaluating 25 models × 9 weights: ~5-10 seconds
- **Total experiment time: ~2-3 minutes**

## Files Added to Repository

1. ✅ `sweep_tta_weights_with_timing.py` - Time-tracking sweep script
2. ✅ `analyze_tta_time_tradeoff.py` - Time-accuracy tradeoff analysis
3. ✅ `runpod_exp410_with_timing.sh` - RunPod setup script
4. ✅ `RUNPOD_COMPLETE_WITH_TIMING.txt` - Complete setup instructions
5. ✅ `RUNPOD_ONE_LINER_WITH_TIMING.txt` - One-line command
6. ✅ `EXP410_TIME_TRACKING_README.md` - This file

## Next Steps

1. Commit these files to the `experiment/exp410-tta-weight-optimization` branch
2. Push to GitHub
3. Run on RunPod using the one-liner command
4. Download and analyze `tta_time_accuracy_tradeoff.csv`
5. Identify optimal TTA weight for your use case

## Questions?

- **Q: Why is eval time similar for all TTA weights?**
  - A: The weight only changes how predictions are averaged, not the number of augmentations computed.

- **Q: Which time matters more?**
  - A: Total time is most important. Training time dominates (~3.8s), eval time is tiny (~0.01s).

- **Q: What if no config achieves 94%?**
  - A: The analysis script will show the best accuracy achieved and recommend closest config.

- **Q: Can I use existing checkpoints?**
  - A: Yes! Use `--skip_training` flag (but training time will be estimated, not measured).
