# ESE 3060 Final Project - Experiments Summary

## Overview

This project contains three main optimization experiments for the airbench94 CIFAR-10 training benchmark. All experiments focus on achieving **≥94% accuracy in minimum time**.

## Experiment Sequence

### Baseline
- **Script:** `airbench94.py`
- **Epochs:** 9.9
- **Accuracy:** 94.01% (with TTA level 2)
- **Time:** ~3.83s on A100
- **TTA Weight:** 0.50 (fixed 50/50 split)

---

## Experiment 310: Lookahead Parameter Sweep

### Objective
Find optimal lookahead optimizer parameters for best accuracy/time tradeoff.

### What It Tests
- 9 lookahead configurations:
  - Different k values: 3, 5, 7, 10
  - Different schedules: constant, linear, cubic
  - Different alpha values: 0.90^5, 0.95^5, 0.98^5
  - Plus baseline (no lookahead)

### How to Run
```bash
python airbench94.py \
    --exp_name exp310-lookahead-param-sweep \
    --desc "Lookahead parameter sweep" \
    --runs 25 \
    --sweep
```

### Expected Output
- Best lookahead configuration for speed/accuracy
- Comparison of all 9 configs
- Time/accuracy tradeoffs

### Files
- `airbench94.py` (modified with LOOKAHEAD_CONFIGS)
- `simple_lookahead_analysis.py` (analysis)

---

## Experiment 410: TTA Weight Optimization ⭐ WITH TIME TRACKING

### Objective
Find optimal Test-Time Augmentation weight that achieves ≥94% accuracy in **minimum time**.

### What It Tests
- 9 TTA weight values: 0.30, 0.35, 0.40, 0.45, **0.50**, 0.55, 0.60, 0.65, 0.70
- For each weight:
  - Train 25 baseline models (**track training time**)
  - Evaluate with that TTA weight (**track eval time**)
  - Log **total time** = training + eval

### Key Innovation
**Time tracking added** to find which TTA weight achieves ≥94% fastest, not just which is most accurate.

### How to Run
```bash
# Using time-tracking version (recommended)
python sweep_tta_weights_with_timing.py --n_models 25

# Analysis
python analyze_tta_time_tradeoff.py
```

### Expected Output
- Optimal TTA weight for time-accuracy tradeoff
- Time savings vs baseline (0.50)
- Success rate for each weight

### Files
- `sweep_tta_weights_with_timing.py` ⭐ (main sweep with timing)
- `analyze_tta_time_tradeoff.py` (time-accuracy analysis)
- `simple_tta_analysis.py` (simple comparison)

---

## Experiment 420: Early Stopping with TTA Compensation 🆕

### Objective
Test if TTA boost allows training for fewer epochs while maintaining ≥94% accuracy.

### Hypothesis
Since TTA boosts accuracy by ~0.8% (93.2% → 94.01%), we can:
1. Train for fewer epochs (9.0-9.8 instead of 9.9)
2. Accept lower raw accuracy
3. Let TTA compensate to reach ≥94%
4. **Save training time**

### What It Tests
- 10 epoch values: 9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9
- For each epoch value:
  - Train 25 models for exactly that many epochs
  - Evaluate WITHOUT TTA (raw accuracy)
  - Evaluate WITH TTA (boosted accuracy)
  - Track training time precisely

### How to Run
```bash
# Basic run (default TTA weight 0.50)
python sweep_early_stop_with_tta.py --runs_per_epoch 25

# With optimal TTA weight from exp410
python sweep_early_stop_with_tta.py --runs_per_epoch 25 --tta_weight 0.55

# Analysis
python analyze_early_stopping.py
```

### Expected Outcomes

**Success Scenario:**
- Find that 9.3 epochs achieves ≥94% with TTA
- Save 0.6 epochs = ~6% time reduction
- **Validate hypothesis:** TTA enables early stopping!

**Failure Scenario:**
- Only 9.9 epochs achieves ≥94%
- Cannot reduce training time
- Learn accuracy/epoch tradeoff curve

### Potential Impact
If we can reduce from 9.9 to 9.0 epochs:
- **Per run:** Save 0.35s (9% faster)
- **1,000 runs:** Save 6 minutes
- **10,000 runs:** Save 1 hour

### Files
- `sweep_early_stop_with_tta.py` (main sweep)
- `analyze_early_stopping.py` (analysis)
- `RUNPOD_EXP420_EARLY_STOPPING.txt` (RunPod setup)
- `EXP420_README.md` (detailed documentation)

---

## Experiment Synergies

### Combining Exp410 + Exp420
If exp410 finds optimal TTA weight (e.g., 0.55), use it in exp420:

```bash
# First: Find best TTA weight
python sweep_tta_weights_with_timing.py --n_models 25
# Result: weight=0.55 is fastest while achieving ≥94%

# Then: Test if better TTA enables earlier stopping
python sweep_early_stop_with_tta.py --runs_per_epoch 25 --tta_weight 0.55
# Result: Can stop at 9.2 epochs instead of 9.9!
```

**Combined optimization:**
- Better TTA weight: saves 0.02s (from exp410)
- Fewer epochs: saves 0.27s (from exp420)
- **Total savings: 0.29s per run (7.6%)**

### Three-Way Optimization (Exp310 + Exp410 + Exp420)

Theoretical maximum speedup:
1. **Exp310:** Optimal lookahead (saves ~0.05s)
2. **Exp410:** Optimal TTA weight (saves ~0.02s)
3. **Exp420:** Fewer epochs (saves ~0.27s)
4. **Combined:** ~0.34s savings = **8.9% faster** while maintaining ≥94%

---

## Running All Experiments on RunPod

### Option 1: Run All Three Sequentially

```bash
cd /workspace && \
rm -rf ese-3060-project && \
git clone https://github.com/PakwhanNK/ese-3060-project.git && \
cd ese-3060-project && \
pip install -r requirements.txt && \
echo "=== EXP310: LOOKAHEAD SWEEP ===" && \
git checkout experiment/exp310-lookahead-param-sweep && \
sed -i 's/get_summary_stats()/compute_statistics()/g' airbench94.py && \
sed -i "s/\['mean_accuracy'\]/['accuracy_mean']/g" airbench94.py && \
sed -i "s/\['std_accuracy'\]/['accuracy_std']/g" airbench94.py && \
sed -i "s/\['mean_time'\]/['time_mean']/g" airbench94.py && \
sed -i "s/\['std_time'\]/['time_std']/g" airbench94.py && \
python airbench94.py --exp_name exp310-lookahead-param-sweep --desc "Lookahead sweep" --runs 25 --sweep && \
echo "=== EXP410: TTA WEIGHT OPTIMIZATION ===" && \
git stash && \
git checkout experiment/exp410-tta-weight-optimization && \
python sweep_tta_weights_with_timing.py --n_models 25 && \
python analyze_tta_time_tradeoff.py && \
echo "=== EXP420: EARLY STOPPING WITH TTA ===" && \
git stash && \
git checkout experiment/exp420-early-stop-tta && \
python sweep_early_stop_with_tta.py --runs_per_epoch 25 && \
python analyze_early_stopping.py && \
echo "=== PACKAGING RESULTS ===" && \
cd /workspace && \
zip -r all-experiments-results.zip ese-3060-project/experiments/ -x "*.pt" "*/checkpoints/*" && \
echo "DONE! Download all-experiments-results.zip"
```

### Option 2: Run Individually

**Exp310 (Lookahead):**
```bash
cd /workspace && rm -rf ese-3060-project && git clone https://github.com/PakwhanNK/ese-3060-project.git && cd ese-3060-project && pip install -r requirements.txt && git checkout experiment/exp310-lookahead-param-sweep && sed -i 's/get_summary_stats()/compute_statistics()/g' airbench94.py && sed -i "s/\['mean_accuracy'\]/['accuracy_mean']/g" airbench94.py && sed -i "s/\['std_accuracy'\]/['accuracy_std']/g" airbench94.py && sed -i "s/\['mean_time'\]/['time_mean']/g" airbench94.py && sed -i "s/\['std_time'\]/['time_std']/g" airbench94.py && python airbench94.py --exp_name exp310-lookahead-param-sweep --desc "Lookahead sweep" --runs 25 --sweep
```

**Exp410 (TTA Weight):**
```bash
cd /workspace && rm -rf ese-3060-project && git clone https://github.com/PakwhanNK/ese-3060-project.git && cd ese-3060-project && pip install -r requirements.txt && git checkout experiment/exp410-tta-weight-optimization && python sweep_tta_weights_with_timing.py --n_models 25 && python analyze_tta_time_tradeoff.py
```

**Exp420 (Early Stopping):**
```bash
cd /workspace && rm -rf ese-3060-project && git clone https://github.com/PakwhanNK/ese-3060-project.git && cd ese-3060-project && pip install -r requirements.txt && git checkout experiment/exp420-early-stop-tta && python sweep_early_stop_with_tta.py --runs_per_epoch 25 && python analyze_early_stopping.py
```

---

## Key Output Files

### Exp310
- `experiments/lookahead_comparison.csv` - All lookahead configs compared

### Exp410
- `experiments/tta_time_accuracy_tradeoff.csv` - TTA weights vs time/accuracy
- `experiments/tta_weight_comparison.csv` - Simple comparison

### Exp420
- `experiments/early_stopping_analysis.csv` - Epoch values vs time/accuracy
- `experiments/exp420-early-stop-tta_comparison.csv` - Summary

### General
- `experiments/comparison.csv` - All experiments compared (from compare_experiments.py)

---

## Analysis Scripts

### General Comparison
```bash
python compare_experiments.py
```
Compares ALL experiments in experiments/ folder.

### Exp-Specific Analysis
```bash
# Lookahead
python simple_lookahead_analysis.py

# TTA weights
python analyze_tta_time_tradeoff.py  # Detailed with time
python simple_tta_analysis.py         # Simple comparison

# Early stopping
python analyze_early_stopping.py
```

---

## Expected Runtime (on A100)

| Experiment | Runs | Time |
|------------|------|------|
| Exp310 | 9 configs × 25 runs = 225 | ~15 min |
| Exp410 | 25 models × 9 weights = 225 evals | ~15 min |
| Exp420 | 10 epochs × 25 runs = 250 | ~18 min |
| **Total** | **700 runs** | **~48 min** |

---

## Success Criteria

### Exp310: Lookahead
- ✅ Find best lookahead config
- ✅ Show time/accuracy tradeoff
- 🎯 Identify if any config is faster while maintaining ≥94%

### Exp410: TTA Weight
- ✅ Find best TTA weight for time-accuracy tradeoff
- ✅ Show time savings vs baseline (0.50)
- 🎯 Achieve ≥94% in minimum total time

### Exp420: Early Stopping
- ✅ Test if TTA enables training with fewer epochs
- ✅ Find minimum epochs for ≥94% with TTA
- 🎯 Save training time while maintaining target accuracy

---

## Files Created

### Experiment Scripts
- `sweep_tta_weights_with_timing.py` ⭐
- `sweep_early_stop_with_tta.py` 🆕

### Analysis Scripts
- `compare_experiments.py`
- `simple_tta_analysis.py`
- `simple_lookahead_analysis.py`
- `analyze_tta_time_tradeoff.py` ⭐
- `analyze_early_stopping.py` 🆕

### Documentation
- `EXP410_TIME_TRACKING_README.md`
- `EXP420_README.md` 🆕
- `EXPERIMENTS_SUMMARY.md` (this file)

### RunPod Setup
- `RUNPOD_ONE_LINER_WITH_TIMING.txt`
- `RUNPOD_COMPLETE_WITH_TIMING.txt`
- `RUNPOD_EXP420_EARLY_STOPPING.txt` 🆕

---

## Next Steps

1. **Commit exp420 files** to new branch
2. **Run all three experiments** on RunPod
3. **Analyze results** to find optimal configuration
4. **Update baseline** if optimizations are successful
5. **Document findings** in final report

---

## Questions?

See individual experiment READMEs for detailed documentation:
- Exp310: Check main airbench94.py comments
- Exp410: See `EXP410_TIME_TRACKING_README.md`
- Exp420: See `EXP420_README.md`
