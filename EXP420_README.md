# Experiment 420: Early Stopping with TTA Compensation

## Motivation

Current baseline (airbench94.py):
- Trains for **9.9 epochs**
- Final accuracy: **94.01%** (with TTA level 2)
- Without TTA: **93.2%** accuracy
- TTA boost: **~0.8%**

**Key Insight:** If TTA consistently boosts accuracy by 0.8%, perhaps we can train for fewer epochs and let TTA compensate to still reach 94%.

## Hypothesis

**TTA allows early stopping while maintaining target accuracy.**

If we stop training at 9.0-9.8 epochs, the model will have lower raw accuracy, but TTA might boost it back to ≥94%.

## Experiment Design

### What We Test

Train models for different epoch values:
- **9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9**
- 25 runs per epoch value
- **250 total runs**

### What We Measure

For each epoch value:
1. **Training time** (how long it takes)
2. **No-TTA accuracy** (raw model performance)
3. **TTA accuracy** (with augmentation boost)
4. **Success rate** (% of runs achieving ≥94% with TTA)

### What We Find

- **Minimum epochs** needed to achieve ≥94% with TTA
- **Time savings** vs baseline (9.9 epochs)
- **Tradeoff curve** between epochs, time, and accuracy

## Potential Benefits

If we can reduce from 9.9 to 9.0 epochs:

| Metric | Baseline (9.9) | Optimized (9.0) | Savings |
|--------|----------------|-----------------|---------|
| Time per run | 3.83s | ~3.48s | 0.35s (9%) |
| 1,000 runs | 3,830s (64 min) | 3,480s (58 min) | 6 minutes |
| 10,000 runs | 10.6 hours | 9.7 hours | **58 minutes** |

## Files in This Experiment

### 1. `sweep_early_stop_with_tta.py`

Main experiment script that:
- Trains models for different epoch values
- Evaluates with and without TTA
- Tracks training time precisely
- Logs all results to ExperimentLogger

**Usage:**
```bash
# Basic run (default TTA weight 0.50)
python sweep_early_stop_with_tta.py --runs_per_epoch 25

# With optimal TTA weight from exp410
python sweep_early_stop_with_tta.py --runs_per_epoch 25 --tta_weight 0.55

# Custom epoch range
python sweep_early_stop_with_tta.py \
    --runs_per_epoch 25 \
    --epoch_start 9.0 \
    --epoch_end 9.5 \
    --epoch_step 0.1
```

**Parameters:**
- `--runs_per_epoch`: Number of runs per epoch value (default: 25)
- `--epoch_start`: Starting epoch value (default: 9.0)
- `--epoch_end`: Ending epoch value (default: 9.9)
- `--epoch_step`: Step size (default: 0.1)
- `--tta_weight`: TTA untrans_weight (default: 0.50)
- `--experiment_name`: Base name for logging (default: exp420-early-stop-tta)

### 2. `analyze_early_stopping.py`

Analysis script that:
- Loads all experiment results
- Identifies minimum epochs for ≥94% accuracy
- Calculates time savings
- Projects savings at scale
- Exports detailed CSV

**Usage:**
```bash
python analyze_early_stopping.py
```

**Output:**
- `experiments/early_stopping_analysis.csv` - Detailed comparison
- Console output with recommendations
- Visualization of epoch vs accuracy trend

### 3. `RUNPOD_EXP420_EARLY_STOPPING.txt`

RunPod setup instructions:
- One-line command for easy execution
- Step-by-step alternative
- Expected runtime and results
- Interpretation guide

## How to Run on RunPod

### Quick Start (Copy-Paste)

```bash
cd /workspace && rm -rf ese-3060-project && git clone https://github.com/PakwhanNK/ese-3060-project.git && cd ese-3060-project && pip install -r requirements.txt && git checkout experiment/exp420-early-stop-tta && git pull origin experiment/exp420-early-stop-tta && python sweep_early_stop_with_tta.py --runs_per_epoch 25 && python analyze_early_stopping.py && cd /workspace && zip -r exp420-results.zip ese-3060-project/experiments/exp420* && echo "DONE!"
```

### Expected Runtime

- **Per run:** ~3.5-3.8s (varies by epoch count)
- **Total runs:** 10 epochs × 25 runs = 250 runs
- **Total time:** ~15-20 minutes on A100

## Understanding Results

### Output Format

```
Epochs     Mean TTA    ±Std       Mean Time   ±Std       ≥94% Rate    Target?    Time Saved
9.0        0.9385      0.0042     3.48s       0.05       82%          ✗ NO       -0.35s (9.1%)
9.3        0.9402      0.0038     3.59s       0.04       85%          ✓ YES      -0.24s (6.3%)
9.9        0.9410      0.0035     3.83s       0.06       88%          ✓ YES      BASELINE
```

### Interpretation

**SUCCESS Example:**
If 9.3 epochs achieves ≥94% with TTA:
- ✅ Can reduce training by 0.6 epochs (6%)
- ✅ Saves ~0.24s per run
- ✅ Hypothesis validated: TTA enables early stopping!

**FAILURE Example:**
If only 9.9 epochs achieves ≥94%:
- ❌ Cannot reduce epochs while maintaining target
- ❌ TTA boost insufficient to compensate for less training
- 📊 But we still learn the tradeoff curve

### Key Metrics to Check

1. **Minimum Epochs for ≥94%**
   - What's the earliest we can stop?
   - How much time does it save?

2. **Success Rate**
   - What % of runs achieve target?
   - Is it reliable or marginal?

3. **Time Savings**
   - Absolute: seconds saved per run
   - Relative: % reduction in time
   - At scale: hours saved for 10k runs

4. **Tradeoff Curve**
   - How does accuracy drop with fewer epochs?
   - Is the relationship linear or exponential?

## Combining with Exp410

If exp410 found an optimal TTA weight (e.g., 0.55 instead of 0.50), you can test whether the improved TTA enables even earlier stopping:

```bash
# First run exp410 to find optimal TTA weight
python sweep_tta_weights_with_timing.py --n_models 25

# Then use that weight in exp420
python sweep_early_stop_with_tta.py --runs_per_epoch 25 --tta_weight 0.55
```

This could find that:
- Baseline (0.50 TTA, 9.9 epochs): 94.01% accuracy
- Optimized (0.55 TTA, 9.2 epochs): 94.05% accuracy, 0.65s faster!

## Output Files

After running, you'll get:

### 1. Summary CSV
`experiments/early_stopping_analysis.csv`
```csv
epochs,mean_tta_acc,std_tta_acc,mean_time,std_time,success_rate,achieves_94,epoch_reduction,time_saved,time_saved_pct,speedup
9.0,0.938500,0.004200,3.480000,0.050000,0.820000,0,0.9,0.350000,9.14,1.10
9.3,0.940200,0.003800,3.590000,0.040000,0.850000,1,0.6,0.240000,6.27,1.07
9.9,0.941000,0.003500,3.830000,0.060000,0.880000,1,0.0,0.000000,0.00,1.00
```

### 2. Individual Experiment Folders
```
experiments/exp420-early-stop-tta_epoch90/  (9.0 epochs)
experiments/exp420-early-stop-tta_epoch93/  (9.3 epochs)
...
experiments/exp420-early-stop-tta_epoch99/  (9.9 epochs)
```

Each contains:
- `summary.json` - Statistics for this epoch value
- `runs.csv` - Individual run data
- `results.json` - Complete results

## Next Steps

### If Hypothesis is Validated

1. **Update baseline training:**
   - Change default epochs from 9.9 to optimal value
   - Update hyperparameters in airbench94.py

2. **Document findings:**
   - Report time savings
   - Update README with new baseline

3. **Consider further optimization:**
   - Can we go even lower with better TTA?
   - What about different LR schedules for fewer epochs?

### If Hypothesis is Rejected

1. **Analyze why:**
   - Is TTA boost inconsistent?
   - Does accuracy drop too steeply?

2. **Alternative approaches:**
   - Test with different TTA levels (0, 1, 2)
   - Try adaptive early stopping based on validation
   - Investigate if specific augmentations help more

## Technical Details

### Time Tracking

Uses CUDA events for precise timing:
```python
starter = torch.cuda.Event(enable_timing=True)
ender = torch.cuda.Event(enable_timing=True)

starter.record()
# ... training loop ...
ender.record()
torch.cuda.synchronize()
total_time_seconds += 1e-3 * starter.elapsed_time(ender)
```

### Epoch Precision

The script trains for **exact** fractional epochs:
- 9.1 epochs = 9 full epochs + 10% of one epoch
- Stops mid-epoch at the correct batch
- Maintains LR schedule continuity

### TTA Evaluation

Evaluates each model twice:
1. **Without TTA** (tta_level=0): raw model accuracy
2. **With TTA** (tta_level=2): augmented accuracy

This lets us see how much TTA actually helps for each epoch value.

## Questions?

**Q: What if all epoch values fail to reach 94%?**
A: The analysis will show the best achieved and recommend alternatives. This would mean TTA boost alone isn't enough.

**Q: Can I test different epoch ranges?**
A: Yes! Use `--epoch_start`, `--epoch_end`, and `--epoch_step` parameters.

**Q: Should I use optimal TTA weight from exp410?**
A: If exp410 found a better weight, definitely use it with `--tta_weight`. Otherwise, default 0.50 is fine.

**Q: What if I want finer granularity?**
A: Try `--epoch_step 0.05` to test 9.0, 9.05, 9.10, etc. (doubles the runs though!)

**Q: Can I run this locally?**
A: Yes, but you need a CUDA GPU. CPU version will work but be very slow.

## References

- Baseline: [airbench94.py](https://github.com/KellerJordan/cifar10-airbench/blob/master/legacy/airbench94.py)
- TTA Implementation: See `infer()` function in airbench94.py
- Related: Exp410 (TTA weight optimization)
