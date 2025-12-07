# Experiment 421: Early Stopping with Optimal LR Schedule + TTA

## Quick Summary

**Tests if the optimal LR schedule (that yields best accuracy) combined with TTA enables earlier training termination compared to baseline LR schedule.**

## LR Schedules Compared

| Schedule | Start | Peak | End | Used In |
|----------|-------|------|-----|---------|
| **Baseline** | 0.15 | 0.21 | **0.07** | exp420 |
| **Optimal** | 0.15 | 0.21 | **0.05** | exp421 ← This experiment |

The optimal schedule was found in previous experiments to yield the **best accuracy without time constraints**.

## Hypothesis

If we use the optimal LR schedule (end=0.05):
- Models may reach higher raw accuracy at same epochs
- TTA boost may allow stopping even earlier than baseline
- **Combined effect: Earlier stopping + better accuracy**

## Run on RunPod

### Option 1: Run Only Exp421 (Optimal LR)

```bash
cd /workspace && rm -rf ese-3060-project && git clone https://github.com/PakwhanNK/ese-3060-project.git && cd ese-3060-project && pip install -r requirements.txt && git checkout experiment/exp421-early-stop-optimal-lr && python sweep_early_stop_optimal_lr.py --runs_per_epoch 25 --lr_end 0.05 && python analyze_early_stopping.py && cd /workspace && zip -r exp421-results.zip ese-3060-project/experiments/exp421* && echo "DONE!"
```

### Option 2: Run Both Exp420 + Exp421 for Comparison ⭐

```bash
cd /workspace && rm -rf ese-3060-project && git clone https://github.com/PakwhanNK/ese-3060-project.git && cd ese-3060-project && pip install -r requirements.txt && git checkout experiment/exp420-early-stop-tta && python sweep_early_stop_with_tta.py --runs_per_epoch 25 && git stash && git checkout experiment/exp421-early-stop-optimal-lr && python sweep_early_stop_optimal_lr.py --runs_per_epoch 25 && python compare_early_stop_experiments.py && cd /workspace && zip -r early-stop-comparison.zip ese-3060-project/experiments/exp42* -x "*.pt" && echo "DONE!"
```

## Expected Results

### Success Case
```
Baseline LR (end=0.07):  Stops at 9.4 epochs, 94.01% accuracy
Optimal LR (end=0.05):   Stops at 9.2 epochs, 94.08% accuracy ← Better!

Winner: OPTIMAL LR
- 0.2 fewer epochs (2% epoch reduction)
- 0.07% higher accuracy
- Enables earlier stopping!
```

### Neutral Case
```
Both schedules stop at same epoch (9.4) with similar accuracy
Conclusion: End LR value doesn't significantly impact early stopping
```

## Key Files

1. **sweep_early_stop_optimal_lr.py** - Main experiment script
2. **compare_early_stop_experiments.py** - Compares exp420 vs exp421
3. **RUNPOD_EXP421_OPTIMAL_LR.txt** - RunPod instructions

## Output

- `experiments/exp421-early-stop-optimal-lr_comparison.csv` - Results
- `experiments/early_stop_lr_comparison.csv` - Side-by-side comparison (if both ran)

## Runtime

- **Single experiment (exp421):** ~18-20 minutes (250 runs)
- **Both experiments (exp420 + exp421):** ~35-40 minutes (500 runs)

## Why This Matters

If optimal LR enables earlier stopping:
- **Production benefit:** Use end=0.05 + stop at fewer epochs
- **Compound optimization:** Combine with exp410 (optimal TTA weight)
- **Maximum speedup:** Optimal LR + optimal TTA + fewer epochs

## Next Steps

After results:
1. Check `compare_early_stop_experiments.py` output
2. Look for "HEAD-TO-HEAD COMPARISON" and "WINNER"
3. If optimal LR wins: Update baseline to use end=0.05
4. If baseline wins: Document why end=0.07 is better for this use case
