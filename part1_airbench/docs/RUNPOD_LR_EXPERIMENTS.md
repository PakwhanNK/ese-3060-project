# RunPod Instructions: LR Time Optimization Experiments

Run three LR schedule experiments to minimize training time while maintaining >=94% accuracy.

## Experiments

1. **exp001-lr-baseline**: start=0.20, peak=0.23, end=0.07 (current best)
2. **exp002-lr-end05-peak18**: start=0.15, peak=0.18, end=0.05
3. **exp003-lr-end05-peak27**: start=0.15, peak=0.27, end=0.05

**Success metric:** Minimize `best_time_94plus` (target: <4.05s vs baseline ~4.06s)

---

## Method 1: Automated (RECOMMENDED)

Run all three experiments automatically with a single command:

```bash
cd /workspace/ese-3060-project
git checkout main
git pull origin main
pip install pandas scipy matplotlib
python run_lr_time_experiments.py
```

This will:
- Run all 3 experiments (25 runs each = 75 total runs)
- Modify LR schedule for each experiment
- Save results to `experiments/exp00X-*/`
- Generate comparison CSV at `experiments/lr_time_comparison.csv`
- Print summary with time improvements

**Expected runtime:** ~5 minutes per run × 75 runs = ~6.25 hours

---

## Method 2: Manual (Step-by-Step)

### Step 1: Setup

```bash
cd /workspace/ese-3060-project
git checkout main
git pull origin main
pip install pandas scipy matplotlib
```

### Step 2: Run exp001-lr-baseline

**Current main branch already has the correct LR params!**

```bash
python airbench94.py --exp_name exp001-lr-baseline \
  --desc "Baseline LR schedule: start=0.20, peak=0.23, end=0.07" \
  --runs 25
```

### Step 3: Run exp002-lr-end05-peak18

**Edit airbench94.py line 411:**

Change:
```python
lr_schedule = triangle(total_train_steps, start=0.2, end=0.07, peak=0.23)
```

To:
```python
lr_schedule = triangle(total_train_steps, start=0.15, end=0.05, peak=0.18)
```

Then run:
```bash
python airbench94.py --exp_name exp002-lr-end05-peak18 \
  --desc "Lower end/peak: start=0.15, peak=0.18, end=0.05" \
  --runs 25
```

### Step 4: Run exp003-lr-end05-peak27

**Edit airbench94.py line 411:**

Change:
```python
lr_schedule = triangle(total_train_steps, start=0.15, end=0.05, peak=0.18)
```

To:
```python
lr_schedule = triangle(total_train_steps, start=0.15, end=0.05, peak=0.27)
```

Then run:
```bash
python airbench94.py --exp_name exp003-lr-end05-peak27 \
  --desc "Lower end, higher peak: start=0.15, peak=0.27, end=0.05" \
  --runs 25
```

### Step 5: Generate Comparison

```bash
python compare_lr_experiments.py
```

This generates:
- `experiments/lr_time_comparison.csv`
- Console output with summary and winner

---

## Quick Test (3 runs instead of 25)

To quickly verify everything works:

```bash
# Automated
python run_lr_time_experiments.py  # Edit RUNS_PER_EXPERIMENT = 3 first

# Manual
python airbench94.py --exp_name exp001-lr-baseline --runs 3
# (edit LR params)
python airbench94.py --exp_name exp002-lr-end05-peak18 --runs 3
# (edit LR params)
python airbench94.py --exp_name exp003-lr-end05-peak27 --runs 3
python compare_lr_experiments.py
```

---

## Expected Outputs

### During Experiments

Each experiment will print:
```
============================================================
EXPERIMENT: exp001-lr-baseline
Description: Baseline LR schedule: start=0.20, peak=0.23, end=0.07
Git commit: abc1234
============================================================

|  run     |  epoch  |  train_loss  |  train_acc  |  val_acc  |  tta_val_acc  |  total_time_seconds  |
|  0       |  eval   |              |             |           |  0.9405       |  4.0612             |
...
```

### Comparison Output

```
============================================================
COMPARISON TABLE
============================================================

experiment_name            runs_94plus  success_rate  best_time_94plus  ...
exp001-lr-baseline        25/25        100.0%        4.0612            ...
exp002-lr-end05-peak18    25/25        100.0%        4.0501            ...
exp003-lr-end05-peak27    24/25        96.0%         4.0689            ...

============================================================
SUMMARY
============================================================
Baseline (exp001-lr-baseline): 4.0612s

🏆 FASTEST EXPERIMENT: exp002-lr-end05-peak18
  Best time (>=94%): 4.0501s
  Time improvement: 0.0111s (0.27%)
  Success rate: 100.0%
  Mean accuracy: 0.9408 ± 0.0011

✓ SUCCESS: Found faster configuration!
```

---

## Troubleshooting

### Missing pandas/scipy

```bash
pip install pandas scipy matplotlib
```

### Results not found

Check if experiments completed:
```bash
ls experiments/exp00*
cat experiments/exp001-lr-baseline/results.json | head -20
```

### Re-run comparison only

```bash
python compare_lr_experiments.py
```

---

## Files Created

- `experiments/exp001-lr-baseline/results.json` - Raw run data
- `experiments/exp001-lr-baseline/summary.json` - Statistics
- `experiments/exp002-lr-end05-peak18/results.json`
- `experiments/exp002-lr-end05-peak18/summary.json`
- `experiments/exp003-lr-end05-peak27/results.json`
- `experiments/exp003-lr-end05-peak27/summary.json`
- `experiments/lr_time_comparison.csv` - Final comparison table

---

## One-Liner (Complete Automated Run)

```bash
cd /workspace/ese-3060-project && git checkout main && git pull && pip install pandas scipy matplotlib && python run_lr_time_experiments.py
```

This is the fastest way to run all experiments!
