# Experiment Standardization Guide

This document defines the standard interfaces and execution patterns for all experiments in this project.

## Experiment Types

### Type 1: Single-Run Experiments
**Purpose:** Test a single configuration with multiple runs for statistical significance

**Branches:** exp000-baseline-100runs, exp100-muon-test, exp300-no-lookup

**Execution:**
```bash
python airbench94.py --exp_name "experiment_name" --description "Experiment description" --runs 25
```

**Arguments:**
- `--exp_name` (str, default='unnamed_exp'): Unique experiment identifier
- `--description` (str, default=''): Human-readable experiment description
- `--runs` (int, default=25): Number of training runs to execute
- `--compare` (str, nargs='+', optional): Compare multiple experiments and exit

**Output:**
- `experiments/{exp_name}/results.json` - Individual run results
- `experiments/{exp_name}/summary.json` - Aggregated statistics
- Prints summary table with mean, std, 95% CI

**Example:**
```bash
# Run experiment
python airbench94.py --exp_name "baseline-test" --description "Testing baseline performance" --runs 25

# Compare multiple experiments
python airbench94.py --compare baseline-test muon-test no-lookup
```

---

### Type 2: Training Parameter Sweeps
**Purpose:** Test multiple hyperparameter configurations with training

**Branches:** exp210-triangle-peak-sweep, exp220-triangle-start-sweep, exp230-triangle-end-sweep, exp310-lookahead-param-sweep

**Execution:**
```bash
python airbench94.py --sweep_name "sweep_identifier" --runs_per_config 10
```

**Arguments:**
- `--sweep_name` (str, required): Base name for the sweep (e.g., 'lr_sweep_peak')
- `--runs_per_config` (int, default=10): Number of runs per configuration
- `--compare` (flag): Generate comparison of existing sweep results and exit

**Configuration:**
- Sweep parameters are hardcoded in the script (e.g., peaks, starts, ends, lookahead configs)
- Each configuration gets a unique experiment name: `{sweep_name}_{param_value}`

**Output:**
- `experiments/{sweep_name}_{config}/results.json` - Results for each config
- `experiments/experiments_comparison.csv` - Comparison table across all configs
- Prints comparison table at the end

**Example:**
```bash
# Run sweep (exp210: peaks [0.15, 0.18, 0.21, ...])
python airbench94.py --sweep_name "peak_sweep_test" --runs_per_config 10

# Compare existing sweep results
python airbench94.py --compare
```

**Sweep Configurations by Branch:**
- **exp210:** Peak values = [0.15, 0.18, 0.21, 0.24, 0.27, 0.30, 0.33, 0.36]
- **exp220:** Start values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
- **exp230:** End values = [0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15]
- **exp310:** Lookahead configs (9 different k/schedule/alpha combinations)

---

### Type 3: Post-Training Evaluation Sweeps
**Purpose:** Train models once, then evaluate with multiple post-training configurations

**Branches:** exp410-tta-weight-optimization

**Why Different:** These experiments separate training from evaluation. Training is expensive, but evaluating with different post-training settings (like TTA weights) is cheap. This allows testing many configurations efficiently.

**Execution:**
```bash
# Phase 1 & 2: Train baseline models and sweep evaluations
python sweep_tta_weights.py --n_models 25

# Phase 3: Statistical analysis
python analyze_tta_weights.py
```

**Arguments (sweep_tta_weights.py):**
- `--n_models` (int, default=25): Number of baseline models to train
- `--checkpoint_dir` (str): Directory to save/load model checkpoints
- `--skip_training` (flag): Skip training, use existing checkpoints
- `--experiment_name` (str): Experiment name for logging

**Arguments (analyze_tta_weights.py):**
- `--experiment_prefix` (str): Prefix of experiment names to analyze
- `--baseline_weight` (float, default=0.50): Baseline configuration
- `--output_dir` (str): Directory for analysis outputs

**Output:**
- `experiments/{exp_name}_tta_weight_{XX}/results.json` - Results per weight
- `experiments/{exp_name}/checkpoints/model_*.pt` - Saved model checkpoints
- `experiments/{exp_name}/comparison_table.txt` - Statistical comparison
- `experiments/{exp_name}/tta_weight_analysis.png` - Visualization
- `experiments/{exp_name}/analysis_summary.txt` - Executive summary

**Example:**
```bash
# Train 25 models and test TTA weights
python sweep_tta_weights.py --n_models 25

# Re-run evaluation with existing checkpoints
python sweep_tta_weights.py --skip_training

# Analyze results
python analyze_tta_weights.py
```

---

## Standard Function Signatures

### main() Function

**Type 1 (Single-Run):**
```python
def main(run: int) -> tuple[float, float]:
    """
    Args:
        run: Run number/ID
    Returns:
        (accuracy, time_seconds)
    """
```

**Type 2 (Training Sweep):**
```python
def main(run: int, config_param: float) -> tuple[float, float]:
    """
    Args:
        run: Run number/ID
        config_param: Configuration parameter being swept
    Returns:
        (accuracy, time_seconds)
    """
```

**Type 3 (Post-Training Sweep):**
- Uses separate training and evaluation functions
- Training: Standard airbench94 training loop
- Evaluation: Custom evaluation with configurable parameters

---

## File Naming Conventions

### Experiment Branches
- Format: `experiment/expXXX-description`
- XXX = 3-digit number (000-999)
- Description: kebab-case description

### Experiment Results
- Format: `experiments/{exp_name}/`
- Required files:
  - `results.json` - Raw run data
  - `summary.json` - Statistics (mean, std, CI)
  - `hyperparameters.json` - Full hyperparameter record

### Checkpoint Files (Type 3 only)
- Format: `experiments/{exp_name}/checkpoints/model_{ID:03d}.pt`
- ID: Zero-padded 3-digit model number

---

## Logging Standard

All experiments must use `ExperimentLogger` from `experiment_logger.py`:

```python
from experiment_logger import ExperimentLogger

logger = ExperimentLogger(
    experiment_name="exp_name",
    experiment_description="Human-readable description",
    hyperparameters=hyp  # Full hyperparameter dict
)

# Log each run
logger.log_run(
    run_id=run,
    accuracy=acc,
    time_seconds=time_sec,
    epochs_completed=hyp['opt']['train_epochs']
)

# Save results
logger.save_summary()
logger.print_summary()
```

---

## Comparison and Analysis

### Compare Multiple Experiments
```python
from experiment_logger import ExperimentAggregator

aggregator = ExperimentAggregator()
aggregator.aggregate_experiments(['exp1', 'exp2', 'exp3'])
```

### Standard Output Format
```
Configuration    | Runs | Mean Acc | Std    | 95% CI
-----------------|------|----------|--------|------------------
config_1         | 25   | 0.9401   | 0.0012 | [0.9396, 0.9406]
config_2         | 25   | 0.9405   | 0.0011 | [0.9401, 0.9409]
```

---

## Quick Reference

| Experiment Type | Branch Examples | Command | Config Method |
|----------------|----------------|---------|---------------|
| Single-Run | exp000, exp100 | `python airbench94.py --exp_name X --runs 25` | Fixed in code |
| Training Sweep | exp210, exp220 | `python airbench94.py --sweep_name X --runs_per_config 10` | Hardcoded list |
| Post-Training Sweep | exp410 | `python sweep_tta_weights.py --n_models 25` | Separate script |

---

## Migration Guide

To standardize an existing experiment:

1. **Identify the experiment type** (Single-Run, Training Sweep, or Post-Training Sweep)
2. **Use the corresponding argument pattern** from this document
3. **Ensure main() matches the standard signature**
4. **Use ExperimentLogger** for all result tracking
5. **Update README** with the standard execution command
6. **Test** with `--runs 1` or `--runs_per_config 1` for quick validation

---

## Notes

- All experiments must be reproducible with the documented commands
- Checkpoint files should be gitignored (tracked via .gitignore)
- Results JSON files should be committed to track experiment history
- Use descriptive experiment names (not just numbers)
