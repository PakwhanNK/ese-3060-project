"""
Run LR Schedule Time Optimization Experiments

Runs three LR schedule configurations to find the fastest training time
while maintaining >=94% accuracy.

Experiments:
1. exp001-lr-baseline: start=0.20, peak=0.23, end=0.07 (current best)
2. exp002-lr-end05-peak18: start=0.15, peak=0.18, end=0.05
3. exp003-lr-end05-peak27: start=0.15, peak=0.27, end=0.05

Success metric: Minimize best_time_94plus (target: <4.05s vs baseline ~4.06s)
"""

import subprocess
import json
import os
from pathlib import Path
import pandas as pd


# Experiment configurations
EXPERIMENTS = [
    {
        'name': 'exp001-lr-baseline',
        'description': 'Baseline LR schedule: start=0.20, peak=0.23, end=0.07',
        'lr_params': {'start': 0.20, 'peak': 0.23, 'end': 0.07}
    },
    {
        'name': 'exp002-lr-end05-peak18',
        'description': 'Lower end/peak: start=0.15, peak=0.18, end=0.05',
        'lr_params': {'start': 0.15, 'peak': 0.18, 'end': 0.05}
    },
    {
        'name': 'exp003-lr-end05-peak27',
        'description': 'Lower end, higher peak: start=0.15, peak=0.27, end=0.05',
        'lr_params': {'start': 0.15, 'peak': 0.27, 'end': 0.05}
    }
]

RUNS_PER_EXPERIMENT = 25


def modify_lr_schedule(start, peak, end):
    """Modify the LR schedule in airbench94.py"""
    import fileinput

    # Read the file
    with open('airbench94.py', 'r') as f:
        lines = f.readlines()

    # Find and replace the lr_schedule line
    for i, line in enumerate(lines):
        if 'lr_schedule = triangle(total_train_steps' in line:
            lines[i] = f'    lr_schedule = triangle(total_train_steps, start={start}, end={end}, peak={peak})\n'
            break

    # Write back
    with open('airbench94.py', 'w') as f:
        f.writelines(lines)

    print(f"✓ Modified LR schedule: start={start}, peak={peak}, end={end}")


def run_experiment(exp_config):
    """Run a single experiment"""
    name = exp_config['name']
    description = exp_config['description']
    lr_params = exp_config['lr_params']

    print(f"\n{'='*80}")
    print(f"RUNNING: {name}")
    print(f"Description: {description}")
    print(f"LR Params: {lr_params}")
    print(f"{'='*80}\n")

    # Modify LR schedule
    modify_lr_schedule(**lr_params)

    # Run experiment
    cmd = [
        'python', 'airbench94.py',
        '--exp_name', name,
        '--desc', description,
        '--runs', str(RUNS_PER_EXPERIMENT)
    ]

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"ERROR: Experiment {name} failed!")
        return False

    print(f"\n✓ Completed {name}\n")
    return True


def load_experiment_results(exp_name):
    """Load results from an experiment"""
    results_file = Path('experiments') / exp_name / 'results.json'

    if not results_file.exists():
        print(f"WARNING: Results file not found for {exp_name}")
        return None

    with open(results_file, 'r') as f:
        data = json.load(f)

    return data


def calculate_metrics(exp_name, results_data):
    """Calculate time metrics for an experiment"""
    if results_data is None:
        return None

    runs = results_data['runs']

    # Filter runs with accuracy >= 94%
    successful_runs = [r for r in runs if r['accuracy'] >= 0.94]
    all_times = [r['time_seconds'] for r in runs]
    successful_times = [r['time_seconds'] for r in successful_runs]
    accuracies = [r['accuracy'] for r in runs]

    if len(successful_times) == 0:
        best_time_94plus = None
        mean_time_94plus = None
    else:
        best_time_94plus = min(successful_times)
        mean_time_94plus = sum(successful_times) / len(successful_times)

    metrics = {
        'experiment_name': exp_name,
        'runs_total': len(runs),
        'runs_94plus': len(successful_runs),
        'success_rate': len(successful_runs) / len(runs) if len(runs) > 0 else 0,
        'best_time_94plus': best_time_94plus,
        'mean_time_94plus': mean_time_94plus,
        'best_time_overall': min(all_times) if all_times else None,
        'mean_time_overall': sum(all_times) / len(all_times) if all_times else None,
        'accuracy_mean': sum(accuracies) / len(accuracies) if accuracies else None,
        'accuracy_min': min(accuracies) if accuracies else None,
        'accuracy_max': max(accuracies) if accuracies else None
    }

    return metrics


def create_comparison_csv(experiment_names):
    """Create comparison CSV and calculate improvements"""

    print(f"\n{'='*80}")
    print("CREATING COMPARISON ANALYSIS")
    print(f"{'='*80}\n")

    # Load results for all experiments
    all_metrics = []

    for exp_name in experiment_names:
        print(f"Loading results for {exp_name}...")
        results = load_experiment_results(exp_name)
        metrics = calculate_metrics(exp_name, results)

        if metrics:
            all_metrics.append(metrics)
            print(f"  ✓ {metrics['runs_94plus']}/{metrics['runs_total']} runs >= 94%")
            if metrics['best_time_94plus']:
                print(f"  ✓ Best time (>=94%): {metrics['best_time_94plus']:.4f}s")

    if len(all_metrics) == 0:
        print("ERROR: No experiment results found!")
        return

    # Create DataFrame
    df = pd.DataFrame(all_metrics)

    # Calculate improvements vs baseline
    baseline = df[df['experiment_name'] == 'exp001-lr-baseline'].iloc[0]
    baseline_time = baseline['best_time_94plus']

    df['time_improvement_pct'] = ((baseline_time - df['best_time_94plus']) / baseline_time * 100).round(2)
    df['time_improvement_abs'] = (baseline_time - df['best_time_94plus']).round(4)

    # Reorder columns for readability
    columns_ordered = [
        'experiment_name',
        'runs_94plus',
        'success_rate',
        'best_time_94plus',
        'mean_time_94plus',
        'time_improvement_abs',
        'time_improvement_pct',
        'accuracy_mean',
        'accuracy_min',
        'accuracy_max'
    ]

    df = df[columns_ordered]

    # Save to CSV
    output_file = Path('experiments') / 'lr_time_comparison.csv'
    df.to_csv(output_file, index=False)

    print(f"\n{'='*80}")
    print("COMPARISON RESULTS")
    print(f"{'='*80}\n")
    print(df.to_string(index=False))
    print(f"\n✓ Saved to: {output_file}")

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Baseline (exp001): {baseline_time:.4f}s")

    best_exp = df.loc[df['best_time_94plus'].idxmin()]
    print(f"\nFastest: {best_exp['experiment_name']}")
    print(f"  Time: {best_exp['best_time_94plus']:.4f}s")
    print(f"  Improvement: {best_exp['time_improvement_abs']:.4f}s ({best_exp['time_improvement_pct']:.2f}%)")
    print(f"  Success rate: {best_exp['success_rate']*100:.1f}%")
    print(f"  Mean accuracy: {best_exp['accuracy_mean']:.4f}")
    print(f"{'='*80}\n")

    return df


def main():
    print(f"\n{'='*80}")
    print("LR SCHEDULE TIME OPTIMIZATION EXPERIMENTS")
    print(f"{'='*80}")
    print(f"Running {len(EXPERIMENTS)} experiments with {RUNS_PER_EXPERIMENT} runs each")
    print(f"Total training runs: {len(EXPERIMENTS) * RUNS_PER_EXPERIMENT}")
    print(f"{'='*80}\n")

    # Run all experiments
    completed_experiments = []

    for i, exp_config in enumerate(EXPERIMENTS, 1):
        print(f"\n[{i}/{len(EXPERIMENTS)}] Starting {exp_config['name']}...")

        success = run_experiment(exp_config)

        if success:
            completed_experiments.append(exp_config['name'])
        else:
            print(f"Skipping {exp_config['name']} due to errors")

    # Create comparison
    if len(completed_experiments) > 0:
        print(f"\n{'='*80}")
        print(f"Completed {len(completed_experiments)}/{len(EXPERIMENTS)} experiments")
        print(f"{'='*80}")

        create_comparison_csv(completed_experiments)
    else:
        print("\nERROR: No experiments completed successfully!")

    print("\n✓ ALL EXPERIMENTS COMPLETE!\n")


if __name__ == "__main__":
    main()
