#!/usr/bin/env python3
"""
Simple script to compare experiment results across different configurations.
Works with the ExperimentLogger output format.
"""

import json
import sys
from pathlib import Path
import pandas as pd


def load_experiment_summary(exp_name):
    """Load summary.json from an experiment folder."""
    exp_dir = Path('experiments') / exp_name
    summary_file = exp_dir / 'summary.json'

    if not summary_file.exists():
        return None

    with open(summary_file, 'r') as f:
        return json.load(f)


def compare_all_experiments():
    """Find and compare all experiments in the experiments/ directory."""
    exp_base = Path('experiments')

    if not exp_base.exists():
        print("No experiments/ directory found")
        return

    # Find all experiment directories (those with summary.json)
    experiments = []
    for exp_dir in sorted(exp_base.iterdir()):
        if exp_dir.is_dir():
            summary = load_experiment_summary(exp_dir.name)
            if summary:
                experiments.append({
                    'name': exp_dir.name,
                    'summary': summary
                })

    if not experiments:
        print("No completed experiments found (no summary.json files)")
        return

    print("\n" + "="*120)
    print("EXPERIMENT COMPARISON")
    print("="*120)
    print(f"Found {len(experiments)} completed experiments\n")

    # Build comparison table
    rows = []
    for exp in experiments:
        name = exp['name']
        stats = exp['summary'].get('statistics', {})

        # Handle both old and new key names
        mean_acc = stats.get('accuracy_mean', stats.get('mean_accuracy', 'N/A'))
        std_acc = stats.get('accuracy_std', stats.get('std_accuracy', 'N/A'))
        mean_time = stats.get('time_mean', stats.get('mean_time', 'N/A'))
        std_time = stats.get('time_std', stats.get('std_time', 'N/A'))
        num_runs = stats.get('num_runs', 'N/A')

        # Get confidence intervals if available
        ci_lower = stats.get('accuracy_ci_95_lower', 'N/A')
        ci_upper = stats.get('accuracy_ci_95_upper', 'N/A')

        rows.append({
            'Experiment': name,
            'Runs': num_runs,
            'Mean Acc': mean_acc,
            'Std Acc': std_acc,
            '95% CI Lower': ci_lower,
            '95% CI Upper': ci_upper,
            'Mean Time (s)': mean_time,
            'Std Time (s)': std_time
        })

    # Create DataFrame for nice formatting
    df = pd.DataFrame(rows)

    # Format numeric columns
    for col in ['Mean Acc', 'Std Acc', '95% CI Lower', '95% CI Upper', 'Mean Time (s)', 'Std Time (s)']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)

    print(df.to_string(index=False))
    print("\n" + "="*120)

    # Find best accuracy
    valid_exps = [e for e in experiments if isinstance(e['summary'].get('statistics', {}).get('accuracy_mean',
                                                                                               e['summary'].get('statistics', {}).get('mean_accuracy')),
                                                       (int, float))]
    if valid_exps:
        best = max(valid_exps, key=lambda x: x['summary']['statistics'].get('accuracy_mean',
                                                                            x['summary']['statistics'].get('mean_accuracy', 0)))
        best_acc = best['summary']['statistics'].get('accuracy_mean', best['summary']['statistics'].get('mean_accuracy'))
        print(f"\nBest Accuracy: {best['name']} - {best_acc:.4f}")

    # Save to CSV
    csv_path = Path('experiments/comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved comparison to {csv_path}")
    print("="*120 + "\n")


def compare_specific_experiments(exp_names):
    """Compare specific experiments by name."""
    print("\n" + "="*100)
    print(f"Comparing {len(exp_names)} experiments")
    print("="*100 + "\n")

    for name in exp_names:
        summary = load_experiment_summary(name)
        if not summary:
            print(f"⚠ {name}: No summary.json found")
            continue

        stats = summary.get('statistics', {})
        mean_acc = stats.get('accuracy_mean', stats.get('mean_accuracy', 'N/A'))
        std_acc = stats.get('accuracy_std', stats.get('std_accuracy', 'N/A'))

        print(f"{name}:")
        print(f"  Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"  Runs: {stats.get('num_runs', 'N/A')}")
        print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Compare specific experiments
        compare_specific_experiments(sys.argv[1:])
    else:
        # Compare all experiments
        compare_all_experiments()
