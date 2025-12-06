"""
Compare LR Schedule Time Optimization Experiments

Analyzes and compares results from exp001, exp002, exp003.
Generates comparison CSV with time improvement metrics.

Usage:
    python compare_lr_experiments.py
"""

import json
from pathlib import Path
import pandas as pd
import sys


def load_experiment_results(exp_name):
    """Load results from an experiment"""
    results_file = Path('experiments') / exp_name / 'results.json'

    if not results_file.exists():
        print(f"WARNING: Results file not found: {results_file}")
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
        'success_rate': f"{len(successful_runs) / len(runs) * 100:.1f}%" if len(runs) > 0 else "0%",
        'success_rate_numeric': len(successful_runs) / len(runs) if len(runs) > 0 else 0,
        'best_time_94plus': best_time_94plus,
        'mean_time_94plus': mean_time_94plus,
        'best_time_overall': min(all_times) if all_times else None,
        'mean_time_overall': sum(all_times) / len(all_times) if all_times else None,
        'accuracy_mean': sum(accuracies) / len(accuracies) if accuracies else None,
        'accuracy_std': pd.Series(accuracies).std() if accuracies else None,
        'accuracy_min': min(accuracies) if accuracies else None,
        'accuracy_max': max(accuracies) if accuracies else None
    }

    return metrics


def main():
    print(f"\n{'='*80}")
    print("LR SCHEDULE TIME OPTIMIZATION - COMPARISON ANALYSIS")
    print(f"{'='*80}\n")

    # Experiments to compare
    experiment_names = [
        'exp001-lr-baseline',
        'exp002-lr-end05-peak18',
        'exp003-lr-end05-peak27'
    ]

    # Load results for all experiments
    all_metrics = []
    missing_experiments = []

    for exp_name in experiment_names:
        print(f"Loading {exp_name}...", end=" ")
        results = load_experiment_results(exp_name)
        metrics = calculate_metrics(exp_name, results)

        if metrics:
            all_metrics.append(metrics)
            print(f"✓ ({metrics['runs_94plus']}/{metrics['runs_total']} runs >= 94%)")
            if metrics['best_time_94plus']:
                print(f"    Best time: {metrics['best_time_94plus']:.4f}s, "
                      f"Mean acc: {metrics['accuracy_mean']:.4f}")
        else:
            missing_experiments.append(exp_name)
            print(f"✗ NOT FOUND")

    if len(all_metrics) == 0:
        print("\nERROR: No experiment results found!")
        print("Please run experiments first:")
        print("  python run_lr_time_experiments.py")
        print("\nOr run manually:")
        print("  python airbench94.py --exp_name exp001-lr-baseline --runs 25")
        sys.exit(1)

    # Create DataFrame
    df = pd.DataFrame(all_metrics)

    # Calculate improvements vs baseline
    baseline_row = df[df['experiment_name'] == 'exp001-lr-baseline']

    if len(baseline_row) == 0:
        print("\nWARNING: Baseline experiment (exp001-lr-baseline) not found!")
        print("Using first experiment as baseline...")
        baseline = df.iloc[0]
    else:
        baseline = baseline_row.iloc[0]

    baseline_time = baseline['best_time_94plus']

    if baseline_time is not None:
        df['time_improvement_abs'] = (baseline_time - df['best_time_94plus']).round(4)
        df['time_improvement_pct'] = ((baseline_time - df['best_time_94plus']) / baseline_time * 100).round(2)
    else:
        df['time_improvement_abs'] = None
        df['time_improvement_pct'] = None

    # Reorder and format columns for display
    display_df = df[[
        'experiment_name',
        'runs_94plus',
        'success_rate',
        'best_time_94plus',
        'mean_time_94plus',
        'time_improvement_abs',
        'time_improvement_pct',
        'accuracy_mean',
        'accuracy_std'
    ]].copy()

    # Format numeric columns
    for col in ['best_time_94plus', 'mean_time_94plus', 'time_improvement_abs', 'accuracy_mean', 'accuracy_std']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")

    for col in ['time_improvement_pct']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")

    # Save detailed CSV with all metrics
    output_file = Path('experiments') / 'lr_time_comparison.csv'
    df.to_csv(output_file, index=False)

    # Print results
    print(f"\n{'='*80}")
    print("COMPARISON TABLE")
    print(f"{'='*80}\n")
    print(display_df.to_string(index=False))

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    if baseline_time:
        print(f"Baseline (exp001-lr-baseline): {baseline_time:.4f}s")

        # Find fastest experiment
        best_idx = df['best_time_94plus'].idxmin()
        best_exp = df.loc[best_idx]

        print(f"\n🏆 FASTEST EXPERIMENT: {best_exp['experiment_name']}")
        print(f"  Best time (>=94%): {best_exp['best_time_94plus']:.4f}s")
        print(f"  Time improvement: {best_exp['time_improvement_abs']:.4f}s ({best_exp['time_improvement_pct']:.2f}%)")
        print(f"  Success rate: {best_exp['success_rate']}")
        print(f"  Mean accuracy: {best_exp['accuracy_mean']:.4f} ± {best_exp['accuracy_std']:.4f}")

        if best_exp['best_time_94plus'] < baseline_time:
            print(f"\n✓ SUCCESS: Found faster configuration!")
        elif best_exp['best_time_94plus'] == baseline_time:
            print(f"\n→ NEUTRAL: No improvement over baseline")
        else:
            print(f"\n✗ SLOWER: Baseline remains fastest")
    else:
        print("WARNING: Could not calculate improvements (baseline has no successful runs)")

    print(f"\n{'='*80}")
    print(f"✓ Full results saved to: {output_file}")
    print(f"{'='*80}\n")

    if missing_experiments:
        print(f"NOTE: Missing results for: {', '.join(missing_experiments)}")
        print(f"Run these experiments to complete the comparison.\n")


if __name__ == "__main__":
    main()
