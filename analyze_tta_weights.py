"""
TTA Weight Analysis Script

Analyzes results from the TTA weight optimization sweep and performs statistical tests
to determine if any weight configuration significantly improves accuracy.

Outputs:
- Comparison table with mean, std, 95% CI for each weight config
- Paired t-tests comparing each config to baseline (0.50)
- Effect sizes (Cohen's d)
- Visualization plots
- Summary report
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt


def load_experiment_results(experiment_name):
    """Load results from an experiment directory."""
    exp_dir = Path('experiments') / experiment_name
    results_file = exp_dir / 'results.json'

    if not results_file.exists():
        return None

    with open(results_file, 'r') as f:
        data = json.load(f)

    # Extract accuracies from runs
    accuracies = [run['accuracy'] for run in data['runs']]
    return np.array(accuracies)


def compute_statistics(accuracies):
    """Compute mean, std, 95% CI for a set of accuracies."""
    mean = np.mean(accuracies)
    std = np.std(accuracies, ddof=1)  # Sample standard deviation
    n = len(accuracies)

    # 95% confidence interval
    se = std / np.sqrt(n)
    ci_margin = stats.t.ppf(0.975, n-1) * se
    ci_lower = mean - ci_margin
    ci_upper = mean + ci_margin

    return {
        'mean': mean,
        'std': std,
        'n': n,
        'se': se,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'ci_margin': ci_margin
    }


def paired_t_test(baseline_accs, test_accs):
    """
    Perform paired t-test between baseline and test accuracies.

    Returns:
        dict with t-statistic, p-value, and Cohen's d effect size
    """
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(test_accs, baseline_accs)

    # Cohen's d for paired samples
    differences = test_accs - baseline_accs
    cohens_d = np.mean(differences) / np.std(differences, ddof=1)

    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'mean_difference': np.mean(differences),
        'significant_05': p_value < 0.05,
        'significant_01': p_value < 0.01
    }


def format_pvalue(p):
    """Format p-value for display."""
    if p < 0.001:
        return "< 0.001"
    elif p < 0.01:
        return f"{p:.3f}"
    else:
        return f"{p:.4f}"


def create_comparison_table(results_dict, baseline_weight=0.50):
    """Create formatted comparison table."""
    # Sort by weight
    weights = sorted(results_dict.keys())

    # Build table
    lines = []
    lines.append("="*120)
    lines.append("TTA WEIGHT OPTIMIZATION RESULTS")
    lines.append("="*120)
    lines.append("")

    # Header
    header = f"{'Weight':<10} {'Mean':<10} {'Std':<10} {'95% CI':<25} {'vs Baseline':<15} {'p-value':<12} {'Sig':<5} {'Effect (d)':<12}"
    lines.append(header)
    lines.append("-"*120)

    baseline_accs = results_dict[baseline_weight]['accuracies']

    for weight in weights:
        data = results_dict[weight]
        stats_data = data['statistics']
        test_result = data.get('test_result', None)

        # Format basic statistics
        mean_str = f"{stats_data['mean']:.4f}"
        std_str = f"{stats_data['std']:.4f}"
        ci_str = f"[{stats_data['ci_lower']:.4f}, {stats_data['ci_upper']:.4f}]"

        if weight == baseline_weight:
            # Baseline row
            row = f"{weight:<10.2f} {mean_str:<10} {std_str:<10} {ci_str:<25} {'BASELINE':<15} {'-':<12} {'-':<5} {'-':<12}"
        else:
            # Test row
            diff = test_result['mean_difference']
            diff_str = f"{diff:+.4f}" if diff >= 0 else f"{diff:.4f}"

            p_str = format_pvalue(test_result['p_value'])

            # Significance markers
            if test_result['significant_01']:
                sig = "**"
            elif test_result['significant_05']:
                sig = "*"
            else:
                sig = ""

            effect_str = f"{test_result['cohens_d']:.4f}"

            row = f"{weight:<10.2f} {mean_str:<10} {std_str:<10} {ci_str:<25} {diff_str:<15} {p_str:<12} {sig:<5} {effect_str:<12}"

        lines.append(row)

    lines.append("-"*120)
    lines.append("")
    lines.append("Legend:")
    lines.append("  * = Significant at p < 0.05")
    lines.append("  ** = Significant at p < 0.01")
    lines.append("  Effect size (Cohen's d): small=0.2, medium=0.5, large=0.8")
    lines.append("="*120)
    lines.append("")

    return "\n".join(lines)


def create_visualization(results_dict, output_dir):
    """Create visualization plots."""
    weights = sorted(results_dict.keys())
    means = [results_dict[w]['statistics']['mean'] for w in weights]
    stds = [results_dict[w]['statistics']['std'] for w in weights]
    cis = [results_dict[w]['statistics']['ci_margin'] for w in weights]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Mean accuracy with error bars
    ax1.errorbar(weights, means, yerr=cis, fmt='o-', capsize=5, capthick=2, markersize=8)
    ax1.axhline(y=results_dict[0.50]['statistics']['mean'], color='r', linestyle='--',
                label='Baseline (0.50)', alpha=0.7)
    ax1.set_xlabel('TTA Untranslated Weight', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('TTA Weight vs Accuracy (with 95% CI)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Difference from baseline
    baseline_mean = results_dict[0.50]['statistics']['mean']
    differences = [results_dict[w]['statistics']['mean'] - baseline_mean for w in weights]

    colors = ['red' if w == 0.50 else 'blue' for w in weights]
    ax2.bar(range(len(weights)), differences, color=colors, alpha=0.6)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.set_xlabel('TTA Untranslated Weight', fontsize=12)
    ax2.set_ylabel('Accuracy Difference from Baseline', fontsize=12)
    ax2.set_title('Improvement over Baseline (0.50)', fontsize=14)
    ax2.set_xticks(range(len(weights)))
    ax2.set_xticklabels([f"{w:.2f}" for w in weights], rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save figure
    output_path = output_dir / 'tta_weight_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization to {output_path}")

    plt.close()


def generate_summary_report(results_dict, output_dir, baseline_weight=0.50):
    """Generate detailed summary report."""
    lines = []

    # Find best configuration
    weights = sorted(results_dict.keys())
    best_weight = max(weights, key=lambda w: results_dict[w]['statistics']['mean'])
    best_mean = results_dict[best_weight]['statistics']['mean']

    baseline_mean = results_dict[baseline_weight]['statistics']['mean']
    improvement = best_mean - baseline_mean

    lines.append("="*80)
    lines.append("TTA WEIGHT OPTIMIZATION - SUMMARY REPORT")
    lines.append("="*80)
    lines.append("")

    lines.append("EXPERIMENT OVERVIEW")
    lines.append("-"*80)
    lines.append(f"  Number of weight configurations tested: {len(weights)}")
    lines.append(f"  Sample size per configuration: {results_dict[baseline_weight]['statistics']['n']}")
    lines.append(f"  Baseline weight: {baseline_weight:.2f}")
    lines.append("")

    lines.append("KEY FINDINGS")
    lines.append("-"*80)
    lines.append(f"  Best performing weight: {best_weight:.2f}")
    lines.append(f"  Best mean accuracy: {best_mean:.4f}")
    lines.append(f"  Baseline mean accuracy: {baseline_mean:.4f}")
    lines.append(f"  Improvement: {improvement:+.4f} ({improvement*100:+.2f}%)")
    lines.append("")

    # Statistical significance
    if best_weight != baseline_weight:
        test_result = results_dict[best_weight]['test_result']
        lines.append("STATISTICAL SIGNIFICANCE (Best vs Baseline)")
        lines.append("-"*80)
        lines.append(f"  t-statistic: {test_result['t_statistic']:.4f}")
        lines.append(f"  p-value: {format_pvalue(test_result['p_value'])}")
        lines.append(f"  Effect size (Cohen's d): {test_result['cohens_d']:.4f}")

        if test_result['significant_01']:
            lines.append(f"  Result: HIGHLY SIGNIFICANT (p < 0.01) **")
        elif test_result['significant_05']:
            lines.append(f"  Result: SIGNIFICANT (p < 0.05) *")
        else:
            lines.append(f"  Result: Not statistically significant")
        lines.append("")

    # Recommendations
    lines.append("RECOMMENDATIONS")
    lines.append("-"*80)

    if best_weight != baseline_weight and results_dict[best_weight].get('test_result', {}).get('significant_05', False):
        lines.append(f"  ✓ Consider using untrans_weight={best_weight:.2f} instead of {baseline_weight:.2f}")
        lines.append(f"  ✓ Expected improvement: {improvement*100:+.2f}% accuracy")
    else:
        lines.append(f"  • No significant improvement found over baseline ({baseline_weight:.2f})")
        lines.append(f"  • Current default weight appears optimal")

    lines.append("")
    lines.append("="*80)

    # Save report
    report_text = "\n".join(lines)
    output_path = output_dir / 'analysis_summary.txt'
    with open(output_path, 'w') as f:
        f.write(report_text)

    print(report_text)
    print(f"\n✓ Saved summary report to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze TTA weight optimization results'
    )
    parser.add_argument('--experiment_prefix', type=str, default='exp410-tta-weight-optimization',
                        help='Prefix of experiment names')
    parser.add_argument('--baseline_weight', type=float, default=0.50,
                        help='Baseline weight to compare against')
    parser.add_argument('--output_dir', type=str, default='experiments/exp410-tta-weight-optimization',
                        help='Directory to save analysis outputs')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("TTA WEIGHT OPTIMIZATION - STATISTICAL ANALYSIS")
    print("="*80 + "\n")

    # Find all experiment directories matching the prefix
    exp_base_dir = Path('experiments')
    exp_dirs = sorted(exp_base_dir.glob(f"{args.experiment_prefix}_tta_weight_*"))

    if len(exp_dirs) == 0:
        print(f"ERROR: No experiment directories found matching pattern:")
        print(f"  {exp_base_dir}/{args.experiment_prefix}_tta_weight_*")
        print("\nPlease run sweep_tta_weights.py first.")
        sys.exit(1)

    print(f"Found {len(exp_dirs)} experiment directories")

    # Load results for each weight configuration
    results_dict = {}

    for exp_dir in exp_dirs:
        # Extract weight from directory name
        # Format: exp410-tta-weight-optimization_tta_weight_XX
        weight_str = exp_dir.name.split('_')[-1]
        try:
            weight = int(weight_str) / 100.0
        except ValueError:
            print(f"Warning: Could not parse weight from {exp_dir.name}, skipping")
            continue

        # Load accuracies
        accuracies = load_experiment_results(exp_dir.name)
        if accuracies is None:
            print(f"Warning: Could not load results from {exp_dir.name}, skipping")
            continue

        # Compute statistics
        stats_data = compute_statistics(accuracies)

        results_dict[weight] = {
            'accuracies': accuracies,
            'statistics': stats_data
        }

        print(f"  Loaded weight={weight:.2f}: {len(accuracies)} runs, mean={stats_data['mean']:.4f}")

    if args.baseline_weight not in results_dict:
        print(f"\nERROR: Baseline weight {args.baseline_weight} not found in results!")
        print(f"Available weights: {sorted(results_dict.keys())}")
        sys.exit(1)

    # Perform paired t-tests against baseline
    baseline_accs = results_dict[args.baseline_weight]['accuracies']

    for weight in results_dict.keys():
        if weight != args.baseline_weight:
            test_accs = results_dict[weight]['accuracies']
            test_result = paired_t_test(baseline_accs, test_accs)
            results_dict[weight]['test_result'] = test_result

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate comparison table
    print("\n")
    table = create_comparison_table(results_dict, baseline_weight=args.baseline_weight)
    print(table)

    # Save table to file
    table_path = output_dir / 'comparison_table.txt'
    with open(table_path, 'w') as f:
        f.write(table)
    print(f"✓ Saved comparison table to {table_path}")

    # Create visualization
    print("\nGenerating visualization...")
    create_visualization(results_dict, output_dir)

    # Generate summary report
    print("\nGenerating summary report...")
    generate_summary_report(results_dict, output_dir, baseline_weight=args.baseline_weight)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nOutputs saved to: {output_dir}/")
    print("  - comparison_table.txt")
    print("  - tta_weight_analysis.png")
    print("  - analysis_summary.txt")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
