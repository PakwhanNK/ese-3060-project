#!/usr/bin/env python3
"""
Simple lookahead parameter comparison - for exp310 results.
"""

import json
from pathlib import Path


def compare_lookahead_configs():
    """Compare all lookahead configurations from exp310."""
    exp_base = Path('experiments')

    # Expected lookahead config names from exp310
    config_names = [
        'no_lookahead',
        'original_k5',
        'k3', 'k7', 'k10',
        'k5_constant', 'k5_linear',
        'k5_alpha_high', 'k5_alpha_low'
    ]

    # Find exp310 experiment directories
    lookahead_dirs = []
    for config in config_names:
        pattern = f'exp310-lookahead-param-sweep*{config}*'
        dirs = list(exp_base.glob(pattern))
        if dirs:
            lookahead_dirs.extend(dirs)

    # Also try pattern matching
    if not lookahead_dirs:
        lookahead_dirs = list(exp_base.glob('exp310-*'))

    if not lookahead_dirs:
        print("No lookahead experiments found.")
        print("Run: python airbench94.py --exp_name exp310-lookahead-param-sweep --desc 'Lookahead sweep' --runs 25 --sweep")
        return

    print("\n" + "="*120)
    print("LOOKAHEAD PARAMETER COMPARISON")
    print("="*120)
    print(f"Found {len(lookahead_dirs)} configurations\n")

    results = []

    for exp_dir in lookahead_dirs:
        # Load summary
        summary_file = exp_dir / 'summary.json'
        if not summary_file.exists():
            continue

        with open(summary_file, 'r') as f:
            summary = json.load(f)

        # Extract config name from experiment name or description
        config_name = exp_dir.name.replace('exp310-lookahead-param-sweep-', '')

        stats = summary.get('statistics', {})
        mean_acc = stats.get('accuracy_mean', stats.get('mean_accuracy'))
        std_acc = stats.get('accuracy_std', stats.get('std_accuracy'))
        mean_time = stats.get('time_mean', stats.get('mean_time'))
        std_time = stats.get('time_std', stats.get('std_time'))
        num_runs = stats.get('num_runs', 0)

        results.append({
            'config': config_name,
            'mean_acc': mean_acc,
            'std_acc': std_acc,
            'mean_time': mean_time,
            'std_time': std_time,
            'runs': num_runs
        })

    # Sort by mean accuracy
    results.sort(key=lambda x: x['mean_acc'] if x['mean_acc'] else 0, reverse=True)

    # Print table
    print(f"{'Config':<20} {'Mean Acc':<12} {'Std Acc':<12} {'Mean Time':<12} {'Runs':<8}")
    print("-"*70)

    for r in results:
        print(f"{r['config']:<20} {r['mean_acc']:<12.4f} {r['std_acc']:<12.4f} "
              f"{r['mean_time']:<12.2f} {r['runs']:<8}")

    # Find best
    if results:
        best = results[0]  # Already sorted
        print("\n" + "="*120)
        print(f"Best Configuration: {best['config']}")
        print(f"  Accuracy: {best['mean_acc']:.4f} ± {best['std_acc']:.4f}")
        print(f"  Time: {best['mean_time']:.2f}s ± {best['std_time']:.2f}s")
        print("="*120 + "\n")

    # Export to CSV
    csv_path = exp_base / 'lookahead_comparison.csv'
    with open(csv_path, 'w') as f:
        f.write("config,mean_accuracy,std_accuracy,mean_time,std_time,num_runs\n")
        for r in results:
            f.write(f"{r['config']},{r['mean_acc']:.6f},{r['std_acc']:.6f},"
                   f"{r['mean_time']:.4f},{r['std_time']:.4f},{r['runs']}\n")

    print(f"✓ Saved to {csv_path}\n")


if __name__ == "__main__":
    compare_lookahead_configs()
