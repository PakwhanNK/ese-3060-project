#!/usr/bin/env python3
"""
Simple TTA weight comparison - works even if analyze_tta_weights.py fails.
"""

import json
from pathlib import Path


def compare_tta_weights():
    """Compare all TTA weight configurations."""
    exp_base = Path('experiments')

    # Find all TTA weight experiment directories
    tta_dirs = sorted(exp_base.glob('exp410-tta-weight-optimization_tta_weight_*'))

    if not tta_dirs:
        print("No TTA weight experiments found.")
        print("Run: python sweep_tta_weights.py --n_models 25")
        return

    print("\n" + "="*100)
    print("TTA WEIGHT COMPARISON")
    print("="*100)
    print(f"Found {len(tta_dirs)} weight configurations\n")

    results = []

    for exp_dir in tta_dirs:
        # Extract weight from directory name
        weight_str = exp_dir.name.split('_')[-1]
        try:
            weight = int(weight_str) / 100.0
        except:
            continue

        # Load summary
        summary_file = exp_dir / 'summary.json'
        if not summary_file.exists():
            continue

        with open(summary_file, 'r') as f:
            summary = json.load(f)

        stats = summary.get('statistics', {})
        mean_acc = stats.get('accuracy_mean', stats.get('mean_accuracy'))
        std_acc = stats.get('accuracy_std', stats.get('std_accuracy'))
        num_runs = stats.get('num_runs', 0)

        results.append({
            'weight': weight,
            'mean': mean_acc,
            'std': std_acc,
            'runs': num_runs
        })

    # Sort by weight
    results.sort(key=lambda x: x['weight'])

    # Print table
    print(f"{'Weight':<10} {'Mean Acc':<12} {'Std Acc':<12} {'Runs':<8} {'vs 0.50':<12}")
    print("-"*60)

    baseline_mean = None
    for r in results:
        if r['weight'] == 0.50:
            baseline_mean = r['mean']
            break

    for r in results:
        diff_str = ""
        if baseline_mean is not None and r['weight'] != 0.50:
            diff = r['mean'] - baseline_mean
            diff_str = f"{diff:+.4f}"
        elif r['weight'] == 0.50:
            diff_str = "BASELINE"

        print(f"{r['weight']:<10.2f} {r['mean']:<12.4f} {r['std']:<12.4f} {r['runs']:<8} {diff_str:<12}")

    # Find best
    if results:
        best = max(results, key=lambda x: x['mean'])
        print("\n" + "="*100)
        print(f"Best Weight: {best['weight']:.2f} - Mean Accuracy: {best['mean']:.4f}")
        if baseline_mean:
            improvement = best['mean'] - baseline_mean
            print(f"Improvement over baseline (0.50): {improvement:+.4f} ({improvement*100:+.2f}%)")
        print("="*100 + "\n")

    # Export to CSV
    csv_path = exp_base / 'tta_weight_comparison.csv'
    with open(csv_path, 'w') as f:
        f.write("weight,mean_accuracy,std_accuracy,num_runs,diff_from_baseline\n")
        for r in results:
            diff = r['mean'] - baseline_mean if baseline_mean else 0
            f.write(f"{r['weight']:.2f},{r['mean']:.6f},{r['std']:.6f},{r['runs']},{diff:.6f}\n")

    print(f"✓ Saved to {csv_path}\n")


if __name__ == "__main__":
    compare_tta_weights()
