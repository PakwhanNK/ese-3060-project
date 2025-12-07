#!/usr/bin/env python3
"""
TTA Weight Time-Accuracy Tradeoff Analysis

Analyzes TTA weight sweep results to find the configuration that achieves
≥94% accuracy in minimum time.

KEY METRICS:
- Accuracy: Must be ≥94% (target threshold)
- Time: Training + Evaluation time (want to minimize)
- Success Rate: Percentage of runs achieving ≥94%

GOAL: Find TTA weight with best time while maintaining ≥94% accuracy
"""

import json
from pathlib import Path
import sys


def analyze_tta_time_tradeoff():
    """Analyze TTA weights to find best time-accuracy tradeoff."""
    exp_base = Path('experiments')

    # Find all TTA weight experiment directories
    tta_dirs = sorted(exp_base.glob('exp410-tta-weight-optimization_tta_weight_*'))

    if not tta_dirs:
        print("No TTA weight experiments found.")
        print("Run: python sweep_tta_weights_with_timing.py --n_models 25")
        return

    print("\n" + "="*120)
    print("TTA WEIGHT TIME-ACCURACY TRADEOFF ANALYSIS")
    print("="*120)
    print(f"Found {len(tta_dirs)} weight configurations")
    print(f"Target: ≥94% accuracy with minimum time\n")

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

        # Get accuracy stats
        mean_acc = stats.get('accuracy_mean', stats.get('mean_accuracy', 0))
        std_acc = stats.get('accuracy_std', stats.get('std_accuracy', 0))

        # Get time stats
        mean_time = stats.get('time_mean', stats.get('mean_time', 0))
        std_time = stats.get('time_std', stats.get('std_time', 0))

        # Get success rate (≥94%)
        success_rate = stats.get('success_rate', 0)
        num_successful = stats.get('num_successful', 0)
        num_runs = stats.get('num_runs', 0)

        results.append({
            'weight': weight,
            'mean_acc': mean_acc,
            'std_acc': std_acc,
            'mean_time': mean_time,
            'std_time': std_time,
            'success_rate': success_rate,
            'num_successful': num_successful,
            'num_runs': num_runs,
            'achieves_target': mean_acc >= 0.94
        })

    if not results:
        print("No valid results found!")
        return

    # Sort by weight
    results.sort(key=lambda x: x['weight'])

    # Find baseline (0.50)
    baseline = next((r for r in results if r['weight'] == 0.50), None)
    baseline_time = baseline['mean_time'] if baseline else None
    baseline_acc = baseline['mean_acc'] if baseline else None

    # Print detailed table
    print(f"{'Weight':<10} {'Mean Acc':<12} {'±Std':<10} {'Mean Time':<12} {'±Std':<10} "
          f"{'≥94% Rate':<12} {'Target?':<10} {'Time Δ':<15}")
    print("-"*120)

    for r in results:
        # Calculate differences from baseline
        if baseline_time:
            time_diff = r['mean_time'] - baseline_time
            time_diff_pct = (time_diff / baseline_time) * 100
            time_diff_str = f"{time_diff:+.2f}s ({time_diff_pct:+.1f}%)"
        else:
            time_diff_str = "-"

        # Mark if achieves target
        target_str = "✓ YES" if r['achieves_target'] else "✗ NO"

        print(f"{r['weight']:<10.2f} {r['mean_acc']:<12.4f} {r['std_acc']:<10.4f} "
              f"{r['mean_time']:<12.2f} {r['std_time']:<10.2f} "
              f"{r['success_rate']:<12.1%} {target_str:<10} {time_diff_str:<15}")

    # Find best configurations
    print("\n" + "="*120)
    print("RECOMMENDATIONS")
    print("="*120)

    # Filter configs that achieve ≥94% mean accuracy
    valid_configs = [r for r in results if r['achieves_target']]

    if not valid_configs:
        print("⚠️  WARNING: No configurations achieved ≥94% mean accuracy!")
        print("    Best accuracy achieved:")
        best_acc = max(results, key=lambda x: x['mean_acc'])
        print(f"    Weight {best_acc['weight']:.2f}: {best_acc['mean_acc']:.4f} "
              f"({best_acc['mean_time']:.2f}s)")
    else:
        # Find fastest config among those achieving ≥94%
        fastest = min(valid_configs, key=lambda x: x['mean_time'])

        print(f"\n1. FASTEST CONFIG ACHIEVING ≥94% ACCURACY:")
        print(f"   Weight: {fastest['weight']:.2f}")
        print(f"   Accuracy: {fastest['mean_acc']:.4f} ± {fastest['std_acc']:.4f}")
        print(f"   Time: {fastest['mean_time']:.2f}s ± {fastest['std_time']:.2f}s")
        print(f"   Success Rate: {fastest['success_rate']:.1%} ({fastest['num_successful']}/{fastest['num_runs']} runs)")

        if baseline_time and fastest['weight'] != 0.50:
            time_saved = baseline_time - fastest['mean_time']
            time_saved_pct = (time_saved / baseline_time) * 100
            print(f"   Time Savings: {time_saved:.2f}s ({time_saved_pct:.1f}%) vs baseline")

        # Find highest accuracy config
        best_acc = max(valid_configs, key=lambda x: x['mean_acc'])

        if best_acc['weight'] != fastest['weight']:
            print(f"\n2. HIGHEST ACCURACY CONFIG (among ≥94%):")
            print(f"   Weight: {best_acc['weight']:.2f}")
            print(f"   Accuracy: {best_acc['mean_acc']:.4f} ± {best_acc['std_acc']:.4f}")
            print(f"   Time: {best_acc['mean_time']:.2f}s ± {best_acc['std_time']:.2f}s")
            print(f"   Success Rate: {best_acc['success_rate']:.1%}")

        # Find most reliable config (highest success rate)
        most_reliable = max(valid_configs, key=lambda x: x['success_rate'])

        if most_reliable['weight'] not in [fastest['weight'], best_acc['weight']]:
            print(f"\n3. MOST RELIABLE CONFIG:")
            print(f"   Weight: {most_reliable['weight']:.2f}")
            print(f"   Accuracy: {most_reliable['mean_acc']:.4f} ± {most_reliable['std_acc']:.4f}")
            print(f"   Time: {most_reliable['mean_time']:.2f}s ± {most_reliable['std_time']:.2f}s")
            print(f"   Success Rate: {most_reliable['success_rate']:.1%}")

    # Compare to baseline
    if baseline:
        print(f"\n4. BASELINE (weight=0.50) PERFORMANCE:")
        print(f"   Accuracy: {baseline['mean_acc']:.4f} ± {baseline['std_acc']:.4f}")
        print(f"   Time: {baseline['mean_time']:.2f}s ± {baseline['std_time']:.2f}s")
        print(f"   Success Rate: {baseline['success_rate']:.1%}")

    # Export detailed CSV
    csv_path = exp_base / 'tta_time_accuracy_tradeoff.csv'
    with open(csv_path, 'w') as f:
        f.write("weight,mean_accuracy,std_accuracy,mean_time,std_time,success_rate,"
                "achieves_94,time_diff_vs_baseline,time_diff_pct\n")
        for r in results:
            time_diff = r['mean_time'] - baseline_time if baseline_time else 0
            time_diff_pct = (time_diff / baseline_time * 100) if baseline_time else 0
            f.write(f"{r['weight']:.2f},{r['mean_acc']:.6f},{r['std_acc']:.6f},"
                   f"{r['mean_time']:.4f},{r['std_time']:.4f},{r['success_rate']:.4f},"
                   f"{int(r['achieves_target'])},{time_diff:.4f},{time_diff_pct:.2f}\n")

    print(f"\n✓ Saved detailed analysis to {csv_path}")
    print("="*120 + "\n")


if __name__ == "__main__":
    analyze_tta_time_tradeoff()
