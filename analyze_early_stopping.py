#!/usr/bin/env python3
"""
Analysis for Experiment 420: Early Stopping with TTA Compensation

Analyzes results to find the minimum epochs needed to achieve ≥94% accuracy
when using TTA, and calculates time savings.
"""

import json
from pathlib import Path
import sys


def analyze_early_stopping():
    """Analyze early stopping experiment results."""
    exp_base = Path('experiments')

    # Find all early stopping experiment directories
    early_stop_dirs = sorted(exp_base.glob('exp420-early-stop-tta_epoch*'))

    if not early_stop_dirs:
        print("No early stopping experiments found.")
        print("Run: python sweep_early_stop_with_tta.py --runs_per_epoch 25")
        return

    print("\n" + "="*120)
    print("EARLY STOPPING WITH TTA COMPENSATION - ANALYSIS")
    print("="*120)
    print(f"Found {len(early_stop_dirs)} epoch configurations")
    print(f"Objective: Find minimum epochs to achieve ≥94% accuracy with TTA\n")

    results = []

    for exp_dir in early_stop_dirs:
        # Extract epoch value from directory name
        # Format: exp420-early-stop-tta_epochXX (where XX is epochs*10)
        epoch_str = exp_dir.name.split('_')[-1].replace('epoch', '')
        try:
            epoch_val = int(epoch_str) / 10.0
        except:
            continue

        # Load summary
        summary_file = exp_dir / 'summary.json'
        if not summary_file.exists():
            continue

        with open(summary_file, 'r') as f:
            summary = json.load(f)

        stats = summary.get('statistics', {})

        # Get TTA accuracy (primary metric)
        mean_tta = stats.get('accuracy_mean', stats.get('mean_accuracy', 0))
        std_tta = stats.get('accuracy_std', stats.get('std_accuracy', 0))

        # Get time stats
        mean_time = stats.get('time_mean', stats.get('mean_time', 0))
        std_time = stats.get('time_std', stats.get('std_time', 0))

        # Get success rate
        success_rate = stats.get('success_rate', 0)
        num_runs = stats.get('num_runs', 0)

        results.append({
            'epochs': epoch_val,
            'mean_tta': mean_tta,
            'std_tta': std_tta,
            'mean_time': mean_time,
            'std_time': std_time,
            'success_rate': success_rate,
            'num_runs': num_runs,
            'achieves_target': mean_tta >= 0.94
        })

    if not results:
        print("No valid results found!")
        return

    # Sort by epoch value
    results.sort(key=lambda x: x['epochs'])

    # Find baseline (9.9 epochs)
    baseline = next((r for r in results if r['epochs'] == 9.9), None)
    baseline_time = baseline['mean_time'] if baseline else None
    baseline_acc = baseline['mean_tta'] if baseline else None

    # Print detailed table
    print(f"{'Epochs':<10} {'Mean TTA':<12} {'±Std':<10} {'Mean Time':<12} {'±Std':<10} "
          f"{'≥94% Rate':<12} {'Target?':<10} {'Time Saved':<20}")
    print("-"*120)

    for r in results:
        # Calculate time savings
        if baseline_time:
            time_saved = baseline_time - r['mean_time']
            time_saved_pct = (time_saved / baseline_time) * 100
            if r['epochs'] == 9.9:
                time_str = "BASELINE"
            else:
                time_str = f"-{time_saved:.2f}s ({time_saved_pct:.1f}%)"
        else:
            time_str = "-"

        # Mark if achieves target
        target_str = "✓ YES" if r['achieves_target'] else "✗ NO"

        print(f"{r['epochs']:<10.1f} {r['mean_tta']:<12.4f} {r['std_tta']:<10.4f} "
              f"{r['mean_time']:<12.2f} {r['std_time']:<10.2f} "
              f"{r['success_rate']:<12.1%} {target_str:<10} {time_str:<20}")

    # Analysis
    print("\n" + "="*120)
    print("FINDINGS & RECOMMENDATIONS")
    print("="*120)

    # Filter configs that achieve ≥94%
    valid_configs = [r for r in results if r['achieves_target']]

    if not valid_configs:
        print("\n⚠️  WARNING: No epoch values achieved ≥94% mean TTA accuracy!")
        print("    Hypothesis REJECTED: Cannot reduce epochs while maintaining ≥94% with TTA")
        best = max(results, key=lambda x: x['mean_tta'])
        print(f"\n    Best achieved: {best['epochs']:.1f} epochs → {best['mean_tta']:.4f} accuracy")
        print(f"    Gap to target: {0.94 - best['mean_tta']:.4f} ({(0.94 - best['mean_tta'])*100:.2f}%)")
    else:
        # Find minimum epochs
        min_epoch = min(valid_configs, key=lambda x: x['epochs'])

        print(f"\n✓ HYPOTHESIS VALIDATED: Can reduce training time with TTA!")
        print(f"\n1. MINIMUM EPOCHS FOR ≥94% ACCURACY:")
        print(f"   Epochs: {min_epoch['epochs']:.1f} (vs 9.9 baseline)")
        print(f"   Epoch Reduction: {9.9 - min_epoch['epochs']:.1f} epochs ({(9.9 - min_epoch['epochs'])/9.9*100:.1f}%)")
        print(f"   TTA Accuracy: {min_epoch['mean_tta']:.4f} ± {min_epoch['std_tta']:.4f}")
        print(f"   Mean Time: {min_epoch['mean_time']:.2f}s ± {min_epoch['std_time']:.2f}s")
        print(f"   Success Rate: {min_epoch['success_rate']:.1%} ({int(min_epoch['success_rate'] * min_epoch['num_runs'])}/{min_epoch['num_runs']} runs)")

        if baseline_time and min_epoch['epochs'] < 9.9:
            time_saved = baseline_time - min_epoch['mean_time']
            time_saved_pct = (time_saved / baseline_time) * 100
            speedup = baseline_time / min_epoch['mean_time']

            print(f"\n2. TIME SAVINGS:")
            print(f"   Baseline (9.9 epochs): {baseline_time:.2f}s")
            print(f"   Optimized ({min_epoch['epochs']:.1f} epochs): {min_epoch['mean_time']:.2f}s")
            print(f"   Time Saved: {time_saved:.2f}s ({time_saved_pct:.1f}%)")
            print(f"   Speedup: {speedup:.2f}x")

            # Project savings at scale
            print(f"\n3. PROJECTED SAVINGS AT SCALE:")
            runs_1000 = time_saved * 1000
            print(f"   For 1,000 runs: {runs_1000:.0f}s ({runs_1000/60:.1f} minutes)")
            runs_10000 = time_saved * 10000
            print(f"   For 10,000 runs: {runs_10000:.0f}s ({runs_10000/3600:.1f} hours)")

        # Check if we can go even lower
        even_lower = [r for r in valid_configs if r['epochs'] < min_epoch['epochs']]
        if even_lower:
            print(f"\n4. ALTERNATIVE CONFIGURATIONS:")
            for r in sorted(even_lower, key=lambda x: x['epochs']):
                ts = baseline_time - r['mean_time'] if baseline_time else 0
                print(f"   {r['epochs']:.1f} epochs: {r['mean_tta']:.4f} accuracy, {r['mean_time']:.2f}s (saves {ts:.2f}s)")

        # Most reliable config
        most_reliable = max(valid_configs, key=lambda x: x['success_rate'])
        if most_reliable['epochs'] != min_epoch['epochs']:
            print(f"\n5. MOST RELIABLE CONFIGURATION:")
            print(f"   Epochs: {most_reliable['epochs']:.1f}")
            print(f"   TTA Accuracy: {most_reliable['mean_tta']:.4f}")
            print(f"   Success Rate: {most_reliable['success_rate']:.1%}")
            print(f"   Mean Time: {most_reliable['mean_time']:.2f}s")

    # Baseline comparison
    if baseline:
        print(f"\n6. BASELINE PERFORMANCE (9.9 epochs):")
        print(f"   TTA Accuracy: {baseline['mean_tta']:.4f} ± {baseline['std_tta']:.4f}")
        print(f"   Time: {baseline['mean_time']:.2f}s ± {baseline['std_time']:.2f}s")
        print(f"   Success Rate: {baseline['success_rate']:.1%}")

    # Plot data points
    print(f"\n7. EPOCH vs ACCURACY TREND:")
    print(f"   {'Epochs':<10} {'TTA Accuracy':<15} {'Trend':<30}")
    print(f"   {'-'*55}")
    for r in sorted(results, key=lambda x: x['epochs']):
        # Simple visualization
        bar_length = int(r['mean_tta'] * 100 - 90)  # Scale from 90-100%
        bar = '█' * bar_length
        print(f"   {r['epochs']:<10.1f} {r['mean_tta']:<15.4f} {bar}")

    # Export detailed CSV
    csv_path = exp_base / 'early_stopping_analysis.csv'
    with open(csv_path, 'w') as f:
        f.write("epochs,mean_tta_acc,std_tta_acc,mean_time,std_time,success_rate,"
                "achieves_94,epoch_reduction,time_saved,time_saved_pct,speedup\n")
        for r in results:
            epoch_reduction = 9.9 - r['epochs']
            time_saved = (baseline_time - r['mean_time']) if baseline_time else 0
            time_saved_pct = (time_saved / baseline_time * 100) if baseline_time else 0
            speedup = (baseline_time / r['mean_time']) if r['mean_time'] > 0 else 0

            f.write(f"{r['epochs']:.1f},{r['mean_tta']:.6f},{r['std_tta']:.6f},"
                   f"{r['mean_time']:.4f},{r['std_time']:.4f},{r['success_rate']:.4f},"
                   f"{int(r['achieves_target'])},{epoch_reduction:.1f},"
                   f"{time_saved:.4f},{time_saved_pct:.2f},{speedup:.4f}\n")

    print(f"\n✓ Saved detailed analysis to {csv_path}")
    print("="*120 + "\n")


if __name__ == "__main__":
    analyze_early_stopping()
