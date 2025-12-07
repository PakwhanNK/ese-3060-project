#!/usr/bin/env python3
"""
Compare Early Stopping Experiments: Baseline LR vs Optimal LR

Compares exp420 (baseline LR: end=0.07) and exp421 (optimal LR: end=0.05)
to see if optimal LR schedule enables earlier stopping with TTA.
"""

import json
from pathlib import Path
import sys


def load_early_stop_results(exp_prefix):
    """Load results from early stopping experiment."""
    exp_base = Path('experiments')
    dirs = sorted(exp_base.glob(f'{exp_prefix}_epoch*'))

    results = []
    for exp_dir in dirs:
        # Extract epoch value
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

        results.append({
            'epochs': epoch_val,
            'mean_tta': stats.get('accuracy_mean', stats.get('mean_accuracy', 0)),
            'std_tta': stats.get('accuracy_std', stats.get('std_accuracy', 0)),
            'mean_time': stats.get('time_mean', stats.get('mean_time', 0)),
            'std_time': stats.get('time_std', stats.get('std_time', 0)),
            'success_rate': stats.get('success_rate', 0),
            'num_runs': stats.get('num_runs', 0),
            'achieves_target': stats.get('accuracy_mean', stats.get('mean_accuracy', 0)) >= 0.94
        })

    return sorted(results, key=lambda x: x['epochs'])


def compare_experiments():
    """Compare baseline and optimal LR early stopping experiments."""

    # Try to load both experiments
    baseline_results = load_early_stop_results('exp420-early-stop-tta')
    optimal_results = load_early_stop_results('exp421-early-stop-optimal-lr')

    if not baseline_results and not optimal_results:
        print("No early stopping experiments found.")
        print("Run: python sweep_early_stop_with_tta.py (exp420)")
        print("And: python sweep_early_stop_optimal_lr.py (exp421)")
        return

    print("\n" + "="*140)
    print("EARLY STOPPING COMPARISON: BASELINE LR vs OPTIMAL LR")
    print("="*140)
    print("Baseline LR: start=0.15, peak=0.21, end=0.07 (exp420)")
    print("Optimal LR:  start=0.15, peak=0.21, end=0.05 (exp421)")
    print("="*140 + "\n")

    if baseline_results and optimal_results:
        print("SIDE-BY-SIDE COMPARISON")
        print("-"*140)
        print(f"{'Epochs':<10} {'Baseline TTA':<15} {'Optimal TTA':<15} {'Δ Acc':<12} "
              f"{'Baseline Time':<15} {'Optimal Time':<15} {'Δ Time':<12}")
        print("-"*140)

        # Create lookup dictionaries
        baseline_dict = {r['epochs']: r for r in baseline_results}
        optimal_dict = {r['epochs']: r for r in optimal_results}

        # Get all epoch values
        all_epochs = sorted(set(baseline_dict.keys()) | set(optimal_dict.keys()))

        for epoch in all_epochs:
            base = baseline_dict.get(epoch)
            opt = optimal_dict.get(epoch)

            if base and opt:
                # Both available
                acc_diff = opt['mean_tta'] - base['mean_tta']
                time_diff = opt['mean_time'] - base['mean_time']

                base_str = f"{base['mean_tta']:.4f}"
                opt_str = f"{opt['mean_tta']:.4f}"
                acc_diff_str = f"{acc_diff:+.4f}"

                base_time_str = f"{base['mean_time']:.2f}s"
                opt_time_str = f"{opt['mean_time']:.2f}s"
                time_diff_str = f"{time_diff:+.2f}s"

                print(f"{epoch:<10.1f} {base_str:<15} {opt_str:<15} {acc_diff_str:<12} "
                      f"{base_time_str:<15} {opt_time_str:<15} {time_diff_str:<12}")

            elif base:
                # Only baseline
                print(f"{epoch:<10.1f} {base['mean_tta']:<15.4f} {'N/A':<15} {'-':<12} "
                      f"{base['mean_time']:<15.2f}s {'N/A':<15} {'-':<12}")

            elif opt:
                # Only optimal
                print(f"{epoch:<10.1f} {'N/A':<15} {opt['mean_tta']:<15.4f} {'-':<12} "
                      f"{'N/A':<15} {opt['mean_time']:<15.2f}s {'-':<12}")

    # Individual summaries
    print("\n" + "="*140)
    print("BASELINE LR SCHEDULE (end=0.07)")
    print("="*140)

    if baseline_results:
        valid_base = [r for r in baseline_results if r['achieves_target']]
        if valid_base:
            min_base = min(valid_base, key=lambda x: x['epochs'])
            print(f"\n✓ Achieves ≥94% with TTA")
            print(f"  Minimum epochs: {min_base['epochs']:.1f}")
            print(f"  Accuracy: {min_base['mean_tta']:.4f} ± {min_base['std_tta']:.4f}")
            print(f"  Time: {min_base['mean_time']:.2f}s ± {min_base['std_time']:.2f}s")
            print(f"  Success rate: {min_base['success_rate']:.1%}")

            baseline_99 = next((r for r in baseline_results if r['epochs'] == 9.9), None)
            if baseline_99 and min_base['epochs'] < 9.9:
                time_saved = baseline_99['mean_time'] - min_base['mean_time']
                print(f"  Time savings vs 9.9 epochs: {time_saved:.2f}s ({time_saved/baseline_99['mean_time']*100:.1f}%)")
        else:
            print("\n✗ Does NOT achieve ≥94% with TTA at any epoch value tested")
    else:
        print("\nNo exp420 results found. Run: python sweep_early_stop_with_tta.py")

    print("\n" + "="*140)
    print("OPTIMAL LR SCHEDULE (end=0.05)")
    print("="*140)

    if optimal_results:
        valid_opt = [r for r in optimal_results if r['achieves_target']]
        if valid_opt:
            min_opt = min(valid_opt, key=lambda x: x['epochs'])
            print(f"\n✓ Achieves ≥94% with TTA")
            print(f"  Minimum epochs: {min_opt['epochs']:.1f}")
            print(f"  Accuracy: {min_opt['mean_tta']:.4f} ± {min_opt['std_tta']:.4f}")
            print(f"  Time: {min_opt['mean_time']:.2f}s ± {min_opt['std_time']:.2f}s")
            print(f"  Success rate: {min_opt['success_rate']:.1%}")

            optimal_99 = next((r for r in optimal_results if r['epochs'] == 9.9), None)
            if optimal_99 and min_opt['epochs'] < 9.9:
                time_saved = optimal_99['mean_time'] - min_opt['mean_time']
                print(f"  Time savings vs 9.9 epochs: {time_saved:.2f}s ({time_saved/optimal_99['mean_time']*100:.1f}%)")
        else:
            print("\n✗ Does NOT achieve ≥94% with TTA at any epoch value tested")
    else:
        print("\nNo exp421 results found. Run: python sweep_early_stop_optimal_lr.py")

    # Direct comparison
    if baseline_results and optimal_results:
        valid_base = [r for r in baseline_results if r['achieves_target']]
        valid_opt = [r for r in optimal_results if r['achieves_target']]

        if valid_base and valid_opt:
            min_base = min(valid_base, key=lambda x: x['epochs'])
            min_opt = min(valid_opt, key=lambda x: x['epochs'])

            print("\n" + "="*140)
            print("HEAD-TO-HEAD COMPARISON")
            print("="*140)

            print(f"\n{'Metric':<30} {'Baseline LR':<25} {'Optimal LR':<25} {'Winner':<20}")
            print("-"*100)

            # Minimum epochs
            winner = "Optimal LR" if min_opt['epochs'] < min_base['epochs'] else "Baseline LR" if min_base['epochs'] < min_opt['epochs'] else "Tie"
            print(f"{'Minimum epochs':<30} {min_base['epochs']:<25.1f} {min_opt['epochs']:<25.1f} {winner:<20}")

            # Accuracy at minimum
            winner = "Optimal LR" if min_opt['mean_tta'] > min_base['mean_tta'] else "Baseline LR" if min_base['mean_tta'] > min_opt['mean_tta'] else "Tie"
            print(f"{'Accuracy at minimum':<30} {min_base['mean_tta']:<25.4f} {min_opt['mean_tta']:<25.4f} {winner:<20}")

            # Time at minimum
            winner = "Optimal LR" if min_opt['mean_time'] < min_base['mean_time'] else "Baseline LR" if min_base['mean_time'] < min_opt['mean_time'] else "Tie"
            print(f"{'Time at minimum':<30} {min_base['mean_time']:<25.2f}s {min_opt['mean_time']:<25.2f}s {winner:<20}")

            # Success rate at minimum
            winner = "Optimal LR" if min_opt['success_rate'] > min_base['success_rate'] else "Baseline LR" if min_base['success_rate'] > min_opt['success_rate'] else "Tie"
            print(f"{'Success rate':<30} {min_base['success_rate']:<25.1%} {min_opt['success_rate']:<25.1%} {winner:<20}")

            # Overall winner
            print(f"\n{'='*100}")
            if min_opt['epochs'] <= min_base['epochs'] and min_opt['mean_tta'] >= min_base['mean_tta']:
                print("🏆 WINNER: OPTIMAL LR SCHEDULE")
                print(f"   Allows stopping at same/earlier epoch with same/better accuracy")
                if min_opt['epochs'] < min_base['epochs']:
                    print(f"   Can stop {min_base['epochs'] - min_opt['epochs']:.1f} epochs earlier!")
                if min_opt['mean_tta'] > min_base['mean_tta']:
                    print(f"   Achieves {(min_opt['mean_tta'] - min_base['mean_tta'])*100:.2f}% higher accuracy!")
            elif min_base['epochs'] < min_opt['epochs']:
                print("🏆 WINNER: BASELINE LR SCHEDULE")
                print(f"   Allows stopping {min_opt['epochs'] - min_base['epochs']:.1f} epochs earlier")
            else:
                print("🤝 TIE: Both schedules perform similarly")

    # Export comparison CSV
    if baseline_results and optimal_results:
        csv_path = Path('experiments') / 'early_stop_lr_comparison.csv'
        with open(csv_path, 'w') as f:
            f.write("epochs,baseline_tta_acc,optimal_tta_acc,acc_diff,baseline_time,optimal_time,time_diff,baseline_achieves,optimal_achieves\n")

            baseline_dict = {r['epochs']: r for r in baseline_results}
            optimal_dict = {r['epochs']: r for r in optimal_results}
            all_epochs = sorted(set(baseline_dict.keys()) | set(optimal_dict.keys()))

            for epoch in all_epochs:
                base = baseline_dict.get(epoch)
                opt = optimal_dict.get(epoch)

                base_acc = base['mean_tta'] if base else float('nan')
                opt_acc = opt['mean_tta'] if opt else float('nan')
                acc_diff = opt_acc - base_acc if (base and opt) else float('nan')

                base_time = base['mean_time'] if base else float('nan')
                opt_time = opt['mean_time'] if opt else float('nan')
                time_diff = opt_time - base_time if (base and opt) else float('nan')

                base_achieves = int(base['achieves_target']) if base else 0
                opt_achieves = int(opt['achieves_target']) if opt else 0

                f.write(f"{epoch:.1f},{base_acc:.6f},{opt_acc:.6f},{acc_diff:.6f},"
                       f"{base_time:.4f},{opt_time:.4f},{time_diff:.4f},"
                       f"{base_achieves},{opt_achieves}\n")

        print(f"\n✓ Saved detailed comparison to {csv_path}")

    print("\n" + "="*140 + "\n")


if __name__ == "__main__":
    compare_experiments()
