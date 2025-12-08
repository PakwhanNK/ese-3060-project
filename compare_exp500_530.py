"""
Compare experiments 500, 510, 520, 530 to see individual and combined effects
"""

import json
from pathlib import Path


def load_results(exp_prefix):
    exp_base = Path('experiments')
    dirs = sorted(exp_base.glob(f'{exp_prefix}_epoch*'))
    results = []
    for exp_dir in dirs:
        epoch_str = exp_dir.name.split('_')[-1].replace('epoch', '')
        try:
            epoch_val = int(epoch_str) / 10.0
        except:
            continue
        summary_file = exp_dir / 'summary.json'
        if not summary_file.exists():
            continue
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        stats = summary.get('statistics', {})
        results.append({
            'epochs': epoch_val,
            'success_rate': stats.get('success_rate', 0),
        })
    return sorted(results, key=lambda x: x['epochs'])


def main():
    exp500 = load_results('exp500-baseline-50pct')
    exp510 = load_results('exp510-optimal-lr')
    exp520 = load_results('exp520-optimal-tta')
    exp530 = load_results('exp530-optimal-both')

    print("\n" + "="*100)
    print("EXPERIMENTS 500-530 COMPARISON")
    print("="*100)
    print(f"{'Epochs':<10} {'Exp500':<15} {'Exp510':<15} {'Exp520':<15} {'Exp530':<15}")
    print(f"{'      ':<10} {'Baseline':<15} {'+Opt LR':<15} {'+Opt TTA':<15} {'+Both':<15}")
    print("-"*100)

    all_epochs = sorted(set(r['epochs'] for results in [exp500, exp510, exp520, exp530] for r in results))

    exp500_dict = {r['epochs']: r for r in exp500}
    exp510_dict = {r['epochs']: r for r in exp510}
    exp520_dict = {r['epochs']: r for r in exp520}
    exp530_dict = {r['epochs']: r for r in exp530}

    for epoch in all_epochs:
        r500 = exp500_dict.get(epoch, {}).get('success_rate', 0)
        r510 = exp510_dict.get(epoch, {}).get('success_rate', 0)
        r520 = exp520_dict.get(epoch, {}).get('success_rate', 0)
        r530 = exp530_dict.get(epoch, {}).get('success_rate', 0)

        s500 = "✓" if r500 >= 0.50 else " "
        s510 = "✓" if r510 >= 0.50 else " "
        s520 = "✓" if r520 >= 0.50 else " "
        s530 = "✓" if r530 >= 0.50 else " "

        print(f"{epoch:<10.1f} {r500:.1%} {s500:<7} {r510:.1%} {s510:<7} {r520:.1%} {s520:<7} {r530:.1%} {s530:<7}")

    print("\n" + "="*100)
    print("MINIMUM EPOCHS FOR 50% SUCCESS RATE")
    print("="*100)

    def find_min(results):
        valid = [r for r in results if r['success_rate'] >= 0.50]
        return min(valid, key=lambda x: x['epochs'])['epochs'] if valid else None

    min500 = find_min(exp500)
    min510 = find_min(exp510)
    min520 = find_min(exp520)
    min530 = find_min(exp530)

    print(f"Exp500 (Baseline):         {min500:.1f} epochs" if min500 else "Exp500 (Baseline):         No 50% achieved")
    print(f"Exp510 (+Optimal LR):      {min510:.1f} epochs" if min510 else "Exp510 (+Optimal LR):      No 50% achieved")
    print(f"Exp520 (+Optimal TTA):     {min520:.1f} epochs" if min520 else "Exp520 (+Optimal TTA):     No 50% achieved")
    print(f"Exp530 (+Both):            {min530:.1f} epochs" if min530 else "Exp530 (+Both):            No 50% achieved")

    if all([min500, min510, min520, min530]):
        print(f"\n" + "="*100)
        print("EPOCH REDUCTION VS BASELINE")
        print("="*100)
        print(f"Optimal LR only:   {min500 - min510:.1f} epochs saved ({(min500-min510)/min500*100:.1f}%)")
        print(f"Optimal TTA only:  {min500 - min520:.1f} epochs saved ({(min500-min520)/min500*100:.1f}%)")
        print(f"Both optimizations: {min500 - min530:.1f} epochs saved ({(min500-min530)/min500*100:.1f}%)")

    csv_path = Path('experiments/exp500_530_comparison.csv')
    with open(csv_path, 'w') as f:
        f.write("epochs,exp500_sr,exp510_sr,exp520_sr,exp530_sr\n")
        for epoch in all_epochs:
            r500 = exp500_dict.get(epoch, {}).get('success_rate', 0)
            r510 = exp510_dict.get(epoch, {}).get('success_rate', 0)
            r520 = exp520_dict.get(epoch, {}).get('success_rate', 0)
            r530 = exp530_dict.get(epoch, {}).get('success_rate', 0)
            f.write(f"{epoch:.1f},{r500:.4f},{r510:.4f},{r520:.4f},{r530:.4f}\n")

    print(f"\n✓ Saved to {csv_path}\n")


if __name__ == "__main__":
    main()
