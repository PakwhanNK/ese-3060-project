"""
Experiment 421: Early Stopping with Optimal LR Schedule + TTA Compensation

HYPOTHESIS: Using the optimal triangular LR schedule (that yields best accuracy)
combined with TTA compensation might allow even earlier stopping than baseline.

OPTIMAL LR SCHEDULE (from previous experiments):
- Start: 0.15
- Peak: 0.21 (at 21% of training)
- End: 0.05  ← Better than baseline 0.07

BASELINE SCHEDULE:
- Start: 0.15
- Peak: 0.21
- End: 0.07

APPROACH:
- Test epoch values: 9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9
- For each epoch value, train N models with OPTIMAL LR schedule
- Evaluate with TTA (using optimal weight if available, else 0.50)
- Track: training time, TTA accuracy, no-TTA accuracy
- Compare to exp420 (baseline LR schedule)

OBJECTIVE: Test if optimal LR + TTA allows earlier stopping than baseline LR + TTA
"""

import os
import sys
import torch
import argparse
from pathlib import Path
from math import ceil

# Import from airbench94.py
from airbench94 import make_net, CifarLoader, evaluate, hyp, init_whitening_conv, LookaheadState
from experiment_logger import ExperimentLogger


def train_model_to_epoch_optimal_lr(target_epochs, run_id, lr_start=0.15, lr_peak=0.21, lr_end=0.05, tta_weight=0.50, verbose=False):
    """
    Train a single model for exactly target_epochs using OPTIMAL LR schedule.

    Args:
        target_epochs: Number of epochs to train for
        run_id: Run identifier for seeding
        lr_start: Starting LR multiplier (default: 0.15)
        lr_peak: Peak LR position as fraction (default: 0.21)
        lr_end: Ending LR multiplier (default: 0.05)
        tta_weight: TTA untrans_weight (default: 0.50)
        verbose: Print progress

    Returns:
        tuple: (model, train_time_seconds, final_train_acc, val_acc, tta_val_acc)
    """
    import torch.nn.functional as F

    # Set seed for reproducibility
    seed = 42 + run_id if isinstance(run_id, int) else 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    batch_size = hyp['opt']['batch_size']
    epochs = target_epochs  # Use target instead of default
    momentum = hyp['opt']['momentum']
    kilostep_scale = 1024 * (1 + 1 / (1 - momentum))
    lr = hyp['opt']['lr'] / kilostep_scale
    wd = hyp['opt']['weight_decay'] * batch_size / kilostep_scale
    lr_biases = lr * hyp['opt']['bias_scaler']

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=hyp['opt']['label_smoothing'], reduction='none')
    test_loader = CifarLoader('cifar10', train=False, batch_size=2000)
    train_loader = CifarLoader('cifar10', train=True, batch_size=batch_size, aug=hyp['aug'])

    total_train_steps = ceil(len(train_loader) * epochs)

    model = make_net()

    # Optimizer setup
    norm_biases = [p for k, p in model.named_parameters() if 'norm' in k and p.requires_grad]
    other_params = [p for k, p in model.named_parameters() if 'norm' not in k and p.requires_grad]
    param_configs = [
        dict(params=norm_biases, lr=lr_biases, weight_decay=wd/lr_biases),
        dict(params=other_params, lr=lr, weight_decay=wd/lr)]
    optimizer = torch.optim.SGD(param_configs, momentum=momentum, nesterov=True)

    # OPTIMAL Learning rate schedule
    def triangle(steps, start=lr_start, end=lr_end, peak=lr_peak):
        """Triangular LR schedule with OPTIMAL parameters."""
        xp = torch.tensor([0, int(peak * steps), steps])
        fp = torch.tensor([start, 1, end])
        x = torch.arange(1 + steps)
        m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
        b = fp[:-1] - (m * xp[:-1])
        indices = torch.sum(torch.ge(x[:, None], xp[None, :]), 1) - 1
        indices = torch.clamp(indices, 0, len(m) - 1)
        return m[indices] * x + b[indices]

    lr_schedule = triangle(total_train_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: lr_schedule[min(i, len(lr_schedule)-1)])

    # Lookahead
    alpha_schedule = 0.95**5 * (torch.arange(total_train_steps+1) / total_train_steps)**3
    lookahead_state = LookaheadState(model)

    # Time tracking
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    total_time_seconds = 0.0

    # Initialize whitening layer
    starter.record()
    train_images = train_loader.normalize(train_loader.images[:5000])
    init_whitening_conv(model[0], train_images)
    ender.record()
    torch.cuda.synchronize()
    total_time_seconds += 1e-3 * starter.elapsed_time(ender)

    # Training loop
    current_steps = 0
    final_train_acc = 0.0

    for epoch in range(ceil(epochs)):
        model[0].bias.requires_grad = (epoch < hyp['opt']['whiten_bias_epochs'])

        starter.record()
        model.train()

        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = loss_fn(outputs, labels).sum()

            # Track final training accuracy
            if current_steps == total_train_steps - 1:
                final_train_acc = (outputs.detach().argmax(1) == labels).float().mean().item()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            current_steps += 1

            # Lookahead update
            if current_steps % 5 == 0:
                lookahead_state.update(model, decay=alpha_schedule[current_steps].item())

            if current_steps >= total_train_steps:
                lookahead_state.update(model, decay=1.0)
                break

        ender.record()
        torch.cuda.synchronize()
        total_time_seconds += 1e-3 * starter.elapsed_time(ender)

        if current_steps >= total_train_steps:
            break

    # Evaluation (time this separately)
    eval_starter = torch.cuda.Event(enable_timing=True)
    eval_ender = torch.cuda.Event(enable_timing=True)

    # Evaluate without TTA
    eval_starter.record()
    val_acc = evaluate(model, test_loader, tta_level=0)
    eval_ender.record()
    torch.cuda.synchronize()
    eval_time_no_tta = 1e-3 * eval_starter.elapsed_time(eval_ender)

    # Evaluate with TTA
    eval_starter.record()
    tta_val_acc = evaluate(model, test_loader, tta_level=2, untrans_weight=tta_weight)
    eval_ender.record()
    torch.cuda.synchronize()
    eval_time_tta = 1e-3 * eval_starter.elapsed_time(eval_ender)

    if verbose:
        print(f"  Epochs={target_epochs:.1f}, Time={total_time_seconds:.2f}s, "
              f"No-TTA={val_acc:.4f}, TTA={tta_val_acc:.4f}")

    return model, total_time_seconds, final_train_acc, val_acc, tta_val_acc


def main():
    parser = argparse.ArgumentParser(
        description='Early Stopping with OPTIMAL LR Schedule + TTA Compensation'
    )
    parser.add_argument('--runs_per_epoch', type=int, default=25,
                        help='Number of runs per epoch value (default: 25)')
    parser.add_argument('--epoch_start', type=float, default=9.0,
                        help='Starting epoch value (default: 9.0)')
    parser.add_argument('--epoch_end', type=float, default=9.9,
                        help='Ending epoch value (default: 9.9)')
    parser.add_argument('--epoch_step', type=float, default=0.1,
                        help='Epoch step size (default: 0.1)')
    parser.add_argument('--lr_start', type=float, default=0.15,
                        help='LR schedule start value (default: 0.15)')
    parser.add_argument('--lr_peak', type=float, default=0.21,
                        help='LR schedule peak position (default: 0.21)')
    parser.add_argument('--lr_end', type=float, default=0.05,
                        help='LR schedule end value (default: 0.05 - OPTIMAL)')
    parser.add_argument('--tta_weight', type=float, default=0.50,
                        help='TTA untrans_weight to use (default: 0.50)')
    parser.add_argument('--experiment_name', type=str, default='exp421-early-stop-optimal-lr',
                        help='Experiment name for logging')

    args = parser.parse_args()

    # Generate epoch values to test
    import numpy as np
    epoch_values = np.arange(args.epoch_start, args.epoch_end + args.epoch_step/2, args.epoch_step)
    epoch_values = [round(e, 1) for e in epoch_values]  # Round to 1 decimal

    print("\n" + "="*100)
    print("EXPERIMENT 421: EARLY STOPPING WITH OPTIMAL LR SCHEDULE + TTA")
    print("="*100)
    print(f"Hypothesis: Optimal LR schedule + TTA enables earlier stopping than baseline")
    print(f"\nLR Schedule (OPTIMAL):")
    print(f"  - Start: {args.lr_start}")
    print(f"  - Peak: {args.lr_peak} (at {int(args.lr_peak*100)}% of training)")
    print(f"  - End: {args.lr_end}  ← OPTIMAL (baseline uses 0.07)")
    print(f"\nConfiguration:")
    print(f"  - Epoch values to test: {epoch_values}")
    print(f"  - Runs per epoch: {args.runs_per_epoch}")
    print(f"  - Total runs: {len(epoch_values)} epochs × {args.runs_per_epoch} runs = {len(epoch_values) * args.runs_per_epoch}")
    print(f"  - TTA weight: {args.tta_weight}")
    print(f"  - Target accuracy: ≥94% (with TTA)")
    print("="*100 + "\n")

    # Store results for each epoch value
    all_results = {}

    for epoch_val in epoch_values:
        print(f"\n{'='*100}")
        print(f"Testing {epoch_val} epochs with OPTIMAL LR ({args.runs_per_epoch} runs)")
        print(f"{'='*100}")

        # Initialize logger for this epoch value
        logger = ExperimentLogger(
            experiment_name=f"{args.experiment_name}_epoch{int(epoch_val*10):02d}",
            experiment_description=f"Optimal LR early stopping: {epoch_val} epochs with TTA",
            hyperparameters={
                **hyp,
                'opt': {**hyp['opt'], 'train_epochs': epoch_val},
                'lr_schedule': {'start': args.lr_start, 'peak': args.lr_peak, 'end': args.lr_end},
                'tta_untrans_weight': args.tta_weight
            }
        )

        times = []
        no_tta_accs = []
        tta_accs = []

        for run_id in range(args.runs_per_epoch):
            model, train_time, final_train_acc, val_acc, tta_val_acc = train_model_to_epoch_optimal_lr(
                epoch_val, run_id,
                lr_start=args.lr_start,
                lr_peak=args.lr_peak,
                lr_end=args.lr_end,
                tta_weight=args.tta_weight,
                verbose=(run_id < 3)
            )

            times.append(train_time)
            no_tta_accs.append(val_acc)
            tta_accs.append(tta_val_acc)

            # Log this run (use TTA accuracy as primary metric)
            logger.log_run(
                run_id=run_id,
                accuracy=tta_val_acc,
                time_seconds=train_time,
                epochs_completed=epoch_val
            )

        # Save summary for this epoch value
        logger.save_summary()

        # Calculate statistics
        mean_time = sum(times) / len(times)
        mean_no_tta = sum(no_tta_accs) / len(no_tta_accs)
        mean_tta = sum(tta_accs) / len(tta_accs)
        success_rate = sum(1 for acc in tta_accs if acc >= 0.94) / len(tta_accs)

        all_results[epoch_val] = {
            'mean_time': mean_time,
            'mean_no_tta': mean_no_tta,
            'mean_tta': mean_tta,
            'success_rate': success_rate,
            'num_runs': args.runs_per_epoch
        }

        achieves_target = "✓ YES" if mean_tta >= 0.94 else "✗ NO"
        print(f"\n  Results: No-TTA={mean_no_tta:.4f}, TTA={mean_tta:.4f}, "
              f"Time={mean_time:.2f}s, ≥94%={success_rate:.1%} {achieves_target}")

    # Final comparison
    print("\n" + "="*100)
    print("RESULTS SUMMARY - OPTIMAL LR SCHEDULE + EARLY STOPPING")
    print("="*100)
    print(f"{'Epochs':<10} {'No-TTA Acc':<12} {'TTA Acc':<12} {'Mean Time':<12} {'≥94% Rate':<12} {'Target?':<10} {'Time Saved':<15}")
    print("-"*100)

    baseline_time = all_results.get(9.9, {}).get('mean_time', None)

    for epoch_val in sorted(all_results.keys()):
        r = all_results[epoch_val]

        achieves = "✓ YES" if r['mean_tta'] >= 0.94 else "✗ NO"

        if baseline_time and epoch_val < 9.9:
            time_saved = baseline_time - r['mean_time']
            time_saved_pct = (time_saved / baseline_time) * 100
            time_str = f"-{time_saved:.2f}s ({time_saved_pct:.1f}%)"
        elif epoch_val == 9.9:
            time_str = "BASELINE"
        else:
            time_str = "-"

        print(f"{epoch_val:<10.1f} {r['mean_no_tta']:<12.4f} {r['mean_tta']:<12.4f} "
              f"{r['mean_time']:<12.2f} {r['success_rate']:<12.1%} {achieves:<10} {time_str:<15}")

    # Find optimal epoch value
    valid_epochs = [(e, r) for e, r in all_results.items() if r['mean_tta'] >= 0.94]

    print("\n" + "="*100)
    print("RECOMMENDATIONS")
    print("="*100)

    if not valid_epochs:
        print("\n⚠️  WARNING: No epoch values achieved ≥94% mean TTA accuracy!")
        best = max(all_results.items(), key=lambda x: x[1]['mean_tta'])
        print(f"  Best achieved: {best[0]:.1f} epochs → {best[1]['mean_tta']:.4f} accuracy")
    else:
        # Find minimum epochs that achieves target
        min_epoch = min(valid_epochs, key=lambda x: x[0])

        print(f"\n1. MINIMUM EPOCHS FOR ≥94% ACCURACY (OPTIMAL LR + TTA):")
        print(f"   Epochs: {min_epoch[0]:.1f}")
        print(f"   TTA Accuracy: {min_epoch[1]['mean_tta']:.4f}")
        print(f"   No-TTA Accuracy: {min_epoch[1]['mean_no_tta']:.4f}")
        print(f"   Mean Time: {min_epoch[1]['mean_time']:.2f}s")
        print(f"   Success Rate: {min_epoch[1]['success_rate']:.1%}")

        if baseline_time and min_epoch[0] < 9.9:
            time_saved = baseline_time - min_epoch[1]['mean_time']
            time_saved_pct = (time_saved / baseline_time) * 100
            print(f"   TIME SAVINGS: {time_saved:.2f}s ({time_saved_pct:.1f}%) vs 9.9 epochs")
            print(f"   Speedup: {baseline_time / min_epoch[1]['mean_time']:.2f}x")

        print(f"\n2. COMPARISON TO BASELINE LR (end=0.07):")
        print(f"   Run exp420 to compare baseline LR schedule")
        print(f"   Expected: Optimal LR may allow stopping at same/earlier epoch with higher accuracy")

    # Export CSV
    csv_path = Path('experiments') / f'{args.experiment_name}_comparison.csv'
    csv_path.parent.mkdir(exist_ok=True)
    with open(csv_path, 'w') as f:
        f.write("epochs,mean_no_tta_acc,mean_tta_acc,mean_time,success_rate,achieves_94,time_saved_vs_baseline,lr_schedule\n")
        for epoch_val in sorted(all_results.keys()):
            r = all_results[epoch_val]
            time_saved = (baseline_time - r['mean_time']) if baseline_time else 0
            achieves = 1 if r['mean_tta'] >= 0.94 else 0
            lr_sched = f"start{args.lr_start}_peak{args.lr_peak}_end{args.lr_end}"
            f.write(f"{epoch_val:.1f},{r['mean_no_tta']:.6f},{r['mean_tta']:.6f},"
                   f"{r['mean_time']:.4f},{r['success_rate']:.4f},{achieves},{time_saved:.4f},{lr_sched}\n")

    print(f"\n✓ Saved comparison to {csv_path}")
    print("="*100 + "\n")


if __name__ == "__main__":
    main()
