"""
TTA Weight Optimization Sweep - WITH TIME TRACKING

This script trains baseline models and evaluates them with different TTA weight
configurations to find the optimal weighting that maintains 94% accuracy with
minimal runtime.

KEY MODIFICATION: Tracks training time + evaluation time for each configuration
to identify which TTA weights achieve 94% accuracy fastest.

STRATEGY:
- Phase 1: Train n baseline models, recording training time for each
- Phase 2: Evaluate each model with different TTA weight configs, timing each eval
- Phase 3: Log total_time = training_time + eval_time for each run

OBJECTIVE: Find TTA weight that achieves ≥94% accuracy in minimum total time
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# Import from airbench94.py
from airbench94 import make_net, CifarLoader, evaluate, hyp, init_whitening_conv
from experiment_logger import ExperimentLogger


def evaluate_with_tta_weights_timed(model, test_loader, weight_configs, verbose=True):
    """
    Evaluate a single model with multiple TTA weight configurations.
    NOW TRACKS TIME FOR EACH EVALUATION.

    Args:
        model: Trained PyTorch model
        test_loader: CIFAR-10 test data loader
        weight_configs: List of untrans_weight values to test
        verbose: Print progress

    Returns:
        Dictionary mapping weight -> (accuracy, eval_time_seconds)
    """
    results = {}

    for untrans_weight in weight_configs:
        if verbose:
            print(f"  Testing untrans_weight={untrans_weight:.2f}...", end=" ")

        # Time the evaluation
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        starter.record()
        acc = evaluate(model, test_loader, tta_level=2, untrans_weight=untrans_weight)
        ender.record()
        torch.cuda.synchronize()

        eval_time = 1e-3 * starter.elapsed_time(ender)  # Convert to seconds

        results[untrans_weight] = (acc, eval_time)

        if verbose:
            print(f"acc={acc:.4f}, time={eval_time:.3f}s")

    return results


def save_model_checkpoint(model, checkpoint_dir, model_id):
    """Save model checkpoint to disk."""
    checkpoint_path = checkpoint_dir / f"model_{model_id:03d}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    return checkpoint_path


def load_model_checkpoint(checkpoint_path):
    """Load model from checkpoint."""
    model = make_net()
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model


def train_baseline_models_with_timing(n_models, checkpoint_dir, verbose=True):
    """
    Train n baseline models using a simplified training loop.
    NOW TRACKS TRAINING TIME FOR EACH MODEL.

    Returns:
        List of tuples: (checkpoint_path, training_time_seconds)
    """
    import torch.nn.functional as F
    from math import ceil
    from airbench94 import LookaheadState

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training hyperparameters (from airbench94.py)
    batch_size = hyp['opt']['batch_size']
    epochs = hyp['opt']['train_epochs']
    momentum = hyp['opt']['momentum']
    kilostep_scale = 1024 * (1 + 1 / (1 - momentum))
    lr = hyp['opt']['lr'] / kilostep_scale
    wd = hyp['opt']['weight_decay'] * batch_size / kilostep_scale
    lr_biases = lr * hyp['opt']['bias_scaler']

    train_loader = CifarLoader('cifar10', train=True, batch_size=batch_size, aug=hyp['aug'])
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=hyp['opt']['label_smoothing'], reduction='none')

    checkpoints_with_times = []

    for model_id in range(n_models):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training baseline model {model_id + 1}/{n_models}")
            print(f"{'='*60}")

        # TIMING: Start tracking training time
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        total_time_seconds = 0.0

        # Create fresh model
        model = make_net()

        # Initialize optimizer
        norm_biases = [p for k, p in model.named_parameters() if 'norm' in k and p.requires_grad]
        other_params = [p for k, p in model.named_parameters() if 'norm' not in k and p.requires_grad]
        param_configs = [
            dict(params=norm_biases, lr=lr_biases, weight_decay=wd/lr_biases),
            dict(params=other_params, lr=lr, weight_decay=wd/lr)
        ]
        optimizer = torch.optim.SGD(param_configs, momentum=momentum, nesterov=True)

        # Learning rate schedule (using default from airbench94)
        total_train_steps = ceil(len(train_loader) * epochs)

        def triangle(steps, start=0.15, end=0.07, peak=0.21):
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

        # Initialize whitening layer (TIMED)
        starter.record()
        train_images = train_loader.normalize(train_loader.images[:5000])
        init_whitening_conv(model[0], train_images)
        ender.record()
        torch.cuda.synchronize()
        total_time_seconds += 1e-3 * starter.elapsed_time(ender)

        # Training loop (TIMED)
        current_steps = 0
        for epoch in range(ceil(epochs)):
            model[0].bias.requires_grad = (epoch < hyp['opt']['whiten_bias_epochs'])

            starter.record()
            model.train()
            for inputs, labels in train_loader:
                outputs = model(inputs)
                loss = loss_fn(outputs, labels).sum()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()

                current_steps += 1

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

        # Save checkpoint
        checkpoint_path = save_model_checkpoint(model, checkpoint_dir, model_id)
        checkpoints_with_times.append((checkpoint_path, total_time_seconds))

        if verbose:
            print(f"✓ Model saved to {checkpoint_path}")
            print(f"✓ Training time: {total_time_seconds:.2f}s")

    return checkpoints_with_times


def main():
    parser = argparse.ArgumentParser(
        description='TTA Weight Optimization Sweep for CIFAR-10 (with time tracking)'
    )
    parser.add_argument('--n_models', type=int, default=25,
                        help='Number of baseline models to train (default: 25)')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='experiments/exp410-tta-weight-optimization/checkpoints',
                        help='Directory to save/load model checkpoints')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip training phase and use existing checkpoints (will estimate training time)')
    parser.add_argument('--experiment_name', type=str, default='exp410-tta-weight-optimization',
                        help='Experiment name for logging')

    args = parser.parse_args()

    # TTA weight configurations to test
    weight_configs = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

    print("\n" + "="*80)
    print("TTA WEIGHT OPTIMIZATION SWEEP (WITH TIME TRACKING)")
    print("="*80)
    print(f"Configuration:")
    print(f"  - Number of models: {args.n_models}")
    print(f"  - TTA weights to test: {weight_configs}")
    print(f"  - Total evaluations: {args.n_models} models × {len(weight_configs)} configs = {args.n_models * len(weight_configs)}")
    print(f"  - Checkpoint directory: {args.checkpoint_dir}")
    print(f"  - TRACKING: Training time + Evaluation time for each config")
    print("="*80 + "\n")

    checkpoint_dir = Path(args.checkpoint_dir)

    # Phase 1: Train baseline models (or load existing) WITH TIMING
    if args.skip_training:
        print("Skipping training phase, loading existing checkpoints...")
        checkpoints = sorted(checkpoint_dir.glob("model_*.pt"))
        if len(checkpoints) == 0:
            print(f"ERROR: No checkpoints found in {checkpoint_dir}")
            print("Please train models first or specify a different checkpoint directory.")
            sys.exit(1)
        print(f"Found {len(checkpoints)} existing checkpoints")
        # Estimate training time (use average from similar experiments)
        estimated_train_time = 3.83  # Default airbench94 time
        checkpoints_with_times = [(cp, estimated_train_time) for cp in checkpoints]
        print(f"Note: Using estimated training time of {estimated_train_time:.2f}s per model")
    else:
        print("Phase 1: Training baseline models WITH TIME TRACKING...")
        checkpoints_with_times = train_baseline_models_with_timing(args.n_models, checkpoint_dir)
        print(f"\n✓ Trained and saved {len(checkpoints_with_times)} baseline models")

    # Phase 2: Evaluate each model with all TTA weight configs WITH TIMING
    print("\n" + "="*80)
    print("Phase 2: Evaluating with different TTA weights (WITH TIME TRACKING)")
    print("="*80)

    # Prepare test loader
    test_loader = CifarLoader('cifar10', train=False, batch_size=2000)

    # Store all results: weight -> list of (accuracy, total_time)
    all_results = {weight: [] for weight in weight_configs}

    for i, (checkpoint_path, train_time) in enumerate(checkpoints_with_times):
        print(f"\nEvaluating model {i+1}/{len(checkpoints_with_times)} ({checkpoint_path.name if isinstance(checkpoint_path, Path) else checkpoint_path})")
        print(f"  Training time: {train_time:.2f}s")

        # Load model
        model = load_model_checkpoint(checkpoint_path)

        # Evaluate with all weight configs (TIMED)
        results = evaluate_with_tta_weights_timed(model, test_loader, weight_configs, verbose=True)

        # Store results with TOTAL time (training + eval)
        for weight, (acc, eval_time) in results.items():
            total_time = train_time + eval_time
            all_results[weight].append((acc, total_time))
            print(f"    weight={weight:.2f}: acc={acc:.4f}, eval_time={eval_time:.3f}s, TOTAL={total_time:.2f}s")

    # Phase 3: Save results for each weight configuration WITH TIME
    print("\n" + "="*80)
    print("Phase 3: Saving results (with time tracking)")
    print("="*80)

    for weight in weight_configs:
        weight_name = f"tta_weight_{int(weight*100):02d}"
        logger = ExperimentLogger(
            experiment_name=f"{args.experiment_name}_{weight_name}",
            experiment_description=f"TTA weight optimization: untrans_weight={weight:.2f} (time-tracked)",
            hyperparameters={
                **hyp,
                'tta_untrans_weight': weight,
                'n_models_tested': len(checkpoints_with_times)
            }
        )

        # Log each model's result as a separate run WITH TIME
        for run_id, (acc, total_time) in enumerate(all_results[weight]):
            logger.log_run(
                run_id=run_id,
                accuracy=acc,
                time_seconds=total_time,  # NOW INCLUDES TRAINING + EVAL TIME
                epochs_completed=hyp['opt']['train_epochs']
            )

        logger.save_summary()

        # Calculate statistics
        accs = [a for a, t in all_results[weight]]
        times = [t for a, t in all_results[weight]]
        mean_acc = sum(accs) / len(accs)
        mean_time = sum(times) / len(times)
        success_rate = sum(1 for a in accs if a >= 0.94) / len(accs)

        print(f"✓ weight={weight:.2f}: mean_acc={mean_acc:.4f}, mean_time={mean_time:.2f}s, "
              f"≥94% rate={success_rate:.1%}")

    # Print summary comparison
    print("\n" + "="*80)
    print("SWEEP COMPLETE - TIME COMPARISON")
    print("="*80)
    print(f"{'Weight':<10} {'Mean Acc':<12} {'Mean Time (s)':<15} {'≥94% Rate':<12} {'vs Baseline':<15}")
    print("-"*80)

    baseline_time = None
    for weight in weight_configs:
        accs = [a for a, t in all_results[weight]]
        times = [t for a, t in all_results[weight]]
        mean_acc = sum(accs) / len(accs)
        mean_time = sum(times) / len(times)
        success_rate = sum(1 for a in accs if a >= 0.94) / len(accs)

        if weight == 0.50:
            baseline_time = mean_time
            diff_str = "BASELINE"
        elif baseline_time:
            diff = mean_time - baseline_time
            diff_str = f"{diff:+.2f}s ({diff/baseline_time*100:+.1f}%)"
        else:
            diff_str = "-"

        print(f"{weight:<10.2f} {mean_acc:<12.4f} {mean_time:<15.2f} {success_rate:<12.1%} {diff_str:<15}")

    print("\n" + "="*80)
    print(f"✓ Evaluated {len(checkpoints_with_times)} models with {len(weight_configs)} weight configs")
    print(f"✓ Total evaluations: {len(checkpoints_with_times) * len(weight_configs)}")
    print(f"\nNext steps:")
    print(f"  1. Run analysis: python simple_tta_analysis.py")
    print(f"  2. Check results in: experiments/{args.experiment_name}_*/")
    print(f"  3. Look for: High accuracy (≥94%) + Low total time")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
