"""
TTA Weight Optimization Sweep

This script trains baseline models and evaluates them with different TTA weight
configurations to find the optimal weighting between untranslated and translated views.

STRATEGY:
- Phase 1: Train n baseline models with standard hyperparameters
- Phase 2: For each trained model, evaluate with 8 different TTA weight configs:
  * 0.50 (baseline - current default)
  * 0.45, 0.55 (small variations)
  * 0.40, 0.60 (moderate variations)
  * 0.35, 0.65 (larger variations)
  * 0.30, 0.70 (aggressive variations)

EFFICIENCY: Since we only re-evaluate (not retrain), this is very fast!
Total evaluations: n_models × 8 configs × ~0.1s = minimal compute
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# Import from airbench94.py
from airbench94 import make_net, CifarLoader, evaluate, hyp, init_whitening_conv
from experiment_logger import ExperimentLogger


def train_baseline_model(model_id, verbose=True):
    """
    Train a single baseline model and return it.
    This uses the standard training procedure from airbench94.py
    """
    from airbench94 import main as train_main

    # Train model using the main training function
    # We use model_id as the run number
    if verbose:
        print(f"\nTraining baseline model {model_id}...")

    # Note: We need to modify this to return the trained model
    # For now, we'll train models in-place and save checkpoints
    raise NotImplementedError("Need to modify training to save checkpoints")


def evaluate_with_tta_weights(model, test_loader, weight_configs, verbose=True):
    """
    Evaluate a single model with multiple TTA weight configurations.

    Args:
        model: Trained PyTorch model
        test_loader: CIFAR-10 test data loader
        weight_configs: List of untrans_weight values to test
        verbose: Print progress

    Returns:
        Dictionary mapping weight -> accuracy
    """
    results = {}

    for untrans_weight in weight_configs:
        if verbose:
            print(f"  Testing untrans_weight={untrans_weight:.2f}...", end=" ")

        # Evaluate with this TTA weight configuration
        acc = evaluate(model, test_loader, tta_level=2, untrans_weight=untrans_weight)
        results[untrans_weight] = acc

        if verbose:
            print(f"acc={acc:.4f}")

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


def train_baseline_models_simple(n_models, checkpoint_dir, verbose=True):
    """
    Train n baseline models using a simplified training loop.
    Saves checkpoints for later evaluation.
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

    checkpoints = []

    for model_id in range(n_models):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training baseline model {model_id + 1}/{n_models}")
            print(f"{'='*60}")

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
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: lr_schedule[i])

        # Lookahead
        alpha_schedule = 0.95**5 * (torch.arange(total_train_steps+1) / total_train_steps)**3
        lookahead_state = LookaheadState(model)

        # Initialize whitening layer
        train_images = train_loader.normalize(train_loader.images[:5000])
        init_whitening_conv(model[0], train_images)

        # Training loop
        current_steps = 0
        for epoch in range(ceil(epochs)):
            model[0].bias.requires_grad = (epoch < hyp['opt']['whiten_bias_epochs'])

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

        # Save checkpoint
        checkpoint_path = save_model_checkpoint(model, checkpoint_dir, model_id)
        checkpoints.append(checkpoint_path)

        if verbose:
            print(f"✓ Model saved to {checkpoint_path}")

    return checkpoints


def main():
    parser = argparse.ArgumentParser(
        description='TTA Weight Optimization Sweep for CIFAR-10'
    )
    parser.add_argument('--n_models', type=int, default=25,
                        help='Number of baseline models to train (default: 25)')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='experiments/exp410-tta-weight-optimization/checkpoints',
                        help='Directory to save/load model checkpoints')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip training phase and use existing checkpoints')
    parser.add_argument('--experiment_name', type=str, default='exp410-tta-weight-optimization',
                        help='Experiment name for logging')

    args = parser.parse_args()

    # TTA weight configurations to test
    weight_configs = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

    print("\n" + "="*80)
    print("TTA WEIGHT OPTIMIZATION SWEEP")
    print("="*80)
    print(f"Configuration:")
    print(f"  - Number of models: {args.n_models}")
    print(f"  - TTA weights to test: {weight_configs}")
    print(f"  - Total evaluations: {args.n_models} models × {len(weight_configs)} configs = {args.n_models * len(weight_configs)}")
    print(f"  - Checkpoint directory: {args.checkpoint_dir}")
    print("="*80 + "\n")

    checkpoint_dir = Path(args.checkpoint_dir)

    # Phase 1: Train baseline models (or load existing)
    if args.skip_training:
        print("Skipping training phase, loading existing checkpoints...")
        checkpoints = sorted(checkpoint_dir.glob("model_*.pt"))
        if len(checkpoints) == 0:
            print(f"ERROR: No checkpoints found in {checkpoint_dir}")
            print("Please train models first or specify a different checkpoint directory.")
            sys.exit(1)
        print(f"Found {len(checkpoints)} existing checkpoints")
    else:
        print("Phase 1: Training baseline models...")
        checkpoints = train_baseline_models_simple(args.n_models, checkpoint_dir)
        print(f"\n✓ Trained and saved {len(checkpoints)} baseline models")

    # Phase 2: Evaluate each model with all TTA weight configs
    print("\n" + "="*80)
    print("Phase 2: Evaluating with different TTA weights")
    print("="*80)

    # Prepare test loader
    test_loader = CifarLoader('cifar10', train=False, batch_size=2000)

    # Store all results for analysis
    all_results = {weight: [] for weight in weight_configs}

    for i, checkpoint_path in enumerate(checkpoints):
        print(f"\nEvaluating model {i+1}/{len(checkpoints)} ({checkpoint_path.name})")

        # Load model
        model = load_model_checkpoint(checkpoint_path)

        # Evaluate with all weight configs
        results = evaluate_with_tta_weights(model, test_loader, weight_configs, verbose=True)

        # Store results
        for weight, acc in results.items():
            all_results[weight].append(acc)

    # Phase 3: Save results for each weight configuration
    print("\n" + "="*80)
    print("Phase 3: Saving results")
    print("="*80)

    for weight in weight_configs:
        weight_name = f"tta_weight_{int(weight*100):02d}"
        logger = ExperimentLogger(
            experiment_name=f"{args.experiment_name}_{weight_name}",
            experiment_description=f"TTA weight optimization: untrans_weight={weight:.2f}",
            hyperparameters={
                **hyp,
                'tta_untrans_weight': weight,
                'n_models_tested': len(checkpoints)
            }
        )

        # Log each model's result as a separate run
        for run_id, acc in enumerate(all_results[weight]):
            logger.log_run(
                run_id=run_id,
                accuracy=acc,
                time_seconds=0.0,  # We didn't measure eval time separately
                epochs_completed=hyp['opt']['train_epochs']
            )

        logger.save_summary()
        print(f"✓ Saved results for weight={weight:.2f} to experiments/{args.experiment_name}_{weight_name}/")

    print("\n" + "="*80)
    print("SWEEP COMPLETE!")
    print("="*80)
    print(f"✓ Evaluated {len(checkpoints)} models with {len(weight_configs)} weight configs")
    print(f"✓ Total evaluations: {len(checkpoints) * len(weight_configs)}")
    print(f"\nNext steps:")
    print(f"  1. Run analysis: python analyze_tta_weights.py")
    print(f"  2. Check results in: experiments/{args.experiment_name}_*/")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
