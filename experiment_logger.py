"""
Enhanced Logging System for CIFAR-10 Speedrun Experiments
Tracks experiment metadata, timing, accuracy, and GPU usage with CSV export
"""

import os
import csv
import json
import torch
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


class ExperimentLogger:
    """
    Comprehensive logging system for ML experiments.
    
    Features:
    - Experiment metadata tracking
    - Per-run logging with timing and accuracy
    - Statistical summaries (mean, median, std, min, max)
    - CSV export for analysis
    - GPU tracking
    - Aggregated results across experiments
    """
    
    def __init__(self, 
                 experiment_name: str,
                 experiment_description: str = "",
                 base_log_dir: str = "experiments",
                 hyperparameters: Optional[Dict] = None):
        """
        Initialize experiment logger.
        
        Args:
            experiment_name: Unique name for this experiment (e.g., "exp001_muon")
            experiment_description: Human-readable description
            base_log_dir: Root directory for all experiments
            hyperparameters: Dict of hyperparameters for this experiment
        """
        self.experiment_name = experiment_name
        self.experiment_description = experiment_description
        self.base_log_dir = Path(base_log_dir)
        self.hyperparameters = hyperparameters or {}
        
        # Create experiment directory
        self.experiment_dir = self.base_log_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Get GPU info
        self.gpu_info = self._get_gpu_info()
        
        # Storage for run data
        self.runs: List[Dict[str, Any]] = []
        self.start_time = datetime.now()
        
        # Create metadata file
        self._save_metadata()
        
    def _get_gpu_info(self) -> Dict[str, str]:
        """Extract GPU information from nvidia-smi."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,driver_version,memory.total', 
                 '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                info = result.stdout.strip().split(',')
                return {
                    'gpu_name': info[0].strip(),
                    'driver_version': info[1].strip(),
                    'memory_total': info[2].strip()
                }
        except Exception as e:
            print(f"Warning: Could not get GPU info: {e}")
        
        return {
            'gpu_name': 'Unknown',
            'driver_version': 'Unknown',
            'memory_total': 'Unknown'
        }
    
    def _save_metadata(self):
        """Save experiment metadata to JSON."""
        metadata = {
            'experiment_name': self.experiment_name,
            'description': self.experiment_description,
            'start_time': self.start_time.isoformat(),
            'gpu_info': self.gpu_info,
            'hyperparameters': self.hyperparameters,
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        }
        
        metadata_path = self.experiment_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def log_run(self, 
                run_id: int,
                accuracy: float,
                time_seconds: float,
                train_loss: Optional[float] = None,
                epochs_completed: Optional[float] = None,
                additional_metrics: Optional[Dict] = None):
        """
        Log a single training run.
        
        Args:
            run_id: Unique identifier for this run
            accuracy: Final test accuracy (0-1 scale)
            time_seconds: Total training time in seconds
            train_loss: Final training loss (optional)
            epochs_completed: Number of epochs completed (optional)
            additional_metrics: Dict of additional metrics to log
        """
        run_data = {
            'run_id': run_id,
            'accuracy': float(accuracy),
            'time_seconds': float(time_seconds),
            'train_loss': float(train_loss) if train_loss is not None else None,
            'epochs_completed': float(epochs_completed) if epochs_completed is not None else None,
            'achieved_target': accuracy >= 0.94,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add any additional metrics
        if additional_metrics:
            run_data.update(additional_metrics)
        
        self.runs.append(run_data)
        
        # Append to per-run CSV immediately (for real-time monitoring)
        self._append_to_runs_csv(run_data)
    
    def _append_to_runs_csv(self, run_data: Dict):
        """Append a single run to the per-run CSV file."""
        csv_path = self.experiment_dir / 'runs_detailed.csv'
        
        # Create header if file doesn't exist
        file_exists = csv_path.exists()
        
        with open(csv_path, 'a', newline='') as f:
            # Get all keys from run_data
            fieldnames = list(run_data.keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(run_data)
    
    def compute_statistics(self) -> Dict[str, Any]:
        """Compute comprehensive statistics across all runs."""
        if not self.runs:
            return {}
        
        # Extract data
        accuracies = torch.tensor([r['accuracy'] for r in self.runs])
        times = torch.tensor([r['time_seconds'] for r in self.runs])
        achieved_target = torch.tensor([r['achieved_target'] for r in self.runs])
        
        # Basic statistics
        stats = {
            'num_runs': len(self.runs),
            'accuracy_mean': float(accuracies.mean()),
            'accuracy_std': float(accuracies.std()),
            'accuracy_min': float(accuracies.min()),
            'accuracy_max': float(accuracies.max()),
            'accuracy_median': float(accuracies.median()),
            'time_mean': float(times.mean()),
            'time_std': float(times.std()),
            'time_min': float(times.min()),
            'time_max': float(times.max()),
            'time_median': float(times.median()),
            'success_rate': float(achieved_target.float().mean()),
            'num_successful': int(achieved_target.sum()),
        }
        
        # Statistics for successful runs (>=94% accuracy)
        if achieved_target.any():
            successful_times = times[achieved_target]
            successful_accs = accuracies[achieved_target]
            
            stats.update({
                'successful_time_mean': float(successful_times.mean()),
                'successful_time_std': float(successful_times.std()),
                'successful_time_min': float(successful_times.min()),
                'successful_time_median': float(successful_times.median()),
                'successful_accuracy_mean': float(successful_accs.mean()),
                'best_run_time': float(successful_times.min()),
                'best_run_id': int(torch.argmin(successful_times)),
            })
        
        return stats
    
    def save_summary(self):
        """Save summary statistics and final results."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        stats = self.compute_statistics()
        
        # Calculate GPU hours used
        total_training_time_seconds = sum(r['time_seconds'] for r in self.runs)
        a100_hours_used = total_training_time_seconds / 3600
        
        summary = {
            'experiment_name': self.experiment_name,
            'description': self.experiment_description,
            'start_time': self.start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'total_duration_seconds': duration,
            'gpu_info': self.gpu_info,
            'a100_hours_used': a100_hours_used,
            'statistics': stats,
            'hyperparameters': self.hyperparameters,
        }
        
        # Save as JSON
        summary_path = self.experiment_dir / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Also save PyTorch checkpoint format
        torch_log = {
            'experiment_name': self.experiment_name,
            'stats': stats,
            'runs': self.runs,
            'hyperparameters': self.hyperparameters,
        }
        torch.save(torch_log, self.experiment_dir / 'results.pt')
        
        return summary
    
    def print_summary(self):
        """Print formatted summary to console."""
        stats = self.compute_statistics()
        
        print("\n" + "="*80)
        print(f"EXPERIMENT SUMMARY: {self.experiment_name}")
        print("="*80)
        print(f"Description: {self.experiment_description}")
        print(f"GPU: {self.gpu_info['gpu_name']}")
        print(f"Number of runs: {stats['num_runs']}")
        print()
        print("ACCURACY STATISTICS:")
        print(f"  Mean:   {stats['accuracy_mean']:.4f} ± {stats['accuracy_std']:.4f}")
        print(f"  Median: {stats['accuracy_median']:.4f}")
        print(f"  Range:  [{stats['accuracy_min']:.4f}, {stats['accuracy_max']:.4f}]")
        print()
        print("TIME STATISTICS:")
        print(f"  Mean:   {stats['time_mean']:.4f}s ± {stats['time_std']:.4f}s")
        print(f"  Median: {stats['time_median']:.4f}s")
        print(f"  Range:  [{stats['time_min']:.4f}s, {stats['time_max']:.4f}s]")
        print()
        print(f"SUCCESS RATE (≥94% accuracy): {stats['success_rate']*100:.1f}% "
              f"({stats['num_successful']}/{stats['num_runs']})")
        
        if stats.get('best_run_time'):
            print()
            print("BEST RUN (≥94% accuracy):")
            print(f"  Time:    {stats['best_run_time']:.4f}s (Run #{stats['best_run_id']})")
            print(f"  Mean:    {stats['successful_time_mean']:.4f}s")
            print(f"  Median:  {stats['successful_time_median']:.4f}s")
        
        print("="*80)


class ExperimentAggregator:
    """
    Aggregate results across multiple experiments for comparison.
    """
    
    def __init__(self, base_log_dir: str = "experiments"):
        self.base_log_dir = Path(base_log_dir)
        
    def load_experiment(self, experiment_name: str) -> Dict:
        """Load summary for a single experiment."""
        summary_path = self.base_log_dir / experiment_name / 'summary.json'
        
        if not summary_path.exists():
            raise FileNotFoundError(f"No summary found for {experiment_name}")
        
        with open(summary_path, 'r') as f:
            return json.load(f)
    
    def aggregate_experiments(self, experiment_names: List[str]) -> None:
        """
        Create aggregated comparison CSV across multiple experiments.
        
        Args:
            experiment_names: List of experiment names to compare
        """
        aggregated_data = []
        
        for exp_name in experiment_names:
            try:
                summary = self.load_experiment(exp_name)
                stats = summary['statistics']
                
                row = {
                    'experiment_name': exp_name,
                    'description': summary.get('description', ''),
                    'num_runs': stats['num_runs'],
                    'gpu': summary['gpu_info']['gpu_name'],
                    'a100_hours_used': summary.get('a100_hours_used', 0),
                    'accuracy_mean': stats['accuracy_mean'],
                    'accuracy_std': stats['accuracy_std'],
                    'accuracy_min': stats['accuracy_min'],
                    'accuracy_max': stats['accuracy_max'],
                    'time_mean': stats['time_mean'],
                    'time_std': stats['time_std'],
                    'time_min': stats['time_min'],
                    'time_median': stats['time_median'],
                    'success_rate': stats['success_rate'],
                    'best_time_94plus': stats.get('successful_time_min', None),
                    'mean_time_94plus': stats.get('successful_time_mean', None),
                }
                
                # Add key hyperparameters if available
                hyp = summary.get('hyperparameters', {})
                if 'opt' in hyp:
                    row['lr'] = hyp['opt'].get('lr', None)
                    row['momentum'] = hyp['opt'].get('momentum', None)
                    row['batch_size'] = hyp['opt'].get('batch_size', None)
                
                aggregated_data.append(row)
                
            except Exception as e:
                print(f"Warning: Could not load {exp_name}: {e}")
        
        # Save aggregated CSV
        csv_path = self.base_log_dir / 'experiments_comparison.csv'
        
        if aggregated_data:
            fieldnames = list(aggregated_data[0].keys())
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(aggregated_data)
            
            print(f"\nAggregated results saved to: {csv_path}")
            
            # Also print comparison table
            self._print_comparison_table(aggregated_data)
    
    def _print_comparison_table(self, data: List[Dict]):
        """Print formatted comparison table."""
        print("\n" + "="*120)
        print("EXPERIMENTS COMPARISON")
        print("="*120)
        print(f"{'Experiment':<25} {'Runs':<6} {'Acc Mean':<10} {'Time Mean':<12} "
              f"{'Best Time':<11} {'Success %':<10} {'A100-hrs':<10}")
        print("-"*120)
        
        for row in data:
            best_time = f"{row['best_time_94plus']:.4f}s" if row['best_time_94plus'] else "N/A"
            print(f"{row['experiment_name']:<25} "
                  f"{row['num_runs']:<6} "
                  f"{row['accuracy_mean']:.4f}    "
                  f"{row['time_mean']:.4f}s     "
                  f"{best_time:<11} "
                  f"{row['success_rate']*100:>6.1f}%    "
                  f"{row['a100_hours_used']:>6.4f}")
        
        print("="*120)


# Example usage function
def example_usage():
    """
    Example of how to use the ExperimentLogger in your training script.
    """
    
    # Initialize logger for your experiment
    logger = ExperimentLogger(
        experiment_name="exp001_muon",
        experiment_description="Testing Muon optimizer with lr=0.02",
        hyperparameters={
            'opt': {
                'optimizer': 'muon',
                'lr': 0.02,
                'momentum': 0.95,
                'batch_size': 1024,
            }
        }
    )
    
    # Run multiple training runs
    for run_id in range(25):
        # Your training code here
        accuracy = 0.94 + torch.randn(1).item() * 0.01  # Example
        time_seconds = 4.1 + torch.randn(1).item() * 0.2  # Example
        train_loss = 0.25  # Example
        
        # Log the run
        logger.log_run(
            run_id=run_id,
            accuracy=accuracy,
            time_seconds=time_seconds,
            train_loss=train_loss,
            epochs_completed=9.9
        )
    
    # Save summary and print results
    logger.save_summary()
    logger.print_summary()
    
    # Later: aggregate multiple experiments
    aggregator = ExperimentAggregator()
    aggregator.aggregate_experiments([
        'exp001_muon',
        'exp002_baseline',
        'exp003_lr_sweep',
    ])


if __name__ == "__main__":
    example_usage()
