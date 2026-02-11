#!/usr/bin/env python3
"""Example: Make predictions with uncertainty estimates.

This script demonstrates how to load a trained BNN model and make
predictions with uncertainty quantification.

Usage:
    python examples/predict_with_uncertainty.py --checkpoint path/to/model.ckpt
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT / "bnn_aenet"))


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load a trained BNN model from checkpoint."""
    from bnn_aenet.models import BNN
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Get hyperparameters
    hparams = ckpt.get('hyper_parameters', {})
    
    # Initialize model
    model = BNN(
        input_size=hparams.get('input_size', {}),
        hidden=hparams.get('hidden', [64, 64]),
        output_size=hparams.get('output_size', 1),
        lr=hparams.get('lr', 0.001),
        prior_scale=hparams.get('prior_scale', 0.1),
        q_scale=hparams.get('q_scale', 0.001),
        obs_scale=hparams.get('obs_scale', 0.5),
        mc_samples_train=hparams.get('mc_samples_train', 2),
        mc_samples_eval=hparams.get('mc_samples_eval', 50),
        dataset_size=hparams.get('dataset_size', 1000),
        guide_type=hparams.get('guide_type', 'lrt'),
    )
    
    # Load weights
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.to(device)
    model.eval()
    
    return model, hparams


def predict_with_uncertainty(model, x, logic, n_samples=50):
    """Make predictions with Monte Carlo sampling for uncertainty.
    
    Args:
        model: Trained BNN model
        x: Input features
        logic: Logic tensor (for atomic summing)
        n_samples: Number of MC samples
    
    Returns:
        mean: Mean prediction
        std: Standard deviation (uncertainty)
        samples: All MC samples
    """
    samples = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            # Forward pass with sampled weights
            pred = model(x, logic)
            samples.append(pred.cpu().numpy())
    
    samples = np.array(samples)
    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0)
    
    return mean, std, samples


def plot_predictions(y_true, y_pred, y_std, output_path=None):
    """Create prediction plot with uncertainty."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Parity plot
    ax1.errorbar(y_true, y_pred, yerr=y_std, fmt='o', alpha=0.5,
                 markersize=4, capsize=0)
    
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax1.plot(lims, lims, 'k--', alpha=0.5)
    ax1.set_xlabel('True Energy (eV/atom)')
    ax1.set_ylabel('Predicted Energy (eV/atom)')
    ax1.set_title('Predictions with Uncertainty')
    ax1.set_aspect('equal')
    
    # Uncertainty calibration
    errors = np.abs(y_true - y_pred)
    ax2.scatter(y_std, errors, alpha=0.5)
    ax2.plot([0, y_std.max()], [0, y_std.max()], 'k--', alpha=0.5, label='y=x')
    ax2.plot([0, y_std.max()], [0, 2*y_std.max()], 'r--', alpha=0.5, label='y=2x')
    ax2.set_xlabel('Predicted Uncertainty (σ)')
    ax2.set_ylabel('Absolute Error')
    ax2.set_title('Error vs Uncertainty')
    ax2.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Predict with BNN uncertainty")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--data-dir", type=str, default="data/QM7/train.in",
                        help="Path to data directory or train.in file")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--n-samples", type=int, default=50,
                        help="Number of MC samples")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for plot")
    
    args = parser.parse_args()
    
    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"
    
    print(f"Loading model from {args.checkpoint}")
    model, hparams = load_model(args.checkpoint, args.device)
    
    print("\nModel loaded successfully!")
    print(f"Guide type: {hparams.get('guide_type', 'unknown')}")
    print(f"MC samples for evaluation: {args.n_samples}")
    
    # Load test data
    from datamodule.aenet_datamodule import AenetDataModule
    
    datamodule = AenetDataModule(data_dir=args.data_dir)
    datamodule.setup(stage='test')
    
    # Get test dataloader
    test_loader = datamodule.test_dataloader()
    
    # Collect predictions
    all_y_true = []
    all_y_pred = []
    all_y_std = []
    
    print("\nMaking predictions...")
    for batch in test_loader:
        # Move to device
        x = batch['x'].to(args.device)
        y = batch['y'].to(args.device)
        logic = batch['logic'].to(args.device)
        
        # Predict with uncertainty
        mean, std, _ = predict_with_uncertainty(model, x, logic, args.n_samples)
        
        all_y_true.extend(y.cpu().numpy().flatten())
        all_y_pred.extend(mean.flatten())
        all_y_std.extend(std.flatten())
    
    # Convert to arrays
    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)
    y_std = np.array(all_y_std)
    
    # Compute metrics
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mae = np.mean(np.abs(y_true - y_pred))
    
    print(f"\nResults:")
    print(f"  RMSE: {rmse:.4f} eV/atom")
    print(f"  MAE:  {mae:.4f} eV/atom")
    print(f"  Mean uncertainty: {np.mean(y_std):.4f}")
    
    # Plot
    plot_predictions(y_true, y_pred, y_std, args.output)


if __name__ == "__main__":
    main()
