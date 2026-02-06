#!/usr/bin/env python3
"""
Run force predictions with BNN_Forces_Aux model and plot results.
"""
import sys
sys.path.insert(0, '/home/g15farris/bin/bayesaenet')
sys.path.insert(0, '/home/g15farris/bin/bayesaenet/bnn_aenet')

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import matplotlib.pyplot as plt

# Set matplotlib backend for headless
plt.switch_backend('Agg')
plt.style.use('seaborn-v0_8-whitegrid')


def load_model_and_datamodule(ckpt_path, train_in_path, device='cuda'):
    """Load model from checkpoint and create datamodule."""
    from bnn_aenet.datamodule.aenet_datamodule import AenetDataModule
    from bnn_aenet.models.bnn import BNN_Forces_Aux
    from bnn_aenet.models.nets.network import NetAtom
    
    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    hparams = ckpt['hyper_parameters']
    
    # Create datamodule
    datamodule = AenetDataModule(
        data_dir=train_in_path,
        batch_size=32,
        device=device
    )
    datamodule.setup(stage='predict')
    
    # Create network with correct sizes
    net = NetAtom(
        input_size=datamodule.input_size,
        hidden_size=datamodule.hidden_size,
        species=datamodule.species,
        active_names=datamodule.active_names,
        alpha=datamodule.alpha,
        e_scaling=datamodule.e_scaling,
        e_shift=datamodule.e_shift,
        device=device
    )
    
    # Create model
    model = BNN_Forces_Aux.load_from_checkpoint(
        ckpt_path,
        net=net,
        map_location=device
    )
    model.eval()
    model.to(device)
    
    return model, datamodule


def run_predictions(model, datamodule, mc_samples=50, device='cuda'):
    """Run predictions on all splits."""
    from lightning.pytorch import Trainer
    
    trainer = Trainer(
        accelerator='gpu' if device == 'cuda' else 'cpu',
        devices=1,
        logger=False,
        enable_progress_bar=True
    )
    
    # Run predictions
    predictions = trainer.predict(model, datamodule=datamodule)
    
    # Convert to DataFrame
    df = pd.DataFrame.from_records(predictions)
    df = df.explode(df.columns.tolist()).reset_index(drop=True)
    
    return df


def plot_results(df, output_dir):
    """Create comprehensive force prediction plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get force data
    true_forces = df['true_forces'].values
    pred_forces = df['pred_forces'].values
    std_forces = df['std_forces'].values
    
    # Check if we have force data
    has_forces = (true_forces[0] is not None and 
                  not (isinstance(true_forces[0], float) and np.isnan(true_forces[0])))
    
    if not has_forces:
        print("No force data in predictions")
        # Plot energy only
        plot_energy_only(df, output_dir)
        return
    
    # Convert to numpy arrays
    true_forces = np.array([f for f in true_forces if f is not None])
    pred_forces = np.array([f for f in pred_forces if f is not None])
    std_forces = np.array([f for f in std_forces if f is not None])
    
    # Compute metrics
    errors = np.abs(pred_forces - true_forces)
    force_mae = np.mean(errors) * 1000  # mHa/Bohr
    force_rmse = np.sqrt(np.mean((pred_forces - true_forces)**2)) * 1000  # mHa/Bohr
    
    print(f"\n=== Force Metrics ===")
    print(f"Force MAE: {force_mae:.2f} mHa/Bohr")
    print(f"Force RMSE: {force_rmse:.2f} mHa/Bohr")
    print(f"Mean uncertainty: {np.mean(std_forces)*1000:.2f} mHa/Bohr")
    
    # 1. Force Parity Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(true_forces * 1000, pred_forces * 1000, alpha=0.3, s=10, c='steelblue')
    lims = [min(true_forces.min(), pred_forces.min()) * 1000,
            max(true_forces.max(), pred_forces.max()) * 1000]
    ax.plot(lims, lims, 'k--', alpha=0.7, lw=2, label='Perfect prediction')
    ax.set_xlabel('True Forces (mHa/Bohr)', fontsize=12)
    ax.set_ylabel('Predicted Forces (mHa/Bohr)', fontsize=12)
    ax.set_title(f'Force Parity Plot\nMAE={force_mae:.2f}, RMSE={force_rmse:.2f} mHa/Bohr', fontsize=14)
    ax.legend()
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(output_dir / 'force_parity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'force_parity.png'}")
    
    # 2. Error vs Uncertainty (key uncertainty plot)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(std_forces * 1000, errors * 1000, alpha=0.3, s=10, c='steelblue')
    max_val = max(std_forces.max(), errors.max()) * 1000
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.7, lw=2, label='σ = |error|')
    ax.set_xlabel('Predicted Uncertainty σ (mHa/Bohr)', fontsize=12)
    ax.set_ylabel('Absolute Error |error| (mHa/Bohr)', fontsize=12)
    ax.set_title('Force Error vs Predicted Uncertainty', fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'force_error_vs_uncertainty.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'force_error_vs_uncertainty.png'}")
    
    # 3. Uncertainty Calibration (binned)
    fig, ax = plt.subplots(figsize=(8, 6))
    n_bins = 10
    bin_edges = np.percentile(std_forces, np.linspace(0, 100, n_bins + 1))
    bin_centers = []
    mean_errors = []
    stds_in_bins = []
    
    for i in range(n_bins):
        mask = (std_forces >= bin_edges[i]) & (std_forces < bin_edges[i+1])
        if mask.sum() > 0:
            bin_centers.append(np.mean(std_forces[mask]))
            mean_errors.append(np.mean(errors[mask]))
            stds_in_bins.append(np.std(errors[mask]))
    
    bin_centers = np.array(bin_centers) * 1000
    mean_errors = np.array(mean_errors) * 1000
    stds_in_bins = np.array(stds_in_bins) * 1000
    
    ax.errorbar(bin_centers, mean_errors, yerr=stds_in_bins, 
                fmt='o', markersize=10, capsize=5, color='steelblue', 
                ecolor='gray', elinewidth=2, capthick=2)
    max_val = max(bin_centers.max(), mean_errors.max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.7, lw=2, label='Perfect calibration')
    ax.set_xlabel('Mean Predicted Uncertainty (mHa/Bohr)', fontsize=12)
    ax.set_ylabel('Mean Absolute Error (mHa/Bohr)', fontsize=12)
    ax.set_title('Force Uncertainty Calibration', fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'force_uncertainty_calibration.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'force_uncertainty_calibration.png'}")
    
    # 4. Error and Uncertainty Distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    ax.hist(errors * 1000, bins=50, alpha=0.7, color='coral', edgecolor='black')
    ax.axvline(np.mean(errors) * 1000, color='red', linestyle='--', lw=2,
               label=f'Mean={np.mean(errors)*1000:.2f}')
    ax.axvline(np.median(errors) * 1000, color='darkred', linestyle=':', lw=2,
               label=f'Median={np.median(errors)*1000:.2f}')
    ax.set_xlabel('Absolute Error (mHa/Bohr)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Force Error Distribution', fontsize=14)
    ax.legend()
    
    ax = axes[1]
    ax.hist(std_forces * 1000, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(np.mean(std_forces) * 1000, color='blue', linestyle='--', lw=2,
               label=f'Mean={np.mean(std_forces)*1000:.2f}')
    ax.axvline(np.median(std_forces) * 1000, color='darkblue', linestyle=':', lw=2,
               label=f'Median={np.median(std_forces)*1000:.2f}')
    ax.set_xlabel('Predicted Uncertainty (mHa/Bohr)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Force Uncertainty Distribution', fontsize=14)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'force_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'force_distributions.png'}")
    
    # 5. Energy plots
    plot_energy_only(df, output_dir)


def plot_energy_only(df, output_dir):
    """Plot energy predictions."""
    output_dir = Path(output_dir)
    
    true_e = df['true'].values.astype(float)
    pred_e = df['preds'].values.astype(float)
    std_e = df['stds'].values.astype(float)
    n_atoms = df['n_atoms'].values.astype(float)
    
    # Per-atom energy
    true_e_atom = true_e / n_atoms * 1000  # meV/atom
    pred_e_atom = pred_e / n_atoms * 1000
    std_e_atom = std_e / n_atoms * 1000
    
    errors = np.abs(pred_e_atom - true_e_atom)
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean((pred_e_atom - true_e_atom)**2))
    
    print(f"\n=== Energy Metrics ===")
    print(f"Energy MAE: {mae:.2f} meV/atom")
    print(f"Energy RMSE: {rmse:.2f} meV/atom")
    
    # Energy parity
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(true_e_atom, pred_e_atom, alpha=0.5, s=30, c='steelblue')
    lims = [min(true_e_atom.min(), pred_e_atom.min()),
            max(true_e_atom.max(), pred_e_atom.max())]
    ax.plot(lims, lims, 'k--', alpha=0.7, lw=2)
    ax.set_xlabel('True Energy (meV/atom)', fontsize=12)
    ax.set_ylabel('Predicted Energy (meV/atom)', fontsize=12)
    ax.set_title(f'Energy Parity Plot\nMAE={mae:.2f}, RMSE={rmse:.2f} meV/atom', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'energy_parity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'energy_parity.png'}")
    
    # Energy uncertainty
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(std_e_atom, errors, alpha=0.5, s=30, c='steelblue')
    max_val = max(std_e_atom.max(), errors.max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.7, lw=2, label='σ = |error|')
    ax.set_xlabel('Predicted Uncertainty (meV/atom)', fontsize=12)
    ax.set_ylabel('Absolute Error (meV/atom)', fontsize=12)
    ax.set_title('Energy Error vs Uncertainty', fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'energy_error_vs_uncertainty.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'energy_error_vs_uncertainty.png'}")


def main():
    parser = argparse.ArgumentParser(description='Run force predictions and plot results')
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--train_in', type=str, 
                        default='/home/g15farris/bin/bayesaenet/data/TiO/train_forces.in',
                        help='Path to train.in file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for plots')
    parser.add_argument('--mc_samples', type=int, default=50,
                        help='Number of MC samples for uncertainty')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    args = parser.parse_args()
    
    ckpt_path = Path(args.ckpt)
    output_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent.parent / 'force_plots'
    
    print(f"=== Force Prediction & Plotting ===")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Output dir: {output_dir}")
    
    # Load model
    print("\nLoading model...")
    model, datamodule = load_model_and_datamodule(
        ckpt_path, args.train_in, args.device
    )
    
    # Run predictions
    print("\nRunning predictions...")
    model.hparams.mc_samples_eval = args.mc_samples
    df = run_predictions(model, datamodule, args.mc_samples, args.device)
    
    # Save predictions
    pred_file = output_dir / 'predictions.csv'
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(pred_file, index=False)
    print(f"Saved predictions: {pred_file}")
    
    # Plot results
    print("\nPlotting results...")
    plot_results(df, output_dir)
    
    print(f"\n=== Done! Results saved to {output_dir} ===")


if __name__ == '__main__':
    main()
