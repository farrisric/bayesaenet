#!/usr/bin/env python3
"""
Plot force training results - errors and uncertainties.
Run after training completes.
"""
import sys
sys.path.insert(0, '/home/g15farris/bin/bayesaenet')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import glob

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10


def load_tensorboard_scalars(log_dir: Path):
    """Load scalar metrics from tensorboard event files."""
    from tensorboard.backend.event_processing import event_accumulator
    
    event_files = list(log_dir.glob('**/events.out.tfevents.*'))
    if not event_files:
        print(f"No event files found in {log_dir}")
        return {}
    
    # Use the most recent event file
    event_file = sorted(event_files)[-1]
    
    ea = event_accumulator.EventAccumulator(str(event_file.parent))
    ea.Reload()
    
    scalars = {}
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        scalars[tag] = {
            'steps': [e.step for e in events],
            'values': [e.value for e in events]
        }
    
    return scalars


def plot_training_curves(scalars: dict, output_dir: Path):
    """Plot training curves for energy and force metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Energy RMSE
    ax = axes[0, 0]
    if 'rmse/train' in scalars:
        ax.plot(scalars['rmse/train']['steps'], scalars['rmse/train']['values'], 
                label='Train', alpha=0.8)
    if 'rmse/val' in scalars:
        ax.plot(scalars['rmse/val']['steps'], scalars['rmse/val']['values'], 
                label='Validation', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Energy RMSE (meV/atom)')
    ax.set_title('Energy RMSE During Training')
    ax.legend()
    ax.set_yscale('log')
    
    # Force RMSE
    ax = axes[0, 1]
    if 'force_rmse/train' in scalars:
        ax.plot(scalars['force_rmse/train']['steps'], scalars['force_rmse/train']['values'], 
                label='Train', alpha=0.8, color='C2')
    if 'force_rmse/val' in scalars:
        ax.plot(scalars['force_rmse/val']['steps'], scalars['force_rmse/val']['values'], 
                label='Validation', alpha=0.8, color='C3')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Force RMSE (mHa/Bohr)')
    ax.set_title('Force RMSE During Training')
    ax.legend()
    ax.set_yscale('log')
    
    # ELBO
    ax = axes[1, 0]
    if 'elbo/train' in scalars:
        ax.plot(scalars['elbo/train']['steps'], scalars['elbo/train']['values'], 
                label='Train ELBO', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('ELBO')
    ax.set_title('ELBO (Energy Loss)')
    ax.legend()
    
    # KL Divergence
    ax = axes[1, 1]
    if 'kl/train' in scalars:
        ax.plot(scalars['kl/train']['steps'], scalars['kl/train']['values'], 
                label='KL Divergence', alpha=0.8, color='C4')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('KL')
    ax.set_title('KL Divergence')
    ax.legend()
    
    plt.tight_layout()
    output_path = output_dir / 'training_curves.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def load_predictions(pred_dir: Path):
    """Load prediction results from parquet files."""
    pred_files = list(pred_dir.glob('*.parquet'))
    if not pred_files:
        # Try CSV
        pred_files = list(pred_dir.glob('*.csv'))
    
    if not pred_files:
        print(f"No prediction files found in {pred_dir}")
        return None
    
    dfs = []
    for f in pred_files:
        try:
            df = pd.read_parquet(f)
        except:
            df = pd.read_csv(f)
        df['split'] = f.stem.split('_')[-1]  # e.g., 'train', 'val', 'test'
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True) if dfs else None


def plot_force_parity(df: pd.DataFrame, output_dir: Path, split='val'):
    """Plot force parity plot - predicted vs true forces."""
    df_split = df[df['split'] == split] if 'split' in df.columns else df
    
    if 'true_forces' not in df_split.columns or 'pred_forces' not in df_split.columns:
        print("No force data in predictions")
        return
    
    # Get force data - handle various formats
    true_forces = df_split['true_forces'].values
    pred_forces = df_split['pred_forces'].values
    
    # Flatten if needed
    if isinstance(true_forces[0], (list, np.ndarray)):
        true_forces = np.concatenate([np.array(f).flatten() for f in true_forces if f is not None])
        pred_forces = np.concatenate([np.array(f).flatten() for f in pred_forces if f is not None])
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Scatter plot
    ax.scatter(true_forces, pred_forces, alpha=0.3, s=10, c='steelblue')
    
    # Perfect prediction line
    lims = [min(true_forces.min(), pred_forces.min()),
            max(true_forces.max(), pred_forces.max())]
    ax.plot(lims, lims, 'k--', alpha=0.7, label='Perfect prediction')
    
    # Compute metrics
    mae = np.mean(np.abs(true_forces - pred_forces))
    rmse = np.sqrt(np.mean((true_forces - pred_forces)**2))
    
    ax.set_xlabel('True Forces (Ha/Bohr)')
    ax.set_ylabel('Predicted Forces (Ha/Bohr)')
    ax.set_title(f'Force Parity Plot ({split})\nMAE={mae*1000:.2f} mHa/Bohr, RMSE={rmse*1000:.2f} mHa/Bohr')
    ax.legend()
    ax.set_aspect('equal')
    
    plt.tight_layout()
    output_path = output_dir / f'force_parity_{split}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_force_uncertainty(df: pd.DataFrame, output_dir: Path, split='val'):
    """Plot force uncertainty analysis."""
    df_split = df[df['split'] == split] if 'split' in df.columns else df
    
    if 'std_forces' not in df_split.columns:
        print("No force uncertainty data")
        return
    
    true_forces = df_split['true_forces'].values
    pred_forces = df_split['pred_forces'].values
    std_forces = df_split['std_forces'].values
    
    # Flatten
    if isinstance(true_forces[0], (list, np.ndarray)):
        true_forces = np.concatenate([np.array(f).flatten() for f in true_forces if f is not None])
        pred_forces = np.concatenate([np.array(f).flatten() for f in pred_forces if f is not None])
        std_forces = np.concatenate([np.array(f).flatten() for f in std_forces if f is not None])
    
    errors = np.abs(pred_forces - true_forces)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Error vs Uncertainty
    ax = axes[0]
    ax.scatter(std_forces * 1000, errors * 1000, alpha=0.3, s=10, c='steelblue')
    ax.plot([0, std_forces.max() * 1000], [0, std_forces.max() * 1000], 'k--', 
            alpha=0.7, label='σ = |error|')
    ax.set_xlabel('Predicted Uncertainty (mHa/Bohr)')
    ax.set_ylabel('Absolute Error (mHa/Bohr)')
    ax.set_title('Force Error vs Uncertainty')
    ax.legend()
    
    # 2. Uncertainty distribution
    ax = axes[1]
    ax.hist(std_forces * 1000, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(np.mean(std_forces) * 1000, color='red', linestyle='--', 
               label=f'Mean={np.mean(std_forces)*1000:.2f}')
    ax.set_xlabel('Predicted Uncertainty (mHa/Bohr)')
    ax.set_ylabel('Count')
    ax.set_title('Force Uncertainty Distribution')
    ax.legend()
    
    # 3. Calibration - binned error vs uncertainty
    ax = axes[2]
    n_bins = 10
    bin_edges = np.percentile(std_forces, np.linspace(0, 100, n_bins + 1))
    bin_centers = []
    mean_errors = []
    
    for i in range(n_bins):
        mask = (std_forces >= bin_edges[i]) & (std_forces < bin_edges[i+1])
        if mask.sum() > 0:
            bin_centers.append(np.mean(std_forces[mask]))
            mean_errors.append(np.mean(errors[mask]))
    
    ax.scatter(np.array(bin_centers) * 1000, np.array(mean_errors) * 1000, 
               s=100, c='steelblue', edgecolor='black', zorder=5)
    ax.plot([0, max(bin_centers) * 1000], [0, max(bin_centers) * 1000], 'k--', 
            alpha=0.7, label='Perfect calibration')
    ax.set_xlabel('Mean Predicted Uncertainty (mHa/Bohr)')
    ax.set_ylabel('Mean Absolute Error (mHa/Bohr)')
    ax.set_title('Force Uncertainty Calibration')
    ax.legend()
    
    plt.tight_layout()
    output_path = output_dir / f'force_uncertainty_{split}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_force_components(df: pd.DataFrame, output_dir: Path, split='val'):
    """Plot force errors by component (x, y, z)."""
    df_split = df[df['split'] == split] if 'split' in df.columns else df
    
    if 'true_forces' not in df_split.columns:
        return
    
    true_forces = df_split['true_forces'].values
    pred_forces = df_split['pred_forces'].values
    
    # Reshape to (N, 3) if needed
    if isinstance(true_forces[0], (list, np.ndarray)):
        true_forces = np.vstack([np.array(f).reshape(-1, 3) for f in true_forces if f is not None])
        pred_forces = np.vstack([np.array(f).reshape(-1, 3) for f in pred_forces if f is not None])
    
    if true_forces.ndim == 1:
        true_forces = true_forces.reshape(-1, 3)
        pred_forces = pred_forces.reshape(-1, 3)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    components = ['x', 'y', 'z']
    
    for i, (ax, comp) in enumerate(zip(axes, components)):
        ax.scatter(true_forces[:, i], pred_forces[:, i], alpha=0.3, s=10)
        
        lims = [min(true_forces[:, i].min(), pred_forces[:, i].min()),
                max(true_forces[:, i].max(), pred_forces[:, i].max())]
        ax.plot(lims, lims, 'k--', alpha=0.7)
        
        mae = np.mean(np.abs(true_forces[:, i] - pred_forces[:, i])) * 1000
        ax.set_xlabel(f'True F_{comp} (Ha/Bohr)')
        ax.set_ylabel(f'Predicted F_{comp} (Ha/Bohr)')
        ax.set_title(f'Force {comp}-component\nMAE={mae:.2f} mHa/Bohr')
        ax.set_aspect('equal')
    
    plt.tight_layout()
    output_path = output_dir / f'force_components_{split}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Plot force training results')
    parser.add_argument('--log_dir', type=str, 
                        default='/home/g15farris/bin/bayesaenet/bnn_aenet/logs/train/runs/tio_forces_long',
                        help='Training log directory')
    parser.add_argument('--pred_dir', type=str, default=None,
                        help='Prediction directory (optional)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for plots')
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir) if args.output_dir else log_dir / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Log directory: {log_dir}")
    print(f"Output directory: {output_dir}")
    
    # Plot training curves from tensorboard
    tb_dir = log_dir / 'tensorboard'
    if tb_dir.exists():
        print("\n=== Loading Tensorboard Logs ===")
        scalars = load_tensorboard_scalars(tb_dir)
        if scalars:
            print(f"Found metrics: {list(scalars.keys())}")
            plot_training_curves(scalars, output_dir)
    
    # Plot predictions if available
    pred_dir = Path(args.pred_dir) if args.pred_dir else None
    if pred_dir is None:
        # Try to find prediction directory
        pred_dirs = list(Path('/home/g15farris/bin/bayesaenet/bnn_aenet/logs/pred/runs').glob('tio_forces*'))
        if pred_dirs:
            pred_dir = sorted(pred_dirs)[-1]
    
    if pred_dir and pred_dir.exists():
        print(f"\n=== Loading Predictions from {pred_dir} ===")
        df = load_predictions(pred_dir)
        if df is not None:
            print(f"Loaded {len(df)} predictions")
            print(f"Columns: {list(df.columns)}")
            
            for split in ['train', 'val', 'test']:
                if 'split' not in df.columns or split in df['split'].values:
                    plot_force_parity(df, output_dir, split)
                    plot_force_uncertainty(df, output_dir, split)
                    plot_force_components(df, output_dir, split)
    else:
        print("\nNo prediction data found. Run predictions first.")
    
    print(f"\n=== Done! Plots saved to {output_dir} ===")


if __name__ == '__main__':
    main()
