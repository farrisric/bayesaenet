import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
from scipy.stats import spearmanr
import torch.nn.functional as F
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

cwd = os.path.dirname(os.path.abspath(__file__))
logs = '/home/g15farris/bin/bayesaenet/bnn_aenet/logs'
method = 'de'  # NN/deterministic training

# QM7 specific parameters (extracted from training logs)
e_scaling = 0.9754923797786934
e_shift = -4.652443333333333

runs = glob.glob(f'{logs}/{method}_pred/runs/*')

if len(runs) == 0:
    print(f"No prediction runs found in {logs}/{method}_pred/runs/")
    print("Please run the prediction scripts first.")
    exit(1)

for run in runs:
    run_name = run.split('/')[-1]
    os.makedirs(f'{cwd}/figs_{run_name}', exist_ok=True)

    data = run + f'/NN_0_val.parquet'
    if not os.path.exists(data):
        print(f'No data for {run_name}')
        continue

    print(f'Loading data for {run_name}')
    rs = pd.read_csv(data)

    # Get total number of structures
    n_total = len(rs)
    
    # Compute approximate splits based on 80/10/10 split
    n_train = int(n_total * 0.8)
    n_valid = int(n_total * 0.1)
    n_test = n_total - n_train - n_valid
    
    train_indices = np.arange(0, n_train)
    valid_indices = np.arange(n_train, n_train + n_valid)
    test_indices = np.arange(n_train + n_valid, n_total)
    
    print(f"  Train: {len(train_indices)} structures")
    print(f"  Valid: {len(valid_indices)} structures")
    print(f"  Test:  {len(test_indices)} structures")

    points = {}
    for indices, split in zip([train_indices, valid_indices, test_indices], ['Train', 'Val', 'Test']):
        if len(indices) == 0:
            continue
            
        n_atoms = rs['n_atoms'].to_numpy()[indices]
        y_true = (rs['true'].to_numpy()[indices] / e_scaling + n_atoms * e_shift) / n_atoms
        y_pred = (rs['preds'].to_numpy()[indices] / e_scaling + n_atoms * e_shift) / n_atoms

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)

        points[split] = [y_true, y_pred, n_atoms, mse, mae, rmse]

    # Prediction vs True Values Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=200)
    axes = axes.flatten()

    for i, (split, ax) in enumerate(zip(points, axes)):
        y_true, y_pred, n_atoms, mse, mae, rmse = points[split]
        
        # Scatter plot
        sc = ax.scatter(y_true, y_pred, alpha=0.3, c=n_atoms, cmap='jet', s=10)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')
        
        ax.set_title(f'{split}\nRMSE={rmse:.4f} eV/atom, MAE={mae:.4f} eV/atom')
        ax.set_xlabel('True Energy (eV/atom)')
        ax.set_ylabel('Predicted Energy (eV/atom)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Add color bar
    cb_ax = axes[2].inset_axes([1.1, 0.05, 0.05, 0.9])
    cbar = fig.colorbar(sc, cax=cb_ax, orientation='vertical')
    cbar.set_label('Number of Atoms')

    fig.tight_layout()
    fig.savefig(f'{cwd}/figs_{run_name}/{run_name}_predictions.png')
    plt.close(fig)
    print(f"  Saved predictions plot")

    # Residuals Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=200, sharex=True, sharey=True)
    axes = axes.flatten()

    for i, (split, ax) in enumerate(zip(points, axes)):
        y_true, y_pred, n_atoms, mse, mae, rmse = points[split]
        err = y_true - y_pred
        
        sc = ax.scatter(y_pred, err, alpha=0.3, c=n_atoms, cmap='jet', s=10)
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        
        ax.set_title(f'{split}\nRMSE={rmse:.4f} eV/atom')
        ax.set_xlabel('Predicted Energy (eV/atom)')
        ax.set_ylabel('Residual (eV/atom)')
        ax.grid(True, alpha=0.3)

    # Add color bar
    cb_ax = axes[2].inset_axes([1.1, 0.05, 0.05, 0.9])
    cbar = fig.colorbar(sc, cax=cb_ax, orientation='vertical')
    cbar.set_label('Number of Atoms')

    fig.tight_layout()
    fig.savefig(f'{cwd}/figs_{run_name}/{run_name}_residuals.png')
    plt.close(fig)
    print(f"  Saved residuals plot")

    # Summary statistics
    print(f"\n  Summary for {run_name}:")
    for split in points:
        y_true, y_pred, n_atoms, mse, mae, rmse = points[split]
        print(f"    {split}: RMSE={rmse:.4f} eV/atom, MAE={mae:.4f} eV/atom")

print("\nPlotting complete!")
print(f"Figures saved in {cwd}/figs_*/")
