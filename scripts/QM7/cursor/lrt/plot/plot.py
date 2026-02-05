import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
from scipy.stats import spearmanr
import torch.nn.functional as F
import torch
import uncertainty_toolbox as uct
import seaborn as sns
from sklearn.metrics import mean_squared_error
import os

cwd = os.path.dirname(os.path.abspath(__file__))
logs = '/home/g15farris/bin/bayesaenet/bnn_aenet/logs'
method = 'lrt'

# QM7 specific parameters (extracted from training logs)
e_scaling = 0.9754923797786934
e_shift = -4.652443333333333

# QM7 dataset info from training logs
# Total structures: ~7200
# Train: 80%, Valid: 10%, Test: 10% (based on config)
# For now, we'll compute indices based on the data available
# You may need to adjust this based on your actual data splits

runs = glob.glob(f'{logs}/{method}_pred/runs/*')

if len(runs) == 0:
    print(f"No prediction runs found in {logs}/{method}_pred/runs/")
    print("Please run the prediction scripts first.")
    exit(1)

for run in runs:
    run_name = run.split('/')[-1]
    os.makedirs(f'{cwd}/figs_{run_name}', exist_ok=True)

    data = run + f'/{method.upper()}_0_val.parquet'
    if not os.path.exists(data):
        print(f'No data for {run_name}')
        continue

    print(f'Loading data for {run_name}')
    rs = pd.read_csv(data)

    # Get total number of structures
    n_total = len(rs)
    
    # Compute approximate splits based on 80/10/10 split
    # Note: This assumes sequential splitting. Adjust if your training used different indices
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
        
        # For BNN, we have standard deviations
        if 'stds' in rs.columns:
            y_std = (rs['stds'].to_numpy()[indices] / e_scaling) / n_atoms
        else:
            y_std = np.zeros_like(y_pred)  # If no uncertainty available

        mae = mean_squared_error(y_true, y_pred)
        
        if np.any(y_std > 0):
            nll = F.gaussian_nll_loss(
                torch.tensor(y_pred),
                torch.tensor(y_true),
                torch.square(torch.tensor(y_std))
            ).item()
        else:
            nll = float('nan')

        points[split] = [y_true, y_pred, y_std, n_atoms, mae, nll]

    # Residuals vs Uncertainty Plot
    fig, axes = plt.subplots(1, 3, figsize=(10, 4), dpi=200, sharex=True, sharey=True)
    axes = axes.flatten()

    sc = None
    for i, (split, ax) in enumerate(zip(points, axes)):
        y_true, y_pred, y_std, n_atoms, mae, nll = points[split]
        err = y_true - y_pred
        sc = ax.scatter(y_std, err, alpha=0.3, c=n_atoms, cmap='jet', s=10)
        l = np.linspace(0, max(y_std) if max(y_std) > 0 else 1, 100)
        for j in range(-2, 3):
            ax.plot(l, l * j, color='black', linestyle='--', alpha=0.3)
        ax.set_title(f'{split}\nMSE={mae:.4f}, NLL={nll:.4f}')
        ax.set_xlabel('Predicted Std Dev (eV/atom)')
        ax.set_ylabel('Residual (eV/atom)')

    # Add color bar only above the third plot (Test)
    if sc is not None:
        cb_ax = axes[2].inset_axes([1.1, 0.05, 0.05, 0.9])
        cbar = fig.colorbar(sc, cax=cb_ax, orientation='vertical')
        cbar.set_label('Number of Atoms')

    fig.tight_layout()
    fig.savefig(f'{cwd}/figs_{run_name}/{run_name}_residuals.png')
    plt.close(fig)
    print(f"  Saved residuals plot")

    # Uncertainty Toolbox Plots
    for split in points:
        y_true, y_pred, y_std, *_ = points[split]
        
        # Only create UQ plots if we have uncertainty estimates
        if np.any(y_std > 0):
            fig1, ax = plt.subplots(2, 3, figsize=(15, 9), dpi=150)
            ax = ax.flat
            uct.plot_intervals(y_pred, y_std, y_true, ax=ax[0])
            uct.plot_intervals_ordered(y_pred, y_std, y_true, ax=ax[1])
            uct.plot_calibration(y_pred, y_std, y_true, ax=ax[2])
            uct.plot_adversarial_group_calibration(y_pred, y_std, y_true, ax=ax[3])
            uct.plot_sharpness(y_std, ax=ax[4])
            uct.plot_residuals_vs_stds(y_pred, y_std, y_true, ax=ax[5])

            fig1.suptitle(f'{split} UQ Plots - {run_name}', fontsize=14)
            fig1.tight_layout()
            fig1.savefig(f'{cwd}/figs_{run_name}/{run_name}_{split}_uq.png')
            plt.close(fig1)
            print(f'  Finished UQ plotting for {split}')
        else:
            print(f'  Skipping UQ plots for {split} (no uncertainty estimates)')

print("\nPlotting complete!")
print(f"Figures saved in {cwd}/figs_*/")
