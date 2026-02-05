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
method = cwd.split('/')[-2]

e_scaling, e_shift = 0.06565926932648217, 6.6588702845000975
path_indices = '/home/g15telari/TiO/Indices/Data100/'
train_indices = np.genfromtxt(path_indices+'train_set_idxes.txt').astype(int)
valid_indices = np.genfromtxt(path_indices+'valid_set_idxes.txt').astype(int)
test_indices = np.genfromtxt(path_indices+'test_set_idxes.txt').astype(int)

runs = glob.glob(f'{logs}/{method}_pred/runs/*')

# ... (all your imports and setup remain unchanged)

for run in runs:
    run_name = run.split('/')[-1]
    os.makedirs(f'{cwd}/figs_{run_name}', exist_ok=True)

    data = run + f'/{method.upper()}_0_val.parquet'
    if not os.path.exists(data):
        print(f'No data for {run_name}')
        continue

    print(f'Loading data for {run_name}')
    rs = pd.read_csv(data)

    points = {}
    for indices, split in zip([train_indices, valid_indices, test_indices], ['Train', 'Val', 'Test']):
        n_atoms = rs['n_atoms'].to_numpy()[indices]
        y_true = (rs['true'].to_numpy()[indices] / e_scaling + n_atoms * e_shift) / n_atoms
        y_pred = (rs['preds'].to_numpy()[indices] / e_scaling + n_atoms * e_shift) / n_atoms
        y_std = (rs['stds'].to_numpy()[indices] / e_scaling) / n_atoms

        mae = mean_squared_error(y_true, y_pred)
        nll = F.gaussian_nll_loss(
            torch.tensor(y_pred),
            torch.tensor(y_true),
            torch.square(torch.tensor(y_std))
        ).item()

        points[split] = [y_true, y_pred, y_std, n_atoms, mae, nll]

    # Residuals vs Uncertainty Plot
    fig, axes = plt.subplots(1, 3, figsize=(10, 4), dpi=200, sharex=True, sharey=True)
    axes = axes.flatten()

    sc = None
    for i, (split, ax) in enumerate(zip(points, axes)):
        y_true, y_pred, y_std, n_atoms, mae, nll = points[split]
        err = y_true - y_pred
        sc = ax.scatter(y_std, err, alpha=0.3, c=n_atoms, cmap='jet', s=10)
        l = np.linspace(0, max(y_std), 100)
        for j in range(-2, 3):
            ax.plot(l, l * j, color='black', linestyle='--', alpha=0.3)
        ax.set_title(f'{split}\nMSE={mae:.4f}, NLL={nll:.4f}')
        ax.set_xlabel('Predicted Std Dev')
        ax.set_ylabel('Residual')

    # Add color bar only above the third plot (Test)
    cb_ax = axes[2].inset_axes([1.1, 0.05,  0.05, 0.9])  # [x, y, width, height] in axis coords
    cbar = fig.colorbar(sc, cax=cb_ax, orientation='vertical')
    cbar.set_label('Number of Atoms')

    # fig.suptitle(f'Residuals vs Uncertainty: {run_name}', fontsize=12)
    fig.tight_layout()
    fig.savefig(f'{cwd}/figs_{run_name}/{run_name}_residuals.png')
    plt.close(fig)

    # Uncertainty Toolbox Plots
    for split in points:
        y_true, y_pred, y_std, *_ = points[split]
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
        print(f'Finished plotting for {split} in {run_name}')