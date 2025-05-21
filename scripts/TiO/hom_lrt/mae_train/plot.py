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

e_scaling, e_shift = 0.06565926932648217, 6.6588702845000975
path_indices = '/home/g15telari/TiO/Indices/Data100/'
train_indices = np.genfromtxt(path_indices+'train_set_idxes.txt').astype(int)
valid_indices = np.genfromtxt(path_indices+'valid_set_idxes.txt').astype(int)
test_indices = np.genfromtxt(path_indices+'test_set_idxes.txt').astype(int)

parquet_path = '/home/g15farris/bin/bayesaenet/bnn_aenet/logs/hom_lrt_pred_0.012/runs/2025-05-21_11-22-26/HLRT_0_val.parquet'
points = dict()
rs = pd.read_csv(parquet_path)
for indices, split in zip([train_indices, valid_indices, test_indices], ['Train', 'Val', 'Test']):
    # Extract and normalize data
    n_atoms = rs['n_atoms'].to_numpy()[indices]
    y_true = (rs['true'].to_numpy()[indices] / e_scaling + n_atoms * e_shift) / n_atoms
    y_pred = (rs['preds'].to_numpy()[indices] / e_scaling + n_atoms * e_shift) / n_atoms
    y_std = (rs['stds'].to_numpy()[indices] / e_scaling)  / n_atoms
    
    mae = mean_squared_error(y_true, y_pred)# float(np.mean(np.abs(y_true - y_pred)))
    nll= F.gaussian_nll_loss(
        torch.tensor(y_pred),
        torch.tensor(y_true),
        torch.square(torch.tensor(y_std))
    ).item()
    points[split] = [y_true, y_pred, y_std, n_atoms, mae, nll]
    

fig, axes = plt.subplots(1, 3, figsize=(9, 3), dpi=200, sharex=True, sharey=True)
    
axes = axes.flatten()  
  
for split, ax in zip(points, axes):
    y_true, y_pred, y_std, n_atoms, mae, nll = points[split]
    err = (y_true - y_pred)
    ax.scatter(y_std, err, alpha=0.3, c=n_atoms, cmap='jet')
    l = np.linspace(0, max(y_std), 100)
    for j in range(-2, 3):
        ax.plot(l, l*j, color='black', linestyle='--', alpha=0.2)
    ax.set_title(f'(mae={mae:.5f})'
                        f'nll={nll:.5f}')    
    
    fig.savefig('/home/g15farris/bin/bayesaenet/scripts/TiO/hom_lrt/mae_train/data.png')
    
    fig1, ax = plt.subplots(2, 3, figsize=(15, 9))
    ax = ax.flat
    uct.plot_intervals(y_pred, y_std, y_true, ax=ax[0])
    uct.plot_intervals_ordered(y_pred, y_std, y_true, ax=ax[1])
    uct.plot_calibration(y_pred, y_std, y_true, ax=ax[2])
    uct.plot_adversarial_group_calibration(y_pred, y_std, y_true, ax=ax[3])
    uct.plot_sharpness(y_std, ax=ax[4])
    uct.plot_residuals_vs_stds(y_pred, y_std, y_true, ax=ax[5])
    
    fig1.savefig(f'/home/g15farris/bin/bayesaenet/scripts/TiO/hom_lrt/mae_train/data_{split}.png')