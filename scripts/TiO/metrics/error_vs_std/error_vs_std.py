import sys
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
import torch

# Add local library path
sys.path.append('/home/g15farris/bin/bayesaenet')
from bnn_aenet.results.metrics import (
    rms_calibration_error,
    sharpness,
    mean_absolute_calibration_error,
    gaussian_nll_loss
)

# Define metrics function
def get_metrics(y_true, y_pred, std):
    mse = ((y_true - y_pred) ** 2).mean().item()
    rmse = torch.sqrt(((y_true - y_pred) ** 2).mean()).item()
    max_error = torch.max(torch.abs(y_true - y_pred)).item()
    rmsce = rms_calibration_error(y_pred, std, y_true)
    sharp = sharpness(std)
    ece = mean_absolute_calibration_error(y_pred, std, y_true).item()
    nll = gaussian_nll_loss(y_pred, y_true, torch.square(std)).item()
    
    return {'mse': mse, 'rmse': rmse, 'maxerr': max_error,
            'rmsce': rmsce, 'sharp': sharp, 'ece': ece, 'nll': nll}

# UQ score function
def compute_uq_score(rmse, rmsce, sharpness, minmax_vals, weights=(0.5, 0.3, 0.2)):
    min_rmse, max_rmse, min_rmsce, max_rmsce, min_sharp, max_sharp = minmax_vals
    rmse_n = (rmse - min_rmse) / (max_rmse - min_rmse + 1e-8)
    rmsce_n = (rmsce - min_rmsce) / (max_rmsce - min_rmsce + 1e-8)
    sharp_n = (sharpness - min_sharp) / (max_sharp - min_sharp + 1e-8)
    a, b, c = weights
    return a * rmse_n + b * rmsce_n + c * sharp_n

# Constants
e_scaling, e_shift = 0.06565926932648217, 6.6588702845000975
log_dir = '/home/g15farris/bin/bayesaenet/bnn_aenet/logs'
sizes = ['20', '100']

# Load paths
directories = {x.split('/')[-1]: None for x in glob.glob(f'{log_dir}/*_pred*')}
for parquet in sorted(glob.glob(f'{log_dir}/*_pred*/runs/**/*parquet', recursive=True)):
    directories[parquet.split('/')[-4]] = parquet

# Models
bnn_models = sorted({key.split('_')[0] for key in directories if not key.startswith("de")})
models = bnn_models + ['de']

# Collect metrics across all runs
all_metrics = []
for model in models:
    for size in sizes:
        if model != 'de':
            model_runs = [name for name in directories if name.startswith(model) and f"_{size}_" in name]
        else:
            model_runs = [path.split('/')[-1] for path in glob.glob(f'{log_dir}/de_pred*') if f"_{size}_" in path]

        for run_name in model_runs:
            if model != 'de':
                parquet = directories[run_name]
                rs = pd.read_csv(parquet)
                n_atoms = rs['n_atoms'].to_numpy()
                y_true = (rs['labels'].to_numpy() / e_scaling + n_atoms * e_shift) / n_atoms
                y_pred = (rs['preds'].to_numpy() / e_scaling + n_atoms * e_shift) / n_atoms
                y_std = (rs['stds'].to_numpy() / e_scaling) / n_atoms
            else:
                deep_ens_dir = os.path.join(log_dir, run_name)
                y_preds = []
                for parquet in glob.glob(f'{deep_ens_dir}/**/*parquet', recursive=True):
                    rs = pd.read_csv(parquet)
                    n_atoms = rs['n_atoms'].to_numpy()
                    y_true = (rs['true'].to_numpy() / e_scaling + n_atoms * e_shift) / n_atoms
                    y_pred = (rs['preds'].to_numpy() / e_scaling + n_atoms * e_shift) / n_atoms
                    y_preds.append(y_pred)
                y_preds = np.array(y_preds)
                y_pred = y_preds.mean(axis=0)
                y_std = y_preds.std(axis=0)

            metrics = get_metrics(torch.tensor(y_true), torch.tensor(y_pred), torch.tensor(y_std))
            all_metrics.append({
                'model': model,
                'size': size,
                'run': run_name,
                'y_true': y_true,
                'y_pred': y_pred,
                'y_std': y_std,
                **metrics
            })

# Compute minmax
rmse_vals = [m['rmse'] for m in all_metrics]
rmsce_vals = [m['rmsce'] for m in all_metrics]
sharp_vals = [m['sharp'] for m in all_metrics]
minmax_vals = (min(rmse_vals), max(rmse_vals),
               min(rmsce_vals), max(rmsce_vals),
               min(sharp_vals), max(sharp_vals))

# Select best UQ run per model/size
best_runs = {}
for model in models:
    for size in sizes:
        best_score = float('inf')
        best_data = None
        for m in all_metrics:
            if m['model'] == model and m['size'] == size:
                uq = compute_uq_score(m['rmse'], m['rmsce'], m['sharp'], minmax_vals)
                if uq < best_score:
                    best_score = uq
                    best_data = m
        if best_data:
            best_runs[(model, size)] = best_data

# Plot
fig, axes = plt.subplots(2, 5, figsize=(20, 5), constrained_layout=True)
axes = axes.flatten()
model_order = ['de', 'hnn', 'fo', 'lrt', 'rad']
plot_idx = 0

for size in sizes:
    for model in model_order:
        if (model, size) not in best_runs:
            continue

        entry = best_runs[(model, size)]
        y_true = entry['y_true']
        y_pred = entry['y_pred']
        y_std = entry['y_std']
        abs_error = y_pred - y_true

        ax = axes[plot_idx]
        ax.scatter(y_std, abs_error, alpha=0.6, edgecolor='k', linewidth=0.2, s=10)

        sigma_range = np.linspace(0, 0.06, 1000)
        for mult, style in zip([1, 2], ['--', ':']):
            ax.plot(sigma_range,  mult * sigma_range, style, color='black')
            ax.plot(sigma_range, -mult * sigma_range, style, color='black')

        ax.set_xlim(0, 0.06)
        ax.set_ylim(-0.06, 0.06)
        ax.set_xlabel(r"$\sigma$ [eV/atom]")
        ax.set_ylabel("MAE [eV/atom]")
        ax.set_title(f"{model.upper()} {size}%\nUQ Score={compute_uq_score(entry['rmse'], entry['rmsce'], entry['sharp'], minmax_vals):.3f}", fontsize=10)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

        plot_idx += 1

plt.savefig("/home/g15farris/bin/bayesaenet/scripts/TiO/metrics/error_vs_std/error_vs_std_uqscore.png",
            dpi=200, bbox_inches="tight")
plt.close()
