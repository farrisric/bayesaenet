import sys
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

# Paths and constants
sys.path.append('/home/g15farris/bin/bayesaenet')
e_scaling, e_shift = 0.06565926932648217, 6.6588702845000975
log_dir = '/home/g15farris/bin/bayesaenet/bnn_aenet/logs'
sizes = ['20', '100']
models = ['hnn']

# Load parquet paths
directories = {x.split('/')[-1]: None for x in glob.glob(f'{log_dir}/*_pred*')}
for parquet in sorted(glob.glob(f'{log_dir}/*_pred*/runs/**/*parquet', recursive=True)):
    directories[parquet.split('/')[-4]] = parquet

# Plotting setup: 3 rows (total, aleatoric, epistemic), 2 columns (20% and 100%)
fig, axs = plt.subplots(3, 2, figsize=(12, 10), sharex=True, sharey=True)
uncertainty_types = ['total', 'aleatoric', 'epistemic']

for i, size in enumerate(sizes):
    y_trues = []
    mus = []
    sigmas = []

    model_runs = [name for name in directories if name.startswith('hnn') and f"_{size}_" in name]
    if len(model_runs) < 2:
        print(f"Not enough runs for hnn {size}%")
        continue

    for run_name in model_runs:
        csv_path = directories[run_name]
        rs = pd.read_csv(csv_path)
        n_atoms = rs['n_atoms'].to_numpy()
        y_true = (rs['labels'].to_numpy() / e_scaling + n_atoms * e_shift) / n_atoms
        y_pred = (rs['preds'].to_numpy() / e_scaling + n_atoms * e_shift) / n_atoms
        y_std = (rs['stds'].to_numpy() / e_scaling) / n_atoms

        y_trues.append(y_true)
        mus.append(y_pred)
        sigmas.append(y_std)

    y_trues = np.array(y_trues[0])  # Same across ensemble
    mus = np.stack(mus)
    sigmas = np.stack(sigmas)

    mu_bar = np.mean(mus, axis=0)
    epistemic = np.mean((mus - mu_bar) ** 2, axis=0)
    aleatoric = np.mean(sigmas ** 2, axis=0)
    total_uncertainty = epistemic + aleatoric
    error = mu_bar - y_trues

    for j, (uncertainty, label) in enumerate(zip(
        [total_uncertainty, aleatoric, epistemic],
        ['Total', 'Aleatoric', 'Epistemic']
    )):
        ax = axs[j, i]
        ax.scatter(np.sqrt(uncertainty), error, alpha=0.6, s=10)
        x_vals = np.linspace(0, np.sqrt(np.max(uncertainty)), 100)
        for k, ls in zip([1, 2], ['--', ':']):
            ax.plot(x_vals, k * x_vals, ls, color='black', label=f'{k}σ-Envelope' if (k == 1 and i == 0) else None)
            ax.plot(x_vals, -k * x_vals, ls, color='black')
        r2 = r2_score(np.abs(error), np.sqrt(uncertainty))
        ax.set_title(f'{size}% - {label} ($R^2$ = {r2:.2f})')
        if i == 0:
            ax.set_ylabel(r'Error [eV/atom]')
        ax.set_xlim(0, 0.06)
        ax.set_ylim(-0.06, 0.06)
        if j == 2:
            ax.set_xlabel(r'$\sigma$ [eV/atom]')
        if j == 0 and i == 0:
            ax.legend()

plt.suptitle('Uncertainty Calibration: Deep Ensemble of HNNs', fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('/home/g15farris/bin/bayesaenet/scripts/TiO/metrics/error_vs_std/hnn_errors/de_hnn_uncertainty_all.png', dpi=300)
plt.show()
