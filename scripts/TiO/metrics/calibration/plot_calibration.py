import sys
sys.path.append('/home/g15farris/bin/bayesaenet')

import uncertainty_toolbox as uct
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
from bnn_aenet.results.metrics import (
    get_proportion_lists_vectorized,
)
from uncertainty_toolbox.metrics_calibration import (
    miscalibration_area_from_proportions,
)
import torch

# Rescaling constants
e_scaling, e_shift = 0.06565926932648217, 6.6588702845000975

# Load prediction files
log_dir = '/home/g15farris/bin/bayesaenet/bnn_aenet/logs'
directories = {x.split('/')[-1]: None for x in glob.glob(f'{log_dir}/*_pred*')}
for parquet in sorted(glob.glob(f'{log_dir}/*_pred*/runs/**/*parquet', recursive=True)):
    directories[parquet.split('/')[-4]] = parquet

# Define dataset sizes and model types
sizes = ['20', '100']
bnn_models = sorted({key.split('_')[0] for key in directories if not key.startswith("de")})
models = bnn_models + ['de']

# Create subplots (rows = sizes, cols = models)
fig, axes = plt.subplots(len(sizes), len(models), figsize=(5 * len(models), 8), sharex=True, sharey=True)

# Iterate over dataset sizes and model types
for row_idx, size in enumerate(sizes):
    for col_idx, model in enumerate(models):
        ax = axes[row_idx, col_idx]

        if model != 'de':
            for name, parquet in directories.items():
                if parquet is None:
                    continue
                if not name.startswith(model) or f"_{size}_" not in name:
                    continue

                rs = pd.read_csv(parquet)
                y_true = (rs['labels'].to_numpy() / e_scaling + rs['n_atoms'].to_numpy() * e_shift) / rs['n_atoms'].to_numpy()
                y_pred = (rs['preds'].to_numpy() / e_scaling + rs['n_atoms'].to_numpy() * e_shift) / rs['n_atoms'].to_numpy()
                y_std = (rs['stds'].to_numpy() / e_scaling) / rs['n_atoms'].to_numpy()
                ep_vars = rs['ep_vars'].to_numpy() / rs['n_atoms'].to_numpy()
                al_vars = rs['al_vars'].to_numpy() / rs['n_atoms'].to_numpy()
                exp_proportions, obs_proportions = get_proportion_lists_vectorized(
                    y_pred, y_std, y_true, prop_type="interval"
                )

                ax.plot(exp_proportions, obs_proportions, label=f"{size}%", linestyle='-')
                ax.fill_between(exp_proportions, exp_proportions, obs_proportions, alpha=0.1)

        else:
            for deep_ens in sorted(glob.glob(f'{log_dir}/de_pred*')):
                if f"_{size}_" not in deep_ens:
                    continue

                y_preds = []
                y_true = None

                for parquet in glob.glob(f'{deep_ens}/**/*parquet', recursive=True):
                    rs = pd.read_csv(parquet)
                    y_true = (rs['true'].to_numpy() / e_scaling + rs['n_atoms'].to_numpy() * e_shift) / rs['n_atoms'].to_numpy()
                    y_pred = (rs['preds'].to_numpy() / e_scaling + rs['n_atoms'].to_numpy() * e_shift) / rs['n_atoms'].to_numpy()
                    y_preds.append(y_pred)

                y_preds = np.array(y_preds)
                y_pred = y_preds.mean(axis=0)
                y_std = y_preds.std(axis=0)

                exp_proportions, obs_proportions = get_proportion_lists_vectorized(
                    y_pred, y_std, y_true, prop_type="interval"
                )

                ax.plot(exp_proportions, obs_proportions, label=f"{size}%", linestyle='-')
                ax.fill_between(exp_proportions, exp_proportions, obs_proportions, alpha=0.1)

        # Add reference line
        ax.plot([0, 1], [0, 1], "--", color="gray")

        # Axis labels and styling
        if row_idx == len(sizes) - 1:
            ax.set_xlabel("Predicted Proportion")
        if col_idx == 0:
            ax.set_ylabel(f"{size}%\nObserved Proportion")
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(model.upper(), fontsize=10)
        ax.grid(True)
        ax.label_outer()

# Save figure
plt.tight_layout()
plt.savefig("/home/g15farris/bin/bayesaenet/scripts/TiO/metrics/calibration/calibration_grid.png", dpi=200)
plt.show()
