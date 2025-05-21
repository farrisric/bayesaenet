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
models = ['lrt', 'fo', 'rad']
output_dir = '/home/g15farris/bin/bayesaenet/scripts/TiO/metrics/error_vs_std/grids'
os.makedirs(output_dir, exist_ok=True)

# Load parquet paths
directories = {x.split('/')[-1]: None for x in glob.glob(f'{log_dir}/*_pred*')}
for parquet in sorted(glob.glob(f'{log_dir}/*_pred*/runs/**/*parquet', recursive=True)):
    directories[parquet.split('/')[-4]] = parquet

# Loop through models and training sizes
for model in models:
    for size in sizes:
        model_runs = [name for name in directories if name.startswith(model) and f"_{size}_" in name]
        model_runs = sorted(model_runs)[:5]  # Take only the first 5 runs

        if len(model_runs) < 5:
            print(f"Skipping {model} {size} — only {len(model_runs)} runs found.")
            continue

        fig, axs = plt.subplots(5, 3, figsize=(15, 10), sharex=True, sharey=True)
        uncertainty_labels = ['Total', 'Epistemic', 'Aleatoric']

        for i, run_name in enumerate(model_runs):
            parquet = directories[run_name]
            rs = pd.read_csv(parquet)
            n_atoms = rs['n_atoms'].to_numpy()
            y_true = (rs['labels'].to_numpy() / e_scaling + n_atoms * e_shift) / n_atoms
            y_pred = (rs['preds'].to_numpy() / e_scaling + n_atoms * e_shift) / n_atoms
            abs_error = y_pred - y_true

            std = (rs['stds'].to_numpy() / e_scaling) / n_atoms
            ep = (np.sqrt(rs['ep_vars'].to_numpy()) / e_scaling) / n_atoms
            al = (np.sqrt(rs['al_vars'].to_numpy()) / e_scaling) / n_atoms

            all_uncertainties = [std, ep, al]

            for j, (uncert, label) in enumerate(zip(all_uncertainties, uncertainty_labels)):
                ax = axs[i, j]
                ax.scatter(uncert, abs_error, alpha=0.6, s=8, edgecolor='k', linewidth=0.2)

                # Envelope lines ±1σ and ±2σ
                sigma_range = np.linspace(0, 0.06, 1000)
                for mult, style in zip([1, 2], ['--', ':']):
                    ax.plot(sigma_range,  mult * sigma_range, style, color='black', linewidth=1)
                    ax.plot(sigma_range, -mult * sigma_range, style, color='black', linewidth=1)

                ax.set_xlim(0, 0.06)
                ax.set_ylim(-0.06, 0.06)
                if i == 4:
                    ax.set_xlabel(f"{label} σ [eV/atom]")
                if j == 0:
                    ax.set_ylabel(f"Run {i+1}\nError [eV/atom]")
                ax.grid(True, linestyle="--", alpha=0.5)


        fig.suptitle(f"{model.upper()} - {size}% (5 runs × 3 uncertainties)", fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fname = f"{model}_{size}_grid.png"
        fig.savefig(os.path.join(output_dir, fname), dpi=150)
        plt.close()
