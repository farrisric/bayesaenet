import sys
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# Define R² computation and regression line
def get_r2_score(y_true, y_pred, y_std):
    regr = LinearRegression()
    y_model = np.abs(y_pred - y_true).reshape(-1, 1)
    x_model = y_std.reshape(-1, 1)
    regr.fit(x_model, y_model)
    y_model_pred = regr.predict(x_model)
    r2 = r2_score(y_model, y_model_pred)
    cal_x = np.linspace(0, np.max(x_model), 1000).reshape(-1, 1)
    cal_y = regr.predict(cal_x)
    return r2, cal_x, cal_y

# Paths and constants
sys.path.append('/home/g15farris/bin/bayesaenet')
e_scaling, e_shift = 0.06565926932648217, 6.6588702845000975
log_dir = '/home/g15farris/bin/bayesaenet/bnn_aenet/logs'
sizes = ['20', '100']

# Load parquet paths
directories = {x.split('/')[-1]: None for x in glob.glob(f'{log_dir}/*_pred*')}
for parquet in sorted(glob.glob(f'{log_dir}/*_pred*/runs/**/*parquet', recursive=True)):
    directories[parquet.split('/')[-4]] = parquet

# Identify models
bnn_models = sorted({key.split('_')[0] for key in directories if not key.startswith("de")})
models = bnn_models + ['de']

# Find best R² runs
best_runs = {}

for model in models:
    for size in sizes:
        best_r2 = -np.inf
        best_data = None

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

            abs_error = y_true - y_pred
            r2, cal_x, cal_y = get_r2_score(y_true, y_pred, y_std)

            if r2 > best_r2:
                best_r2 = r2
                best_data = (y_true, y_pred, y_std, cal_x, cal_y, r2)

        best_runs[(model, size)] = best_data

# Plot best runs into 2x5 grid
fig, axes = plt.subplots(2, 5, figsize=(20, 8), constrained_layout=True)
axes = axes.flatten()

model_order = ['de', 'hnn', 'fo', 'lrt', 'rad']
plot_idx = 0

for size_idx, size in enumerate(sizes):
    for model in model_order:
        if (model, size) not in best_runs:
            continue
        
        y_true, y_pred, y_std, cal_x, cal_y, r2 = best_runs[(model, size)]
        abs_error = y_true - y_pred
        ax = axes[plot_idx]

        ax.scatter(y_std, abs_error, alpha=0.6, edgecolor='k', linewidth=0.2, s=10)
        
        # --- Envelope lines start ---
        sigma_range = np.linspace(0, ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else y_std.max(), 1000)
        for mult, style, label in zip([1, 2], ['--', ':'], ['1σ-Envelope', '2σ-Envelope']):
            ax.plot(sigma_range,  sigma_range-sigma_range, style, color='black', linewidth=1, label=label)
            ax.plot(sigma_range,  mult * sigma_range, style, color='black', linewidth=1, label=label)
            ax.plot(sigma_range, -mult * sigma_range, style, color='black', linewidth=1)

        if plot_idx == 0:
            ax.legend(fontsize=9)
        # --- Envelope lines end ---

        lim = 0.2#max(y_std.max(), abs_error.max())
        #ax.plot(cal_x, cal_y, linestyle='--', color='dimgray', linewidth=2)
        ax.set_xlim(0, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel(r"$\sigma$ [eV/atom]", fontsize=10)
        ax.set_ylabel("MAE [eV/atom]", fontsize=10)
        ax.set_title(f"{model.upper()} {size}%\n$R^2$={r2:.2f}", fontsize=10)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

        # Add zoom inset
        inset_ax = inset_axes(ax, width="30%", height="30%", loc="upper left")
        for mult, style, label in zip([1, 2], ['--', ':'], ['1σ-Envelope', '2σ-Envelope']):
            inset_ax.plot(sigma_range,  sigma_range-sigma_range, style, color='black', linewidth=1, label=label)
            inset_ax.plot(sigma_range,  mult * sigma_range, style, color='black', linewidth=1, label=label)
            inset_ax.plot(sigma_range, -mult * sigma_range, style, color='black', linewidth=1)
        inset_ax.scatter(y_std, abs_error, alpha=0.6, edgecolor='k', linewidth=0.2, s=10)
        zoom_lim = ax.get_xticks()[1] / 2
        #inset_ax.plot([0, zoom_lim], [0, zoom_lim], linestyle='--', color='dimgray')
        inset_ax.set_xlim(-0.005, zoom_lim)
        inset_ax.set_ylim(-zoom_lim, zoom_lim)
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        mark_inset(ax, inset_ax, loc1=2, loc2=4, fc="none", ec="0.5")

        plot_idx += 1


fig.savefig("/home/g15farris/bin/bayesaenet/scripts/TiO/metrics/error_vs_std/error_vs_std.png",
            dpi=200, bbox_inches="tight")
plt.close()
