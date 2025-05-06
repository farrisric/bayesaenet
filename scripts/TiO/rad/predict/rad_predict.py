import sys
sys.path.append('/home/g15farris/bin/forks/bayesaenet/src')

import uncertainty_toolbox as uct
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
import glob

parquets = sorted(glob.glob('/home/g15farris/bin/bayesaenet/bnn_aenet/logs/rad_pred*/**/*parquet', recursive=True))

fig, ax = plt.subplots(9, 2, figsize=(16, 9*6))
fig.tight_layout(pad=4.0)
e_scaling, e_shift = 0.06565926932648217, 6.6588702845000975

for i, parquet in enumerate(parquets):
    rs = pd.read_csv(parquet)
    y_true = (rs['labels'].to_numpy()/e_scaling + rs['n_atoms'].to_numpy()*e_shift)/rs['n_atoms'].to_numpy()
    y_pred = (rs['preds'].to_numpy()/e_scaling + rs['n_atoms'].to_numpy()*e_shift)/rs['n_atoms'].to_numpy()
    y_std = (rs['stds'].to_numpy()/e_scaling)/rs['n_atoms'].to_numpy()
    name_file = parquet.split('/')[-4]
    train_percentage = name_file.split('_')[-2]
    name = f"BNN LRT - Train Perc: {train_percentage}%"

    regr = LinearRegression()
    
    x_model = abs(y_pred-y_true).reshape(-1,1)
    y_model = y_std.reshape(-1,1)

    regr.fit(x_model, y_model)
    y_model_pred = regr.predict(x_model)
    x_modelito = np.linspace(0, max(x_model), 100).reshape(-1,1)
    
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    fig.tight_layout(pad=4.0)
    
    ax[0].scatter(x_model, y_model, alpha=0.3, label='Data points')
    ax[0].plot(x_modelito, regr.predict(x_modelito), '--', color='tab:orange', label='Linear fit')
    ax[0].grid(True)
    ax[0].legend()
    
    r2 = r2_score(y_model, y_model_pred)
    print('R2 score: {:.3f}'.format(r2))
    ax[0].set_title(f'{name} R2 score: {r2:.3f}', fontsize=16)
    ax[0].set_xlabel('Mean Absolute Error (MAE)', fontsize=14)
    ax[0].set_ylabel('Standard Deviation', fontsize=14)
    ax[0].tick_params(axis='both', which='major', labelsize=12)
    
    uct.plot_calibration(y_pred, y_std, y_true, ax=ax[1])
    ax[1].grid(True)
    ax[1].legend()
    ax[1].set_title(f'Calibration Plot - {name}', fontsize=16)
    ax[1].set_xlabel('Predicted Probability', fontsize=14)
    ax[1].set_ylabel('Observed Frequency', fontsize=14)
    ax[1].tick_params(axis='both', which='major', labelsize=12)
    
    # Calculate and print RMSE, MAE, and MAX error
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    max_error = np.max(abs(y_true - y_pred))
    
    print(f'RMSE: {rmse:.3f}')
    print(f'MAE: {mae:.3f}')
    print(f'MAX error: {max_error:.3f}')
    
    fig, ax = plt.subplots(1, 3, figsize=(24, 6))
    fig.tight_layout(pad=4.0)
    
    ax[0].scatter(x_model, y_model, alpha=0.3, label='Data points')
    ax[0].plot(x_modelito, regr.predict(x_modelito), '--', color='tab:orange', label='Linear fit')
    ax[0].grid(True)
    ax[0].legend()
    
    r2 = r2_score(y_model, y_model_pred)
    print('R2 score: {:.3f}'.format(r2))
    ax[0].set_title(f'{name} R2 score: {r2:.3f}', fontsize=16)
    ax[0].set_xlabel('Mean Absolute Error (MAE)', fontsize=14)
    ax[0].set_ylabel('Standard Deviation', fontsize=14)
    ax[0].tick_params(axis='both', which='major', labelsize=12)
    
    uct.plot_calibration(y_pred, y_std, y_true, ax=ax[1])
    ax[1].grid(True)
    ax[1].legend()
    ax[1].set_title(f'Calibration Plot - {name}', fontsize=16)
    ax[1].set_xlabel('Predicted Probability', fontsize=14)
    ax[1].set_ylabel('Observed Frequency', fontsize=14)
    ax[1].tick_params(axis='both', which='major', labelsize=12)
    
    ax[2].scatter(y_true, y_pred, alpha=0.3, label='Data points')
    ax[2].plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], '--', color='tab:orange', label='Ideal fit')
    ax[2].grid(True)
    ax[2].legend()
    ax[2].set_title(f'Parity Plot - {name}', fontsize=16)
    ax[2].set_xlabel('True Values', fontsize=14)
    ax[2].set_ylabel('Predicted Values', fontsize=14)
    ax[2].tick_params(axis='both', which='major', labelsize=12)

    # Add text annotations for RMSE, MAE, and MAX error
    textstr = '\n'.join((
        f'RMSE: {rmse:.3f} eV',
        f'MAE: {mae:.3f} eV',
        f'MAX error: {max_error:.3f} eV',
    ))
    fig.text(0.75, 0.5, textstr, ha='center', fontsize=14)
    
    fig.subplots_adjust(top=0.85)
    fig.suptitle('Commitee Prediction Analysis', fontsize=20)
    plt.savefig(f'/home/g15farris/bin/bayesaenet/scripts/TiO/rad/predict/{name_file}_{train_percentage}.png', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    ax = ax.flat
    uct.plot_intervals(y_pred, y_std, y_true, ax=ax[0])
    uct.plot_intervals_ordered(y_pred, y_std, y_true, ax=ax[1])
    uct.plot_calibration(y_pred, y_std, y_true, ax=ax[2])
    uct.plot_adversarial_group_calibration(y_pred, y_std, y_true, ax=ax[3])
    uct.plot_sharpness(y_std, ax=ax[4])
    uct.plot_residuals_vs_stds(y_pred, y_std, y_true, ax=ax[5])
    plt.savefig(f'/home/g15farris/bin/bayesaenet/scripts/TiO/rad/predict/{name_file}_{train_percentage}_uct.png', dpi=300)
