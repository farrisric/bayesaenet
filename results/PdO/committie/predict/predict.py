import sys
sys.path.append('/home/g15farris/bin/forks/bayesaenet/src')

import uncertainty_toolbox as uct
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import glob

fig, ax = plt.subplots(9, 2, figsize=(16, 6*9))
fig.tight_layout(pad=4.0)

for i, train_percentage in enumerate(
    sorted(glob.glob('/home/g15farris/bin/bayesaenet/bnn_aenet/logs/predict_commitie_*perc/')
    )):
    parquets = [x for x in glob.glob(f'{train_percentage}/**/*parquet', recursive=True)]
    y_preds = []
    for parquet in parquets:
        rs = pd.read_parquet(parquet)

        y_true = rs['true'].to_numpy()
        y_pred = rs['preds'].to_numpy()
        
        y_preds.append(y_pred)
        
    y_std = np.std(y_preds, axis=0)

    error_matrix = abs(y_true - y_preds)
    mean_error = np.mean(error_matrix, axis=0)

    test_percentage = int(100/15336*len(y_true))
    train_percentage = 90 - test_percentage

    name = f"Commitee - Train Perc: {train_percentage}%"
    
    regr = LinearRegression()
    
    x_model = abs(y_pred-y_true).reshape(-1,1)
    y_model = y_std.reshape(-1,1)
    
    regr.fit(x_model, y_model)
    y_model_pred = regr.predict(x_model)
    x_modelito = np.linspace(0, max(x_model), 100).reshape(-1,1)
    
    row = i
    
    ax[row, 0].scatter(x_model, y_model, alpha=0.3, label='Data points')
    ax[row, 0].plot(x_modelito, regr.predict(x_modelito), '--', color='tab:orange', label='Linear fit')
    ax[row, 0].grid(True)
    ax[row, 0].legend()
    
    r2 = r2_score(y_model, y_model_pred)
    print('R2 score: {:.3f}'.format(r2))
    ax[row, 0].set_title(f'{name} R2 score: {r2:.3f}', fontsize=16)
    ax[row, 0].set_xlabel('Mean Absolute Error (MAE)', fontsize=14)
    ax[row, 0].set_ylabel('Standard Deviation', fontsize=14)
    ax[row, 0].tick_params(axis='both', which='major', labelsize=12)
    
    uct.plot_calibration(y_pred, y_std, y_true, ax=ax[row, 1])
    ax[row, 1].grid(True)
    ax[row, 1].legend()
    ax[row, 1].set_title(f'Calibration Plot - {name}', fontsize=16)
    ax[row, 1].set_xlabel('Predicted Probability', fontsize=14)
    ax[row, 1].set_ylabel('Observed Frequency', fontsize=14)
    ax[row, 1].tick_params(axis='both', which='major', labelsize=12)

fig.subplots_adjust(top=0.95)
fig.suptitle('Commitee Prediction Analysis', fontsize=20)
fig.savefig(f'/home/g15farris/bin/bayesaenet/results/PdO/committie/predict/combined_corr.png', dpi=300)
