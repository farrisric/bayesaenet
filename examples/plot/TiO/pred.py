import sys
sys.path.append('/home/g15farris/bin/forks/bayesaenet/src')

#from bnn_aenet.utils.miscellaneous import ResultSaver
import uncertainty_toolbox as uct
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import glob


work_dir = '/home/g15farris/bin/forks/bayesaenet/src/logs/predict/runs/2024-07-09_09-01-00'

parquets = [x for x in glob.glob(f'{work_dir}/*parquet')] 

for parquet in parquets:
    print(parquet)
    rs = pd.read_parquet(parquet)
    #rs = ResultSaver(work_dir,parquets)

    y_true = rs['true'].to_numpy()
    y_pred = rs['preds'].to_numpy()
    y_std = rs['stds'].to_numpy()
    name = parquet.split('/')[-1].split('.')[0]
    
    regr = LinearRegression()
    
    x_model = abs(y_pred-y_true).reshape(-1,1)
    y_model = y_std.reshape(-1,1)
    
    regr.fit(x_model, y_model)
    y_model_pred = regr.predict(x_model)
    x_modelito = np.linspace(0, max(x_model), 100).reshape(-1,1)
    
    
    fig, ax = plt.subplots(1,2, figsize=(6,3), sharey=True)
    
    ax[0].scatter(x_model, y_model, alpha=0.3)
    ax[0].plot(x_modelito, regr.predict(x_modelito), '--', color='tab:orange', label='f(x)=x')
    
    
    print('R2 score: {:.3f}'.format(r2_score(y_model, y_model_pred)))
    ax[1].scatter(x_model, y_model, alpha=0.3, label='Predictions')
    
    ax[1].plot(x_modelito, regr.predict(x_modelito), '--', color='tab:orange', label='f(x)=x')
    
    ax[1].set_xlim(-0.1,1)
    ax[0].set_title('{} R2 score: {:.3f}'.format(name, r2_score(y_model, y_model_pred)))
    ax[1].set_xlabel('MAE')
    ax[0].set_xlabel('MAE')
    ax[0].set_ylabel('std. dev.')
    ax[1].legend()
    fig.tight_layout()
    #fig.savefig(f'predict/RAD_correlation_{name}_5perc.png')
    fig.savefig(f'{name}_corr.png')
    
    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    ax = ax.flat
    uct.plot_intervals(y_pred, y_std, y_true, ax=ax[0])
    uct.plot_intervals_ordered(y_pred, y_std, y_true, ax=ax[1])
    uct.plot_calibration(y_pred, y_std, y_true, ax=ax[2])
    uct.plot_adversarial_group_calibration(y_pred, y_std, y_true, ax=ax[3])
    uct.plot_sharpness(y_std, ax=ax[4])
    uct.plot_residuals_vs_stds(y_pred, y_std, y_true, ax=ax[5])
    
    
    fig.savefig(f'{name}_uct.png')
