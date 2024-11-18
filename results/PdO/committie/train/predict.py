#from bnn_aenet.utils.miscellaneous import ResultSaver
import uncertainty_toolbox as uct
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import glob


work_dir = '/home/g15farris/bin/bayesaenet/bnn_aenet/logs/predict/runs/2024-11-14_15-16-07'
work_dir = '/home/g15farris/bin/bayesaenet/bnn_aenet/logs/predict/runs/2024-11-14_17-24-21'

parquets = [x for x in glob.glob(f'{work_dir}/*parquet')] 

y_true_matrix = []
y_pred_matrix = []

for parquet in parquets:
    rs = pd.read_parquet(parquet)
    y_true_matrix.append(rs['true'].to_numpy())
    y_pred_matrix.append(rs['preds'].to_numpy())

y_true_matrix = np.array(y_true_matrix)
y_pred_matrix = np.array(y_pred_matrix)

y_true_std = np.std(y_true_matrix, axis=0)
y_pred_std = np.std(y_pred_matrix, axis=0)

error_matrix = abs(y_pred_matrix - y_true_matrix)
mean_error = np.mean(error_matrix, axis=0)

perc_test = int(100/15336*y_true_matrix.shape[1])
prec_train = 90 - perc_test

regr = LinearRegression()
    
x_model = mean_error.reshape(-1,1)
y_model = y_pred_std.reshape(-1,1)

regr.fit(x_model, y_model)
y_model_pred = regr.predict(x_model)
x_modelito = np.linspace(0, max(x_model), 100).reshape(-1,1)

fig, ax = plt.subplots(1, figsize=(6,3), sharey=True)

ax.scatter(x_model, y_model, alpha=0.3)
ax.plot(x_modelito, regr.predict(x_modelito), '--', color='tab:orange', label='f(x)=x')

ax.set_title('Perc. Train. {}% R2 score: {:.3f}'.format(prec_train, r2_score(y_model, y_model_pred)))
ax.set_xlabel('MAE')
ax.set_ylabel('std. dev.')
fig.tight_layout()
    
fig.savefig('mean_error_vs_std.png')
