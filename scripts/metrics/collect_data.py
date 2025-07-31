import sys
sys.path.append('/home/g15farris/bin/bayesaenet')

import uncertainty_toolbox as uct
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
import glob
from bnn_aenet.results.metrics import (rms_calibration_error, 
                                       sharpness,
                                       mean_absolute_calibration_error,
                                       gaussian_nll_loss)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
from scipy.stats import spearmanr
import torch.nn.functional as F
import torch
import uncertainty_toolbox as uct
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

def get_metrics(y_true, y_pred, std):
    mae = mean_absolute_error(y_true, y_pred)
    mse = ((y_true - y_pred) ** 2).mean().item()
    rmse = torch.sqrt(((y_true - y_pred) ** 2).mean()).item()
    max_error = torch.max(torch.abs(y_true - y_pred)).item()
    rmsce = rms_calibration_error(y_pred, std, y_true).item()
    sharp = sharpness(std).item()
    ece = mean_absolute_calibration_error(y_pred, std, y_true).item()
    nll = gaussian_nll_loss(y_pred, y_true, torch.square(std)).item()
    r2 = get_r2_score(y_true, y_pred, std)
    
    return {'mae': mae,
            'rmse': rmse,
            'maxerr' : max_error,
            'r2score': r2,
            'rmsce': rmsce,
            'sharp': sharp,
            'ece': ece,
            'nll': nll}

def get_r2_score(y_true, y_pred, y_std):
    regr = LinearRegression()
    x_model = abs(y_pred-y_true).reshape(-1,1)
    y_model = y_std.reshape(-1,1)

    regr.fit(x_model, y_model)
    y_model_pred = regr.predict(x_model)
    r2 = r2_score(y_model, y_model_pred)
    return r2



def analyze_uncertainty_and_error(y_true, y_pred, y_std):
    """
    Analyze uncertainty and error, compute quartile thresholds, and generate scatter plot.

    Parameters:
    y_true (numpy.ndarray): True values.
    y_pred (numpy.ndarray): Predicted values.
    y_std (numpy.ndarray): Standard deviations of predictions.

    Returns:
    float: Percentage of high-uncertainty points falling in the top error quartile.
    """
    errors = abs(y_true - y_pred)
    uncertainties = y_std

    q3_error = np.percentile(errors, 75)
    q3_uncertainty = np.percentile(uncertainties, 75)

    high_error = errors > q3_error
    high_uncertainty = uncertainties > q3_uncertainty
    high_both = high_error & high_uncertainty

    n_overlap = np.sum(high_both)
    n_high_uncertainty = np.sum(high_uncertainty)

    percent_overlap = 100 * n_overlap / n_high_uncertainty
    return percent_overlap

cwd = os.path.dirname(os.path.abspath(__file__))
logs = '/home/g15farris/bin/bayesaenet/bnn_aenet/logs'

e_scaling, e_shift = 0.06565926932648217, 6.6588702845000975
path_indices = '/home/g15telari/TiO/Indices/Data100/'
train_indices = np.genfromtxt(path_indices+'train_set_idxes.txt').astype(int)
valid_indices = np.genfromtxt(path_indices+'valid_set_idxes.txt').astype(int)
test_indices = np.genfromtxt(path_indices+'test_set_idxes.txt').astype(int)

methods = ['lrt', 'fo', 'rad']
sizes = ['big',  'small']

all_metrics_df = {x : [] for x in ['Train', 'Val', 'Test']}

for method in methods:
    runs = glob.glob(f'{logs}/{method}_pred/runs/*best*')
    for run in runs:
        if run.split('/')[-1].split('_')[1] == 'small':
            size = 'small'
        else:
            size = 'big'
            
        print(f"Processing: {run}")
        data = run + f'/{method.upper()}_0_val.parquet'
        rs = pd.read_csv(data)
        print(run)
        for indices, split in zip([train_indices, valid_indices, test_indices], ['Train', 'Val', 'Test']):
            n_atoms = rs['n_atoms'].to_numpy()[indices]
            y_true = (rs['true'].to_numpy()[indices] / e_scaling + n_atoms * e_shift) / n_atoms
            y_pred = (rs['preds'].to_numpy()[indices] / e_scaling + n_atoms * e_shift) / n_atoms
            y_std = (rs['stds'].to_numpy()[indices] / e_scaling) / n_atoms

            metrics = get_metrics(torch.tensor(y_true), torch.tensor(y_pred), torch.tensor(y_std))
            metrics['Overlap'] = analyze_uncertainty_and_error(y_true, y_pred, y_std)
            metrics['Method'] = method
            metrics['Size'] = size
            metrics['Run'] = os.path.basename(run[-1])
            metrics['Split'] = split
            all_metrics_df[split].append(metrics)
                
# Create DataFrame and save to CSV
for split in all_metrics_df:
    metrics_df = pd.DataFrame(all_metrics_df[split])
    metrics_df.to_csv(f"/home/g15farris/bin/bayesaenet/scripts/metrics/uq_metrics_{split}.csv", index=False)
    print("Metrics saved to uq_metrics_summary.csv")
