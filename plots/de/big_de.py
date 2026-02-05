import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
from scipy.stats import spearmanr
import torch.nn.functional as F
import torch
import uncertainty_toolbox as uct
import seaborn as sns
from sklearn.metrics import mean_squared_error
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

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

    # Compute quartile thresholds
    q3_error = np.percentile(errors, 75)
    q3_uncertainty = np.percentile(uncertainties, 75)

    # Boolean masks
    high_error = errors > q3_error
    high_uncertainty = uncertainties > q3_uncertainty
    high_both = high_error & high_uncertainty

    n_overlap = np.sum(high_both)
    n_high_uncertainty = np.sum(high_uncertainty)

    percent_overlap = 100 * n_overlap / n_high_uncertainty
   
    return percent_overlap

def read(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [float(line.split('=')[1]) for line in lines if 'model.obs_scale' in line]

e_scaling, e_shift = 0.06565926932648217, 6.6588702845000975
path_indices = '/home/riccardo/tmp/Data100/'
train_indices = np.genfromtxt(path_indices+'train_set_idxes.txt').astype(int)
valid_indices = np.genfromtxt(path_indices+'valid_set_idxes.txt').astype(int)
test_indices = np.genfromtxt(path_indices+'test_set_idxes.txt').astype(int)

points = {}
for run in sorted(glob.glob('DE/de_pred_100_*/runs/2025-03*', recursive=True)):
     
    points[run_name] = {}
    parquet_path = glob.glob(f"{run}/**/*.parquet", recursive=True)   
    
    y_preds = []
    
    for parquet in parquet_path:
        rs = pd.read_csv(parquet)
        n_atoms = rs['n_atoms'].to_numpy()
        y_true_all = (rs['true'].to_numpy()/e_scaling + n_atoms*e_shift)/n_atoms
        y_pred = (rs['preds'].to_numpy()/e_scaling + n_atoms*e_shift)/n_atoms
        y_preds.append(y_pred)
    y_pred_all = np.array(y_preds).mean(axis=0)
    y_std_all = np.std(y_preds, axis=0)
    
    split='Test'
        
    mae = mean_squared_error(y_true_all, y_pred_all)# float(np.mean(np.abs(y_true_all - y_pred_all)))
    nll_scaled = F.gaussian_nll_loss(
        torch.tensor(y_pred_all),
        torch.tensor(y_true_all),
        torch.square(torch.tensor(y_std_all))
    ).item()
    points[run_name][split] = [y_true_all, y_pred_all, y_std_all, mae, nll_scaled, n_atoms]

    a = analyze_uncertainty_and_error(y_true_all, y_pred_all, y_std_all)
    mae = abs(y_true_all - y_pred_all).mean()
    print('de', 'big', a, mae)
    
    