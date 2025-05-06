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
import torch

e_scaling, e_shift = 0.06565926932648217, 6.6588702845000975

def get_metrics(y_true, y_pred, std):
    mse = ((y_true - y_pred) ** 2).mean().item()
    rmse = torch.sqrt(((y_true - y_pred) ** 2).mean()).item()
    max_error = torch.max(torch.abs(y_true - y_pred)).item()
    rmsce = rms_calibration_error(y_pred, std, y_true)
    sharp = sharpness(std)
    ece = mean_absolute_calibration_error(y_pred, std, y_true).item()
    nll = gaussian_nll_loss(y_pred, y_true, torch.square(std)).item()
    r2 = get_r2_score(y_true, y_pred, std)
    
    return {'mse': mse,
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

outfile =  open(f'/home/g15farris/bin/bayesaenet/scripts/TiO/metrics/metrics.txt', 'w')
outfile.write('Model\tSize\tR2\tRMSE\tMAE\tMAXE\tRMSCE\tSharpness\tECE\tNLL\n')

log_dir = '/home/g15farris/bin/bayesaenet/bnn_aenet/logs'
directories = {x.split('/')[-1] : None for x in glob.glob(f'{log_dir}/*_pred*')}

for parquet in sorted(glob.glob(f'{log_dir}/*_pred*/runs/**/*parquet', recursive=True)):
    directories[parquet.split('/')[-4]] = parquet


for parquet in directories.values():
    name_file = parquet.split('/')[-4]
    model = name_file.split('_')[0]
    train_percentage = name_file.split('_')[-2]
    
    if model == 'de':
        continue
        
    rs = pd.read_csv(parquet)
    y_true = (rs['labels'].to_numpy()/e_scaling + rs['n_atoms'].to_numpy()*e_shift)/rs['n_atoms'].to_numpy()
    y_pred = (rs['preds'].to_numpy()/e_scaling + rs['n_atoms'].to_numpy()*e_shift)/rs['n_atoms'].to_numpy()
    y_std = (rs['stds'].to_numpy()/e_scaling)/rs['n_atoms'].to_numpy()
    
    metrics = get_metrics(torch.tensor(y_true), torch.tensor(y_pred), torch.tensor(y_std))
    
    out = f"{model}\t{train_percentage}\t{metrics['r2score']}\t{metrics['rmse']}\t{metrics['mse']}\t{metrics['maxerr']}\
        \t{metrics['rmsce']}\t{metrics['sharp']}\t{metrics['ece']}\t{metrics['nll']}\n"
    outfile.write(out)
    
for deep_ens in sorted(glob.glob('/home/g15farris/bin/bayesaenet/bnn_aenet/logs/de_pred*')):
    y_preds = []
    name_file = deep_ens.split('/')[-1]
    model = name_file.split('_')[0]
    train_percentage = name_file.split('_')[-2]
    print(name_file)
    for parquet in glob.glob(f'{deep_ens}/**/*parquet', recursive=True):
        rs = pd.read_csv(parquet)
        y_true = (rs['true'].to_numpy()/e_scaling + rs['n_atoms'].to_numpy()*e_shift)/rs['n_atoms'].to_numpy()
        y_pred = (rs['preds'].to_numpy()/e_scaling + rs['n_atoms'].to_numpy()*e_shift)/rs['n_atoms'].to_numpy()
        y_preds.append(y_pred)
        
    y_std = np.std(y_preds, axis=0)
    y_pred = np.array(y_preds).mean(axis=0)
    
    metrics = get_metrics(torch.tensor(y_true), torch.tensor(y_pred), torch.tensor(y_std))
    
    out = f"{model}\t{train_percentage}\t{metrics['r2score']}\t{metrics['rmse']}\t{metrics['mse']}\t{metrics['maxerr']}\
        \t{metrics['rmsce']}\t{metrics['sharp']}\t{metrics['ece']}\t{metrics['nll']}\n"
        
    outfile.write(out)
    