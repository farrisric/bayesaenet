import sys
sys.path.append('/home/g15farris/bin/bayesaenet')

import uncertainty_toolbox as uct
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import warnings
from bnn_aenet.results.metrics import (rms_calibration_error, 
                                       sharpness,
                                       mean_absolute_calibration_error,
                                       gaussian_nll_loss)
import torch

warnings.filterwarnings("ignore")
e_scaling, e_shift = 0.06565926932648217, 6.6588702845000975
def get_metrics(y_true, y_pred, std):
    mse = ((y_true - y_pred) ** 2).mean().item()
    rmse = torch.sqrt(((y_true - y_pred) ** 2).mean()).item()
    rmsce = rms_calibration_error(y_pred, std, y_true)
    sharp = sharpness(std)
    ece = mean_absolute_calibration_error(y_pred, std, y_true).item()
    nll = gaussian_nll_loss(y_pred, y_true, torch.square(std)).item()
    
    print(uct.mean_absolute_calibration_error(y_pred.numpy(), std.numpy(), y_true.numpy()))
    
    return {'mse': mse, 'rmse': rmse, 'rmsce': rmsce, 'sharp': sharp, 'ece': ece, 'nll': nll}

def process_results(results, bnn):
    parquets = sorted(glob.glob(f'/home/g15farris/bin/bayesaenet/bnn_aenet/logs/*pred*_{bnn}_80perc/**/*parquet', recursive=True))
    
    for parquet in parquets:
        rs = pd.read_csv(parquet)
        try:
            y_true = (rs['labels'].to_numpy()/e_scaling + rs['n_atoms'].to_numpy()*e_shift)/rs['n_atoms'].to_numpy()
        except:
            y_true = (rs['true'].to_numpy()/e_scaling + rs['n_atoms'].to_numpy()*e_shift)/rs['n_atoms'].to_numpy()
        y_pred = (rs['preds'].to_numpy()/e_scaling + rs['n_atoms'].to_numpy()*e_shift)/rs['n_atoms'].to_numpy()
        std = torch.tensor(rs['stds'], device='cpu')
        results[bnn] = get_metrics(torch.tensor(y_true), torch.tensor(y_pred), torch.tensor(std))

def process_results_de(results, bnn):
    for train_percentage_path in sorted(glob.glob('/home/g15farris/bin/bayesaenet/bnn_aenet/logs/TiO_pred_de_*perc/')):
        parquets = glob.glob(f'{train_percentage_path}/**/*parquet', recursive=True)
        print(parquets)
        y_preds = []
        for parquet in parquets:
            rs = pd.read_csv(parquet)
            try:
                y_true = (rs['labels'].to_numpy()/e_scaling + rs['n_atoms'].to_numpy()*e_shift)/rs['n_atoms'].to_numpy()
            except:
                y_true = (rs['true'].to_numpy()/e_scaling + rs['n_atoms'].to_numpy()*e_shift)/rs['n_atoms'].to_numpy()
            y_pred = (rs['preds'].to_numpy()/e_scaling + rs['n_atoms'].to_numpy()*e_shift)/rs['n_atoms'].to_numpy()
            y_preds.append(y_pred)

        y_pred = torch.tensor(y_preds, device='cpu')
        std = torch.std(y_pred, axis=0)
        y_pred = y_pred.mean(axis=0)
        results[bnn] = get_metrics(torch.tensor(y_true), torch.tensor(y_pred), torch.tensor(std))
        
def normalize_metrics(results):
    metrics = ['mse', 'rmse', 'rmsce', 'sharp', 'ece', 'nll']
    min_vals = {metric: min(results[bnn][metric] for bnn in results) for metric in metrics}
    max_vals = {metric: max(results[bnn][metric] for bnn in results) for metric in metrics}
    
    for bnn in results:
        for metric in metrics:
            if max_vals[metric] > min_vals[metric]:  # Avoid division by zero
                results[bnn][metric] = (results[bnn][metric] - min_vals[metric]) / (max_vals[metric] - min_vals[metric])
            else:
                results[bnn][metric] = 0

def plot_radar(results):
    metrics = ['mse', 'rmse', 'rmsce', 'sharp', 'ece', 'nll']
    labels = ['MSE', 'RMSE', 'RMSCE', 'Sharpness', 'ECE', 'NLL']
    bnn_methods = results.keys()
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the radar chart
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    
    for bnn in bnn_methods:
        values = [results[bnn][metric] for metric in metrics]
        values += values[:1]
        ax.plot(angles, values, label=bnn)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_ylim(1.05, 0)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    
    plt.legend()
    plt.title("Comparison of Normalized Metrics at 80% Training")
    
    plt.savefig('/home/g15farris/bin/bayesaenet/results/TiO/bnn_predictions/radar_plot_80.png')

def main():
    results = {}
    
    for method in ['de', 'hnn', 'lrt', 'fo', 'rad']:
        if method == 'de':
            process_results_de(results, method)
        else:
            process_results(results, method)
    
    for method in results.keys():
        print(method, results[method]['sharp'], results[method]['ece'])
    normalize_metrics(results)
    plot_radar(results)

if __name__ == '__main__':
    main()
    