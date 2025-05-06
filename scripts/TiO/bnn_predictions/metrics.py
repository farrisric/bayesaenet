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
    rmsce = rms_calibration_error(y_pred, std, y_true)
    sharp = sharpness(std)
    ece = mean_absolute_calibration_error(y_pred, std, y_true).item()
    nll = gaussian_nll_loss(y_pred, y_true, torch.square(std)).item()
    
    return {'mse': mse, 'rmse': rmse, 'rmsce': rmsce, 'sharp': sharp, 'ece': ece, 'nll': nll}


def process_de_results(results):
    for train_percentage_path in sorted(glob.glob('/home/g15farris/bin/bayesaenet/bnn_aenet/logs/TiO_pred_de_*perc/')):
        parquets = glob.glob(f'{train_percentage_path}/**/*parquet', recursive=True)
        y_preds = []
        for parquet in parquets:
            rs = pd.read_csv(parquet)

            y_true = (rs['true'].to_numpy()/e_scaling + rs['n_atoms'].to_numpy()*e_shift)/rs['n_atoms'].to_numpy()
            y_pred = (rs['preds'].to_numpy()/e_scaling + rs['n_atoms'].to_numpy()*e_shift)/rs['n_atoms'].to_numpy()
            
            y_preds.append(y_pred)
            
        y_std = np.std(y_preds, axis=0)
        y_pred = y_preds.mean(axis=0)

        print(y_true.shape, y_pred.shape, y_std.shape)
        results['de'][train_percentage] = get_metrics(torch.tensor(y_true), torch.tensor(y_pred), torch.tensor(y_std))


def process_other_results(results, bnn):
    parquets = sorted(glob.glob(f'/home/g15farris/bin/bayesaenet/bnn_aenet/logs/predict_{bnn}_*perc/**/*parquet', recursive=True))

    for parquet in parquets:
        train_percentage = int(parquet.split('/')[-4].split('_')[-1].replace('perc', ''))
        rs = pd.read_csv(parquet)
        try:
            y_true = torch.tensor(rs['labels'], device='cpu')
        except:
            y_true = torch.tensor(rs['true'], device='cpu')
        y_pred = torch.tensor(rs['preds'], device='cpu')
        std = torch.tensor(rs['stds'], device='cpu')
        results[bnn][train_percentage] = get_metrics(y_true, y_pred, std)


def plot_results(results):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metrics = ['mse', 'rmse', 'rmsce', 'sharp', 'ece', 'nll']
    titles = ['MSE', 'RMSE', 'RMS Calibration Error', 'Sharpness', 'MA Calibration Error', 'NLL']

    for i, metric in enumerate(metrics):
        ax = axes[i // 3, i % 3]
        d = 0
        for bnn in results.keys():
            train_percentages = sorted(results[bnn].keys())
            values = [results[bnn][tp][metric] for tp in train_percentages]
            ax.bar(np.array(train_percentages)+d, values, label=bnn, width=2)
            d += 1
        ax.set_title(titles[i])
        ax.set_xlabel('Training Percentage')
        ax.set_ylabel(metric.upper())
    ax.legend()

    plt.tight_layout()
    plt.savefig('/home/g15farris/bin/bayesaenet/results/TiO/bnn_predictions/metrics.png')


def main():
    results = {bnn: {} for bnn in ['de', 'mcd', 'hnn', 'lrt', 'fo', 'rad']}

    process_de_results(results)

    for bnn in ['lrt', 'fo', 'rad', 'mcd', 'hnn']:
        process_other_results(results, bnn)

    plot_results(results)


if __name__ == '__main__':
    main()
