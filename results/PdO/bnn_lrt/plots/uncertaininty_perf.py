import sys
sys.path.append('/home/g15farris/bin/forks/bayesaenet/src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
import uncertainty_toolbox as uct
import glob 

# Modified functions to plot on axes
def compute_correlation(y_true, y_pred, y_std):
    abs_error = abs(y_pred - y_true)
    pearson_corr, p_value_pearson = pearsonr(y_std.flatten(), abs_error)
    spearman_corr, p_value_spearman = spearmanr(y_std.flatten(), abs_error)
    return abs_error, pearson_corr, p_value_pearson, spearman_corr, p_value_spearman

def regression_analysis(abs_error, y_std):
    regr = LinearRegression()
    regr.fit(abs_error.reshape(-1, 1), y_std.reshape(-1, 1))
    y_pred_std = regr.predict(abs_error.reshape(-1, 1))
    r2 = r2_score(y_std, y_pred_std)
    return regr, r2

def plot_relationship(abs_error, y_std, regr, ax):
    x_modelito = np.linspace(0, max(abs_error), 100).reshape(-1, 1)
    ax.scatter(abs_error, y_std, alpha=0.5, color='tab:blue', label='Data points')
    ax.plot(x_modelito, regr.predict(x_modelito), '--', color='tab:orange', label='Linear fit')
    ax.set_xlabel('Absolute Error (|Prediction - True|)', fontsize=12)
    ax.set_ylabel('Predicted Uncertainty (σ)', fontsize=12)
    ax.set_title('Uncertainty vs Absolute Error', fontsize=14)
    ax.grid(True)
    ax.legend()

def plot_calibration(y_pred, y_std, y_true, ax):
    uct.plot_calibration(y_pred, y_std, y_true, ax=ax)
    ax.set_title('Calibration Plot', fontsize=14)

def plot_normalized_errors(y_pred, y_true, y_std, ax):
    normalized_error = (y_pred - y_true) / (y_std + 1.9663562574521685)
    ax.hist(normalized_error, bins=50, density=True, alpha=0.7, color='tab:green', label='Normalized Errors')
    ax.axvline(0, color='k', linestyle='--', label='Mean')
    ax.set_title('Histogram of Normalized Errors', fontsize=14)
    ax.set_xlabel('(Prediction - True) / Uncertainty', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend()

def compute_uce(y_pred, y_std, y_true):
    return uct.mean_absolute_calibration_error(y_pred, y_std, y_true)

# Master function to create and save combined plots with annotations
def analyze_uncertainty(y_true, y_pred, y_std, train_percentage, output_dir):
    abs_error, pearson_corr, p_value_pearson, spearman_corr, p_value_spearman = compute_correlation(y_true, y_pred, y_std)
    regr, r2 = regression_analysis(abs_error, y_std)
    uce = compute_uce(y_pred, y_std, y_true)

    # Create combined plot
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    plot_relationship(abs_error, y_std, regr, ax[0])
    plot_calibration(y_pred, y_std, y_true, ax[1])
    plot_normalized_errors(y_pred, y_true, y_std, ax[2])

    # Add annotations with text output
    text_output = (
        f"Pearson Corr: {pearson_corr:.3f} (p={p_value_pearson:.3e})   "
        f"Spearman Corr: {spearman_corr:.3f} (p={p_value_spearman:.3e})\n"
        f"R²: {r2:.3f}   "
        f"UCE: {uce:.3f}"
    )
    fig.text(0.5, 0.005, text_output, ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

    fig.suptitle(f'BNN LRT Train Percentage: {train_percentage}%', fontsize=16)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # Save figure
    output_path = f'{output_dir}/combined_plot_train_{train_percentage}.png'
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Saved plot for {train_percentage}% train data: {output_path}")

# Main loop
if __name__ == "__main__":
    output_dir = '/home/g15farris/bin/bayesaenet/results/PdO/bnn_lrt/plots'
    parquets = sorted(glob.glob('/home/g15farris/bin/bayesaenet/bnn_aenet/logs/predict_lrt_*perc/**/*parquet', recursive=True))

    for i, parquet in enumerate(parquets):
        rs = pd.read_parquet(parquet)

        y_true = rs['true'].to_numpy()
        y_pred = rs['preds'].to_numpy()
        y_std = rs['stds'].to_numpy() - 1.9663562574521685
        
        # Compute train percentage
        test_percentage = int(100 / 15336 * len(y_true))
        train_percentage = 90 - test_percentage

        analyze_uncertainty(y_true, y_pred, y_std, train_percentage, output_dir)
