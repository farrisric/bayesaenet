import sys
sys.path.append('/home/g15farris/bin/forks/bayesaenet/src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
import uncertainty_toolbox as uct

parquets = sorted(glob.glob('/home/g15farris/bin/bayesaenet/bnn_aenet/logs/predict_lrt_*perc/**/*parquet', recursive=True))

for i, parquet in enumerate(parquets):
    rs = pd.read_parquet(parquet)

    y_true = rs['true'].to_numpy()
    y_pred = rs['preds'].to_numpy()
    y_std = rs['stds'].to_numpy() - 1.9663562574521685

# Example: Replace these with your actual arrays
# y_true = np.array([...])
# y_pred = np.array([...])
# y_std = np.array([...])

# Step 1: Correlation Coefficients
def compute_correlation(y_true, y_pred, y_std):
    abs_error = abs(y_pred - y_true)
    pearson_corr, p_value_pearson = pearsonr(y_std.flatten(), abs_error)
    spearman_corr, p_value_spearman = spearmanr(y_std.flatten(), abs_error)
    print(f"Pearson correlation: {pearson_corr:.3f} (p-value: {p_value_pearson:.3e})")
    print(f"Spearman correlation: {spearman_corr:.3f} (p-value: {p_value_spearman:.3e})")
    return abs_error, pearson_corr, spearman_corr

# Step 2: Regression Analysis
def regression_analysis(abs_error, y_std):
    regr = LinearRegression()
    regr.fit(abs_error.reshape(-1, 1), y_std.reshape(-1, 1))
    y_pred_std = regr.predict(abs_error.reshape(-1, 1))
    r2 = r2_score(y_std, y_pred_std)
    print(f"R² score: {r2:.3f}")
    return regr, r2

# Step 3: Visualization
def plot_relationship(abs_error, y_std, regr):
    plt.figure(figsize=(8, 6))
    plt.scatter(abs_error, y_std, alpha=0.5, color='tab:blue', label='Data points')
    x_modelito = np.linspace(0, max(abs_error), 100).reshape(-1, 1)
    plt.plot(x_modelito, regr.predict(x_modelito), '--', color='tab:orange', label='Linear fit')
    plt.xlabel('Absolute Error (|Prediction - True|)', fontsize=14)
    plt.ylabel('Predicted Uncertainty (σ)', fontsize=14)
    plt.title('Uncertainty vs Absolute Error', fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.show()

# Step 4: Reliability Diagram
def plot_calibration(y_pred, y_std, y_true):
    uct.plot_calibration(y_pred, y_std, y_true)
    plt.title('Calibration Plot', fontsize=16)
    plt.show()

# Step 5: Histogram of Normalized Errors
def plot_normalized_errors(y_pred, y_true, y_std):
    normalized_error = (y_pred - y_true) / y_std
    plt.figure(figsize=(8, 6))
    plt.hist(normalized_error, bins=50, density=True, alpha=0.7, color='tab:green', label='Normalized Errors')
    plt.axvline(0, color='k', linestyle='--', label='Mean')
    plt.title('Histogram of Normalized Errors', fontsize=16)
    plt.xlabel('(Prediction - True) / Uncertainty', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.legend()
    plt.show()

# Step 6: Uncertainty Calibration Error (UCE)
def compute_uce(y_pred, y_std, y_true):
    uce = uct.get_uncertainty_calibration_error(y_pred, y_std, y_true)
    print(f"Uncertainty Calibration Error (UCE): {uce:.3f}")
    return uce

# Master Function to Run All Checks
def analyze_uncertainty(y_true, y_pred, y_std):
    print("Analyzing Uncertainty and Error Correlation...\n")
    abs_error, pearson_corr, spearman_corr = compute_correlation(y_true, y_pred, y_std)
    regr, r2 = regression_analysis(abs_error, y_std)
    plot_relationship(abs_error, y_std, regr)
    plot_calibration(y_pred, y_std, y_true)
    plot_normalized_errors(y_pred, y_true, y_std)
    uce = compute_uce(y_pred, y_std, y_true)

    # Summary
    print("\nSummary:")
    print(f"Pearson Correlation: {pearson_corr:.3f}")
    print(f"Spearman Correlation: {spearman_corr:.3f}")
    print(f"R² score: {r2:.3f}")
    print(f"Uncertainty Calibration Error (UCE): {uce:.3f}")

# Run the analysis
# Replace y_true, y_pred, and y_std with your actual data
# analyze_uncertainty(y_true, y_pred, y_std)
