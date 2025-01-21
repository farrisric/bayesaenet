import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

for perc in [5] + list(range(10, 81, 10)):
    # Define paths
    input_path = f'/home/g15farris/bin/bayesaenet/bnn_aenet/logs/predict_commitie_{perc}perc/runs/*'
    output_path = '/home/g15farris/bin/bayesaenet/results/PdO/committie/metrics'
    output_file = os.path.join(output_path, f'{perc}perc.png')

    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Load Parquet files
    parquets = glob.glob(os.path.join(input_path, '*parquet'))

    if not parquets:
        raise FileNotFoundError(f"No Parquet files found in {input_path}")

    # Collect predictions
    y_preds = []
    for parquet in parquets:
        try:
            rs = pd.read_parquet(parquet)
            y_true = rs['true'].to_numpy()
            y_preds.append(rs['preds'].to_numpy())
        except Exception as e:
            print(f"Error reading {parquet}: {e}")
            
    y_preds = np.array(y_preds)
    # print(y_true[:10])

    # for i in y_preds[:,:10]:
    #     print(list(i))
    # # Calculate mean and standard deviation of predictions
    
    y_pred = np.mean(y_preds, axis=0)
    y_std = np.std(y_preds, axis=0)/rs['n_atoms'].to_numpy()

    # Calculate metrics
    mae = np.mean(np.abs(y_pred - y_true)/rs['n_atoms'])
    rmse = np.sqrt(np.mean((y_pred - y_true)/rs['n_atoms'] ** 2))

    # Create subplots
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    # Subplot 1: Predictions with error bars
    ax[0].scatter(y_true, y_pred, color='tab:blue', alpha=0.7, label=f'Predictions (MAE: {mae:.3f}\nRMSE: {rmse:.3f})')
    ax[0].errorbar(y_true, y_pred, yerr=y_std, fmt='o', color='tab:orange', alpha=0.5, label='Uncertainty (±σ)')
    ax[0].axline((0, 0), slope=1, color='gray', linestyle='--', label='Ideal Prediction')
    ax[0].set_xlabel('DFT eV/atom', fontsize=14)
    ax[0].set_ylabel('Aenet eV/atom', fontsize=14)
    ax[0].set_title('Prediction vs True Values with Uncertainty', fontsize=16)
    ax[0].legend(
        loc='upper left', 
        fontsize=10
    )
    ax[0].grid(True)

    # Subplot 2: Correlation between absolute error and std
    abs_error = np.abs(y_pred - y_true)/rs['n_atoms'].to_numpy()

    # Fit a linear regression model
    model = LinearRegression()
    y_std_reshaped = y_std.reshape(-1, 1)  # Reshape for sklearn compatibility
    model.fit(y_std_reshaped, abs_error)
    y_fit = model.predict(y_std_reshaped)

    # Compute R²
    r2 = r2_score(abs_error, y_fit)

    # Scatter plot with regression line
    ax[1].scatter(y_std, abs_error, color='tab:red', alpha=0.7, label='Error vs Std')
    ax[1].plot(y_std, y_fit, color='tab:orange', linestyle='--', label=f'Fit Line (R²: {r2:.3f})')
    ax[1].set_xlabel('Standard Deviation (σ)', fontsize=14)
    ax[1].set_ylabel('Absolute Error', fontsize=14)
    ax[1].set_title('Error vs Std with Regression Line', fontsize=16)
    ax[1].legend()
    ax[1].grid(True)
    ax[1].set_ylim(0,100)
    ax[1].set_xlim(0,100)
    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Plot saved to {output_file}")
