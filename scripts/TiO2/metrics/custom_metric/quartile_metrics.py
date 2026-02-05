import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import pandas as pd


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
    # print(f"{percent_overlap:.2f}% of high-uncertainty points fall in the top error quartile.")
    return percent_overlap
    
for method in methods:
    for size in sizes:
        if size == 'big':
            runs = glob.glob(f'{logs}/{method}_pred/runs/*[0-9]')
        else:
            runs = glob.glob(f'{logs}/{method}_pred/runs/*{size}')
        
        for run in runs:
            data = run + f'/{method.upper()}_0_val.parquet'
            rs = pd.read_csv(data)
            for indices, split in zip([train_indices, valid_indices, test_indices], ['Train', 'Val', 'Test']):
                if split != 'Test':
                    continue
                n_atoms = rs['n_atoms'].to_numpy()[indices]
                y_true = (rs['true'].to_numpy()[indices] / e_scaling + n_atoms * e_shift) / n_atoms
                y_pred = (rs['preds'].to_numpy()[indices] / e_scaling + n_atoms * e_shift) / n_atoms
                y_std = (rs['stds'].to_numpy()[indices] / e_scaling) / n_atoms

                
                percent_overlap = analyze_uncertainty_and_error(y_true, y_pred, y_std)
                mae = abs(y_true - y_pred).mean()
                print(method, size, percent_overlap, mae)

          
errors = abs(y_true - y_pred)   
uncertainties = y_std           
# Compute quartile thresholds
q3_error = np.percentile(errors, 75)
q3_uncertainty = np.percentile(uncertainties, 75)

# Boolean masks
high_error = errors > q3_error
high_uncertainty = uncertainties > q3_uncertainty
high_both = high_error & high_uncertainty

high_both = high_error & high_uncertainty
n_overlap = np.sum(high_both)
n_high_uncertainty = np.sum(high_uncertainty)

percent_overlap = 100 * n_overlap / n_high_uncertainty
print(f"{percent_overlap:.2f}% of high-uncertainty points fall in the top error quartile.")
# Scatter plot with color coding
plt.figure(figsize=(8, 6))
sns.scatterplot(x=uncertainties, y=errors, hue=high_both, palette={True: "red", False: "gray"}, alpha=0.7)
plt.axvline(q3_uncertainty, color='blue', linestyle='--', label='Q3 Uncertainty')
plt.axhline(q3_error, color='green', linestyle='--', label='Q3 Error')
plt.title("Uncertainty vs Error with Top Quartile Highlight")
plt.xlabel("Uncertainty")
plt.ylabel("Absolute Error")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
plt.savefig(f'{cwd}/test.png')
plt.close()

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Histogram for uncertainties
ax[0].hist(uncertainties, bins=30, color='skyblue', edgecolor='black')
ax[0].axvline(q3_uncertainty, color='red', linestyle='--', label='Q3 Uncertainty')
ax[0].set_title("Histogram of Uncertainty Estimates")
ax[0].set_xlabel("Uncertainty")
ax[0].set_ylabel("Frequency")
ax[0].legend()
ax[0].grid(True)

# Histogram for errors
ax[1].hist(errors, bins=30, color='salmon', edgecolor='black')
ax[1].axvline(q3_error, color='red', linestyle='--', label='Q3 Error')
ax[1].set_title("Histogram of Error Estimates")
ax[1].set_xlabel("Error")
ax[1].set_ylabel("Frequency")
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.savefig(f'{cwd}/hist.png')

# Calculate histogram data
counts, bin_edges = np.histogram(uncertainties, bins=100, density=True)
cdf = np.cumsum(counts)
cdf_normalized = cdf / cdf[-1] *100 # normalize to 1

perc_tresh = 70
threshold_uncertainty = np.percentile(uncertainties, perc_tresh)
# Extract data above the threshold
high_uncertainty_mask = uncertainties > threshold_uncertainty
high_uncertainty_values = uncertainties[high_uncertainty_mask]

fig, ax1 = plt.subplots(figsize=(8, 6))

# Histogram on primary y-axis
color_hist = 'salmon'
ax1.hist(uncertainties, bins=30, color=color_hist, edgecolor='black')
ax1.set_xlabel("Uncertainty")
ax1.set_ylabel("Frequency", color=color_hist)
ax1.tick_params(axis='y', labelcolor=color_hist)
ax1.grid(True)

# Twin axis for CDF
ax2 = ax1.twinx()
color_cdf = 'blue'
ax2.plot(bin_edges[1:], cdf_normalized, color=color_cdf, label='CDF of Uncertainty')
ax2.axvline(threshold_uncertainty, color='red', linestyle='--', label='70% CDF Threshold')
ax2.set_ylabel("Cumulative Density", color=color_cdf)
ax2.tick_params(axis='y', labelcolor=color_cdf)

# Title and legend
fig.suptitle("Histogram and Cumulative Distribution of Uncertainty")
fig.tight_layout()
fig.subplots_adjust(top=0.92)
fig.legend(loc="upper left", bbox_to_anchor=(0.15, 0.85))

# Save figure
cumm_path = os.path.join(cwd, 'cumm.png')
fig.savefig(cumm_path)
plt.show()


threshold_error_70 = np.percentile(errors, perc_tresh)
high_error_mask = errors >= threshold_error_70

# Check overlap
high_both_mask = high_uncertainty_mask & high_error_mask
percent_overlap = 100 * np.sum(high_both_mask) / np.sum(high_uncertainty_mask)
print(percent_overlap)