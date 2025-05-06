import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv("/home/g15farris/bin/bayesaenet/scripts/TiO/metrics/metrics.txt", sep="\t")
df.columns = df.columns.str.strip()
df['Size'] = df['Size'].astype(str)
df['Model'] = df['Model'].str.upper()
# Set seaborn theme
sns.set(style="whitegrid", font_scale=1.5, rc={"axes.labelsize":18, "axes.titlesize":18,
                                               "legend.fontsize":14, "xtick.labelsize":14, "ytick.labelsize":14})

# Metrics to plot
remaining_metrics = ["ECE", "RMSCE", "Sharpness"]

# Desired model and size orders
model_order = ['DE', 'HNN', 'FO', 'LRT', 'RAD']
size_order = ['20', '100']

# Choose a colorblind-friendly palette
palette = sns.color_palette("colorblind")[:2]

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True)
axes = axes.flatten()

# Generate boxplots
for idx, metric in enumerate(remaining_metrics):
    sns.boxplot(
        data=df, x="Model", y=metric, hue="Size",
        ax=axes[idx], order=model_order, hue_order=size_order,
        palette=palette, width=0.6, fliersize=3
    )
    axes[idx].set_title(f"{metric} by Model and Dataset Size", pad=15)
    axes[idx].set_xlabel("Model", labelpad=10)
    axes[idx].set_ylabel(metric, labelpad=10)
    axes[idx].grid(axis='y', linestyle='--', alpha=0.7)
    axes[idx].legend(title="Dataset Size", loc="upper right", frameon=True)

# Global settings
plt.tight_layout()
plt.savefig(
    "/home/g15farris/bin/bayesaenet/scripts/TiO/metrics/unc_metrics/figure_uncertainty_metrics_paper.png",
    dpi=200)
plt.show()
