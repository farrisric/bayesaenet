import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load the CSV
df = pd.read_csv("/home/g15farris/bin/bayesaenet/scripts/metrics/uq_metrics_summary.csv")

# Ensure output folder exists
output_dir = "/home/g15farris/bin/bayesaenet/scripts/metrics/uq_plots"
os.makedirs(output_dir, exist_ok=True)

# Convert 'Split' to categorical for consistent ordering
df['Split'] = pd.Categorical(df['Split'], categories=['Train', 'Val', 'Test'], ordered=True)

# Identify metric columns (exclude metadata)
metadata_cols = ['Method', 'Size', 'Run', 'Split']
metric_cols = [col for col in df.columns if col not in metadata_cols]

# 1. Boxplots: Metric by Method and Size
for metric in metric_cols:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Method', y=metric, hue='Size')
    plt.title(f"{metric} by Method and Size")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{metric}_boxplot.png")
    plt.close()

# 2. Lineplots: Metric across Train/Val/Test
for metric in metric_cols:
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='Split', y=metric, hue='Method', style='Size', markers=True, dashes=False)
    plt.title(f"{metric} across Splits")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{metric}_split_lineplot.png")
    plt.close()

# 3. Optional: Pairplot of selected metrics
selected_metrics = ['mse', 'rmse', 'r2score', 'ece', 'nll']
subset = df[metadata_cols + [m for m in selected_metrics if m in df.columns]]
sns.pairplot(subset, hue='Method', corner=True, diag_kind='kde')
plt.suptitle("Pairplot of Selected UQ Metrics", y=1.02)
plt.savefig(f"{output_dir}/pairplot_selected_metrics.png")
plt.close()

print(f"Plots saved to: {output_dir}")
