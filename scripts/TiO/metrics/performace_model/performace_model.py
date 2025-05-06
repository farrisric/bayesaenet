import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv("/home/g15farris/bin/bayesaenet/scripts/TiO/metrics/metrics.txt", sep="\t")
df.columns = df.columns.str.strip()
df['Size'] = df['Size'].astype(str)
df['Model'] = df['Model'].str.upper()

# Define metrics for the figure
metrics = ["RMSE", "NLL"]
titles = [
          "Root Mean Squared Error (RMSE)",
          "Negative Log-Likelihood (NLL)"]

# Model and size order
model_order = ['DE', 'HNN', 'FO', 'LRT', 'RAD']
size_order = ['20', '100']

# Seaborn styling
sns.set_theme(style="whitegrid", font_scale=1.6)
palette = sns.color_palette("colorblind")[:2]

# Create subplots
fig, axes = plt.subplots(2, 1, figsize=(6, 12), sharex=True)

axes = axes.flatten()

for idx, metric in enumerate(metrics):
    sns.boxplot(
        data=df, x="Model", y=metric, hue="Size", ax=axes[idx],
        order=model_order, hue_order=size_order, palette=palette,
        width=0.5, fliersize=2
    )
    axes[idx].set_xlabel("Model", labelpad=10)
    axes[idx].set_ylabel(f"{metric} [eV/atom]" if metric != "NLL" else "NLL", labelpad=10)
    axes[idx].set_title(titles[idx], pad=15, fontsize=20)
    axes[idx].grid(axis='y', linestyle='--', alpha=0.6)
    axes[idx].tick_params(axis='x', rotation=45)

# Remove individual legends
for ax in axes:
    ax.get_legend().remove()

# Add global legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels, title="Dataset Size (%)",
    loc='upper center', bbox_to_anchor=(0.5, 1.03), ncol=2, frameon=False, fontsize=20, title_fontsize=20
)

# Final layout
plt.tight_layout()
plt.savefig("/home/g15farris/bin/bayesaenet/scripts/TiO/metrics/performace_model/figure1_rmse_mae_maxe_nll_paper.png", dpi=200, bbox_inches="tight")
plt.show()
