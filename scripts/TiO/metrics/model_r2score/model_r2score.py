import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv("/home/g15farris/bin/bayesaenet/scripts/TiO/metrics/metrics.txt", sep="\t")
df.columns = df.columns.str.strip()
df['Size'] = df['Size'].astype(str)
df['Model'] = df['Model'].str.upper()

# Define metrics for Figure 1
metrics = ["MAE", "RMSE", "MAXE"]
titles = ["Root Mean Squared Error (RMSE)",
          "Mean Absolute Error (MAE)",
          "Maximum Absolute Error (MAXE)"]

# Model order
model_order = ['DE', 'HNN', 'FO', 'LRT', 'RAD']
size_order = ['20', '100']

# Set seaborn theme
sns.set_theme(style="whitegrid", font_scale=1.5)
palette = sns.color_palette("colorblind")[:2]  # Only two colors needed

# --- First figure (RMSE, MAE, MAXE) ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True)

for idx, metric in enumerate(metrics):
    sns.boxplot(
        data=df, x="Model", y=metric, hue="Size", ax=axes[idx],
        order=model_order, hue_order=size_order, palette=palette,
        width=0.6, fliersize=3
    )
    axes[idx].set_xlabel("Model")
    axes[idx].set_ylabel(f"{metric} [eV/atom]")
    axes[idx].grid(axis='y', linestyle='--', alpha=0.7)
    axes[idx].legend(title="Dataset Size (%)", loc="upper right")

plt.tight_layout()
plt.savefig("/home/g15farris/bin/bayesaenet/scripts/TiO/metrics/model_r2score/figure1_rmse_mae_maxe_paper.png", dpi=600)
plt.show()

# --- Second figure (R2 Scores) ---
plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df, x="Model", y="R2", hue="Size",
    order=model_order, hue_order=size_order, palette=palette,
    width=0.6, fliersize=3
)

plt.axhline(1.0, linestyle="--", color="gray", linewidth=1)
plt.xlabel("Model")
plt.ylabel("R² Score")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title="Dataset Size (%)", loc="upper left")

plt.tight_layout()
plt.savefig("/home/g15farris/bin/bayesaenet/scripts/TiO/metrics/model_r2score/figure_r2_scores_paper.png", dpi=600)
plt.show()
