import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def read_metrics(metrics_file):
    df = pd.read_csv(metrics_file, sep='\t')
    return df

df = read_metrics('/home/g15farris/bin/bayesaenet/scripts/TiO/metrics/metrics.txt')
df = df.drop(columns=["MAXE"])
df = df.drop(columns=["NLL"])
df = df.drop(columns=["R2"])


#df['Size'] = df['Size'].astype(str)
df['Model'] = df['Model'].str.upper()

model_order = ['DE', 'HNN', 'FO', 'LRT', 'RAD']
size_order = ['20', '100']

fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=False)
for i, size in enumerate([20, 100]):
    df_size = df[df["Size"] == size]
    df_melted = df_size.melt(id_vars=["Model", "Size"], var_name="Metric", value_name="Value")
    sns.boxplot(data=df_melted, x="Metric", y="Value", hue="Model", ax=ax[i])
    ax[i].set_title(f"Size = {size}%")
    ax[i].set_ylabel("Value")
    ax[i].set_xlabel("Model")
    ax[i].legend(title="Metric")
plt.tight_layout()
fig.savefig('/home/g15farris/bin/bayesaenet/scripts/TiO/metrics/all_metrics/metrics_barplot.png')
