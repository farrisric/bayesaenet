import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add bayesaenet path
sys.path.append('/home/g15farris/bin/bayesaenet')

# Constants for rescaling
e_scaling, e_shift = 0.06565926932648217, 6.6588702845000975

# Load parquet paths
log_dir = '/home/g15farris/bin/bayesaenet/bnn_aenet/logs'
directories = {x.split('/')[-1]: None for x in glob.glob(f'{log_dir}/*_pred*')}
for parquet in sorted(glob.glob(f'{log_dir}/*_pred*/runs/**/*parquet', recursive=True)):
    directories[parquet.split('/')[-4]] = parquet

# Define models and data sizes
models = ['lrt', 'fo', 'rad']
sizes = ['20', '100']

# Container for results
decomp_data = []

# Extract variances
for model in models:
    for size in sizes:
        ep_vars_all = []
        al_vars_all = []

        for name, parquet in directories.items():
            if parquet is None:
                continue
            if not name.startswith(model) or f"_{size}_" not in name:
                continue

            df = pd.read_csv(parquet)
            n_atoms = df['n_atoms'].to_numpy()
            ep_vars = (df['ep_vars'].to_numpy() / e_scaling) / n_atoms
            al_vars = (df['al_vars'].to_numpy() / e_scaling) / n_atoms
            ep_vars_all.extend(ep_vars)
            al_vars_all.extend(al_vars)

        if ep_vars_all and al_vars_all:
            decomp_data.append({
                'Model': model.upper(),
                'Size': size + '%',
                'Epistemic': np.mean(ep_vars_all),
                'Aleatoric': np.mean(al_vars_all)
            })

# Create DataFrame
decomp_df = pd.DataFrame(decomp_data)
decomp_melted = decomp_df.melt(id_vars=['Model', 'Size'], var_name='Type', value_name='Variance')

# Plot with subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=False)

for i, size in enumerate(['20%', '100%']):
    sns.barplot(
        data=decomp_melted[decomp_melted['Size'] == size],
        x="Model", y="Variance", hue="Type", ax=axes[i], errorbar=None,
        palette="muted", dodge=True
    )
    axes[i].set_title(f"{size} Training Data")
    axes[i].set_ylabel("Average Per-Atom Variance" if i == 0 else "")
    axes[i].set_xlabel("Model")
    axes[i].grid(True, axis='y')
    axes[i].legend(title="Uncertainty Type")

plt.suptitle("Decomposition of Predictive Uncertainty by Dataset Size", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("/home/g15farris/bin/bayesaenet/scripts/TiO/metrics/unc_decomposition/uncertainty_decomposition_by_size.png", dpi=300)
plt.show()
