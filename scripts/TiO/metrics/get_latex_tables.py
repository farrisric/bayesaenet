import pandas as pd

# Load data
df = pd.read_csv("/home/g15farris/bin/bayesaenet/scripts/TiO/metrics/metrics.txt", sep="\t")
df.columns = df.columns.str.strip()
df["Model"] = df["Model"].str.upper()
df["Size"] = df["Size"].astype(str)

# Group and aggregate
summary = df.groupby(["Model", "Size"]).agg(["mean", "std"]).round(4)
summary_20 = summary.xs("20", level="Size")
summary_100 = summary.xs("100", level="Size")

# Function to convert to LaTeX format
def to_latex_table_simple(df, caption, label):
    means = df.xs("mean", axis=1, level=1)
    stds = df.xs("std", axis=1, level=1)
    formatted = means.copy()
    for col in means.columns:
        formatted[col] = means[col].map('{:.4f}'.format) + " ± " + stds[col].map('{:.4f}'.format)
    latex = formatted.to_latex(escape=False, caption=caption, label=label)
    return latex

# Split into performance and uncertainty metrics
perf_metrics = ["MAE", "RMSE", "MAXE", "NLL"]
unc_metrics = ["R2", "ECE", "RMSCE", "Sharpness"]

# Generate LaTeX tables
latex_perf_20 = to_latex_table_simple(summary_20[perf_metrics], "Performance metrics (MAE, RMSE, MAXE, NLL) for each model (20\\% dataset).", "tab:performance_20")
latex_perf_100 = to_latex_table_simple(summary_100[perf_metrics], "Performance metrics (MAE, RMSE, MAXE, NLL) for each model (100\\% dataset).", "tab:performance_100")
latex_unc_20 = to_latex_table_simple(summary_20[unc_metrics], "Uncertainty quantification metrics (R$^2$, ECE, RMSCE, Sharpness) for each model (20\\% dataset).", "tab:uncertainty_20")
latex_unc_100 = to_latex_table_simple(summary_100[unc_metrics], "Uncertainty quantification metrics (R$^2$, ECE, RMSCE, Sharpness) for each model (100\\% dataset).", "tab:uncertainty_100")

# Write all to a .tex file
full_tex = "\n\n".join([latex_perf_20, latex_perf_100, latex_unc_20, latex_unc_100])
output_path = "/home/g15farris/bin/bayesaenet/scripts/TiO/metrics/summary_tables.tex"
with open(output_path, "w") as f:
    f.write(full_tex)

output_path
