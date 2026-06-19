#!/usr/bin/env python
"""Create plots from TiO2 analysis results."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from analysis.plotting import plot_performance_comparison, plot_sharpness

# Set plotting style
plt.rcParams.update({"font.size": 14, "axes.labelsize": 16, "figure.dpi": 150})
sns.set_context("talk", font_scale=1.2)


def main():
    print("=" * 80)
    print("Creating TiO2 Plots")
    print("=" * 80)

    # Load results
    results_path = Path(__file__).parent / "results" / "uq_metrics_Test.csv"
    if not results_path.exists():
        print(f"\n✗ Results file not found: {results_path}")
        print("  Run 'python run_analysis.py' first!")
        return

    df = pd.read_csv(results_path)
    print(f"\nLoaded {len(df)} experiments")

    # Create figures directory
    fig_dir = Path(__file__).parent / "figures"
    fig_dir.mkdir(exist_ok=True)

    methods = sorted(df["Method"].unique())
    sizes = sorted(df["Size"].unique())

    print(f"Methods: {methods}")
    print(f"Sizes: {sizes}")

    # Performance plots
    print("\n" + "=" * 80)
    print("Creating performance plots...")
    print("=" * 80)

    perf_metrics = ["mae", "rmse", "maxerr"]
    for metric in perf_metrics:
        if metric in df.columns:
            print(f"  Plotting {metric}...")
            try:
                figs = plot_performance_comparison(
                    df,
                    metrics=[metric],
                    methods=methods,
                    sizes=sizes,
                    figsize=(12, 6),
                    save_path=str(fig_dir / f"performance_{metric}.png"),
                )
                print(f"  ✓ Saved: figures/performance_{metric}.png")
            except Exception as e:
                print(f"  ✗ Error: {e}")

    # UQ plots (only for methods with uncertainty)
    uq_methods = [m for m in methods if m != "nn"]
    df_uq = df[df["Method"].isin(uq_methods)]

    if len(df_uq) > 0:
        print("\n" + "=" * 80)
        print("Creating UQ plots...")
        print("=" * 80)

        uq_metrics = ["overlap", "sharp", "nll"]
        for metric in uq_metrics:
            if metric in df_uq.columns:
                print(f"  Plotting {metric}...")
                try:
                    figs = plot_performance_comparison(
                        df_uq,
                        metrics=[metric],
                        methods=uq_methods,
                        sizes=sizes,
                        figsize=(12, 6),
                        save_path=str(fig_dir / f"uq_{metric}.png"),
                    )
                    print(f"  ✓ Saved: figures/uq_{metric}.png")
                except Exception as e:
                    print(f"  ✗ Error: {e}")

        # Sharpness distribution
        if "sharp" in df_uq.columns:
            print(f"  Plotting sharpness distribution...")
            try:
                fig = plot_sharpness(
                    df_uq,
                    methods=uq_methods,
                    sizes=sizes,
                    figsize=(12, 6),
                    save_path=str(fig_dir / "sharpness_distribution.png"),
                )
                print(f"  ✓ Saved: figures/sharpness_distribution.png")
                plt.close(fig)
            except Exception as e:
                print(f"  ✗ Error: {e}")

    # Summary table
    print("\n" + "=" * 80)
    print("Creating summary table...")
    print("=" * 80)

    summary_cols = [c for c in ["mae", "rmse", "overlap", "nll"] if c in df.columns]
    summary = df.groupby(["Method", "Size"])[summary_cols].agg(["mean", "std"])
    summary = summary.round(4)

    summary_path = fig_dir / "summary_table.csv"
    summary.to_csv(summary_path)
    print(f"✓ Saved: {summary_path}")

    print("\n" + "=" * 80)
    print("Summary Statistics:")
    print("=" * 80)
    print(summary)

    print("\n" + "=" * 80)
    print("Done! All plots saved to: figures/")
    print("=" * 80)


if __name__ == "__main__":
    main()
