"""Performance comparison plots."""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List


def plot_performance_comparison(
    df: pd.DataFrame,
    metrics: List[str],
    methods: List[str],
    sizes: List[str],
    figsize=(15, 5),
    save_path=None
):
    """Create catplots comparing performance metrics across methods and sizes.
    
    Args:
        df: DataFrame with columns ['Method', 'Size', 'Run', metric columns...]
        metrics: List of metric names to plot
        methods: List of method names
        sizes: List of size labels
        figsize: Figure size per metric
        save_path: Path template for saving (will add metric name)
    """
    sns.set_style("whitegrid")
    sns.set_context("talk")
    
    figures = {}
    
    for metric in metrics:
        g = sns.catplot(
            data=df,
            x='Method',
            y=metric,
            hue='Size',
            kind='violin',
            order=methods,
            hue_order=sizes,
            height=figsize[1],
            aspect=figsize[0] / figsize[1],
            palette='Set2'
        )
        
        g.set_axis_labels("Method", metric.upper())
        g.legend.set_title("Data Size")
        g.fig.suptitle(f'{metric.upper()} Comparison', y=1.02, fontsize=16)
        
        if save_path:
            path = save_path.replace('{metric}', metric)
            g.savefig(path, dpi=300, bbox_inches='tight')
            plt.close(g.fig)
        
        figures[metric] = g
    
    return figures
