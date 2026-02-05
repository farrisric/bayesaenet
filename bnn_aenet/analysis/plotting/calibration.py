"""Calibration and sharpness plots."""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List
import pandas as pd

try:
    import uncertainty_toolbox as uct
    HAS_UCT = True
except ImportError:
    HAS_UCT = False


def plot_calibration(
    results: Dict,
    figsize=(15, 9),
    save_path=None
):
    """Create calibration plots using uncertainty_toolbox.
    
    Args:
        results: Dictionary with keys (method, size, run) -> (y_true, y_pred, y_std)
        figsize: Figure size
        save_path: Path to save figure (optional)
    """
    if not HAS_UCT:
        raise ImportError("uncertainty_toolbox required for calibration plots")
    
    fig, axes = plt.subplots(2, 3, figsize=figsize, dpi=150)
    axes = axes.flat
    
    # Aggregate all results
    all_y_true = []
    all_y_pred = []
    all_y_std = []
    
    for (method, size, run), (y_true, y_pred, y_std, n_atoms) in results.items():
        all_y_true.append(y_true)
        all_y_pred.append(y_pred)
        all_y_std.append(y_std)
    
    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)
    y_std = np.concatenate(all_y_std)
    
    # Create calibration plots
    uct.plot_intervals(y_pred, y_std, y_true, ax=axes[0])
    uct.plot_intervals_ordered(y_pred, y_std, y_true, ax=axes[1])
    uct.plot_calibration(y_pred, y_std, y_true, ax=axes[2])
    uct.plot_adversarial_group_calibration(y_pred, y_std, y_true, ax=axes[3])
    uct.plot_sharpness(y_std, ax=axes[4])
    uct.plot_residuals_vs_stds(y_pred, y_std, y_true, ax=axes[5])
    
    fig.suptitle('Uncertainty Quantification Analysis', fontsize=16)
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    return fig


def plot_sharpness(
    df: pd.DataFrame,
    methods: List[str],
    sizes: List[str],
    figsize=(12, 6),
    save_path=None
):
    """Create violin plots of sharpness (uncertainty distribution) by method and size.
    
    Args:
        df: DataFrame with columns ['Method', 'Size', 'sharp']
        methods: List of method names to plot
        sizes: List of size labels to plot
        figsize: Figure size
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(1, len(sizes), figsize=figsize, sharey=True)
    if len(sizes) == 1:
        axes = [axes]
    
    # Set style
    sns.set_context("talk", font_scale=1.2)
    palette = sns.color_palette("tab10", n_colors=len(methods))
    
    for ax, size_label in zip(axes, sizes):
        df_size = df[df['Size'] == size_label]
        
        sns.violinplot(
            data=df_size,
            x='Method',
            y='sharp',
            ax=ax,
            palette=palette,
            order=methods
        )
        
        ax.set_title(f'Data Size: {size_label}')
        ax.set_xlabel('Method')
        ax.set_ylabel('Sharpness (eV/atom)' if ax == axes[0] else '')
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Uncertainty Sharpness Comparison', fontsize=16, y=1.02)
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    return fig
