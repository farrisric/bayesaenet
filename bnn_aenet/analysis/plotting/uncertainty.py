"""Uncertainty quality plots."""
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple


def plot_residuals_vs_uncertainty(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    n_atoms: np.ndarray,
    title: str = "Residuals vs Uncertainty",
    figsize=(10, 6),
    save_path=None
):
    """Create scatter plot of residuals vs predicted uncertainty.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        y_std: Predicted uncertainties
        n_atoms: Number of atoms per structure
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=200)
    
    # Compute residuals
    residuals = y_true - y_pred
    
    # Create scatter plot colored by number of atoms
    sc = ax.scatter(
        y_std,
        residuals,
        c=n_atoms,
        cmap='jet',
        alpha=0.3,
        s=10
    )
    
    # Add reference lines for ±1σ, ±2σ
    x_range = np.linspace(0, max(y_std), 100)
    for n_sigma in range(-2, 3):
        linestyle = '--' if n_sigma != 0 else '-'
        linewidth = 1 if n_sigma != 0 else 2
        ax.plot(
            x_range,
            x_range * n_sigma,
            color='black',
            linestyle=linestyle,
            alpha=0.5,
            linewidth=linewidth
        )
    
    # Add colorbar
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Number of Atoms')
    
    # Labels and title
    ax.set_xlabel('Predicted Std Dev (eV/atom)')
    ax.set_ylabel('Residual (eV/atom)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    return fig
