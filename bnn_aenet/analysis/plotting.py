"""Publication-quality plotting functions for BNN-AENET.

Provides functions to create:
- Parity plots (predicted vs true)
- Residual plots
- Uncertainty calibration plots
- Training curves from TensorBoard logs
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional, Dict, List, Tuple
from pathlib import Path

# Publication style settings
FIGSIZE_SINGLE = (4, 3.5)
FIGSIZE_DOUBLE = (8, 3.5)
FIGSIZE_SQUARE = (4, 4)
DPI = 300
FONT_SIZE = 10
LABEL_SIZE = 12

# Color scheme for methods
METHOD_COLORS = {
    "lrt": "#1f77b4",      # Blue
    "flipout": "#ff7f0e",  # Orange
    "radial": "#2ca02c",   # Green
    "de": "#d62728",       # Red
    "nn": "#9467bd",       # Purple
}

METHOD_LABELS = {
    "lrt": "LRT",
    "flipout": "Flipout",
    "radial": "Radial",
    "de": "Deep Ensemble",
    "nn": "NN",
}


def setup_plot_style():
    """Set up matplotlib style for publication."""
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'axes.labelsize': LABEL_SIZE,
        'axes.titlesize': LABEL_SIZE,
        'xtick.labelsize': FONT_SIZE,
        'ytick.labelsize': FONT_SIZE,
        'legend.fontsize': FONT_SIZE - 1,
        'figure.dpi': DPI,
        'savefig.dpi': DPI,
        'savefig.bbox': 'tight',
        'axes.linewidth': 1.0,
        'lines.linewidth': 1.5,
        'lines.markersize': 4,
    })


def plot_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "",
    xlabel: str = "True",
    ylabel: str = "Predicted",
    color: str = "#1f77b4",
    alpha: float = 0.6,
    show_metrics: bool = True,
    show_diagonal: bool = True,
    unit: str = "eV/atom",
) -> plt.Axes:
    """Create a parity plot (predicted vs true).
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        y_std: Standard deviations for error bars (optional)
        ax: Matplotlib axes (creates new if None)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        color: Marker color
        alpha: Marker transparency
        show_metrics: Whether to show RMSE/MAE
        show_diagonal: Whether to show y=x line
        unit: Unit for metrics
    
    Returns:
        Matplotlib axes
    """
    setup_plot_style()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Plot data
    if y_std is not None:
        y_std = np.asarray(y_std).flatten()
        ax.errorbar(
            y_true, y_pred, yerr=y_std,
            fmt='o', color=color, alpha=alpha,
            markersize=3, capsize=0, elinewidth=0.5
        )
    else:
        ax.scatter(y_true, y_pred, c=color, alpha=alpha, s=15, edgecolors='none')
    
    # Diagonal line
    if show_diagonal:
        lims = [
            min(y_true.min(), y_pred.min()),
            max(y_true.max(), y_pred.max())
        ]
        margin = 0.05 * (lims[1] - lims[0])
        lims = [lims[0] - margin, lims[1] + margin]
        ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1, zorder=0)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
    
    # Metrics
    if show_metrics:
        from .metrics import compute_energy_metrics
        metrics = compute_energy_metrics(y_true, y_pred, y_std)
        text = f"RMSE: {metrics['rmse']:.4f} {unit}\nMAE: {metrics['mae']:.4f} {unit}"
        ax.text(
            0.05, 0.95, text,
            transform=ax.transAxes,
            verticalalignment='top',
            fontsize=FONT_SIZE - 1,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')
    
    return ax


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Residuals",
    xlabel: str = "True",
    ylabel: str = "Residual (True - Pred)",
    color: str = "#1f77b4",
    show_uncertainty: bool = True,
) -> plt.Axes:
    """Create a residual plot.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        y_std: Standard deviations (optional)
        ax: Matplotlib axes
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        color: Marker color
        show_uncertainty: Whether to show uncertainty bounds
    
    Returns:
        Matplotlib axes
    """
    setup_plot_style()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    residuals = y_true - y_pred
    
    ax.scatter(y_true, residuals, c=color, alpha=0.6, s=15, edgecolors='none')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1)
    
    if show_uncertainty and y_std is not None:
        y_std = np.asarray(y_std).flatten()
        # Sort by y_true for better visualization
        idx = np.argsort(y_true)
        ax.fill_between(
            y_true[idx], -2*y_std[idx], 2*y_std[idx],
            alpha=0.2, color=color, label='2σ'
        )
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    return ax


def plot_uncertainty_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "Uncertainty Calibration",
    color: str = "#1f77b4",
    label: str = "",
    n_bins: int = 20,
) -> plt.Axes:
    """Create an uncertainty calibration plot.
    
    Shows observed vs expected confidence coverage.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        y_std: Standard deviations
        ax: Matplotlib axes
        title: Plot title
        color: Line color
        label: Line label for legend
        n_bins: Number of confidence levels
    
    Returns:
        Matplotlib axes
    """
    setup_plot_style()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    
    from .metrics import compute_calibration_curve
    expected, observed = compute_calibration_curve(
        y_true, y_pred, y_std, n_bins=n_bins
    )
    
    # Plot diagonal (perfect calibration)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
    
    # Plot calibration curve
    ax.plot(expected, observed, 'o-', color=color, label=label or 'Model', markersize=4)
    
    # Fill area between curve and diagonal
    ax.fill_between(
        expected, expected, observed,
        alpha=0.2, color=color
    )
    
    ax.set_xlabel('Expected Confidence')
    ax.set_ylabel('Observed Confidence')
    ax.set_title(title)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='lower right')
    
    return ax


def plot_training_curves(
    log_dir: Path,
    metrics: List[str] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Training Curves",
    smooth_window: int = 10,
) -> plt.Axes:
    """Plot training curves from TensorBoard logs.
    
    Args:
        log_dir: Path to TensorBoard log directory
        metrics: List of metrics to plot (defaults to loss, rmse)
        ax: Matplotlib axes
        title: Plot title
        smooth_window: Window for smoothing
    
    Returns:
        Matplotlib axes
    """
    setup_plot_style()
    
    if metrics is None:
        metrics = ['rmse/train', 'rmse/val']
    
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        log_dir = Path(log_dir)
        event_files = list(log_dir.glob('events.out.tfevents.*'))
        
        if not event_files:
            ax.text(0.5, 0.5, 'No TensorBoard logs found',
                    ha='center', va='center', transform=ax.transAxes)
            return ax
        
        ea = EventAccumulator(str(log_dir))
        ea.Reload()
        
        available_tags = ea.Tags().get('scalars', [])
        
        for metric in metrics:
            if metric in available_tags:
                events = ea.Scalars(metric)
                steps = [e.step for e in events]
                values = [e.value for e in events]
                
                # Smooth if requested
                if smooth_window > 1 and len(values) > smooth_window:
                    kernel = np.ones(smooth_window) / smooth_window
                    values_smooth = np.convolve(values, kernel, mode='valid')
                    steps_smooth = steps[smooth_window-1:]
                    ax.plot(steps_smooth, values_smooth, label=metric)
                else:
                    ax.plot(steps, values, label=metric)
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.legend()
        ax.set_yscale('log')
        
    except ImportError:
        ax.text(0.5, 0.5, 'TensorBoard not installed',
                ha='center', va='center', transform=ax.transAxes)
    
    return ax


def plot_method_comparison(
    results: Dict[str, Dict],
    metric: str = "rmse",
    dataset: str = "",
    ax: Optional[plt.Axes] = None,
    show_values: bool = True,
) -> plt.Axes:
    """Create a bar plot comparing methods.
    
    Args:
        results: Dict mapping method names to metric dicts
        metric: Metric to compare
        dataset: Dataset name for title
        ax: Matplotlib axes
        show_values: Whether to show values on bars
    
    Returns:
        Matplotlib axes
    """
    setup_plot_style()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    
    methods = list(results.keys())
    values = [results[m].get(metric, 0) for m in methods]
    colors = [METHOD_COLORS.get(m.lower(), '#333333') for m in methods]
    labels = [METHOD_LABELS.get(m.lower(), m) for m in methods]
    
    bars = ax.bar(range(len(methods)), values, color=colors)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    if show_values:
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(
                f'{val:.4f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=FONT_SIZE - 2
            )
    
    ax.set_ylabel(metric.upper())
    ax.set_title(f'{metric.upper()} Comparison{" - " + dataset if dataset else ""}')
    
    return ax


def create_publication_figure(
    results_dict: Dict,
    output_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (10, 8),
) -> plt.Figure:
    """Create a multi-panel publication figure.
    
    Creates a 2x2 grid with:
    - Parity plots for energy
    - Residual plots
    - Calibration curves
    - Method comparison
    
    Args:
        results_dict: Dictionary with prediction results
        output_path: Path to save figure (optional)
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    setup_plot_style()
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Extract data
    y_true = results_dict.get('y_true', [])
    y_pred = results_dict.get('y_pred', [])
    y_std = results_dict.get('y_std', None)
    
    # Parity plot
    plot_parity(y_true, y_pred, y_std, ax=axes[0, 0], title='Energy Parity')
    
    # Residuals
    plot_residuals(y_true, y_pred, y_std, ax=axes[0, 1], title='Residuals')
    
    # Calibration (if uncertainty available)
    if y_std is not None:
        plot_uncertainty_calibration(y_true, y_pred, y_std, ax=axes[1, 0])
    else:
        axes[1, 0].text(0.5, 0.5, 'No uncertainty data', ha='center', va='center')
    
    # Histogram of errors
    if len(y_true) > 0:
        errors = np.asarray(y_true) - np.asarray(y_pred)
        axes[1, 1].hist(errors, bins=30, color='#1f77b4', alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(x=0, color='r', linestyle='--', alpha=0.7)
        axes[1, 1].set_xlabel('Error')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Error Distribution')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=DPI, bbox_inches='tight')
    
    return fig
