"""Plot force predictions with uncertainties and error analysis."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import warnings

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')


def load_predictions(pred_file: Path) -> pd.DataFrame:
    """Load prediction results from parquet or CSV file.
    
    Auto-detects format even if extension is wrong.
    """
    pred_file = Path(pred_file)
    
    # Try parquet first
    try:
        return pd.read_parquet(pred_file)
    except Exception:
        pass
    
    # Try CSV (handles .parquet files that are actually CSV)
    try:
        return pd.read_csv(pred_file)
    except Exception:
        pass
    
    raise ValueError(f"Could not read file as parquet or CSV: {pred_file}")


def plot_energy_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Energy Predictions",
    unit: str = "eV/atom"
) -> plt.Axes:
    """Plot energy parity plot with optional error bars.
    
    Args:
        y_true: True energy values
        y_pred: Predicted energy values
        y_std: Prediction uncertainties (optional)
        ax: Matplotlib axes (created if None)
        title: Plot title
        unit: Energy unit for labels
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    # Compute metrics
    errors = y_pred - y_true
    rmse = np.sqrt(np.mean(errors**2))
    mae = np.mean(np.abs(errors))
    
    # Plot parity line
    all_vals = np.concatenate([y_true, y_pred])
    min_val, max_val = all_vals.min(), all_vals.max()
    margin = 0.05 * (max_val - min_val)
    line_range = [min_val - margin, max_val + margin]
    ax.plot(line_range, line_range, 'k--', lw=1, alpha=0.5, label='Perfect')
    
    # Plot predictions
    if y_std is not None:
        # Color by uncertainty
        scatter = ax.scatter(y_true, y_pred, c=y_std, cmap='viridis', 
                            alpha=0.7, edgecolors='none', s=20)
        cbar = plt.colorbar(scatter, ax=ax, label=f'Uncertainty ({unit})')
    else:
        ax.scatter(y_true, y_pred, alpha=0.6, edgecolors='none', s=20, c='steelblue')
    
    # Labels and formatting
    ax.set_xlabel(f'True Energy ({unit})', fontsize=11)
    ax.set_ylabel(f'Predicted Energy ({unit})', fontsize=11)
    ax.set_title(f'{title}\nRMSE: {rmse:.4f}, MAE: {mae:.4f} {unit}', fontsize=12)
    ax.set_xlim(line_range)
    ax.set_ylim(line_range)
    ax.set_aspect('equal')
    
    return ax


def plot_force_parity(
    f_true: np.ndarray,
    f_pred: np.ndarray,
    f_std: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Force Predictions",
    unit: str = "mHa/Bohr"
) -> plt.Axes:
    """Plot force parity plot (all components).
    
    Args:
        f_true: True force components (flattened)
        f_pred: Predicted force components (flattened)
        f_std: Force uncertainties (optional)
        ax: Matplotlib axes
        title: Plot title
        unit: Force unit for labels
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    # Compute metrics
    errors = f_pred - f_true
    rmse = np.sqrt(np.mean(errors**2))
    mae = np.mean(np.abs(errors))
    
    # Plot parity line
    all_vals = np.concatenate([f_true, f_pred])
    min_val, max_val = all_vals.min(), all_vals.max()
    margin = 0.1 * (max_val - min_val)
    line_range = [min_val - margin, max_val + margin]
    ax.plot(line_range, line_range, 'k--', lw=1, alpha=0.5, label='Perfect')
    
    # Subsample if too many points
    n_points = len(f_true)
    if n_points > 10000:
        idx = np.random.choice(n_points, 10000, replace=False)
        f_true_plot = f_true[idx]
        f_pred_plot = f_pred[idx]
        f_std_plot = f_std[idx] if f_std is not None else None
    else:
        f_true_plot, f_pred_plot, f_std_plot = f_true, f_pred, f_std
    
    # Plot predictions
    if f_std_plot is not None:
        scatter = ax.scatter(f_true_plot, f_pred_plot, c=f_std_plot, cmap='plasma',
                            alpha=0.5, edgecolors='none', s=10)
        plt.colorbar(scatter, ax=ax, label=f'Uncertainty ({unit})')
    else:
        ax.scatter(f_true_plot, f_pred_plot, alpha=0.3, edgecolors='none', s=10, c='coral')
    
    # Labels
    ax.set_xlabel(f'True Force Component ({unit})', fontsize=11)
    ax.set_ylabel(f'Predicted Force Component ({unit})', fontsize=11)
    ax.set_title(f'{title}\nRMSE: {rmse:.4f}, MAE: {mae:.4f} {unit}', fontsize=12)
    ax.set_xlim(line_range)
    ax.set_ylim(line_range)
    ax.set_aspect('equal')
    
    return ax


def plot_force_magnitude_parity(
    f_true: np.ndarray,
    f_pred: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "Force Magnitudes"
) -> plt.Axes:
    """Plot parity of force magnitudes (per atom).
    
    Args:
        f_true: True force components (N*3 or Nx3)
        f_pred: Predicted force components
        ax: Matplotlib axes
        title: Plot title
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    # Reshape to Nx3 if needed
    f_true = f_true.flatten()
    f_pred = f_pred.flatten()
    
    if len(f_true) % 3 != 0:
        warnings.warn("Force array length not divisible by 3")
        return ax
    
    n_atoms = len(f_true) // 3
    f_true_vec = f_true.reshape(n_atoms, 3)
    f_pred_vec = f_pred.reshape(n_atoms, 3)
    
    # Compute magnitudes
    mag_true = np.linalg.norm(f_true_vec, axis=1)
    mag_pred = np.linalg.norm(f_pred_vec, axis=1)
    
    # Compute metrics
    errors = mag_pred - mag_true
    rmse = np.sqrt(np.mean(errors**2))
    mae = np.mean(np.abs(errors))
    
    # Plot
    all_vals = np.concatenate([mag_true, mag_pred])
    min_val, max_val = 0, all_vals.max()
    margin = 0.05 * max_val
    line_range = [0, max_val + margin]
    
    ax.plot(line_range, line_range, 'k--', lw=1, alpha=0.5)
    ax.scatter(mag_true, mag_pred, alpha=0.5, s=15, c='forestgreen', edgecolors='none')
    
    ax.set_xlabel('True |F| (mHa/Bohr)', fontsize=11)
    ax.set_ylabel('Predicted |F| (mHa/Bohr)', fontsize=11)
    ax.set_title(f'{title}\nRMSE: {rmse:.4f}, MAE: {mae:.4f}', fontsize=12)
    ax.set_xlim(line_range)
    ax.set_ylim(line_range)
    ax.set_aspect('equal')
    
    return ax


def plot_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "Error Distribution",
    unit: str = "",
    bins: int = 50
) -> plt.Axes:
    """Plot histogram of prediction errors.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        ax: Matplotlib axes
        title: Plot title
        unit: Unit label
        bins: Number of histogram bins
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    
    errors = y_pred - y_true
    
    ax.hist(errors, bins=bins, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    
    # Add statistics
    mean_err = np.mean(errors)
    std_err = np.std(errors)
    ax.axvline(mean_err, color='red', linestyle='--', label=f'Mean: {mean_err:.4f}')
    ax.axvline(mean_err - std_err, color='orange', linestyle=':', alpha=0.7)
    ax.axvline(mean_err + std_err, color='orange', linestyle=':', alpha=0.7, 
               label=f'±Std: {std_err:.4f}')
    
    ax.set_xlabel(f'Error {unit}', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    
    return ax


def plot_uncertainty_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "Uncertainty Calibration"
) -> plt.Axes:
    """Plot uncertainty calibration curve.
    
    For well-calibrated uncertainties, points should follow the diagonal.
    
    Args:
        y_true: True values
        y_pred: Predicted means
        y_std: Predicted uncertainties
        ax: Matplotlib axes
        title: Plot title
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    
    # Compute z-scores
    z_scores = np.abs(y_pred - y_true) / np.maximum(y_std, 1e-10)
    
    # Expected coverage at various confidence levels
    confidence_levels = np.linspace(0.1, 0.99, 20)
    expected_coverage = []
    observed_coverage = []
    
    for conf in confidence_levels:
        # For Gaussian: proportion within k*sigma where k = norm.ppf((1+conf)/2)
        from scipy.stats import norm
        k = norm.ppf((1 + conf) / 2)
        expected_coverage.append(conf)
        observed_coverage.append(np.mean(z_scores <= k))
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
    ax.plot(expected_coverage, observed_coverage, 'o-', color='steelblue', 
            markersize=4, label='Observed')
    ax.fill_between(expected_coverage, observed_coverage, expected_coverage, 
                   alpha=0.2, color='steelblue')
    
    ax.set_xlabel('Expected Coverage', fontsize=11)
    ax.set_ylabel('Observed Coverage', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(fontsize=9)
    ax.set_aspect('equal')
    
    return ax


def plot_force_components(
    f_true: np.ndarray,
    f_pred: np.ndarray,
    f_std: Optional[np.ndarray] = None,
    fig: Optional[plt.Figure] = None
) -> plt.Figure:
    """Plot force parity for each component (x, y, z).
    
    Args:
        f_true: True force components (flattened N*3)
        f_pred: Predicted force components
        f_std: Force uncertainties (optional)
        fig: Matplotlib figure
    
    Returns:
        Matplotlib figure
    """
    if fig is None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    else:
        axes = fig.subplots(1, 3)
    
    f_true = f_true.flatten()
    f_pred = f_pred.flatten()
    if f_std is not None:
        f_std = f_std.flatten()
    
    components = ['x', 'y', 'z']
    colors = ['#e74c3c', '#2ecc71', '#3498db']
    
    for i, (comp, color) in enumerate(zip(components, colors)):
        comp_true = f_true[i::3]
        comp_pred = f_pred[i::3]
        comp_std = f_std[i::3] if f_std is not None else None
        
        ax = axes[i]
        
        # Compute metrics
        errors = comp_pred - comp_true
        rmse = np.sqrt(np.mean(errors**2))
        mae = np.mean(np.abs(errors))
        
        # Plot
        all_vals = np.concatenate([comp_true, comp_pred])
        min_val, max_val = all_vals.min(), all_vals.max()
        margin = 0.1 * (max_val - min_val)
        line_range = [min_val - margin, max_val + margin]
        
        ax.plot(line_range, line_range, 'k--', lw=1, alpha=0.5)
        ax.scatter(comp_true, comp_pred, alpha=0.4, s=10, c=color, edgecolors='none')
        
        ax.set_xlabel(f'True F_{comp} (mHa/Bohr)', fontsize=10)
        ax.set_ylabel(f'Pred F_{comp} (mHa/Bohr)', fontsize=10)
        ax.set_title(f'Force {comp.upper()}\nRMSE: {rmse:.4f}, MAE: {mae:.4f}', fontsize=11)
        ax.set_xlim(line_range)
        ax.set_ylim(line_range)
        ax.set_aspect('equal')
    
    fig.tight_layout()
    return fig


def plot_comprehensive_force_analysis(
    pred_file: Path,
    output_dir: Optional[Path] = None,
    show: bool = True,
    prefix: str = ""
) -> Dict[str, plt.Figure]:
    """Generate comprehensive force prediction analysis plots.
    
    Args:
        pred_file: Path to prediction parquet/CSV file
        output_dir: Directory to save plots (None = don't save)
        show: Whether to display plots
        prefix: Prefix for saved filenames
    
    Returns:
        Dictionary of figure names to Figure objects
    """
    # Load data
    df = load_predictions(pred_file)
    
    # Extract data
    y_true = df['true'].values
    y_pred = df['preds'].values
    y_std = df['stds'].values if 'stds' in df.columns else None
    
    has_forces = 'true_forces' in df.columns and 'pred_forces' in df.columns
    
    figures = {}
    
    # Figure 1: Energy and Force Parity Plots
    if has_forces:
        fig1 = plt.figure(figsize=(14, 5))
        gs = GridSpec(1, 3, figure=fig1, wspace=0.3)
        
        ax1 = fig1.add_subplot(gs[0])
        plot_energy_parity(y_true, y_pred, y_std, ax=ax1, title="Energy")
        
        # Collect force data
        f_true_list = []
        f_pred_list = []
        f_std_list = []
        
        for idx in df.index:
            true_f = df.loc[idx, 'true_forces']
            pred_f = df.loc[idx, 'pred_forces']
            
            if true_f is not None and pred_f is not None:
                if isinstance(true_f, (list, np.ndarray)) and len(true_f) > 0:
                    f_true_list.append(np.array(true_f).flatten())
                    f_pred_list.append(np.array(pred_f).flatten())
                    if 'std_forces' in df.columns:
                        std_f = df.loc[idx, 'std_forces']
                        if std_f is not None and isinstance(std_f, (list, np.ndarray)):
                            f_std_list.append(np.array(std_f).flatten())
        
        if len(f_true_list) > 0:
            f_true = np.concatenate(f_true_list)
            f_pred = np.concatenate(f_pred_list)
            f_std = np.concatenate(f_std_list) if len(f_std_list) > 0 else None
            
            ax2 = fig1.add_subplot(gs[1])
            plot_force_parity(f_true, f_pred, f_std, ax=ax2, title="Force Components")
            
            ax3 = fig1.add_subplot(gs[2])
            plot_force_magnitude_parity(f_true, f_pred, ax=ax3, title="Force Magnitudes")
        
        fig1.suptitle(f'Prediction Quality: {pred_file.stem}', fontsize=13, y=1.02)
        fig1.tight_layout()
        figures['parity'] = fig1
        
        # Figure 2: Error Distributions
        if len(f_true_list) > 0:
            fig2 = plt.figure(figsize=(12, 4))
            
            ax1 = fig2.add_subplot(131)
            plot_error_distribution(y_true, y_pred, ax=ax1, title="Energy Errors", 
                                   unit="(eV/atom)")
            
            ax2 = fig2.add_subplot(132)
            plot_error_distribution(f_true, f_pred, ax=ax2, title="Force Errors",
                                   unit="(mHa/Bohr)")
            
            # Force magnitude errors
            n_atoms = len(f_true) // 3
            f_true_vec = f_true.reshape(n_atoms, 3)
            f_pred_vec = f_pred.reshape(n_atoms, 3)
            mag_true = np.linalg.norm(f_true_vec, axis=1)
            mag_pred = np.linalg.norm(f_pred_vec, axis=1)
            
            ax3 = fig2.add_subplot(133)
            plot_error_distribution(mag_true, mag_pred, ax=ax3, title="|F| Errors",
                                   unit="(mHa/Bohr)")
            
            fig2.tight_layout()
            figures['errors'] = fig2
        
        # Figure 3: Force Components
        if len(f_true_list) > 0:
            fig3 = plot_force_components(f_true, f_pred, f_std)
            fig3.suptitle('Force Component Analysis', fontsize=13, y=1.02)
            figures['components'] = fig3
        
        # Figure 4: Uncertainty Calibration
        if y_std is not None and f_std is not None:
            fig4 = plt.figure(figsize=(10, 4.5))
            
            ax1 = fig4.add_subplot(121)
            plot_uncertainty_calibration(y_true, y_pred, y_std, ax=ax1, 
                                        title="Energy UQ Calibration")
            
            ax2 = fig4.add_subplot(122)
            plot_uncertainty_calibration(f_true, f_pred, f_std, ax=ax2,
                                        title="Force UQ Calibration")
            
            fig4.tight_layout()
            figures['calibration'] = fig4
    
    else:
        # Energy-only plots
        fig1 = plt.figure(figsize=(10, 4.5))
        
        ax1 = fig1.add_subplot(121)
        plot_energy_parity(y_true, y_pred, y_std, ax=ax1)
        
        ax2 = fig1.add_subplot(122)
        plot_error_distribution(y_true, y_pred, ax=ax2, title="Energy Errors")
        
        fig1.tight_layout()
        figures['energy'] = fig1
        
        if y_std is not None:
            fig2 = plt.figure(figsize=(5, 4.5))
            plot_uncertainty_calibration(y_true, y_pred, y_std, title="Energy UQ Calibration")
            fig2.tight_layout()
            figures['calibration'] = fig2
    
    # Save figures
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for name, fig in figures.items():
            filename = f"{prefix}{name}.png" if prefix else f"{name}.png"
            fig.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
            print(f"Saved: {output_dir / filename}")
    
    if show:
        plt.show()
    
    return figures


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot force prediction analysis')
    parser.add_argument('pred_file', type=str, help='Path to prediction parquet/CSV file')
    parser.add_argument('--output', '-o', type=str, help='Output directory for plots')
    parser.add_argument('--prefix', type=str, default='', help='Filename prefix')
    parser.add_argument('--no-show', action='store_true', help='Do not display plots')
    
    args = parser.parse_args()
    
    pred_file = Path(args.pred_file)
    output_dir = Path(args.output) if args.output else None
    
    print(f"Analyzing: {pred_file}")
    figures = plot_comprehensive_force_analysis(
        pred_file,
        output_dir=output_dir,
        show=not args.no_show,
        prefix=args.prefix
    )
    
    print(f"\nGenerated {len(figures)} figures")


if __name__ == '__main__':
    main()
