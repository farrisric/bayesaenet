#!/usr/bin/env python3
"""Analyze results and create publication figures.

This script processes all trained models and creates:
- Summary statistics table
- Parity plots for all methods
- Uncertainty calibration comparisons
- Method comparison bar charts
- LaTeX-formatted tables

Usage:
    python scripts/final/analyze_results.py --logs-dir bnn_aenet/logs/final
    python scripts/final/analyze_results.py --dataset qm7 --output-dir plots/qm7
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT / "bnn_aenet"))

from analysis.metrics import (
    compute_energy_metrics,
    compute_force_metrics,
    compute_uncertainty_metrics,
)
from analysis.plotting import (
    setup_plot_style,
    plot_parity,
    plot_residuals,
    plot_uncertainty_calibration,
    plot_training_curves,
    plot_method_comparison,
    create_publication_figure,
    METHOD_COLORS,
    METHOD_LABELS,
    FIGSIZE_SINGLE,
    FIGSIZE_DOUBLE,
    DPI,
)


def load_predictions(pred_path: Path) -> Dict:
    """Load prediction results from parquet or csv file."""
    if pred_path.suffix == '.parquet':
        df = pd.read_parquet(pred_path)
    elif pred_path.suffix == '.csv':
        df = pd.read_csv(pred_path)
    else:
        raise ValueError(f"Unsupported format: {pred_path.suffix}")
    
    results = {
        'y_true': df['y_true'].values if 'y_true' in df else df['target'].values,
        'y_pred': df['y_pred'].values if 'y_pred' in df else df['prediction'].values,
    }
    
    # Load uncertainty if available
    for col in ['y_std', 'std', 'uncertainty', 'epistemic_std']:
        if col in df.columns:
            results['y_std'] = df[col].values
            break
    
    return results


def find_prediction_files(logs_dir: Path, method: str, dataset: str) -> List[Path]:
    """Find prediction files for a method and dataset."""
    patterns = [
        f"*{method}*{dataset}*/predictions*.parquet",
        f"*{method}*{dataset}*/pred*.parquet",
        f"*{dataset}*{method}*/predictions*.parquet",
        f"*{method}*{dataset}*.parquet",
    ]
    
    files = []
    for pattern in patterns:
        files.extend(logs_dir.rglob(pattern))
    
    return list(set(files))


def analyze_method(
    pred_files: List[Path],
    method: str,
    dataset: str,
    split: str = "test"
) -> Dict:
    """Analyze predictions for a single method."""
    all_results = []
    
    for pred_file in pred_files:
        if split.lower() in pred_file.stem.lower():
            try:
                results = load_predictions(pred_file)
                all_results.append(results)
            except Exception as e:
                print(f"Warning: Could not load {pred_file}: {e}")
    
    if not all_results:
        return {}
    
    # Aggregate results (for ensembles, compute mean predictions)
    if len(all_results) == 1:
        agg_results = all_results[0]
    else:
        # For Deep Ensemble, average predictions
        y_true = all_results[0]['y_true']
        y_preds = np.array([r['y_pred'] for r in all_results])
        y_pred_mean = np.mean(y_preds, axis=0)
        y_pred_std = np.std(y_preds, axis=0)  # Epistemic uncertainty
        
        agg_results = {
            'y_true': y_true,
            'y_pred': y_pred_mean,
            'y_std': y_pred_std,
        }
    
    # Compute metrics
    y_std = agg_results.get('y_std')
    metrics = compute_energy_metrics(
        agg_results['y_true'],
        agg_results['y_pred'],
        y_std
    )
    
    if y_std is not None:
        uq_metrics = compute_uncertainty_metrics(
            agg_results['y_true'],
            agg_results['y_pred'],
            y_std
        )
        metrics.update(uq_metrics)
    
    return {
        'predictions': agg_results,
        'metrics': metrics,
        'method': method,
        'dataset': dataset,
        'n_files': len(all_results),
    }


def create_summary_table(results: Dict[str, Dict], output_path: Path = None) -> pd.DataFrame:
    """Create summary statistics table."""
    rows = []
    
    for key, data in results.items():
        if 'metrics' not in data:
            continue
        
        method = data.get('method', key)
        dataset = data.get('dataset', '')
        metrics = data['metrics']
        
        row = {
            'Method': METHOD_LABELS.get(method.lower(), method),
            'Dataset': dataset.upper(),
            'RMSE': f"{metrics.get('rmse', 0):.4f}",
            'MAE': f"{metrics.get('mae', 0):.4f}",
            'R²': f"{metrics.get('r2', 0):.4f}",
            'MaxErr': f"{metrics.get('max_err', 0):.4f}",
        }
        
        # Add UQ metrics if available
        if 'nll' in metrics:
            row['NLL'] = f"{metrics['nll']:.4f}"
        if 'picp_95' in metrics:
            row['PICP (95%)'] = f"{metrics['picp_95']:.2%}"
        if 'ece' in metrics:
            row['ECE'] = f"{metrics['ece']:.4f}"
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Summary table saved to {output_path}")
    
    return df


def export_latex_table(df: pd.DataFrame, output_path: Path):
    """Export summary table as LaTeX."""
    latex = df.to_latex(index=False, escape=False)
    
    with open(output_path, 'w') as f:
        f.write(latex)
    
    print(f"LaTeX table saved to {output_path}")


def create_comparison_plots(
    results: Dict[str, Dict],
    output_dir: Path,
    dataset: str = ""
):
    """Create method comparison plots."""
    setup_plot_style()
    
    # Collect metrics for comparison
    methods = []
    rmse_values = []
    mae_values = []
    nll_values = []
    
    for key, data in results.items():
        if 'metrics' not in data:
            continue
        
        methods.append(data.get('method', key))
        rmse_values.append(data['metrics'].get('rmse', 0))
        mae_values.append(data['metrics'].get('mae', 0))
        nll_values.append(data['metrics'].get('nll', np.nan))
    
    # RMSE comparison
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    colors = [METHOD_COLORS.get(m.lower(), '#333333') for m in methods]
    labels = [METHOD_LABELS.get(m.lower(), m) for m in methods]
    
    bars = ax.bar(range(len(methods)), rmse_values, color=colors)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('RMSE (eV/atom)')
    ax.set_title(f'RMSE Comparison{" - " + dataset if dataset else ""}')
    
    for bar, val in zip(bars, rmse_values):
        ax.annotate(f'{val:.4f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
    
    plt.tight_layout()
    fig.savefig(output_dir / f'rmse_comparison_{dataset.lower()}.png', dpi=DPI)
    plt.close()
    
    # NLL comparison (if available)
    valid_nll = [(m, v) for m, v in zip(methods, nll_values) if not np.isnan(v)]
    if valid_nll:
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        methods_nll, nll_vals = zip(*valid_nll)
        colors_nll = [METHOD_COLORS.get(m.lower(), '#333333') for m in methods_nll]
        labels_nll = [METHOD_LABELS.get(m.lower(), m) for m in methods_nll]
        
        bars = ax.bar(range(len(methods_nll)), nll_vals, color=colors_nll)
        ax.set_xticks(range(len(methods_nll)))
        ax.set_xticklabels(labels_nll, rotation=45, ha='right')
        ax.set_ylabel('Negative Log-Likelihood')
        ax.set_title(f'NLL Comparison{" - " + dataset if dataset else ""}')
        
        plt.tight_layout()
        fig.savefig(output_dir / f'nll_comparison_{dataset.lower()}.png', dpi=DPI)
        plt.close()


def create_parity_plots(
    results: Dict[str, Dict],
    output_dir: Path,
    dataset: str = ""
):
    """Create parity plots for all methods."""
    setup_plot_style()
    
    n_methods = len([k for k in results if 'predictions' in results[k]])
    if n_methods == 0:
        return
    
    # Grid layout
    ncols = min(3, n_methods)
    nrows = (n_methods + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    if n_methods == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    idx = 0
    for key, data in results.items():
        if 'predictions' not in data:
            continue
        
        method = data.get('method', key)
        pred = data['predictions']
        color = METHOD_COLORS.get(method.lower(), '#1f77b4')
        label = METHOD_LABELS.get(method.lower(), method)
        
        plot_parity(
            pred['y_true'],
            pred['y_pred'],
            pred.get('y_std'),
            ax=axes[idx],
            title=label,
            color=color,
        )
        idx += 1
    
    # Hide unused axes
    for i in range(idx, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'Energy Predictions{" - " + dataset if dataset else ""}', y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / f'parity_all_{dataset.lower()}.png', dpi=DPI)
    plt.close()


def create_calibration_plots(
    results: Dict[str, Dict],
    output_dir: Path,
    dataset: str = ""
):
    """Create calibration plots for methods with uncertainty."""
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
    
    for key, data in results.items():
        if 'predictions' not in data:
            continue
        
        pred = data['predictions']
        if 'y_std' not in pred or pred['y_std'] is None:
            continue
        
        method = data.get('method', key)
        color = METHOD_COLORS.get(method.lower(), '#1f77b4')
        label = METHOD_LABELS.get(method.lower(), method)
        
        from analysis.metrics import compute_calibration_curve
        expected, observed = compute_calibration_curve(
            pred['y_true'],
            pred['y_pred'],
            pred['y_std']
        )
        
        ax.plot(expected, observed, 'o-', color=color, label=label, markersize=4)
    
    ax.set_xlabel('Expected Confidence')
    ax.set_ylabel('Observed Confidence')
    ax.set_title(f'Uncertainty Calibration{" - " + dataset if dataset else ""}')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    fig.savefig(output_dir / f'calibration_{dataset.lower()}.png', dpi=DPI)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze BNN-AENET results")
    parser.add_argument(
        "--logs-dir", type=Path, default=Path("bnn_aenet/logs/final"),
        help="Directory containing prediction logs"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("plots/final"),
        help="Output directory for plots and tables"
    )
    parser.add_argument(
        "--dataset", type=str, choices=["qm7", "tio2", "all"], default="all",
        help="Dataset to analyze"
    )
    parser.add_argument(
        "--split", type=str, default="test",
        help="Data split to analyze (train, val, test)"
    )
    
    args = parser.parse_args()
    
    # Setup
    logs_dir = PROJECT_ROOT / args.logs_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("BNN-AENET Results Analysis")
    print(f"{'='*60}")
    print(f"Logs directory: {logs_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")
    print(f"{'='*60}\n")
    
    # Methods and datasets to analyze
    methods = ["lrt", "fo", "rad", "de"]
    datasets = ["qm7", "tio2"] if args.dataset == "all" else [args.dataset]
    
    all_results = {}
    
    for dataset in datasets:
        print(f"\nAnalyzing {dataset.upper()}...")
        dataset_results = {}
        
        for method in methods:
            print(f"  Processing {method.upper()}...", end=" ")
            
            pred_files = find_prediction_files(logs_dir, method, dataset)
            
            if not pred_files:
                print("No predictions found")
                continue
            
            results = analyze_method(pred_files, method, dataset, args.split)
            
            if results:
                dataset_results[f"{method}_{dataset}"] = results
                metrics = results.get('metrics', {})
                print(f"RMSE: {metrics.get('rmse', 0):.4f}, "
                      f"MAE: {metrics.get('mae', 0):.4f}")
            else:
                print("Analysis failed")
        
        all_results.update(dataset_results)
        
        # Create plots for this dataset
        if dataset_results:
            dataset_output = output_dir / dataset
            dataset_output.mkdir(exist_ok=True)
            
            create_parity_plots(dataset_results, dataset_output, dataset)
            create_comparison_plots(dataset_results, dataset_output, dataset)
            create_calibration_plots(dataset_results, dataset_output, dataset)
    
    # Create summary table
    if all_results:
        summary_df = create_summary_table(all_results, output_dir / "summary.csv")
        export_latex_table(summary_df, output_dir / "summary.tex")
        
        print(f"\n{'='*60}")
        print("Summary Statistics")
        print(f"{'='*60}")
        print(summary_df.to_string(index=False))
    
    print(f"\n\nAnalysis complete! Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
