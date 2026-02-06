"""Analyze predictions with force data and compute comprehensive metrics."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import argparse
import json

from .metrics import compute_combined_metrics


def load_predictions(pred_file: Path) -> pd.DataFrame:
    """Load prediction results from parquet or CSV file.
    
    Args:
        pred_file: Path to prediction file
        
    Returns:
        DataFrame with predictions
    """
    if pred_file.suffix == '.parquet':
        df = pd.read_parquet(pred_file)
    elif pred_file.suffix == '.csv':
        df = pd.read_csv(pred_file)
    else:
        raise ValueError(f"Unsupported file format: {pred_file.suffix}")
    
    return df


def compute_metrics_from_predictions(
    df: pd.DataFrame,
    has_forces: bool = True,
    per_component: bool = False
) -> Dict[str, float]:
    """Compute all metrics from prediction DataFrame.
    
    Args:
        df: DataFrame with columns: true, preds, stds, true_forces, pred_forces, std_forces
        has_forces: Whether force data is present
        per_component: Whether to compute per-component force metrics
        
    Returns:
        Dictionary of all computed metrics
    """
    # Extract energy data
    y_true = df['true'].values
    y_pred = df['preds'].values
    y_std = df['stds'].values if 'stds' in df.columns else None
    
    # Extract force data if available
    f_true = None
    f_pred = None
    f_std = None
    
    if has_forces and 'true_forces' in df.columns and 'pred_forces' in df.columns:
        # Handle cases where forces might be None or NaN
        force_mask = df['true_forces'].notna() & df['pred_forces'].notna()
        
        if force_mask.sum() > 0:
            # Concatenate all force components
            f_true_list = []
            f_pred_list = []
            f_std_list = [] if 'std_forces' in df.columns else None
            
            for idx in df[force_mask].index:
                true_f = df.loc[idx, 'true_forces']
                pred_f = df.loc[idx, 'pred_forces']
                
                # Handle different storage formats
                if isinstance(true_f, (list, np.ndarray)):
                    f_true_list.append(np.array(true_f))
                    f_pred_list.append(np.array(pred_f))
                    if f_std_list is not None and 'std_forces' in df.columns:
                        std_f = df.loc[idx, 'std_forces']
                        f_std_list.append(np.array(std_f) if isinstance(std_f, (list, np.ndarray)) else std_f)
            
            if len(f_true_list) > 0:
                f_true = np.concatenate(f_true_list)
                f_pred = np.concatenate(f_pred_list)
                if f_std_list is not None and len(f_std_list) > 0:
                    f_std = np.concatenate(f_std_list)
    
    # Compute combined metrics
    metrics = compute_combined_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_std=y_std,
        f_true=f_true,
        f_pred=f_pred,
        f_std=f_std
    )
    
    # Add summary statistics
    metrics['n_structures'] = len(df)
    if f_true is not None:
        metrics['n_force_components'] = len(f_true)
        metrics['n_atoms_with_forces'] = len(f_true) // 3 if len(f_true) % 3 == 0 else np.nan
    
    return metrics


def analyze_prediction_file(
    pred_file: Path,
    output_file: Optional[Path] = None,
    verbose: bool = True
) -> Dict[str, float]:
    """Analyze a single prediction file and compute metrics.
    
    Args:
        pred_file: Path to prediction parquet/CSV file
        output_file: Optional path to save metrics JSON
        verbose: Whether to print metrics
        
    Returns:
        Dictionary of computed metrics
    """
    if verbose:
        print(f"\nAnalyzing: {pred_file.name}")
        print("=" * 60)
    
    # Load predictions
    df = load_predictions(pred_file)
    
    # Check if force data exists
    has_forces = 'true_forces' in df.columns and 'pred_forces' in df.columns
    
    if verbose:
        print(f"Structures: {len(df)}")
        print(f"Has forces: {has_forces}")
    
    # Compute metrics
    metrics = compute_metrics_from_predictions(df, has_forces=has_forces)
    
    # Print metrics
    if verbose:
        print("\nEnergy Metrics:")
        print("-" * 40)
        for key in ['mae', 'rmse', 'maxerr', 'r2score']:
            if key in metrics:
                print(f"  {key:15s}: {metrics[key]:.6f}")
        
        if 'sharp' in metrics:
            print("\nEnergy Uncertainty Metrics:")
            print("-" * 40)
            for key in ['sharp', 'overlap', 'nll']:
                if key in metrics:
                    print(f"  {key:15s}: {metrics[key]:.6f}")
        
        if has_forces and 'force_mae' in metrics:
            print("\nForce Metrics:")
            print("-" * 40)
            for key in ['force_mae', 'force_rmse', 'force_maxerr', 'force_r2']:
                if key in metrics:
                    print(f"  {key:15s}: {metrics[key]:.6f}")
            
            if 'force_mag_mae' in metrics:
                print("\nForce Vector Metrics:")
                print("-" * 40)
                for key in ['force_mag_mae', 'force_mag_rmse', 'force_angular_mae']:
                    if key in metrics:
                        print(f"  {key:15s}: {metrics[key]:.6f}")
            
            if 'force_sharp' in metrics:
                print("\nForce Uncertainty Metrics:")
                print("-" * 40)
                for key in ['force_sharp', 'force_overlap', 'force_nll']:
                    if key in metrics:
                        print(f"  {key:15s}: {metrics[key]:.6f}")
    
    # Save to file if requested
    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        if verbose:
            print(f"\nMetrics saved to: {output_file}")
    
    return metrics


def analyze_multiple_runs(
    pred_dir: Path,
    pattern: str = "*_val.parquet",
    output_file: Optional[Path] = None
) -> pd.DataFrame:
    """Analyze multiple prediction files and create summary.
    
    Args:
        pred_dir: Directory containing prediction files
        pattern: Glob pattern to match prediction files
        output_file: Optional path to save summary CSV
        
    Returns:
        DataFrame with metrics for each run
    """
    pred_files = sorted(pred_dir.glob(pattern))
    
    if len(pred_files) == 0:
        raise ValueError(f"No files matching pattern '{pattern}' found in {pred_dir}")
    
    print(f"Found {len(pred_files)} prediction files")
    
    # Analyze each file
    results = []
    for pred_file in pred_files:
        try:
            metrics = analyze_prediction_file(pred_file, verbose=False)
            metrics['file'] = pred_file.name
            results.append(metrics)
        except Exception as e:
            print(f"Error analyzing {pred_file.name}: {e}")
            continue
    
    # Create summary DataFrame
    df_summary = pd.DataFrame(results)
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS (Mean ± Std)")
    print("=" * 80)
    
    # Energy metrics
    print("\nEnergy Metrics:")
    for key in ['mae', 'rmse', 'maxerr', 'r2score']:
        if key in df_summary.columns:
            mean_val = df_summary[key].mean()
            std_val = df_summary[key].std()
            print(f"  {key:15s}: {mean_val:.6f} ± {std_val:.6f}")
    
    # Force metrics
    if 'force_mae' in df_summary.columns:
        print("\nForce Metrics:")
        for key in ['force_mae', 'force_rmse', 'force_maxerr', 'force_r2']:
            if key in df_summary.columns:
                mean_val = df_summary[key].mean()
                std_val = df_summary[key].std()
                print(f"  {key:15s}: {mean_val:.6f} ± {std_val:.6f}")
    
    # Save summary
    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df_summary.to_csv(output_file, index=False)
        print(f"\nSummary saved to: {output_file}")
    
    return df_summary


def main():
    parser = argparse.ArgumentParser(description='Analyze BNN predictions with forces')
    parser.add_argument('pred_path', type=str, help='Path to prediction file or directory')
    parser.add_argument('--pattern', type=str, default='*_val.parquet',
                       help='Glob pattern for multiple files')
    parser.add_argument('--output', type=str, help='Output file for metrics')
    parser.add_argument('--multiple', action='store_true',
                       help='Analyze multiple files in directory')
    
    args = parser.parse_args()
    
    pred_path = Path(args.pred_path)
    output_path = Path(args.output) if args.output else None
    
    if args.multiple or pred_path.is_dir():
        # Analyze multiple files
        analyze_multiple_runs(pred_path, pattern=args.pattern, output_file=output_path)
    else:
        # Analyze single file
        analyze_prediction_file(pred_path, output_file=output_path)


if __name__ == '__main__':
    main()
