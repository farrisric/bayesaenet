#!/usr/bin/env python
"""
Example script showing how to analyze force predictions from BNN_Forces_Aux models.

Usage:
    # Analyze a single prediction file
    python analyze_forces_example.py /path/to/predictions/bnn_forces_aux_000_val.parquet
    
    # Analyze all predictions in a directory
    python analyze_forces_example.py /path/to/predictions/ --multiple
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bnn_aenet.analysis.analyze_force_predictions import (
    analyze_prediction_file,
    analyze_multiple_runs
)


def example_single_file():
    """Example: Analyze a single prediction file."""
    # Replace with your actual prediction file path
    pred_file = Path("bnn_aenet/logs/lrt_hps_gpu/runs/pred_lrt_forces_aux_0/bnn_forces_aux_000_val.parquet")
    
    if not pred_file.exists():
        print(f"File not found: {pred_file}")
        print("This is just an example. Replace with your actual file path.")
        return
    
    # Analyze and save metrics
    metrics = analyze_prediction_file(
        pred_file,
        output_file=pred_file.parent / f"{pred_file.stem}_metrics.json",
        verbose=True
    )
    
    print(f"\nComputed {len(metrics)} metrics:")
    print(f"  - Energy RMSE: {metrics['rmse']:.4f}")
    if 'force_rmse' in metrics:
        print(f"  - Force RMSE: {metrics['force_rmse']:.4f}")
        print(f"  - Force magnitude MAE: {metrics.get('force_mag_mae', 'N/A')}")


def example_multiple_files():
    """Example: Analyze multiple prediction files (e.g., ensemble runs)."""
    # Replace with your actual directory
    pred_dir = Path("bnn_aenet/logs/lrt_hps_gpu/runs/")
    
    if not pred_dir.exists():
        print(f"Directory not found: {pred_dir}")
        print("This is just an example. Replace with your actual directory path.")
        return
    
    # Analyze all validation predictions
    df_summary = analyze_multiple_runs(
        pred_dir,
        pattern="*_val.parquet",  # Match all validation predictions
        output_file=pred_dir / "force_predictions_summary.csv"
    )
    
    print(f"\nAnalyzed {len(df_summary)} runs")
    print(f"Average energy RMSE: {df_summary['rmse'].mean():.4f}")
    if 'force_rmse' in df_summary.columns:
        print(f"Average force RMSE: {df_summary['force_rmse'].mean():.4f}")


def quick_comparison():
    """Example: Compare energy-only vs force-trained models."""
    # Paths to predictions from different models
    energy_only_pred = Path("bnn_aenet/logs/lrt_train/runs/pred_0/bnn_lrt_000_val.parquet")
    with_forces_pred = Path("bnn_aenet/logs/lrt_forces_train/runs/pred_0/bnn_forces_aux_000_val.parquet")
    
    results = {}
    
    for name, pred_file in [("Energy-Only", energy_only_pred), ("With Forces", with_forces_pred)]:
        if pred_file.exists():
            print(f"\n{'='*60}")
            print(f"Model: {name}")
            print(f"{'='*60}")
            metrics = analyze_prediction_file(pred_file, verbose=True)
            results[name] = metrics
        else:
            print(f"\n{name} predictions not found: {pred_file}")
    
    # Compare
    if len(results) == 2:
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        
        for metric in ['rmse', 'mae', 'sharp', 'force_rmse', 'force_mae']:
            if metric in results["Energy-Only"] and metric in results["With Forces"]:
                val1 = results["Energy-Only"][metric]
                val2 = results["With Forces"][metric]
                diff = val2 - val1
                print(f"{metric:20s}: {val1:.6f} -> {val2:.6f} (Δ={diff:+.6f})")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze force predictions - Examples')
    parser.add_argument('pred_path', type=str, nargs='?',
                       help='Path to prediction file or directory')
    parser.add_argument('--multiple', action='store_true',
                       help='Analyze multiple files in directory')
    parser.add_argument('--compare', action='store_true',
                       help='Run comparison example')
    
    args = parser.parse_args()
    
    if args.compare:
        quick_comparison()
    elif args.pred_path:
        pred_path = Path(args.pred_path)
        if args.multiple or pred_path.is_dir():
            pred_dir = pred_path if pred_path.is_dir() else pred_path.parent
            df_summary = analyze_multiple_runs(
                pred_dir,
                pattern="*_val.parquet",
                output_file=pred_dir / "metrics_summary.csv"
            )
        else:
            metrics = analyze_prediction_file(
                pred_path,
                output_file=pred_path.parent / f"{pred_path.stem}_metrics.json",
                verbose=True
            )
    else:
        print("Examples of how to use force prediction analysis:\n")
        print("1. Analyze single file:")
        print("   example_single_file()")
        print("\n2. Analyze multiple files:")
        print("   example_multiple_files()")
        print("\n3. Compare models:")
        print("   quick_comparison()")
        print("\nRun with --help for CLI usage")
