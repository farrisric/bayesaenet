#!/usr/bin/env python
"""Quick QM7 analysis script"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# QM7 parameters (from training logs)
E_SCALING = 0.9754923797786934
E_SHIFT = -4.652443333333333

def denormalize_energy(vals, n_atoms):
    """Denormalize energy values."""
    return (vals / E_SCALING + E_SHIFT) * n_atoms

def denormalize_std(vals, n_atoms):
    """Denormalize standard deviation (no shift for std)."""
    return (vals / E_SCALING) * n_atoms

def compute_metrics(y_true, y_pred, y_std=None):
    """Compute prediction metrics."""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    max_err = np.max(np.abs(y_true - y_pred))
    
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'maxerr': max_err,
        'r2': 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    }
    
    if y_std is not None:
        # UQ metrics
        residuals = np.abs(y_true - y_pred)
        overlap = np.mean(residuals <= y_std) * 100
        sharpness = np.mean(y_std)
        
        # NLL (assuming Gaussian)
        nll = 0.5 * np.mean(np.log(2 * np.pi * y_std**2) + (residuals / y_std)**2)
        
        metrics.update({
            'overlap': overlap,
            'sharp': sharpness,
            'nll': nll
        })
    
    return metrics

def analyze_method(method_name, pred_dir):
    """Analyze predictions for a method."""
    print(f"\n{'='*80}")
    print(f"{method_name.upper()} Results")
    print(f"{'='*80}")
    
    results = []
    
    # Find all run directories
    run_dirs = sorted([d for d in pred_dir.iterdir() if d.is_dir() and d.name.startswith('pred_')])
    
    if not run_dirs:
        print(f"No prediction runs found in {pred_dir}")
        return None
    
    print(f"Found {len(run_dirs)} runs")
    
    for run_dir in run_dirs:
        run_name = run_dir.name
        
        # Find parquet file(s)
        parquet_files = list(run_dir.glob("*.parquet"))
        
        if not parquet_files:
            print(f"  ✗ {run_name}: No parquet files found")
            continue
        
        try:
            # Load data
            if method_name == 'de' and len(parquet_files) > 1:
                # Deep ensemble - load all members
                y_preds = []
                for pf in sorted(parquet_files):
                    df = pd.read_csv(pf)
                    pred_denorm = denormalize_energy(df['preds'].values, df['n_atoms'].values)
                    y_preds.append(pred_denorm)
                
                # Ensemble statistics
                y_pred = np.mean(y_preds, axis=0)
                y_std = np.std(y_preds, axis=0)
                
                # True values from first file
                df = pd.read_csv(parquet_files[0])
                y_true = denormalize_energy(df['true'].values, df['n_atoms'].values)
                
            else:
                # Single file
                df = pd.read_csv(parquet_files[0])
                y_true = denormalize_energy(df['true'].values, df['n_atoms'].values)
                y_pred = denormalize_energy(df['preds'].values, df['n_atoms'].values)
                
                # Check for std column (BNN methods) - could be 'std' or 'stds'
                if 'stds' in df.columns:
                    y_std = denormalize_std(df['stds'].values, df['n_atoms'].values)
                elif 'std' in df.columns:
                    y_std = denormalize_std(df['std'].values, df['n_atoms'].values)
                else:
                    y_std = None
            
            # Compute metrics
            metrics = compute_metrics(y_true, y_pred, y_std)
            metrics['run'] = run_name
            results.append(metrics)
            
            # Print
            has_uq = y_std is not None
            if has_uq:
                print(f"  ✓ {run_name:30s} MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}, "
                      f"Overlap={metrics['overlap']:.1f}%, Sharp={metrics['sharp']:.3f}")
            else:
                print(f"  ✓ {run_name:30s} MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}, "
                      f"MaxErr={metrics['maxerr']:.3f}")
            
        except Exception as e:
            print(f"  ✗ {run_name}: {e}")
    
    if not results:
        return None
    
    # Summary statistics
    df_results = pd.DataFrame(results)
    print(f"\n{'-'*80}")
    print(f"Summary Statistics (n={len(results)} runs):")
    print(f"{'-'*80}")
    
    metrics_to_show = ['mae', 'rmse', 'maxerr']
    if 'overlap' in df_results.columns:
        metrics_to_show.extend(['overlap', 'sharp', 'nll'])
    
    for metric in metrics_to_show:
        if metric in df_results.columns:
            mean_val = df_results[metric].mean()
            std_val = df_results[metric].std()
            print(f"  {metric.upper():10s}: {mean_val:.4f} ± {std_val:.4f}")
    
    return df_results

def main():
    print("="*80)
    print("QM7 Prediction Analysis")
    print("="*80)
    
    logs_dir = Path("/home/g15farris/bin/bayesaenet/bnn_aenet/logs")
    
    all_results = {}
    
    # Analyze DE
    de_dir = logs_dir / "de_pred" / "runs"
    if de_dir.exists():
        df_de = analyze_method('de', de_dir)
        if df_de is not None:
            all_results['de'] = df_de
    
    # Analyze LRT
    lrt_dir = logs_dir / "lrt_pred" / "runs"
    if lrt_dir.exists():
        df_lrt = analyze_method('lrt', lrt_dir)
        if df_lrt is not None:
            all_results['lrt'] = df_lrt
    
    # Analyze NN (if exists)
    nn_dir = logs_dir / "nn_pred" / "runs"
    if nn_dir.exists():
        df_nn = analyze_method('nn', nn_dir)
        if df_nn is not None:
            all_results['nn'] = df_nn
    
    # Overall comparison
    if len(all_results) > 1:
        print(f"\n{'='*80}")
        print("Method Comparison")
        print(f"{'='*80}")
        
        for method, df in all_results.items():
            mae_mean = df['mae'].mean()
            mae_std = df['mae'].std()
            rmse_mean = df['rmse'].mean()
            rmse_std = df['rmse'].std()
            
            print(f"\n{method.upper():5s}: MAE={mae_mean:.4f}±{mae_std:.4f}  RMSE={rmse_mean:.4f}±{rmse_std:.4f}")
            if 'overlap' in df.columns:
                print(f"       UQ: Overlap={df['overlap'].mean():.1f}%, Sharpness={df['sharp'].mean():.4f}")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

if __name__ == '__main__':
    main()
