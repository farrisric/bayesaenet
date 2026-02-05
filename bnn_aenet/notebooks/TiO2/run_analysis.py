#!/usr/bin/env python
"""Run TiO2 analysis - compute metrics for all experiments."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from analysis.config import TIO2_CONFIG
from analysis.data_loader import PredictionLoader
from analysis.metrics import compute_all_metrics
import pandas as pd
from tqdm import tqdm

def main():
    print("=" * 80)
    print("TiO2 Results Analysis")
    print("=" * 80)
    
    # Initialize
    loader = PredictionLoader(TIO2_CONFIG)
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)
    
    # Check configuration
    print(f"\nDataset: {TIO2_CONFIG.name}")
    print(f"Logs dir: {TIO2_CONFIG.logs_dir}")
    print(f"Logs exist: {TIO2_CONFIG.logs_dir.exists()}")
    print(f"Methods: {TIO2_CONFIG.methods}")
    
    # Check available runs
    print("\n" + "=" * 80)
    print("Available runs:")
    print("=" * 80)
    for method in TIO2_CONFIG.methods:
        print(f"\n{method.upper()}:")
        for size in ['high', 'low']:
            try:
                runs = loader.get_available_runs(method, size)
                print(f"  {size:5s}: {runs}")
            except Exception as e:
                print(f"  {size:5s}: Error - {e}")
    
    # Collect results
    print("\n" + "=" * 80)
    print("Computing metrics...")
    print("=" * 80)
    
    results = []
    failed = []
    
    for method in tqdm(TIO2_CONFIG.methods, desc="Methods"):
        for size_label, size_key in [('high', 'high'), ('low', 'low')]:
            # Get available runs
            try:
                available_runs = loader.get_available_runs(method, size_key)
            except Exception as e:
                print(f"\n✗ Could not get runs for {method} {size_label}: {e}")
                continue
            
            if not available_runs:
                print(f"\n✗ No runs found for {method} {size_label}")
                continue
            
            for run in available_runs:
                try:
                    # Load predictions
                    y_true, y_pred, y_std, n_atoms = loader.load_method_predictions(
                        method, size_key, run, split='test'
                    )
                    
                    # Compute metrics
                    metrics = compute_all_metrics(y_true, y_pred, y_std)
                    
                    # Store results
                    metrics.update({
                        'Method': method,
                        'Size': size_label,
                        'Run': run,
                        'Split': 'Test'
                    })
                    results.append(metrics)
                    
                    tqdm.write(f"✓ {method:4s} {size_label:4s} run {run}")
                    
                except Exception as e:
                    error_msg = f"{method} {size_label} run {run}: {e}"
                    tqdm.write(f"✗ {error_msg}")
                    failed.append(error_msg)
    
    # Save results
    print("\n" + "=" * 80)
    print("Saving results...")
    print("=" * 80)
    
    if results:
        df = pd.DataFrame(results)
        output_path = output_dir / 'uq_metrics_Test.csv'
        df.to_csv(output_path, index=False)
        print(f"\n✓ Metrics saved to: {output_path}")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        
        # Summary
        print("\n" + "=" * 80)
        print("Summary Statistics:")
        print("=" * 80)
        summary = df.groupby(['Method', 'Size'])[['mae', 'rmse']].agg(['mean', 'std'])
        print(summary)
    else:
        print("\n✗ No results to save!")
    
    if failed:
        print(f"\n✗ Failed experiments: {len(failed)}")
        with open(output_dir / 'failed_experiments.txt', 'w') as f:
            f.write("\n".join(failed))
    
    print("\n" + "=" * 80)
    print(f"Done! Processed {len(results)} experiments")
    print("=" * 80)

if __name__ == '__main__':
    main()
