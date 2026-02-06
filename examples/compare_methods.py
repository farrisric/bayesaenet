#!/usr/bin/env python3
"""Example: Compare BNN methods on a dataset.

This script trains and compares different BNN methods (LRT, Flipout, Radial)
and Deep Ensemble on a given dataset.

Usage:
    python examples/compare_methods.py --dataset qm7 --epochs 500
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT / "bnn_aenet"))

import hydra
from omegaconf import OmegaConf


def train_method(method: str, dataset: str, seed: int, max_epochs: int) -> dict:
    """Train a single method and return metrics."""
    
    # Map method to experiment config
    method_to_config = {
        "lrt": f"final/lrt_{dataset}",
        "fo": f"final/fo_{dataset}",
        "rad": f"final/rad_{dataset}",
        "de": f"final/de_{dataset}",
    }
    
    exp_config = method_to_config.get(method)
    if exp_config is None:
        raise ValueError(f"Unknown method: {method}")
    
    # Build overrides
    overrides = [
        f"experiment={exp_config}",
        f"seed={seed}",
        f"run_name={method}_{dataset}_comparison",
        f"trainer.max_epochs={max_epochs}",
        "callbacks.early_stopping.patience=50",
    ]
    
    # Initialize Hydra
    with hydra.initialize(config_path="../bnn_aenet/configs", version_base="1.3"):
        cfg = hydra.compose(config_name="train", overrides=overrides)
    
    print(f"\nTraining {method.upper()} on {dataset.upper()}...")
    
    from tasks.train import train
    
    try:
        metrics = train(cfg)
        return {
            "method": method,
            "dataset": dataset,
            "seed": seed,
            **metrics
        }
    except Exception as e:
        print(f"Error training {method}: {e}")
        return {
            "method": method,
            "dataset": dataset,
            "seed": seed,
            "error": str(e)
        }


def plot_comparison(results: list, output_path: str = None):
    """Plot method comparison."""
    df = pd.DataFrame(results)
    
    # Filter out errors
    df = df[~df.get('error', pd.Series([None]*len(df))).notna()]
    
    if len(df) == 0:
        print("No successful results to plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # RMSE comparison
    metrics_rmse = ['test/rmse', 'val/rmse']
    for metric in metrics_rmse:
        if metric in df.columns:
            rmse_col = metric
            break
    else:
        rmse_col = None
    
    if rmse_col:
        methods = df['method'].values
        rmse_values = df[rmse_col].values
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        axes[0].bar(range(len(methods)), rmse_values, color=colors[:len(methods)])
        axes[0].set_xticks(range(len(methods)))
        axes[0].set_xticklabels([m.upper() for m in methods], rotation=45)
        axes[0].set_ylabel('RMSE (eV/atom)')
        axes[0].set_title('RMSE Comparison')
        
        for i, (m, v) in enumerate(zip(methods, rmse_values)):
            axes[0].annotate(f'{v:.4f}', xy=(i, v), ha='center',
                            va='bottom', fontsize=8)
    
    # NLL comparison (if available)
    nll_col = None
    for col in ['test/nll', 'val/nll', 'nll']:
        if col in df.columns:
            nll_col = col
            break
    
    if nll_col:
        nll_values = df[nll_col].values
        valid = ~np.isnan(nll_values)
        
        if valid.sum() > 0:
            axes[1].bar(range(valid.sum()), nll_values[valid], color=colors[:valid.sum()])
            axes[1].set_xticks(range(valid.sum()))
            axes[1].set_xticklabels([m.upper() for m in df['method'].values[valid]], rotation=45)
            axes[1].set_ylabel('Negative Log-Likelihood')
            axes[1].set_title('NLL Comparison')
    else:
        axes[1].text(0.5, 0.5, 'NLL not available', ha='center', va='center')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Compare BNN methods")
    parser.add_argument("--dataset", type=str, default="qm7",
                        choices=["qm7", "tio2"],
                        help="Dataset to use")
    parser.add_argument("--methods", type=str, nargs="+",
                        default=["lrt", "fo", "rad"],
                        help="Methods to compare")
    parser.add_argument("--epochs", type=int, default=500,
                        help="Max training epochs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for comparison plot")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("BNN Method Comparison")
    print("=" * 60)
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Methods: {[m.upper() for m in args.methods]}")
    print(f"Max epochs: {args.epochs}")
    print(f"Seed: {args.seed}")
    print("=" * 60)
    
    results = []
    
    for method in args.methods:
        result = train_method(method, args.dataset, args.seed, args.epochs)
        results.append(result)
        
        if "error" not in result:
            print(f"\n{method.upper()} Results:")
            for key, value in result.items():
                if key not in ["method", "dataset", "seed"]:
                    print(f"  {key}: {value}")
    
    # Save results
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"comparison_{args.dataset}_{timestamp}.csv"
    df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")
    
    # Plot comparison
    plot_comparison(results, args.output)
    
    print("\n" + "=" * 60)
    print("Comparison Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
