#!/usr/bin/env python
"""
Plot comparison of HPS results across models.

Usage:
    python scripts/plot_hps_comparison.py
    python scripts/plot_hps_comparison.py --output figures/hps_comparison.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd


def load_all_studies():
    """Load all HPS studies."""
    base_path = Path(__file__).parent.parent / "bnn_aenet" / "results"
    
    studies_info = {
        "NN": (base_path / "nn" / "nn_forces.db", "nn_forces"),
        "LRT": (base_path / "bayesian" / "bnn_lrt_forces.db", "bnn_lrt_forces"),
        "Flipout": (base_path / "bayesian" / "bnn_fo_forces.db", "bnn_fo_forces"),
        "Radial": (base_path / "bayesian" / "bnn_rad_forces.db", "bnn_rad_forces"),
    }
    
    studies = {}
    for name, (db_path, study_name) in studies_info.items():
        if db_path.exists():
            try:
                storage = f"sqlite:///{db_path}"
                studies[name] = optuna.load_study(study_name=study_name, storage=storage)
            except Exception as e:
                print(f"Could not load {name}: {e}")
    
    return studies


def plot_optimization_history(studies: dict, ax):
    """Plot optimization history for all models."""
    colors = {"NN": "C0", "LRT": "C1", "Flipout": "C2", "Radial": "C3"}
    
    for name, study in studies.items():
        completed_trials = [t for t in study.trials 
                          if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed_trials:
            continue
        
        values = [t.value for t in completed_trials]
        best_values = np.minimum.accumulate(values)
        
        ax.plot(range(len(values)), best_values, 
                label=name, color=colors.get(name, "gray"), linewidth=2)
        ax.scatter(range(len(values)), values, 
                   color=colors.get(name, "gray"), alpha=0.3, s=20)
    
    ax.set_xlabel("Trial", fontsize=12)
    ax.set_ylabel("Best Total RMSE (Val)", fontsize=12)
    ax.set_title("Hyperparameter Optimization Progress", fontsize=14)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)


def plot_best_values_comparison(studies: dict, ax):
    """Bar chart comparing best values across models."""
    models = []
    best_values = []
    colors = []
    color_map = {"NN": "C0", "LRT": "C1", "Flipout": "C2", "Radial": "C3"}
    
    for name, study in studies.items():
        completed_trials = [t for t in study.trials 
                          if t.state == optuna.trial.TrialState.COMPLETE]
        if completed_trials:
            models.append(name)
            best_values.append(study.best_trial.value)
            colors.append(color_map.get(name, "gray"))
    
    bars = ax.bar(models, best_values, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, value in zip(bars, best_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.4f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel("Best Total RMSE (Val)", fontsize=12)
    ax.set_title("Best HPS Results by Model", fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')


def plot_parameter_importance(studies: dict, ax):
    """Plot parameter importance (if available)."""
    try:
        # Use the first available study for importance
        for name, study in studies.items():
            if len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]) >= 5:
                importance = optuna.importance.get_param_importances(study)
                
                params = list(importance.keys())[:8]  # Top 8 params
                values = [importance[p] for p in params]
                
                ax.barh(params, values, color='steelblue', edgecolor='black')
                ax.set_xlabel("Importance", fontsize=12)
                ax.set_title(f"Parameter Importance ({name})", fontsize=14)
                ax.grid(True, alpha=0.3, axis='x')
                return
        
        ax.text(0.5, 0.5, "Not enough trials\nfor importance analysis",
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.set_title("Parameter Importance", fontsize=14)
    except Exception as e:
        ax.text(0.5, 0.5, f"Could not compute\nimportance: {e}",
                ha='center', va='center', fontsize=10, transform=ax.transAxes)
        ax.set_title("Parameter Importance", fontsize=14)


def plot_trial_distribution(studies: dict, ax):
    """Plot distribution of trial outcomes."""
    models = []
    completed = []
    failed = []
    pruned = []
    
    for name, study in studies.items():
        models.append(name)
        trials = study.trials
        completed.append(len([t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]))
        failed.append(len([t for t in trials if t.state == optuna.trial.TrialState.FAIL]))
        pruned.append(len([t for t in trials if t.state == optuna.trial.TrialState.PRUNED]))
    
    x = np.arange(len(models))
    width = 0.25
    
    ax.bar(x - width, completed, width, label='Completed', color='green', alpha=0.7)
    ax.bar(x, failed, width, label='Failed', color='red', alpha=0.7)
    ax.bar(x + width, pruned, width, label='Pruned', color='orange', alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Number of Trials", fontsize=12)
    ax.set_title("Trial Outcomes by Model", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')


def main():
    parser = argparse.ArgumentParser(description="Plot HPS comparison")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path")
    args = parser.parse_args()
    
    studies = load_all_studies()
    
    if not studies:
        print("No HPS studies found!")
        return
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Hyperparameter Search Analysis", fontsize=16, fontweight='bold')
    
    plot_optimization_history(studies, axes[0, 0])
    plot_best_values_comparison(studies, axes[0, 1])
    plot_parameter_importance(studies, axes[1, 0])
    plot_trial_distribution(studies, axes[1, 1])
    
    plt.tight_layout()
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
