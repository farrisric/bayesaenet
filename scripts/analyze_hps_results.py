#!/usr/bin/env python
"""
Analyze Optuna HPS results and extract best hyperparameters.

Usage:
    python scripts/analyze_hps_results.py
    python scripts/analyze_hps_results.py --model lrt
    python scripts/analyze_hps_results.py --all
"""

import argparse
import sqlite3
from pathlib import Path

import optuna
import pandas as pd


def load_study(db_path: str, study_name: str) -> optuna.Study:
    """Load an Optuna study from SQLite database."""
    storage = f"sqlite:///{db_path}"
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        return study
    except Exception as e:
        print(f"Error loading study {study_name}: {e}")
        return None


def analyze_study(study: optuna.Study, model_name: str):
    """Analyze and print study results."""
    print(f"\n{'='*60}")
    print(f"Model: {model_name.upper()}")
    print(f"{'='*60}")
    
    if study is None:
        print("Study not found or failed to load.")
        return None
    
    # Study statistics
    trials = study.trials
    completed = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    failed = [t for t in trials if t.state == optuna.trial.TrialState.FAIL]
    pruned = [t for t in trials if t.state == optuna.trial.TrialState.PRUNED]
    
    print(f"\nTrials: {len(trials)} total")
    print(f"  - Completed: {len(completed)}")
    print(f"  - Failed: {len(failed)}")
    print(f"  - Pruned: {len(pruned)}")
    
    if len(completed) == 0:
        print("No completed trials yet.")
        return None
    
    # Best trial
    best_trial = study.best_trial
    print(f"\nBest Trial: #{best_trial.number}")
    print(f"Best Value (total_rmse/val): {best_trial.value:.6f}")
    
    print(f"\nBest Hyperparameters:")
    for key, value in best_trial.params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6g}")
        else:
            print(f"  {key}: {value}")
    
    # Top 5 trials
    print(f"\nTop 5 Trials:")
    sorted_trials = sorted(completed, key=lambda t: t.value)[:5]
    for i, trial in enumerate(sorted_trials, 1):
        print(f"  {i}. Trial #{trial.number}: {trial.value:.6f}")
    
    return best_trial.params


def get_all_studies():
    """Get all available HPS study paths."""
    base_path = Path(__file__).parent.parent / "bnn_aenet" / "results"
    
    studies = {
        "nn": (base_path / "nn" / "nn_forces.db", "nn_forces"),
        "lrt": (base_path / "bayesian" / "bnn_lrt_forces.db", "bnn_lrt_forces"),
        "fo": (base_path / "bayesian" / "bnn_fo_forces.db", "bnn_fo_forces"),
        "rad": (base_path / "bayesian" / "bnn_rad_forces.db", "bnn_rad_forces"),
    }
    
    return studies


def export_best_params(all_params: dict, output_path: str = None):
    """Export best parameters to a YAML file for easy reuse."""
    if output_path is None:
        output_path = Path(__file__).parent.parent / "bnn_aenet" / "configs" / "best_hps.yaml"
    
    import yaml
    
    with open(output_path, 'w') as f:
        yaml.dump(all_params, f, default_flow_style=False)
    
    print(f"\nBest parameters exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze HPS results")
    parser.add_argument("--model", type=str, choices=["nn", "lrt", "fo", "rad"],
                        help="Specific model to analyze")
    parser.add_argument("--all", action="store_true", 
                        help="Analyze all models")
    parser.add_argument("--export", action="store_true",
                        help="Export best params to YAML")
    args = parser.parse_args()
    
    studies = get_all_studies()
    all_best_params = {}
    
    if args.model:
        models_to_analyze = [args.model]
    elif args.all:
        models_to_analyze = list(studies.keys())
    else:
        # Default: analyze all available
        models_to_analyze = list(studies.keys())
    
    for model_name in models_to_analyze:
        db_path, study_name = studies[model_name]
        
        if not db_path.exists():
            print(f"\nDatabase not found for {model_name}: {db_path}")
            continue
        
        study = load_study(str(db_path), study_name)
        best_params = analyze_study(study, model_name)
        
        if best_params:
            all_best_params[model_name] = best_params
    
    if args.export and all_best_params:
        export_best_params(all_best_params)
    
    # Summary comparison
    if len(all_best_params) > 1:
        print(f"\n{'='*60}")
        print("SUMMARY COMPARISON")
        print(f"{'='*60}")
        
        for model, params in all_best_params.items():
            study = load_study(str(studies[model][0]), studies[model][1])
            if study and study.best_trial:
                print(f"{model.upper():6s}: {study.best_trial.value:.6f}")


if __name__ == "__main__":
    main()
