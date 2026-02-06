#!/usr/bin/env python
"""
Train a model using the best hyperparameters from HPS.

Usage:
    python scripts/train_best_model.py --model lrt --dataset TiO_Data100
    python scripts/train_best_model.py --model nn --dataset QM7_Data10 --epochs 10000
"""

import argparse
import subprocess
import sys
from pathlib import Path

import optuna


def get_best_params(model: str) -> dict:
    """Load best parameters from Optuna study."""
    base_path = Path(__file__).parent.parent / "bnn_aenet" / "results"
    
    studies = {
        "nn": (base_path / "nn" / "nn_forces.db", "nn_forces"),
        "lrt": (base_path / "bayesian" / "bnn_lrt_forces.db", "bnn_lrt_forces"),
        "fo": (base_path / "bayesian" / "bnn_fo_forces.db", "bnn_fo_forces"),
        "rad": (base_path / "bayesian" / "bnn_rad_forces.db", "bnn_rad_forces"),
    }
    
    if model not in studies:
        raise ValueError(f"Unknown model: {model}. Choose from: {list(studies.keys())}")
    
    db_path, study_name = studies[model]
    
    if not db_path.exists():
        raise FileNotFoundError(f"HPS database not found: {db_path}")
    
    storage = f"sqlite:///{db_path}"
    study = optuna.load_study(study_name=study_name, storage=storage)
    
    return study.best_trial.params


def build_hydra_overrides(model: str, params: dict, dataset: str, epochs: int) -> list:
    """Build Hydra override arguments from parameters."""
    overrides = []
    
    # Model-specific experiment config
    experiment_map = {
        "nn": "nn_forces",
        "lrt": "bnn_lrt_forces_aux",
        "fo": "bnn_fo_forces_aux",
        "rad": "bnn_rad_forces_aux",
    }
    
    overrides.append(f"experiment={experiment_map[model]}")
    overrides.append(f"datamodule={dataset}")
    overrides.append(f"trainer.max_epochs={epochs}")
    
    # Add best hyperparameters
    for key, value in params.items():
        if key == "batch_size":
            overrides.append(f"datamodule.batch_size={value}")
        elif key == "lr":
            if model == "nn":
                overrides.append(f"model.optimizer.lr={value}")
            else:
                overrides.append(f"model.lr={value}")
        elif key == "weight_decay":
            overrides.append(f"model.optimizer.weight_decay={value}")
        elif key in ["prior_scale", "q_scale", "obs_scale", "force_weight", "mc_samples_train", "pretrain_epochs"]:
            overrides.append(f"model.{key}={value}")
    
    # GPU settings
    overrides.append("trainer.accelerator=gpu")
    overrides.append("trainer.devices=1")
    
    # Disable mixed precision for LRT
    if model != "lrt":
        overrides.append("+trainer.precision=16-mixed")
    
    return overrides


def main():
    parser = argparse.ArgumentParser(description="Train model with best HPS parameters")
    parser.add_argument("--model", type=str, required=True,
                        choices=["nn", "lrt", "fo", "rad"],
                        help="Model to train")
    parser.add_argument("--dataset", type=str, default="TiO_Data100",
                        help="Dataset config (default: TiO_Data100)")
    parser.add_argument("--epochs", type=int, default=10000,
                        help="Training epochs (default: 10000)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print command without executing")
    args = parser.parse_args()
    
    print(f"Loading best parameters for {args.model}...")
    params = get_best_params(args.model)
    
    print(f"\nBest parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    
    overrides = build_hydra_overrides(args.model, params, args.dataset, args.epochs)
    
    cmd = ["python", "-m", "bnn_aenet.tasks.train"] + overrides
    
    print(f"\nCommand:")
    print(" ".join(cmd))
    
    if not args.dry_run:
        print(f"\nStarting training...")
        subprocess.run(cmd, cwd=Path(__file__).parent.parent)


if __name__ == "__main__":
    main()
