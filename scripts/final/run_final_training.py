#!/usr/bin/env python3
"""Master script for running final training of all methods on all datasets.

This script orchestrates the final training runs for the publication:
- Methods: LRT, Flipout, Radial (BNN), Deep Ensemble (NN)
- Datasets: QM7, TiO2

Usage:
    # Run all experiments
    python scripts/final/run_final_training.py --all

    # Run specific method
    python scripts/final/run_final_training.py --method lrt --dataset qm7

    # Run deep ensemble with 5 members
    python scripts/final/run_final_training.py --method de --dataset qm7 --n_ensemble 5

    # Dry run (print commands without executing)
    python scripts/final/run_final_training.py --all --dry-run
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Configuration
METHODS = ["lrt", "fo", "rad", "de"]
DATASETS = ["qm7", "tio2"]
N_ENSEMBLE_DEFAULT = 5
BASE_SEEDS = [42, 123, 456, 789, 1024]  # Seeds for ensemble members

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
TRAIN_SCRIPT = PROJECT_ROOT / "bnn_aenet" / "tasks" / "train.py"
LOGS_DIR = PROJECT_ROOT / "bnn_aenet" / "logs" / "final"


def get_experiment_name(method: str, dataset: str) -> str:
    """Get the experiment config name."""
    return f"final/{method}_{dataset}"


def get_run_name(method: str, dataset: str, seed: int, member_id: int = None) -> str:
    """Generate a unique run name."""
    timestamp = datetime.now().strftime("%Y%m%d")
    if member_id is not None:
        return f"{method}_{dataset}_member{member_id}_seed{seed}_{timestamp}"
    return f"{method}_{dataset}_seed{seed}_{timestamp}"


def build_train_command(
    method: str,
    dataset: str,
    seed: int,
    member_id: int = None,
    gpu_id: int = 0
) -> list:
    """Build the training command."""
    experiment = get_experiment_name(method, dataset)
    run_name = get_run_name(method, dataset, seed, member_id)
    
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        f"experiment={experiment}",
        f"seed={seed}",
        f"run_name={run_name}",
        f"trainer.devices=[{gpu_id}]",
        f"paths.log_dir={LOGS_DIR}",
        f"hydra.run.dir={LOGS_DIR}/runs/{run_name}",
    ]
    
    return cmd


def run_training(
    method: str,
    dataset: str,
    seed: int,
    member_id: int = None,
    gpu_id: int = 0,
    dry_run: bool = False
) -> int:
    """Run a single training job."""
    cmd = build_train_command(method, dataset, seed, member_id, gpu_id)
    
    print("\n" + "=" * 60)
    print(f"Training: {method.upper()} on {dataset.upper()}")
    if member_id is not None:
        print(f"Ensemble member: {member_id}")
    print(f"Seed: {seed}")
    print(f"GPU: {gpu_id}")
    print("Command:", " ".join(cmd))
    print("=" * 60 + "\n")
    
    if dry_run:
        print("[DRY RUN] Would execute the above command")
        return 0
    
    # Set environment
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROJECT_ROOT}/bnn_aenet:{env.get('PYTHONPATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["HYDRA_FULL_ERROR"] = "1"
    
    # Run training
    result = subprocess.run(cmd, env=env, cwd=PROJECT_ROOT)
    return result.returncode


def run_deep_ensemble(
    dataset: str,
    n_ensemble: int,
    seeds: list,
    gpu_id: int = 0,
    dry_run: bool = False
) -> int:
    """Run deep ensemble training with multiple members."""
    print(f"\n{'#' * 60}")
    print(f"# Deep Ensemble Training: {n_ensemble} members on {dataset.upper()}")
    print(f"{'#' * 60}")
    
    for i, seed in enumerate(seeds[:n_ensemble]):
        result = run_training(
            method="de",
            dataset=dataset,
            seed=seed,
            member_id=i,
            gpu_id=gpu_id,
            dry_run=dry_run
        )
        if result != 0 and not dry_run:
            print(f"Error training ensemble member {i}")
            return result
    
    return 0


def run_bnn_methods(
    dataset: str,
    seed: int = 42,
    gpu_id: int = 0,
    dry_run: bool = False
) -> int:
    """Run all BNN methods on a dataset."""
    for method in ["lrt", "fo", "rad"]:
        result = run_training(
            method=method,
            dataset=dataset,
            seed=seed,
            gpu_id=gpu_id,
            dry_run=dry_run
        )
        if result != 0 and not dry_run:
            print(f"Error training {method} on {dataset}")
            return result
    return 0


def run_all(
    n_ensemble: int = N_ENSEMBLE_DEFAULT,
    seeds: list = None,
    gpu_id: int = 0,
    dry_run: bool = False
) -> int:
    """Run all experiments."""
    if seeds is None:
        seeds = BASE_SEEDS
    
    base_seed = seeds[0]
    
    print("\n" + "#" * 60)
    print("# FINAL TRAINING: ALL METHODS x ALL DATASETS")
    print("#" * 60)
    print(f"\nMethods: {METHODS}")
    print(f"Datasets: {DATASETS}")
    print(f"BNN seed: {base_seed}")
    print(f"Ensemble seeds: {seeds[:n_ensemble]}")
    print(f"GPU: {gpu_id}")
    
    for dataset in DATASETS:
        # Run BNN methods
        result = run_bnn_methods(
            dataset=dataset,
            seed=base_seed,
            gpu_id=gpu_id,
            dry_run=dry_run
        )
        if result != 0 and not dry_run:
            return result
        
        # Run Deep Ensemble
        result = run_deep_ensemble(
            dataset=dataset,
            n_ensemble=n_ensemble,
            seeds=seeds,
            gpu_id=gpu_id,
            dry_run=dry_run
        )
        if result != 0 and not dry_run:
            return result
    
    print("\n" + "#" * 60)
    print("# ALL TRAINING COMPLETE")
    print("#" * 60)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Run final training for publication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--all", action="store_true",
        help="Run all methods on all datasets"
    )
    parser.add_argument(
        "--method", type=str, choices=METHODS,
        help="Specific method to train"
    )
    parser.add_argument(
        "--dataset", type=str, choices=DATASETS,
        help="Specific dataset to use"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for BNN training (default: 42)"
    )
    parser.add_argument(
        "--n-ensemble", type=int, default=N_ENSEMBLE_DEFAULT,
        help=f"Number of ensemble members for DE (default: {N_ENSEMBLE_DEFAULT})"
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="GPU ID to use (default: 0)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.all and (args.method is None or args.dataset is None):
        parser.error("Either --all or both --method and --dataset are required")
    
    # Create logs directory
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.all:
        return run_all(
            n_ensemble=args.n_ensemble,
            seeds=BASE_SEEDS,
            gpu_id=args.gpu,
            dry_run=args.dry_run
        )
    
    if args.method == "de":
        return run_deep_ensemble(
            dataset=args.dataset,
            n_ensemble=args.n_ensemble,
            seeds=BASE_SEEDS,
            gpu_id=args.gpu,
            dry_run=args.dry_run
        )
    else:
        return run_training(
            method=args.method,
            dataset=args.dataset,
            seed=args.seed,
            gpu_id=args.gpu,
            dry_run=args.dry_run
        )


if __name__ == "__main__":
    sys.exit(main())
