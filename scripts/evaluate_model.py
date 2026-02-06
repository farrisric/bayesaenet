#!/usr/bin/env python
"""
Evaluate a trained model on test set and compute metrics.

Usage:
    python scripts/evaluate_model.py --checkpoint path/to/model.ckpt --model lrt --dataset TiO_Data100
"""

import argparse
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from bnn_aenet.datamodule.aenet_datamodule import AenetDataModule
from bnn_aenet.models.bnn import BNN_Forces_Aux, NN_Forces
from bnn_aenet.analysis.metrics import (
    compute_energy_metrics,
    compute_force_metrics,
    compute_uq_metrics,
)


def load_model(checkpoint_path: str, model_type: str):
    """Load model from checkpoint."""
    if model_type == "nn":
        model = NN_Forces.load_from_checkpoint(checkpoint_path)
    else:
        model = BNN_Forces_Aux.load_from_checkpoint(checkpoint_path)
    
    model.eval()
    return model


def get_data_path(dataset: str) -> str:
    """Get data directory path from dataset config name."""
    dataset_paths = {
        "TiO_Data100": "data/TiO",
        "TiO_Data20": "data/TiO",
        "QM7_Data10": "data/QM7",
        "QM7_Data20": "data/QM7",
    }
    
    base = Path(__file__).parent.parent
    return str(base / dataset_paths.get(dataset, f"data/{dataset.split('_')[0]}"))


def evaluate(model, datamodule, model_type: str, mc_samples: int = 50):
    """Run evaluation and compute metrics."""
    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
    )
    
    # Get predictions
    predictions = trainer.predict(model, datamodule.test_dataloader())
    
    # Aggregate predictions
    all_true_energy = []
    all_pred_energy = []
    all_std_energy = []
    all_true_forces = []
    all_pred_forces = []
    all_std_forces = []
    all_n_atoms = []
    
    for batch_pred in predictions:
        all_true_energy.extend(batch_pred["true"])
        all_pred_energy.extend(batch_pred["preds"])
        all_n_atoms.extend(batch_pred["n_atoms"])
        
        if "stds" in batch_pred:
            all_std_energy.extend(batch_pred["stds"])
        
        if "true_forces" in batch_pred:
            all_true_forces.extend(batch_pred["true_forces"])
            all_pred_forces.extend(batch_pred["pred_forces"])
            if "force_stds" in batch_pred:
                all_std_forces.extend(batch_pred["force_stds"])
    
    # Convert to numpy
    true_energy = np.array(all_true_energy)
    pred_energy = np.array(all_pred_energy)
    std_energy = np.array(all_std_energy) if all_std_energy else None
    n_atoms = np.array(all_n_atoms)
    
    # Compute energy metrics
    print("\n" + "="*60)
    print("ENERGY METRICS")
    print("="*60)
    
    energy_metrics = compute_energy_metrics(true_energy, pred_energy, n_atoms)
    for name, value in energy_metrics.items():
        print(f"  {name}: {value:.6f}")
    
    # Compute UQ metrics if available
    if std_energy is not None and len(std_energy) > 0:
        print("\n" + "="*60)
        print("UNCERTAINTY QUANTIFICATION METRICS (Energy)")
        print("="*60)
        
        uq_metrics = compute_uq_metrics(true_energy, pred_energy, std_energy)
        for name, value in uq_metrics.items():
            print(f"  {name}: {value:.6f}")
    
    # Compute force metrics
    if all_true_forces:
        print("\n" + "="*60)
        print("FORCE METRICS")
        print("="*60)
        
        true_forces = np.concatenate(all_true_forces)
        pred_forces = np.concatenate(all_pred_forces)
        
        force_metrics = compute_force_metrics(true_forces, pred_forces)
        for name, value in force_metrics.items():
            print(f"  {name}: {value:.6f}")
        
        if all_std_forces:
            print("\n" + "="*60)
            print("UNCERTAINTY QUANTIFICATION METRICS (Forces)")
            print("="*60)
            
            std_forces = np.concatenate(all_std_forces)
            force_uq_metrics = compute_uq_metrics(
                true_forces.flatten(), 
                pred_forces.flatten(), 
                std_forces.flatten()
            )
            for name, value in force_uq_metrics.items():
                print(f"  {name}: {value:.6f}")
    
    # Return all metrics as dict
    all_metrics = {
        "energy": energy_metrics,
        "energy_uq": uq_metrics if std_energy is not None else {},
    }
    
    if all_true_forces:
        all_metrics["force"] = force_metrics
        if all_std_forces:
            all_metrics["force_uq"] = force_uq_metrics
    
    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--model", type=str, required=True,
                        choices=["nn", "lrt", "fo", "rad"],
                        help="Model type")
    parser.add_argument("--dataset", type=str, default="TiO_Data100",
                        help="Dataset config name")
    parser.add_argument("--mc-samples", type=int, default=50,
                        help="MC samples for BNN prediction (default: 50)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV file for metrics")
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, args.model)
    
    # Setup datamodule
    data_path = get_data_path(args.dataset)
    split_config = args.dataset.split("_")[-1] if "_" in args.dataset else None
    
    print(f"Loading data from: {data_path}")
    print(f"Split config: {split_config}")
    
    datamodule = AenetDataModule(
        data_dir=data_path,
        batch_size=256,
        split_config=split_config,
    )
    datamodule.setup("test")
    
    # Evaluate
    metrics = evaluate(model, datamodule, args.model, args.mc_samples)
    
    # Save metrics
    if args.output:
        # Flatten metrics dict
        flat_metrics = {}
        for category, cat_metrics in metrics.items():
            for name, value in cat_metrics.items():
                flat_metrics[f"{category}_{name}"] = value
        
        df = pd.DataFrame([flat_metrics])
        df.to_csv(args.output, index=False)
        print(f"\nMetrics saved to: {args.output}")


if __name__ == "__main__":
    main()
