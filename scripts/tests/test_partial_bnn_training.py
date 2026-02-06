#!/usr/bin/env python3
"""
Test PartialBNN with actual training on TiO2 dataset.

Compares:
1. Full BNN (all layers Bayesian)
2. Last-layer BNN (only output layers Bayesian)
3. First-last BNN (input and output layers Bayesian)

Tests both energy-only and force training variants.
"""

import sys
import os
import time
import torch

sys.path.insert(0, '/home/g15farris/bin/bayesaenet')

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping

from bnn_aenet.datamodule.aenet_datamodule import AenetDataModule
from bnn_aenet.models.bnn import PartialBNN, PartialBNN_Forces_Aux, BNN, BNN_Forces_Aux


def test_partial_bnn_training():
    """Test PartialBNN training on TiO2 dataset."""
    print("=" * 70)
    print("Testing Partial BNN Training on TiO2")
    print("=" * 70)
    
    # Seed for reproducibility
    L.seed_everything(42)
    
    # Load TiO2 data - need to pass path to train.in file
    train_in_path = "/home/g15farris/bin/bayesaenet/data/TiO/train.in"
    dm = AenetDataModule(
        data_dir=train_in_path,
        batch_size=64,
        split_config="Data20",  # Use 20% split for faster testing
    )
    dm.setup(stage='fit')
    
    # Get network config from datamodule
    dataset_size = len(dm.train_dataloader()) * 64
    
    print(f"\nDataset: TiO2 (20% split)")
    print(f"Train batches: {len(dm.train_dataloader())}")
    print(f"Val batches: {len(dm.val_dataloader())}")
    print(f"Network architecture: {dm.hidden_size}")
    
    # Test configurations
    configs = [
        ("Full BNN", "all", BNN),
        ("Last-layer BNN", "last", PartialBNN),
        ("First-last BNN", "first_last", PartialBNN),
    ]
    
    results = []
    
    for name, bayesian_layers, model_class in configs:
        print(f"\n{'='*70}")
        print(f"Testing: {name} (bayesian_layers='{bayesian_layers}')")
        print("="*70)
        
        # Create fresh network copy
        from bnn_aenet.models.nets.network import NetAtom
        net = NetAtom(
            input_size=dm.input_size,
            hidden_size=dm.hidden_size,
            species=dm.species,
            active_names=dm.active_names,
            alpha=dm.alpha,
            device='cpu',
            e_scaling=dm.e_scaling,
            e_shift=dm.e_shift,
        )
        
        # Create model
        if model_class == BNN:
            model = BNN(
                net=net,
                lr=0.001,
                pretrain_epochs=0,
                mc_samples_train=2,
                mc_samples_eval=5,
                dataset_size=dataset_size,
                fit_context="lrt",
                prior_loc=0.0,
                prior_scale=0.1,
                guide="normal",
                q_scale=0.001,
                obs_scale=0.5,
                name=name,
            )
        else:
            model = model_class(
                net=net,
                lr=0.001,
                pretrain_epochs=0,
                mc_samples_train=2,
                mc_samples_eval=5,
                dataset_size=dataset_size,
                fit_context="lrt",
                prior_loc=0.0,
                prior_scale=0.1,
                guide="normal",
                q_scale=0.001,
                obs_scale=0.5,
                bayesian_layers=bayesian_layers,
                name=name,
            )
        
        # Get param counts for PartialBNN
        if hasattr(model, 'get_bayesian_param_count'):
            counts = model.get_bayesian_param_count()
            print(f"Bayesian params: {counts['bayesian_params']} / {counts['total_params']} ({counts['bayesian_fraction']:.1%})")
        
        # Train
        trainer = L.Trainer(
            max_epochs=5,
            accelerator='cpu',
            enable_progress_bar=True,
            enable_model_summary=False,
            logger=False,
            callbacks=[
                EarlyStopping(monitor='rmse/val', patience=10, mode='min'),
            ],
        )
        
        start_time = time.time()
        trainer.fit(model, dm)
        elapsed = time.time() - start_time
        
        # Get final metrics
        val_rmse = trainer.callback_metrics.get('rmse/val', float('nan'))
        if hasattr(val_rmse, 'item'):
            val_rmse = val_rmse.item()
        
        results.append({
            'name': name,
            'config': bayesian_layers,
            'val_rmse': val_rmse,
            'time': elapsed,
            'bayesian_frac': counts['bayesian_fraction'] if hasattr(model, 'get_bayesian_param_count') else 1.0,
        })
        
        print(f"Val RMSE: {val_rmse:.4f} meV/atom")
        print(f"Training time: {elapsed:.1f}s")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Config':<20} {'Val RMSE':>12} {'Time':>10} {'Bayesian %':>12}")
    print("-"*70)
    for r in results:
        print(f"{r['name']:<20} {r['val_rmse']:>12.4f} {r['time']:>10.1f}s {r['bayesian_frac']:>12.1%}")
    
    print("\nAll tests completed successfully!")
    return results


def test_partial_bnn_forces_training():
    """Test PartialBNN_Forces_Aux training on TiO2 dataset with forces."""
    print("\n" + "=" * 70)
    print("Testing Partial BNN with Forces Training on TiO2")
    print("=" * 70)
    
    L.seed_everything(42)
    
    # Load TiO2 data with forces - need to pass path to train.in file
    train_in_path = "/home/g15farris/bin/bayesaenet/data/TiO/train.in"
    dm = AenetDataModule(
        data_dir=train_in_path,
        batch_size=32,  # Smaller batch for force training
        split_config="Data20",
    )
    dm.setup(stage='fit')
    
    dataset_size = len(dm.train_dataloader()) * 32
    
    # Check if force data is available
    sample_batch = next(iter(dm.train_dataloader()))
    has_forces = len(sample_batch) > 15 and sample_batch[15] is not None
    
    if not has_forces:
        print("Force data not available in TiO2 dataset, skipping force test")
        return None
    
    print(f"Force data available: {has_forces}")
    
    configs = [
        ("Full BNN + Forces", "all", BNN_Forces_Aux),
        ("Last-layer BNN + Forces", "last", PartialBNN_Forces_Aux),
    ]
    
    results = []
    
    for name, bayesian_layers, model_class in configs:
        print(f"\n{'='*70}")
        print(f"Testing: {name}")
        print("="*70)
        
        from bnn_aenet.models.nets.network import NetAtom
        net = NetAtom(
            input_size=dm.input_size,
            hidden_size=dm.hidden_size,
            species=dm.species,
            active_names=dm.active_names,
            alpha=dm.alpha,
            device='cpu',
            e_scaling=dm.e_scaling,
            e_shift=dm.e_shift,
        )
        
        if model_class == BNN_Forces_Aux:
            model = BNN_Forces_Aux(
                net=net,
                lr=0.001,
                pretrain_epochs=0,
                mc_samples_train=2,
                mc_samples_eval=5,
                dataset_size=dataset_size,
                fit_context="lrt",
                prior_loc=0.0,
                prior_scale=0.1,
                guide="normal",
                q_scale=0.001,
                obs_scale=0.5,
                force_weight=1.0,
                name=name,
            )
        else:
            model = PartialBNN_Forces_Aux(
                net=net,
                lr=0.001,
                pretrain_epochs=0,
                mc_samples_train=2,
                mc_samples_eval=5,
                dataset_size=dataset_size,
                fit_context="lrt",
                prior_loc=0.0,
                prior_scale=0.1,
                guide="normal",
                q_scale=0.001,
                obs_scale=0.5,
                bayesian_layers=bayesian_layers,
                force_weight=1.0,
                name=name,
            )
        
        if hasattr(model, 'get_bayesian_param_count'):
            counts = model.get_bayesian_param_count()
            print(f"Bayesian params: {counts['bayesian_params']} / {counts['total_params']} ({counts['bayesian_fraction']:.1%})")
        
        trainer = L.Trainer(
            max_epochs=3,  # Fewer epochs for force training (slower)
            accelerator='cpu',
            enable_progress_bar=True,
            enable_model_summary=False,
            logger=False,
        )
        
        start_time = time.time()
        trainer.fit(model, dm)
        elapsed = time.time() - start_time
        
        val_rmse = trainer.callback_metrics.get('rmse/val', float('nan'))
        force_rmse = trainer.callback_metrics.get('force_rmse/val', float('nan'))
        if hasattr(val_rmse, 'item'):
            val_rmse = val_rmse.item()
        if hasattr(force_rmse, 'item'):
            force_rmse = force_rmse.item()
        
        results.append({
            'name': name,
            'val_rmse': val_rmse,
            'force_rmse': force_rmse,
            'time': elapsed,
        })
        
        print(f"Energy RMSE: {val_rmse:.4f} meV/atom")
        print(f"Force RMSE: {force_rmse:.4f} mHa/Bohr")
        print(f"Training time: {elapsed:.1f}s")
    
    print("\nForce training tests completed!")
    return results


if __name__ == "__main__":
    # Test energy-only training
    test_partial_bnn_training()
    
    # Test force training (if data available)
    print("\n" + "="*70)
    test_partial_bnn_forces_training()
