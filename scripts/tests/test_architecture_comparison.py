#!/usr/bin/env python3
"""
Architecture Comparison Test: 15:15 vs 25:25

Compares training performance, convergence, and final accuracy
of different network architectures on TiO2 and QM7 datasets.
"""

import os
import sys
import time
import json
from functools import partial
from datetime import datetime

# Add project root to path
sys.path.insert(0, '/home/g15farris/bin/bayesaenet')

import torch
import numpy as np
import lightning.pytorch as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

# Import library components
from bnn_aenet.datamodule.aenet_datamodule import AenetDataModule
from bnn_aenet.models.nets.network import NetAtom
from bnn_aenet.models.bnn import NN, NN_Forces, BNN, BNN_Forces_Aux

# Reproducibility
L.seed_everything(42, workers=True)

# Configuration
EPOCHS = 50  # Longer training for proper comparison
BATCH_SIZE = 64
PATIENCE = 15  # Early stopping patience

# Architectures to test
ARCHITECTURES = {
    '15:15': {'hidden': [15, 15], 'activation': ['tanh', 'tanh']},
    '25:25': {'hidden': [25, 25], 'activation': ['tanh', 'tanh']},
    '15:15:15': {'hidden': [15, 15, 15], 'activation': ['tanh', 'tanh', 'tanh']},
}

# Datasets to test
DATASETS = {
    'TiO2': {
        'train_in': '/home/g15farris/bin/bayesaenet/data/TiO/train.in',
        'split_config': 'Data100',
    },
    'QM7': {
        'train_in': '/home/g15farris/bin/bayesaenet/data/QM7/train.in', 
        'split_config': 'Data10',
    },
}


def create_net_with_architecture(dm, arch_config):
    """Create NetAtom with specified architecture."""
    n_species = len(dm.species)
    hidden_size = [arch_config['hidden'] for _ in range(n_species)]
    active_names = [arch_config['activation'] for _ in range(n_species)]
    
    net = NetAtom(
        input_size=dm.input_size,
        hidden_size=hidden_size,
        species=dm.species,
        active_names=active_names,
        alpha=dm.alpha,
        device='cpu',
        e_scaling=dm.e_scaling,
        e_shift=dm.e_shift,
    )
    return net


def count_parameters(net):
    """Count trainable parameters."""
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def train_model(dm, net, model_type='NN', max_epochs=EPOCHS):
    """Train a model and return metrics."""
    start_time = time.time()
    
    # Create model
    if model_type == 'NN':
        model = NN(
            net=net,
            optimizer=partial(torch.optim.Adam, lr=1e-3),
            name=f"NN_{net.hidden_size[0]}"
        )
    elif model_type == 'NN_Forces':
        model = NN_Forces(
            net=net,
            optimizer=partial(torch.optim.Adam, lr=1e-3),
            force_weight=1.0,
            alpha=0.1,
            name=f"NN_Forces_{net.hidden_size[0]}"
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='rmse/val', patience=PATIENCE, mode='min'),
    ]
    
    # Trainer
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator='cpu',
        devices=1,
        callbacks=callbacks,
        enable_progress_bar=True,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False,
    )
    
    # Train
    trainer.fit(model, dm)
    
    training_time = time.time() - start_time
    
    # Get final metrics
    final_train_rmse = trainer.callback_metrics.get('rmse/train', float('nan'))
    final_val_rmse = trainer.callback_metrics.get('rmse/val', float('nan'))
    epochs_trained = trainer.current_epoch + 1
    
    return {
        'train_rmse': float(final_train_rmse) if hasattr(final_train_rmse, 'item') else float(final_train_rmse),
        'val_rmse': float(final_val_rmse) if hasattr(final_val_rmse, 'item') else float(final_val_rmse),
        'epochs': epochs_trained,
        'training_time': training_time,
        'time_per_epoch': training_time / epochs_trained,
        'parameters': count_parameters(net),
    }


def run_comparison():
    """Run the full architecture comparison."""
    print("=" * 70)
    print("ARCHITECTURE COMPARISON TEST")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    results = {}
    
    for dataset_name, dataset_config in DATASETS.items():
        print(f"\n{'=' * 70}")
        print(f"DATASET: {dataset_name}")
        print(f"{'=' * 70}")
        
        results[dataset_name] = {}
        
        # Load datamodule
        print(f"\nLoading datamodule from: {dataset_config['train_in']}")
        dm = AenetDataModule(
            data_dir=dataset_config['train_in'],
            batch_size=BATCH_SIZE,
            split_config=dataset_config['split_config'],
        )
        dm.setup(stage='fit')
        
        print(f"  Species: {dm.species}")
        print(f"  Input size: {dm.input_size}")
        # Get sizes from dataloaders
        train_size = len(dm.train_dataloader()) * BATCH_SIZE
        val_size = len(dm.val_dataloader()) * BATCH_SIZE
        print(f"  Train batches: {len(dm.train_dataloader())}")
        print(f"  Val batches: {len(dm.val_dataloader())}")
        
        for arch_name, arch_config in ARCHITECTURES.items():
            print(f"\n{'-' * 50}")
            print(f"ARCHITECTURE: {arch_name}")
            print(f"{'-' * 50}")
            
            # Create network
            net = create_net_with_architecture(dm, arch_config)
            n_params = count_parameters(net)
            print(f"  Parameters: {n_params:,}")
            
            # Train NN
            print(f"\n  Training NN ({EPOCHS} max epochs, patience={PATIENCE})...")
            try:
                metrics = train_model(dm, net, model_type='NN', max_epochs=EPOCHS)
                
                print(f"    Epochs trained: {metrics['epochs']}")
                print(f"    Final train RMSE: {metrics['train_rmse']:.4f} meV/atom")
                print(f"    Final val RMSE: {metrics['val_rmse']:.4f} meV/atom")
                print(f"    Training time: {metrics['training_time']:.2f}s")
                print(f"    Time per epoch: {metrics['time_per_epoch']:.2f}s")
                
                results[dataset_name][arch_name] = {
                    'NN': metrics,
                    'parameters': n_params,
                }
            except Exception as e:
                print(f"    ERROR: {e}")
                results[dataset_name][arch_name] = {'error': str(e)}
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for dataset_name in results:
        print(f"\n{dataset_name}:")
        print(f"  {'Architecture':<12} {'Params':>10} {'Val RMSE':>12} {'Time':>10} {'Epochs':>8}")
        print(f"  {'-'*12} {'-'*10} {'-'*12} {'-'*10} {'-'*8}")
        
        for arch_name in results[dataset_name]:
            if 'error' in results[dataset_name][arch_name]:
                print(f"  {arch_name:<12} {'ERROR':>10}")
                continue
            
            metrics = results[dataset_name][arch_name].get('NN', {})
            params = results[dataset_name][arch_name].get('parameters', 0)
            val_rmse = metrics.get('val_rmse', float('nan'))
            train_time = metrics.get('training_time', 0)
            epochs = metrics.get('epochs', 0)
            
            print(f"  {arch_name:<12} {params:>10,} {val_rmse:>12.4f} {train_time:>10.1f}s {epochs:>8}")
    
    # Save results
    results_file = '/home/g15farris/bin/bayesaenet/scripts/tests/architecture_comparison_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    print("\n" + "=" * 70)
    print("ARCHITECTURE COMPARISON COMPLETE")
    print("=" * 70)
    
    return results


if __name__ == '__main__':
    run_comparison()
