#!/usr/bin/env python3
"""
Test PartialBNN uncertainty quality.

Compares uncertainty metrics:
- NLL (Negative Log-Likelihood) - lower is better
- RMSCE (Root Mean Squared Calibration Error) - lower is better  
- Sharpness - lower means tighter predictions (but must be calibrated)
"""

import sys
import os
import numpy as np
import torch

sys.path.insert(0, '/home/g15farris/bin/bayesaenet')

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping

from bnn_aenet.datamodule.aenet_datamodule import AenetDataModule
from bnn_aenet.models.bnn import PartialBNN, BNN
from bnn_aenet.results.metrics import rms_calibration_error, sharpness


def evaluate_uncertainty(model, dm, mc_samples=50):
    """Evaluate uncertainty quality on validation set."""
    model.eval()
    
    all_preds = []
    all_stds = []
    all_targets = []
    all_n_atoms = []
    
    with torch.no_grad():
        for batch in dm.val_dataloader():
            x = batch[10], batch[12]  # E_DESCRP, E_LOGIC_REDUCE
            y = batch[11]  # E_ENERGY
            n_atoms = batch[14]  # E_N_ATOM
            
            # Get predictions with uncertainty
            loc, scale = model.bnn.predict(x[0], x[1], num_predictions=mc_samples)
            
            all_preds.append(loc.cpu().numpy())
            all_stds.append(scale.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_n_atoms.append(n_atoms.cpu().numpy())
    
    preds = np.concatenate(all_preds).flatten()
    stds = np.concatenate(all_stds).flatten()
    targets = np.concatenate(all_targets).flatten()
    n_atoms = np.concatenate(all_n_atoms).flatten()
    
    # Per-atom predictions and targets
    preds_per_atom = preds / n_atoms
    targets_per_atom = targets / n_atoms
    stds_per_atom = stds / n_atoms
    
    # Metrics
    rmse = np.sqrt(np.mean((preds_per_atom - targets_per_atom)**2)) * 1000  # meV/atom
    
    # NLL (Negative Log-Likelihood)
    nll = 0.5 * np.mean(np.log(2 * np.pi * stds_per_atom**2) + 
                        ((preds_per_atom - targets_per_atom)**2) / (stds_per_atom**2))
    
    # Calibration and sharpness
    rmsce = rms_calibration_error(
        torch.tensor(preds_per_atom), 
        torch.tensor(stds_per_atom), 
        torch.tensor(targets_per_atom)
    )
    sharp = sharpness(torch.tensor(stds_per_atom))
    
    # Mean uncertainty
    mean_std = np.mean(stds_per_atom) * 1000  # meV/atom
    
    return {
        'rmse': rmse,
        'nll': nll,
        'rmsce': rmsce.item() if hasattr(rmsce, 'item') else rmsce,
        'sharpness': sharp.item() if hasattr(sharp, 'item') else sharp,
        'mean_std': mean_std,
    }


def main():
    print("=" * 70)
    print("Testing Partial BNN Uncertainty Quality on TiO2")
    print("=" * 70)
    
    L.seed_everything(42)
    
    # Load TiO2 data
    train_in_path = "/home/g15farris/bin/bayesaenet/data/TiO/train.in"
    dm = AenetDataModule(
        data_dir=train_in_path,
        batch_size=64,
        split_config="Data20",
    )
    dm.setup(stage='fit')
    
    dataset_size = len(dm.train_dataloader()) * 64
    
    print(f"\nDataset: TiO2 (20% split)")
    print(f"Train batches: {len(dm.train_dataloader())}")
    print(f"Val batches: {len(dm.val_dataloader())}")
    
    # Test configurations - train for more epochs to get meaningful uncertainty
    configs = [
        ("Full BNN (all)", "all"),
        ("Last-layer BNN", "last"),
        ("First-last BNN", "first_last"),
        ("First layer BNN", "first"),
    ]
    
    results = []
    
    for name, bayesian_layers in configs:
        print(f"\n{'='*70}")
        print(f"Training: {name}")
        print("="*70)
        
        # Create fresh network
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
        if bayesian_layers == "all":
            model = BNN(
                net=net,
                lr=0.001,
                pretrain_epochs=0,
                mc_samples_train=2,
                mc_samples_eval=20,
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
            model = PartialBNN(
                net=net,
                lr=0.001,
                pretrain_epochs=0,
                mc_samples_train=2,
                mc_samples_eval=20,
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
        
        # Get param counts
        if hasattr(model, 'get_bayesian_param_count'):
            counts = model.get_bayesian_param_count()
            bayesian_frac = counts['bayesian_fraction']
            print(f"Bayesian params: {counts['bayesian_params']} / {counts['total_params']} ({bayesian_frac:.1%})")
        else:
            bayesian_frac = 1.0
        
        # Train for 15 epochs to get meaningful uncertainty estimates
        trainer = L.Trainer(
            max_epochs=15,
            accelerator='cpu',
            enable_progress_bar=True,
            enable_model_summary=False,
            logger=False,
            callbacks=[
                EarlyStopping(monitor='rmse/val', patience=20, mode='min'),
            ],
        )
        
        trainer.fit(model, dm)
        
        # Evaluate uncertainty
        print("\nEvaluating uncertainty quality...")
        metrics = evaluate_uncertainty(model, dm, mc_samples=50)
        metrics['bayesian_frac'] = bayesian_frac
        metrics['name'] = name
        results.append(metrics)
        
        print(f"  RMSE: {metrics['rmse']:.2f} meV/atom")
        print(f"  NLL: {metrics['nll']:.4f}")
        print(f"  RMSCE (calibration): {metrics['rmsce']:.4f}")
        print(f"  Sharpness: {metrics['sharpness']:.6f}")
        print(f"  Mean uncertainty: {metrics['mean_std']:.2f} meV/atom")
    
    # Summary
    print("\n" + "="*70)
    print("UNCERTAINTY QUALITY SUMMARY")
    print("="*70)
    print(f"{'Config':<20} {'RMSE':>8} {'NLL':>10} {'RMSCE':>10} {'Sharp':>10} {'Mean σ':>10} {'Bayes%':>8}")
    print("-"*70)
    for r in results:
        print(f"{r['name']:<20} {r['rmse']:>8.2f} {r['nll']:>10.4f} {r['rmsce']:>10.4f} {r['sharpness']:>10.6f} {r['mean_std']:>10.2f} {r['bayesian_frac']:>8.1%}")
    
    print("\n" + "="*70)
    print("INTERPRETATION:")
    print("="*70)
    print("- NLL (Negative Log-Likelihood): Lower is better - measures how well")
    print("  predicted distributions match actual outcomes")
    print("- RMSCE (Calibration Error): Lower is better - 0.0 means perfectly")
    print("  calibrated (predicted uncertainty matches actual errors)")
    print("- Sharpness: Lower is better (tighter predictions), but only if")
    print("  calibrated - overconfident models have low sharpness but high RMSCE")
    print("- Mean σ: Average predicted uncertainty in meV/atom")
    
    # Find best model for each metric
    print("\n" + "="*70)
    print("BEST MODEL BY METRIC:")
    print("="*70)
    
    best_rmse = min(results, key=lambda x: x['rmse'])
    best_nll = min(results, key=lambda x: x['nll'])
    best_rmsce = min(results, key=lambda x: x['rmsce'])
    
    print(f"  Best RMSE: {best_rmse['name']} ({best_rmse['rmse']:.2f} meV/atom)")
    print(f"  Best NLL: {best_nll['name']} ({best_nll['nll']:.4f})")
    print(f"  Best Calibration (RMSCE): {best_rmsce['name']} ({best_rmsce['rmsce']:.4f})")
    
    return results


if __name__ == "__main__":
    main()
