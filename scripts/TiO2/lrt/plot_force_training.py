#!/usr/bin/env python
"""
Plot energy and force predictions with uncertainties during/after BNN_Forces_Aux training.
"""

import sys
import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, '/home/g15farris/bin/bayesaenet')

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_tensorboard_scalars(log_dir):
    """Load scalar data from TensorBoard event files."""
    event_files = glob.glob(os.path.join(log_dir, '**/events.out.*'), recursive=True)
    if not event_files:
        return None
    
    event_file = max(event_files, key=os.path.getmtime)
    ea = EventAccumulator(event_file)
    ea.Reload()
    
    scalars = {}
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        scalars[tag] = {
            'steps': [e.step for e in events],
            'values': [e.value for e in events]
        }
    return scalars


def plot_training_curves(scalars, save_path):
    """Plot training curves for energy and force RMSE."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Energy RMSE
    ax = axes[0, 0]
    if 'rmse/train' in scalars and 'rmse/val' in scalars:
        ax.plot(scalars['rmse/train']['values'], label='Train', color='steelblue', alpha=0.8)
        ax.plot(scalars['rmse/val']['values'], label='Validation', color='coral', alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Energy RMSE (meV/atom)')
        ax.set_title('Energy RMSE during Training')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    # Force RMSE
    ax = axes[0, 1]
    if 'force_rmse/train' in scalars and 'force_rmse/val' in scalars:
        ax.plot(scalars['force_rmse/train']['values'], label='Train', color='steelblue', alpha=0.8)
        ax.plot(scalars['force_rmse/val']['values'], label='Validation', color='coral', alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Force RMSE (mHa/Bohr)')
        ax.set_title('Force RMSE during Training')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # ELBO and KL
    ax = axes[1, 0]
    if 'elbo/train' in scalars:
        ax.plot(scalars['elbo/train']['values'], label='ELBO', color='forestgreen', alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('ELBO')
        ax.set_title('Evidence Lower Bound (ELBO)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Alpha and KL
    ax = axes[1, 1]
    if 'kl/train' in scalars:
        ax.plot(scalars['kl/train']['values'], label='KL Divergence', color='purple', alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('KL')
        ax.set_title('KL Divergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Add alpha info if available
    if 'alpha' in scalars and len(scalars['alpha']['values']) > 0:
        alpha_val = scalars['alpha']['values'][-1]
        fig.suptitle(f'BNN_Forces_Aux Training (α={alpha_val:.2f})', fontsize=14, fontweight='bold')
    else:
        fig.suptitle('BNN_Forces_Aux Training', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves to {save_path}")


def run_predictions(checkpoint_path, data_dir, device='cuda'):
    """Load model and run predictions to get energy and force with uncertainties."""
    import pyro
    from pyro.infer import SVI, TraceMeanField_ELBO
    
    # Clear Pyro param store
    pyro.clear_param_store()
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Get hyperparameters
    hparams = ckpt.get('hyper_parameters', {})
    
    # Initialize datamodule
    from bnn_aenet.datamodule.aenet_datamodule import AenetDataModule
    
    datamodule = AenetDataModule(
        data_dir=data_dir,
        batch_size=32,
        device=device
    )
    datamodule.setup(stage='predict')
    
    # Initialize model
    from bnn_aenet.models.nets.network import NetAtom
    from bnn_aenet.models.bnn import BNN_Forces_Aux
    
    net = NetAtom(
        input_size=datamodule.input_size,
        hidden_size=datamodule.hidden_size,
        species=datamodule.species,
        active_names=datamodule.active_names,
        alpha=datamodule.alpha,
        e_scaling=datamodule.e_scaling,
        e_shift=datamodule.e_shift,
        device=device
    )
    
    model = BNN_Forces_Aux(
        net=net,
        lr=hparams.get('lr', 0.0001),
        pretrain_epochs=hparams.get('pretrain_epochs', 0),
        mc_samples_train=hparams.get('mc_samples_train', 2),
        mc_samples_eval=hparams.get('mc_samples_eval', 20),
        dataset_size=hparams.get('dataset_size', 600),
        fit_context=hparams.get('fit_context', 'lrt'),
        prior_loc=hparams.get('prior_loc', 0),
        prior_scale=hparams.get('prior_scale', 0.1),
        guide=hparams.get('guide', 'normal'),
        q_scale=hparams.get('q_scale', 0.001),
        obs_scale=hparams.get('obs_scale', 0.5),
        force_weight=hparams.get('force_weight', 1.0),
        force_lr_scale=hparams.get('force_lr_scale', 0.1),
        scale_lr_factor=hparams.get('scale_lr_factor', 0.5)
    )
    
    model.to(device)
    
    # Initialize BNN (creates the TyXe BNN structure)
    model.define_bnn()
    
    # Initialize optimizer and SVI (needed for the param store structure)
    model.optimizer = pyro.optim.ClippedAdam({'lr': hparams.get('lr', 0.0001), 'betas': [0.95, 0.999], 'clip_norm': 15})
    model.loss = TraceMeanField_ELBO(hparams.get('mc_samples_train', 2))
    model.svi = SVI(
        pyro.poutine.scale(model.bnn.model, scale=1.0/hparams.get('dataset_size', 600)),
        pyro.poutine.scale(model.bnn.guide, scale=1.0/hparams.get('dataset_size', 600)),
        model.optimizer,
        model.loss
    )
    
    # Run a dummy forward pass to initialize the guide in param store
    # This creates the proper param store structure
    if isinstance(datamodule.input_size, dict):
        input_dim = list(datamodule.input_size.values())[0]
    else:
        input_dim = datamodule.input_size[0] if isinstance(datamodule.input_size, list) else datamodule.input_size
    
    dummy_x = torch.randn(2, input_dim).to(device)
    dummy_logic = torch.ones(2, 1).to(device)
    try:
        with torch.no_grad():
            _ = model.bnn.model(dummy_x, dummy_logic)
    except:
        pass  # Ignore errors, just need to initialize structure
    
    # Load state dict (strict=False to handle any minor mismatches)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.eval()
    
    # Initialize bnn_no_obs (needed for predict_step)
    model.bnn_no_obs = pyro.poutine.block(model.bnn, hide=["obs"])
    
    # Run predictions on validation and test sets
    results = {}
    
    for split_name, dataloader in [('val', datamodule.val_dataloader()), 
                                    ('test', datamodule.test_dataloader())]:
        all_preds = []
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            batch = [b.to(device) if isinstance(b, torch.Tensor) else 
                    [t.to(device) for t in b] if isinstance(b, list) and len(b) > 0 and isinstance(b[0], torch.Tensor) else b 
                    for b in batch]
            
            pred = model.predict_step(batch, batch_idx)
            all_preds.append(pred)
        
        # Aggregate predictions
        results[split_name] = {
            'true_energy': np.concatenate([p['true'] for p in all_preds]),
            'pred_energy': np.concatenate([p['preds'] for p in all_preds]),
            'std_energy': np.concatenate([p['stds'] for p in all_preds]),
            'n_atoms': np.concatenate([p['n_atoms'] for p in all_preds]),
        }
        
        # Check if force data is available
        if all_preds[0].get('true_forces') is not None:
            results[split_name]['true_forces'] = np.concatenate([p['true_forces'] for p in all_preds])
            results[split_name]['pred_forces'] = np.concatenate([p['pred_forces'] for p in all_preds])
            results[split_name]['std_forces'] = np.concatenate([p['std_forces'] for p in all_preds])
    
    return results, datamodule.alpha


def plot_energy_predictions(results, alpha, save_path):
    """Plot energy predictions vs true values with uncertainty."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, (split_name, data) in enumerate(results.items()):
        ax = axes[idx]
        
        true_e = data['true_energy'].flatten()
        pred_e = data['pred_energy'].flatten()
        std_e = data['std_energy'].flatten()
        n_atoms = data['n_atoms'].flatten()
        
        # Convert to per-atom (meV/atom)
        true_e_atom = true_e / n_atoms * 1000
        pred_e_atom = pred_e / n_atoms * 1000
        std_e_atom = std_e / n_atoms * 1000
        
        # Parity plot with uncertainty coloring
        scatter = ax.scatter(true_e_atom, pred_e_atom, c=std_e_atom, cmap='viridis', 
                            alpha=0.7, s=30, edgecolors='none')
        cbar = plt.colorbar(scatter, ax=ax, label='Uncertainty (meV/atom)')
        
        # Perfect prediction line
        min_val = min(true_e_atom.min(), pred_e_atom.min())
        max_val = max(true_e_atom.max(), pred_e_atom.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5, alpha=0.7, label='Perfect')
        
        # Metrics
        rmse = np.sqrt(np.mean((pred_e_atom - true_e_atom)**2))
        mae = np.mean(np.abs(pred_e_atom - true_e_atom))
        r2 = 1 - np.sum((pred_e_atom - true_e_atom)**2) / np.sum((true_e_atom - true_e_atom.mean())**2)
        
        ax.set_xlabel('True Energy (meV/atom)', fontsize=12)
        ax.set_ylabel('Predicted Energy (meV/atom)', fontsize=12)
        ax.set_title(f'{split_name.capitalize()} Set\nRMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.4f}')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    
    fig.suptitle(f'Energy Predictions with Uncertainty (α={alpha:.2f})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved energy predictions to {save_path}")


def plot_force_predictions(results, alpha, save_path):
    """Plot force predictions vs true values with uncertainty."""
    # Check if force data is available
    if 'true_forces' not in results.get('val', {}):
        print("No force data available for plotting")
        return
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    for split_idx, (split_name, data) in enumerate(results.items()):
        if 'true_forces' not in data:
            continue
            
        true_f = data['true_forces']
        pred_f = data['pred_forces']
        std_f = data['std_forces']
        
        # Skip if all NaN
        if np.all(np.isnan(true_f)):
            continue
        
        # Parity plot (all components)
        ax = fig.add_subplot(gs[split_idx, 0])
        
        # Subsample if too many points
        n_points = len(true_f)
        max_points = 5000
        if n_points > max_points:
            idx = np.random.choice(n_points, max_points, replace=False)
            true_f_plot = true_f[idx]
            pred_f_plot = pred_f[idx]
            std_f_plot = std_f[idx]
        else:
            true_f_plot = true_f
            pred_f_plot = pred_f
            std_f_plot = std_f
        
        scatter = ax.scatter(true_f_plot, pred_f_plot, c=std_f_plot, cmap='plasma',
                            alpha=0.5, s=10, edgecolors='none')
        plt.colorbar(scatter, ax=ax, label='Uncertainty (mHa/Bohr)')
        
        # Perfect line
        min_val = min(true_f.min(), pred_f.min())
        max_val = max(true_f.max(), pred_f.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1, alpha=0.7)
        
        # Metrics
        rmse = np.sqrt(np.mean((pred_f - true_f)**2)) * 1000  # mHa/Bohr
        mae = np.mean(np.abs(pred_f - true_f)) * 1000
        
        ax.set_xlabel('True Force (Ha/Bohr)', fontsize=10)
        ax.set_ylabel('Predicted Force (Ha/Bohr)', fontsize=10)
        ax.set_title(f'{split_name.capitalize()}: RMSE={rmse:.2f}, MAE={mae:.2f} mHa/Bohr')
        ax.grid(True, alpha=0.3)
        
        # Error histogram
        ax = fig.add_subplot(gs[split_idx, 1])
        errors = (pred_f - true_f) * 1000  # mHa/Bohr
        ax.hist(errors, bins=50, density=True, alpha=0.7, color='coral', edgecolor='white')
        ax.axvline(0, color='black', linestyle='--', lw=1)
        ax.axvline(errors.mean(), color='red', linestyle='-', lw=1.5, label=f'Mean: {errors.mean():.2f}')
        ax.set_xlabel('Force Error (mHa/Bohr)', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'{split_name.capitalize()}: Error Distribution')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Uncertainty vs Error
        ax = fig.add_subplot(gs[split_idx, 2])
        abs_errors = np.abs(errors)
        uncertainties = std_f * 1000  # mHa/Bohr
        
        # Bin by uncertainty
        n_bins = 10
        bins = np.percentile(uncertainties, np.linspace(0, 100, n_bins + 1))
        bin_centers = []
        bin_errors = []
        bin_error_stds = []
        
        for i in range(n_bins):
            mask = (uncertainties >= bins[i]) & (uncertainties < bins[i + 1])
            if mask.sum() > 10:
                bin_centers.append((bins[i] + bins[i + 1]) / 2)
                bin_errors.append(abs_errors[mask].mean())
                bin_error_stds.append(abs_errors[mask].std())
        
        if bin_centers:
            ax.errorbar(bin_centers, bin_errors, yerr=bin_error_stds, fmt='o-', 
                       color='steelblue', capsize=3, capthick=1, markersize=6)
            
            # Perfect calibration line
            max_unc = max(bin_centers)
            ax.plot([0, max_unc], [0, max_unc], 'k--', alpha=0.5, label='Perfect calibration')
        
        ax.set_xlabel('Predicted Uncertainty (mHa/Bohr)', fontsize=10)
        ax.set_ylabel('Actual |Error| (mHa/Bohr)', fontsize=10)
        ax.set_title(f'{split_name.capitalize()}: Uncertainty Calibration')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'Force Predictions with Uncertainty (α={alpha:.2f})', fontsize=14, fontweight='bold')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved force predictions to {save_path}")


def main():
    # Configuration
    run_dir = '/home/g15farris/bin/bayesaenet/bnn_aenet/logs/train/runs/tio_forces_long'
    data_dir = '/home/g15farris/bin/bayesaenet/data/TiO/train_forces.in'
    output_dir = os.path.join(run_dir, 'plots')
    os.makedirs(output_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 1. Plot training curves from TensorBoard
    print("\n=== Plotting Training Curves ===")
    tb_dir = os.path.join(run_dir, 'tensorboard')
    scalars = load_tensorboard_scalars(tb_dir)
    if scalars:
        plot_training_curves(scalars, os.path.join(output_dir, 'training_curves.png'))
        
        # Print summary
        print("\nTraining Summary:")
        for tag in ['rmse/train', 'rmse/val', 'force_rmse/train', 'force_rmse/val', 'alpha']:
            if tag in scalars and len(scalars[tag]['values']) > 0:
                first = scalars[tag]['values'][0]
                last = scalars[tag]['values'][-1]
                n_epochs = len(scalars[tag]['values'])
                if tag == 'alpha':
                    print(f"  {tag}: {last:.3f}")
                else:
                    pct = ((last - first) / abs(first) * 100) if first != 0 else 0
                    print(f"  {tag}: {first:.2f} -> {last:.2f} ({pct:+.1f}%) [{n_epochs} epochs]")
    
    # 2. Load checkpoint and run predictions
    print("\n=== Running Predictions ===")
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    ckpts = glob.glob(os.path.join(ckpt_dir, '*.ckpt'))
    if not ckpts:
        print("No checkpoints found!")
        return
    
    # Use latest checkpoint
    latest_ckpt = max(ckpts, key=os.path.getmtime)
    print(f"Using checkpoint: {os.path.basename(latest_ckpt)}")
    
    try:
        results, alpha = run_predictions(latest_ckpt, data_dir, device)
        
        # 3. Plot energy predictions
        print("\n=== Plotting Energy Predictions ===")
        plot_energy_predictions(results, alpha, os.path.join(output_dir, 'energy_predictions.png'))
        
        # 4. Plot force predictions
        print("\n=== Plotting Force Predictions ===")
        plot_force_predictions(results, alpha, os.path.join(output_dir, 'force_predictions.png'))
        
        print(f"\n=== Done! Plots saved to {output_dir} ===")
        
    except Exception as e:
        print(f"Error running predictions: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
