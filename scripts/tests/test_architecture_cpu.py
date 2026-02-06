#!/usr/bin/env python
"""
CPU-based architecture tests for BNN-AENET.
Tests model initialization, forward passes, and data loading on CPU.

Run on iqtc12 (no GPU required).
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np


def test_netatom_architectures():
    """Test different NetAtom architectures."""
    from bnn_aenet.models.nets.network import NetAtom
    
    print("\n" + "="*60)
    print("Testing NetAtom Architectures")
    print("="*60)
    
    # Test different network sizes
    architectures = [
        {"hidden_size": [[15, 15], [15, 15]], "name": "15:15"},
        {"hidden_size": [[25, 25], [25, 25]], "name": "25:25"},
        {"hidden_size": [[50, 25], [50, 25]], "name": "50:25"},
        {"hidden_size": [[64, 32, 16], [64, 32, 16]], "name": "64:32:16"},
    ]
    
    results = []
    
    for arch in architectures:
        print(f"\nTesting {arch['name']} architecture...")
        
        net = NetAtom(
            input_size=[70, 70],  # TiO2 descriptor size
            hidden_size=arch["hidden_size"],
            species=["Ti", "O"],
            active_names=[["tanh"] * len(arch["hidden_size"][0])] * 2,
            alpha=0.1,
            device="cpu",
            e_scaling=1.0,
            e_shift=0.0
        )
        
        # Count parameters
        n_params = sum(p.numel() for p in net.parameters())
        
        # Time forward pass
        batch_size = 64
        n_atoms = 96  # Typical TiO2 structure
        x = torch.randn(batch_size * n_atoms // 2, 70)  # Half Ti, half O
        
        # Create reduction logic tensor
        logic = torch.ones(batch_size, n_atoms // 2)
        
        start = time.time()
        n_forward = 100
        for _ in range(n_forward):
            with torch.no_grad():
                try:
                    _ = net(x, logic)
                except Exception as e:
                    print(f"  Forward failed: {e}")
                    break
        elapsed = time.time() - start
        
        results.append({
            "name": arch["name"],
            "params": n_params,
            "time_per_forward_ms": elapsed / n_forward * 1000
        })
        
        print(f"  Parameters: {n_params:,}")
        print(f"  Forward time: {elapsed/n_forward*1000:.2f} ms")
    
    return results


def test_data_loading():
    """Test data loading for TiO2."""
    print("\n" + "="*60)
    print("Testing Data Loading")
    print("="*60)
    
    from bnn_aenet.datamodule.aenet_datamodule import AenetDataModule
    
    data_dir = Path(__file__).parent.parent.parent / "data" / "TiO"
    
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return
    
    print(f"\nLoading from: {data_dir}")
    
    # Test different batch sizes
    batch_sizes = [32, 64, 128, 256, 512]
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        try:
            dm = AenetDataModule(
                data_dir=str(data_dir),
                batch_size=batch_size,
                split_config="Data100",
                device="cpu"
            )
            dm.setup("fit")
            
            # Time batch loading
            train_loader = dm.train_dataloader()
            
            start = time.time()
            n_batches = 0
            for batch in train_loader:
                n_batches += 1
                if n_batches >= 10:
                    break
            elapsed = time.time() - start
            
            print(f"  Train batches: {len(train_loader)}")
            print(f"  Time per batch: {elapsed/n_batches*1000:.2f} ms")
            
        except Exception as e:
            print(f"  Error: {e}")


def test_bnn_initialization():
    """Test BNN model initialization."""
    print("\n" + "="*60)
    print("Testing BNN Initialization")
    print("="*60)
    
    from bnn_aenet.models.bnn import BNN
    from bnn_aenet.models.nets.network import NetAtom
    
    # Create network
    net = NetAtom(
        input_size=[70, 70],
        hidden_size=[[15, 15], [15, 15]],
        species=["Ti", "O"],
        active_names=[["tanh", "tanh"], ["tanh", "tanh"]],
        alpha=0.1,
        device="cpu",
        e_scaling=1.0,
        e_shift=0.0
    )
    
    # Test different BNN configurations
    configs = [
        {"fit_context": "lrt", "guide": "normal", "name": "LRT"},
        {"fit_context": "flipout", "guide": "normal", "name": "Flipout"},
        # {"fit_context": "", "guide": "radial", "name": "Radial"},  # Skip - requires special handling
    ]
    
    for config in configs:
        print(f"\nTesting {config['name']}...")
        
        try:
            model = BNN(
                net=net,
                lr=1e-3,
                pretrain_epochs=0,
                mc_samples_train=1,
                mc_samples_eval=10,
                dataset_size=1000,
                fit_context=config["fit_context"],
                prior_loc=0.0,
                prior_scale=0.3,
                guide=config["guide"],
                q_scale=0.001,
                obs_scale=0.5,
            )
            
            print(f"  Initialized successfully")
            
            # Count BNN params (will be set after define_bnn)
            n_params = sum(p.numel() for p in net.parameters())
            print(f"  Network params: {n_params}")
            
        except Exception as e:
            print(f"  Error: {e}")


def test_force_computation():
    """Test force computation logic."""
    print("\n" + "="*60)
    print("Testing Force Computation")
    print("="*60)
    
    from bnn_aenet.models.nets.network import NetAtom
    
    net = NetAtom(
        input_size=[70, 70],
        hidden_size=[[15, 15], [15, 15]],
        species=["Ti", "O"],
        active_names=[["tanh", "tanh"], ["tanh", "tanh"]],
        alpha=0.1,
        device="cpu",
        e_scaling=1.0,
        e_shift=0.0
    )
    
    # Create mock force data
    batch_size = 4
    n_atoms = 12
    
    # Descriptors
    group_descrp = [
        torch.randn(batch_size * n_atoms // 2, 70, requires_grad=True),  # Ti
        torch.randn(batch_size * n_atoms // 2, 70, requires_grad=True),  # O
    ]
    
    # Logic reduction
    logic_reduce = [
        torch.ones(batch_size, n_atoms // 2),
        torch.ones(batch_size, n_atoms // 2),
    ]
    
    # Descriptor derivatives (mock)
    group_sfderiv_i = [
        torch.randn(batch_size * n_atoms // 2, 70, 3),
        torch.randn(batch_size * n_atoms // 2, 70, 3),
    ]
    
    group_sfderiv_j = [
        torch.randn(batch_size * n_atoms // 2, 10, 70, 3),  # 10 neighbors
        torch.randn(batch_size * n_atoms // 2, 10, 70, 3),
    ]
    
    # Neighbor indices
    group_indices_F = [
        torch.zeros(batch_size * n_atoms // 2, 10).long(),
        torch.zeros(batch_size * n_atoms // 2, 10).long(),
    ]
    
    group_indices_F_i = [
        torch.arange(batch_size * n_atoms // 2).long(),
        torch.arange(batch_size * n_atoms // 2).long(),
    ]
    
    try:
        # Forward with forces
        forces, energies = net.forward_F(
            group_descrp, 
            logic_reduce,
            group_sfderiv_i,
            group_sfderiv_j,
            group_indices_F,
            group_indices_F_i
        )
        
        print(f"  Energy shape: {energies.shape}")
        print(f"  Force shape: {forces.shape}")
        print(f"  Force computation: OK")
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()


def test_metrics():
    """Test metrics computation."""
    print("\n" + "="*60)
    print("Testing Metrics")
    print("="*60)
    
    from bnn_aenet.analysis.metrics import (
        compute_energy_metrics,
        compute_uncertainty_metrics,
    )
    
    # Create mock predictions
    n_samples = 100
    true_energy = np.random.randn(n_samples) * 0.5 + 1.0
    pred_energy = true_energy + np.random.randn(n_samples) * 0.1
    std_energy = np.abs(np.random.randn(n_samples) * 0.05 + 0.1)
    n_atoms = np.random.randint(10, 100, n_samples)
    
    print("\nEnergy metrics:")
    energy_metrics = compute_energy_metrics(true_energy, pred_energy, n_atoms)
    for name, value in energy_metrics.items():
        print(f"  {name}: {value:.6f}")
    
    print("\nUQ metrics:")
    uq_metrics = compute_uncertainty_metrics(true_energy, pred_energy, std_energy)
    for name, value in uq_metrics.items():
        print(f"  {name}: {value:.6f}")


def main():
    print("="*60)
    print("BNN-AENET Architecture Tests (CPU)")
    print("="*60)
    
    # Run tests
    test_netatom_architectures()
    test_bnn_initialization()
    test_force_computation()
    test_metrics()
    test_data_loading()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)


if __name__ == "__main__":
    main()
