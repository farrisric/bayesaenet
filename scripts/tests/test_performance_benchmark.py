#!/usr/bin/env python
"""
Performance benchmarks for BNN-AENET.
Tests training speed, data loading, and inference.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np


def benchmark_data_loading():
    """Benchmark data loading performance."""
    from bnn_aenet.datamodule.aenet_datamodule import AenetDataModule
    
    print("\n" + "="*60)
    print("DATA LOADING BENCHMARKS")
    print("="*60)
    
    data_path = "/home/g15farris/bin/bayesaenet/data/TiO/train.in"
    
    batch_sizes = [32, 64, 128, 256, 512]
    
    for batch_size in batch_sizes:
        dm = AenetDataModule(
            data_dir=data_path,
            batch_size=batch_size,
            split_config="Data100",
            device="cpu"
        )
        
        # Time setup
        start = time.time()
        dm.setup("fit")
        setup_time = time.time() - start
        
        # Time iteration
        train_loader = dm.train_dataloader()
        
        start = time.time()
        n_batches = 0
        for batch in train_loader:
            n_batches += 1
        iter_time = time.time() - start
        
        print(f"\nBatch size: {batch_size}")
        print(f"  Setup time: {setup_time:.3f}s")
        print(f"  Iteration time: {iter_time:.3f}s")
        print(f"  Batches: {n_batches}")
        print(f"  Time per batch: {iter_time/n_batches*1000:.2f}ms")


def benchmark_nn_training_step():
    """Benchmark NN training step."""
    from bnn_aenet.models.bnn import NN_Forces
    from bnn_aenet.datamodule.aenet_datamodule import AenetDataModule
    import lightning as L
    
    print("\n" + "="*60)
    print("NN TRAINING STEP BENCHMARKS")
    print("="*60)
    
    data_path = "/home/g15farris/bin/bayesaenet/data/TiO/train.in"
    
    dm = AenetDataModule(
        data_dir=data_path,
        batch_size=64,
        split_config="Data100",
        device="cpu"
    )
    dm.setup("fit")
    
    # Get network from datamodule
    net = dm.tin.net
    net.alpha = torch.tensor(0.1)
    
    model = NN_Forces(
        net=net,
        lr=1e-3,
        force_weight=1.0
    )
    
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    
    # Warm up
    for _ in range(5):
        model.training_step(batch, 0)
    
    # Benchmark
    n_steps = 50
    start = time.time()
    for _ in range(n_steps):
        model.training_step(batch, 0)
    elapsed = time.time() - start
    
    print(f"\nNN_Forces training step:")
    print(f"  Steps: {n_steps}")
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Time per step: {elapsed/n_steps*1000:.2f}ms")


def benchmark_bnn_prediction():
    """Benchmark BNN prediction with different MC samples."""
    from bnn_aenet.models.bnn import BNN_Forces_Aux
    from bnn_aenet.datamodule.aenet_datamodule import AenetDataModule
    
    print("\n" + "="*60)
    print("BNN PREDICTION BENCHMARKS")
    print("="*60)
    
    data_path = "/home/g15farris/bin/bayesaenet/data/TiO/train.in"
    
    dm = AenetDataModule(
        data_dir=data_path,
        batch_size=32,
        split_config="Data100",
        device="cpu"
    )
    dm.setup("fit")
    
    net = dm.tin.net
    net.alpha = torch.tensor(0.1)
    
    mc_samples_list = [1, 5, 10, 20, 50]
    
    for mc_samples in mc_samples_list:
        model = BNN_Forces_Aux(
            net=net,
            lr=1e-3,
            pretrain_epochs=0,
            mc_samples_train=1,
            mc_samples_eval=mc_samples,
            dataset_size=6000,
            fit_context="flipout",
            prior_loc=0.0,
            prior_scale=0.3,
            guide="normal",
            q_scale=0.001,
            obs_scale=0.5,
            force_weight=1.0,
        )
        
        # Initialize BNN
        model.define_bnn()
        
        val_loader = dm.val_dataloader()
        batch = next(iter(val_loader))
        
        # Warm up
        for _ in range(3):
            try:
                model.predict_step(batch, 0)
            except Exception:
                pass
        
        # Benchmark
        n_steps = 10
        start = time.time()
        for _ in range(n_steps):
            try:
                model.predict_step(batch, 0)
            except Exception:
                pass
        elapsed = time.time() - start
        
        print(f"\nMC samples: {mc_samples}")
        print(f"  Time per predict: {elapsed/n_steps*1000:.2f}ms")


def benchmark_metrics_computation():
    """Benchmark metrics computation."""
    from bnn_aenet.analysis.metrics import (
        compute_energy_metrics,
        compute_force_metrics,
        compute_uncertainty_metrics,
    )
    
    print("\n" + "="*60)
    print("METRICS COMPUTATION BENCHMARKS")
    print("="*60)
    
    sizes = [100, 500, 1000, 5000, 10000]
    
    for n in sizes:
        np.random.seed(42)
        
        true_e = np.random.randn(n)
        pred_e = true_e + np.random.randn(n) * 0.05
        std_e = np.abs(np.random.randn(n) * 0.03 + 0.05)
        n_atoms = np.random.randint(10, 100, n)
        
        true_f = np.random.randn(n, 3)
        pred_f = true_f + np.random.randn(n, 3) * 0.01
        
        # Energy metrics
        start = time.time()
        for _ in range(100):
            compute_energy_metrics(true_e, pred_e, n_atoms)
        e_time = (time.time() - start) / 100 * 1000
        
        # Force metrics
        start = time.time()
        for _ in range(100):
            compute_force_metrics(true_f, pred_f)
        f_time = (time.time() - start) / 100 * 1000
        
        # UQ metrics
        start = time.time()
        for _ in range(100):
            compute_uncertainty_metrics(true_e, pred_e, std_e)
        uq_time = (time.time() - start) / 100 * 1000
        
        print(f"\nSize: {n}")
        print(f"  Energy metrics: {e_time:.3f}ms")
        print(f"  Force metrics: {f_time:.3f}ms")
        print(f"  UQ metrics: {uq_time:.3f}ms")


def benchmark_architecture_forward():
    """Benchmark forward pass for different architectures using real data."""
    from bnn_aenet.models.nets.network import NetAtom
    from bnn_aenet.datamodule.aenet_datamodule import AenetDataModule
    
    print("\n" + "="*60)
    print("ARCHITECTURE FORWARD PASS BENCHMARKS")
    print("="*60)
    
    data_path = "/home/g15farris/bin/bayesaenet/data/TiO/train.in"
    
    dm = AenetDataModule(
        data_dir=data_path,
        batch_size=64,
        split_config="Data100",
        device="cpu"
    )
    dm.setup("fit")
    
    # Get a batch
    batch = next(iter(dm.train_dataloader()))
    
    # Get input size from original network
    input_size = dm.tin.net.input_size
    species = dm.tin.net.species
    
    architectures = [
        {"hidden": [[15, 15], [15, 15]], "name": "15:15"},
        {"hidden": [[25, 25], [25, 25]], "name": "25:25"},
        {"hidden": [[50, 25], [50, 25]], "name": "50:25"},
        {"hidden": [[64, 32, 16], [64, 32, 16]], "name": "64:32:16"},
    ]
    
    for arch in architectures:
        n_layers = len(arch["hidden"][0])
        
        net = NetAtom(
            input_size=input_size,
            hidden_size=arch["hidden"],
            species=species,
            active_names=[["tanh"] * n_layers] * len(species),
            alpha=0.1,
            device="cpu",
            e_scaling=dm.tin.net.e_scaling,
            e_shift=dm.tin.net.e_shift
        )
        
        n_params = sum(p.numel() for p in net.parameters())
        
        # Get energy data from batch
        x_descrp = batch[10]  # E_DESCRP
        x_logic = batch[12]   # E_LOGIC_REDUCE
        
        # Warm up
        with torch.no_grad():
            for _ in range(10):
                _ = net(x_descrp, x_logic)
        
        # Benchmark
        n_forward = 100
        start = time.time()
        with torch.no_grad():
            for _ in range(n_forward):
                _ = net(x_descrp, x_logic)
        elapsed = time.time() - start
        
        print(f"\n{arch['name']} (params: {n_params:,})")
        print(f"  Forward time: {elapsed/n_forward*1000:.3f}ms")
        print(f"  Throughput: {n_forward/elapsed:.1f} forward/s")


def main():
    print("="*60)
    print("BNN-AENET PERFORMANCE BENCHMARKS")
    print("="*60)
    
    benchmark_data_loading()
    benchmark_architecture_forward()
    benchmark_metrics_computation()
    
    # These require more setup - run if time permits
    try:
        benchmark_nn_training_step()
    except Exception as e:
        print(f"\nNN training benchmark failed: {e}")
    
    print("\n" + "="*60)
    print("BENCHMARKS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
