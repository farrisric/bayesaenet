#!/usr/bin/env python
"""
Integration tests for BNN-AENET training on CPU.
Tests actual training loops with small datasets.
"""

import sys
import time
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint


def create_net_from_datamodule(dm):
    """Helper to create a NetAtom from datamodule parameters."""
    from bnn_aenet.models.nets.network import NetAtom
    
    return NetAtom(
        input_size=dm.input_size,
        hidden_size=dm.hidden_size,
        species=dm.species,
        active_names=dm.active_names,
        alpha=dm.alpha,
        device="cpu",
        e_scaling=dm.e_scaling,
        e_shift=dm.e_shift
    )


def test_nn_training_loop():
    """Test NN training loop on TiO2 for a few epochs."""
    print("\n" + "="*60)
    print("TEST: NN Training Loop (5 epochs)")
    print("="*60)
    
    from bnn_aenet.models.bnn import NN
    from bnn_aenet.datamodule.aenet_datamodule import AenetDataModule
    
    data_path = "/home/g15farris/bin/bayesaenet/data/TiO/train.in"
    
    dm = AenetDataModule(
        data_dir=data_path,
        batch_size=64,
        split_config="Data100",
        device="cpu"
    )
    dm.setup("fit")
    
    net = create_net_from_datamodule(dm)
    net.alpha = torch.tensor(0.1)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    model = NN(net=net, optimizer=optimizer)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = L.Trainer(
            max_epochs=5,
            accelerator="cpu",
            enable_progress_bar=True,
            default_root_dir=tmpdir,
            enable_checkpointing=False,
            logger=False,
        )
        
        start = time.time()
        trainer.fit(model, dm)
        elapsed = time.time() - start
    
    print(f"  Training time: {elapsed:.2f}s")
    print(f"  Epochs: 5")
    print(f"  Time per epoch: {elapsed/5:.2f}s")
    return {"status": "success", "time": elapsed}


def test_nn_forces_training_loop():
    """Test NN_Forces training loop on TiO2."""
    print("\n" + "="*60)
    print("TEST: NN_Forces Training Loop (5 epochs)")
    print("="*60)
    
    from bnn_aenet.models.bnn import NN_Forces
    from bnn_aenet.datamodule.aenet_datamodule import AenetDataModule
    
    data_path = "/home/g15farris/bin/bayesaenet/data/TiO/train.in"
    
    dm = AenetDataModule(
        data_dir=data_path,
        batch_size=64,
        split_config="Data100",
        device="cpu"
    )
    dm.setup("fit")
    
    net = create_net_from_datamodule(dm)
    net.alpha = torch.tensor(0.1)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    model = NN_Forces(net=net, optimizer=optimizer, force_weight=1.0)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = L.Trainer(
            max_epochs=5,
            accelerator="cpu",
            enable_progress_bar=True,
            default_root_dir=tmpdir,
            enable_checkpointing=False,
            logger=False,
        )
        
        start = time.time()
        trainer.fit(model, dm)
        elapsed = time.time() - start
    
    print(f"  Training time: {elapsed:.2f}s")
    print(f"  Time per epoch: {elapsed/5:.2f}s")
    return {"status": "success", "time": elapsed}


def test_bnn_flipout_training_loop():
    """Test BNN Flipout training loop."""
    print("\n" + "="*60)
    print("TEST: BNN Flipout Training Loop (3 epochs)")
    print("="*60)
    
    from bnn_aenet.models.bnn import BNN
    from bnn_aenet.datamodule.aenet_datamodule import AenetDataModule
    
    data_path = "/home/g15farris/bin/bayesaenet/data/TiO/train.in"
    
    dm = AenetDataModule(
        data_dir=data_path,
        batch_size=64,
        split_config="Data100",
        device="cpu"
    )
    dm.setup("fit")
    
    net = create_net_from_datamodule(dm)
    net.alpha = torch.tensor(0.1)
    
    model = BNN(
        net=net,
        lr=1e-3,
        pretrain_epochs=0,
        mc_samples_train=1,
        mc_samples_eval=5,
        dataset_size=len(dm.train_dataloader().dataset),
        fit_context="flipout",
        prior_loc=0.0,
        prior_scale=0.3,
        guide="normal",
        q_scale=0.001,
        obs_scale=0.5,
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = L.Trainer(
            max_epochs=3,
            accelerator="cpu",
            enable_progress_bar=True,
            default_root_dir=tmpdir,
            enable_checkpointing=False,
            logger=False,
        )
        
        start = time.time()
        trainer.fit(model, dm)
        elapsed = time.time() - start
    
    print(f"  Training time: {elapsed:.2f}s")
    print(f"  Time per epoch: {elapsed/3:.2f}s")
    return {"status": "success", "time": elapsed}


def test_bnn_lrt_training_loop():
    """Test BNN LRT training loop."""
    print("\n" + "="*60)
    print("TEST: BNN LRT Training Loop (3 epochs)")
    print("="*60)
    
    from bnn_aenet.models.bnn import BNN
    from bnn_aenet.datamodule.aenet_datamodule import AenetDataModule
    
    data_path = "/home/g15farris/bin/bayesaenet/data/TiO/train.in"
    
    dm = AenetDataModule(
        data_dir=data_path,
        batch_size=64,
        split_config="Data100",
        device="cpu"
    )
    dm.setup("fit")
    
    net = create_net_from_datamodule(dm)
    net.alpha = torch.tensor(0.1)
    
    model = BNN(
        net=net,
        lr=1e-3,
        pretrain_epochs=0,
        mc_samples_train=1,
        mc_samples_eval=5,
        dataset_size=len(dm.train_dataloader().dataset),
        fit_context="lrt",
        prior_loc=0.0,
        prior_scale=0.3,
        guide="normal",
        q_scale=0.001,
        obs_scale=0.5,
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = L.Trainer(
            max_epochs=3,
            accelerator="cpu",
            enable_progress_bar=True,
            default_root_dir=tmpdir,
            enable_checkpointing=False,
            logger=False,
        )
        
        start = time.time()
        trainer.fit(model, dm)
        elapsed = time.time() - start
    
    print(f"  Training time: {elapsed:.2f}s")
    print(f"  Time per epoch: {elapsed/3:.2f}s")
    return {"status": "success", "time": elapsed}


def test_bnn_forces_aux_training_loop():
    """Test BNN_Forces_Aux training loop."""
    print("\n" + "="*60)
    print("TEST: BNN_Forces_Aux Training Loop (3 epochs)")
    print("="*60)
    
    from bnn_aenet.models.bnn import BNN_Forces_Aux
    from bnn_aenet.datamodule.aenet_datamodule import AenetDataModule
    
    data_path = "/home/g15farris/bin/bayesaenet/data/TiO/train.in"
    
    dm = AenetDataModule(
        data_dir=data_path,
        batch_size=64,
        split_config="Data100",
        device="cpu"
    )
    dm.setup("fit")
    
    net = create_net_from_datamodule(dm)
    net.alpha = torch.tensor(0.1)
    
    model = BNN_Forces_Aux(
        net=net,
        lr=1e-3,
        pretrain_epochs=0,
        mc_samples_train=1,
        mc_samples_eval=5,
        dataset_size=len(dm.train_dataloader().dataset),
        fit_context="flipout",
        prior_loc=0.0,
        prior_scale=0.3,
        guide="normal",
        q_scale=0.001,
        obs_scale=0.5,
        force_weight=1.0,
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = L.Trainer(
            max_epochs=3,
            accelerator="cpu",
            enable_progress_bar=True,
            default_root_dir=tmpdir,
            enable_checkpointing=False,
            logger=False,
        )
        
        start = time.time()
        trainer.fit(model, dm)
        elapsed = time.time() - start
    
    print(f"  Training time: {elapsed:.2f}s")
    print(f"  Time per epoch: {elapsed/3:.2f}s")
    return {"status": "success", "time": elapsed}


def test_checkpoint_save_load():
    """Test checkpoint saving and loading."""
    print("\n" + "="*60)
    print("TEST: Checkpoint Save/Load")
    print("="*60)
    
    from bnn_aenet.models.bnn import NN
    from bnn_aenet.datamodule.aenet_datamodule import AenetDataModule
    
    data_path = "/home/g15farris/bin/bayesaenet/data/TiO/train.in"
    
    dm = AenetDataModule(
        data_dir=data_path,
        batch_size=64,
        split_config="Data100",
        device="cpu"
    )
    dm.setup("fit")
    
    net = create_net_from_datamodule(dm)
    net.alpha = torch.tensor(0.1)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    model = NN(net=net, optimizer=optimizer)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_cb = ModelCheckpoint(
            dirpath=tmpdir,
            filename="test-{epoch:02d}",
            save_last=True,
        )
        
        trainer = L.Trainer(
            max_epochs=2,
            accelerator="cpu",
            enable_progress_bar=False,
            default_root_dir=tmpdir,
            callbacks=[checkpoint_cb],
            logger=False,
        )
        
        trainer.fit(model, dm)
        
        # Check checkpoint exists
        ckpt_path = Path(tmpdir) / "last.ckpt"
        assert ckpt_path.exists(), "Checkpoint not saved"
        
        # Load checkpoint
        net_loaded = create_net_from_datamodule(dm)
        loaded = NN.load_from_checkpoint(
            ckpt_path,
            net=net_loaded,
            optimizer=torch.optim.Adam(net_loaded.parameters(), lr=1e-3)
        )
        
        print(f"  Checkpoint saved: {ckpt_path.name}")
        print(f"  Checkpoint size: {ckpt_path.stat().st_size / 1024:.1f} KB")
        print(f"  Load successful: True")
    
    return {"status": "success"}


def test_different_batch_sizes():
    """Test training with different batch sizes."""
    print("\n" + "="*60)
    print("TEST: Different Batch Sizes")
    print("="*60)
    
    from bnn_aenet.models.bnn import NN
    from bnn_aenet.datamodule.aenet_datamodule import AenetDataModule
    
    data_path = "/home/g15farris/bin/bayesaenet/data/TiO/train.in"
    batch_sizes = [32, 64, 128, 256]
    
    results = {}
    
    for bs in batch_sizes:
        dm = AenetDataModule(
            data_dir=data_path,
            batch_size=bs,
            split_config="Data100",
            device="cpu"
        )
        dm.setup("fit")
        
        net = create_net_from_datamodule(dm)
        net.alpha = torch.tensor(0.1)
        
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        model = NN(net=net, optimizer=optimizer)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = L.Trainer(
                max_epochs=2,
                accelerator="cpu",
                enable_progress_bar=False,
                default_root_dir=tmpdir,
                enable_checkpointing=False,
                logger=False,
            )
            
            start = time.time()
            trainer.fit(model, dm)
            elapsed = time.time() - start
        
        results[bs] = elapsed
        print(f"  Batch size {bs}: {elapsed:.2f}s")
    
    return {"status": "success", "results": results}


def test_qm7_dataset():
    """Test training on QM7 dataset."""
    print("\n" + "="*60)
    print("TEST: QM7 Dataset Training (3 epochs)")
    print("="*60)
    
    from bnn_aenet.models.bnn import NN
    from bnn_aenet.datamodule.aenet_datamodule import AenetDataModule
    
    data_path = "/home/g15farris/bin/bayesaenet/data/QM7/train.in"
    
    dm = AenetDataModule(
        data_dir=data_path,
        batch_size=64,
        split_config="Data10",
        device="cpu"
    )
    dm.setup("fit")
    
    net = create_net_from_datamodule(dm)
    net.alpha = torch.tensor(0.1)
    
    print(f"  Species: {net.species}")
    print(f"  Input size: {net.input_size}")
    print(f"  Train batches: {len(dm.train_dataloader())}")
    
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    model = NN(net=net, optimizer=optimizer)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = L.Trainer(
            max_epochs=3,
            accelerator="cpu",
            enable_progress_bar=True,
            default_root_dir=tmpdir,
            enable_checkpointing=False,
            logger=False,
        )
        
        start = time.time()
        trainer.fit(model, dm)
        elapsed = time.time() - start
    
    print(f"  Training time: {elapsed:.2f}s")
    return {"status": "success", "time": elapsed}


def main():
    print("="*60)
    print("BNN-AENET INTEGRATION TESTS")
    print("="*60)
    
    L.seed_everything(42, workers=True)
    
    results = {}
    
    tests = [
        ("nn_training", test_nn_training_loop),
        ("nn_forces_training", test_nn_forces_training_loop),
        ("bnn_flipout_training", test_bnn_flipout_training_loop),
        ("bnn_lrt_training", test_bnn_lrt_training_loop),
        ("bnn_forces_aux_training", test_bnn_forces_aux_training_loop),
        ("checkpoint_save_load", test_checkpoint_save_load),
        ("different_batch_sizes", test_different_batch_sizes),
        ("qm7_dataset", test_qm7_dataset),
    ]
    
    for name, test_func in tests:
        try:
            result = test_func()
            results[name] = result
        except Exception as e:
            print(f"\n  FAILED: {e}")
            results[name] = {"status": "failed", "error": str(e)}
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(1 for r in results.values() if r.get("status") == "success")
    failed = len(results) - passed
    
    print(f"\nPassed: {passed}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")
    
    for name, result in results.items():
        status = "PASS" if result.get("status") == "success" else "FAIL"
        print(f"  [{status}] {name}")
    
    print("\n" + "="*60)
    print("INTEGRATION TESTS COMPLETE")
    print("="*60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
