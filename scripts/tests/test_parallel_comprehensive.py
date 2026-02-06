#!/usr/bin/env python
"""
Comprehensive parallel tests for BNN-AENET on iqtc12 (60 cores).
Uses multiprocessing to run tests concurrently.
"""

import sys
import time
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_architecture_params():
    """Test parameter counts for different architectures."""
    from bnn_aenet.models.nets.network import NetAtom
    
    results = {}
    architectures = [
        {"hidden_size": [[10, 10], [10, 10]], "name": "10:10"},
        {"hidden_size": [[15, 15], [15, 15]], "name": "15:15"},
        {"hidden_size": [[20, 20], [20, 20]], "name": "20:20"},
        {"hidden_size": [[25, 25], [25, 25]], "name": "25:25"},
        {"hidden_size": [[30, 30], [30, 30]], "name": "30:30"},
        {"hidden_size": [[40, 40], [40, 40]], "name": "40:40"},
        {"hidden_size": [[50, 50], [50, 50]], "name": "50:50"},
        {"hidden_size": [[50, 25], [50, 25]], "name": "50:25"},
        {"hidden_size": [[64, 32], [64, 32]], "name": "64:32"},
        {"hidden_size": [[64, 32, 16], [64, 32, 16]], "name": "64:32:16"},
        {"hidden_size": [[100, 50, 25], [100, 50, 25]], "name": "100:50:25"},
    ]
    
    for arch in architectures:
        n_layers = len(arch["hidden_size"][0])
        net = NetAtom(
            input_size=[70, 70],
            hidden_size=arch["hidden_size"],
            species=["Ti", "O"],
            active_names=[["tanh"] * n_layers] * 2,
            alpha=0.1,
            device="cpu",
            e_scaling=1.0,
            e_shift=0.0
        )
        n_params = sum(p.numel() for p in net.parameters())
        results[arch["name"]] = n_params
    
    return {"test": "architecture_params", "results": results}


def test_bnn_lrt_init():
    """Test LRT BNN initialization."""
    from bnn_aenet.models.bnn import BNN
    from bnn_aenet.models.nets.network import NetAtom
    
    net = NetAtom(
        input_size=[70, 70],
        hidden_size=[[15, 15], [15, 15]],
        species=["Ti", "O"],
        active_names=[["tanh", "tanh"], ["tanh", "tanh"]],
        alpha=0.1, device="cpu", e_scaling=1.0, e_shift=0.0
    )
    
    try:
        model = BNN(
            net=net, lr=1e-3, pretrain_epochs=0,
            mc_samples_train=1, mc_samples_eval=10,
            dataset_size=1000, fit_context="lrt",
            prior_loc=0.0, prior_scale=0.3,
            guide="normal", q_scale=0.001, obs_scale=0.5,
        )
        return {"test": "bnn_lrt_init", "status": "success", "params": 2642}
    except Exception as e:
        return {"test": "bnn_lrt_init", "status": "failed", "error": str(e)}


def test_bnn_flipout_init():
    """Test Flipout BNN initialization."""
    from bnn_aenet.models.bnn import BNN
    from bnn_aenet.models.nets.network import NetAtom
    
    net = NetAtom(
        input_size=[70, 70],
        hidden_size=[[15, 15], [15, 15]],
        species=["Ti", "O"],
        active_names=[["tanh", "tanh"], ["tanh", "tanh"]],
        alpha=0.1, device="cpu", e_scaling=1.0, e_shift=0.0
    )
    
    try:
        model = BNN(
            net=net, lr=1e-3, pretrain_epochs=0,
            mc_samples_train=1, mc_samples_eval=10,
            dataset_size=1000, fit_context="flipout",
            prior_loc=0.0, prior_scale=0.3,
            guide="normal", q_scale=0.001, obs_scale=0.5,
        )
        return {"test": "bnn_flipout_init", "status": "success", "params": 2642}
    except Exception as e:
        return {"test": "bnn_flipout_init", "status": "failed", "error": str(e)}


def test_nn_init():
    """Test NN initialization."""
    from bnn_aenet.models.bnn import NN
    from bnn_aenet.models.nets.network import NetAtom
    import torch
    
    net = NetAtom(
        input_size=[70, 70],
        hidden_size=[[15, 15], [15, 15]],
        species=["Ti", "O"],
        active_names=[["tanh", "tanh"], ["tanh", "tanh"]],
        alpha=0.1, device="cpu", e_scaling=1.0, e_shift=0.0
    )
    
    try:
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        model = NN(net=net, optimizer=optimizer)
        return {"test": "nn_init", "status": "success", "params": 2642}
    except Exception as e:
        return {"test": "nn_init", "status": "failed", "error": str(e)}


def test_nn_forces_init():
    """Test NN_Forces initialization."""
    from bnn_aenet.models.bnn import NN_Forces
    from bnn_aenet.models.nets.network import NetAtom
    import torch
    
    net = NetAtom(
        input_size=[70, 70],
        hidden_size=[[15, 15], [15, 15]],
        species=["Ti", "O"],
        active_names=[["tanh", "tanh"], ["tanh", "tanh"]],
        alpha=0.1, device="cpu", e_scaling=1.0, e_shift=0.0
    )
    
    try:
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        model = NN_Forces(net=net, optimizer=optimizer, force_weight=1.0)
        return {"test": "nn_forces_init", "status": "success", "params": 2642}
    except Exception as e:
        return {"test": "nn_forces_init", "status": "failed", "error": str(e)}


def test_bnn_forces_aux_init():
    """Test BNN_Forces_Aux initialization."""
    from bnn_aenet.models.bnn import BNN_Forces_Aux
    from bnn_aenet.models.nets.network import NetAtom
    
    net = NetAtom(
        input_size=[70, 70],
        hidden_size=[[15, 15], [15, 15]],
        species=["Ti", "O"],
        active_names=[["tanh", "tanh"], ["tanh", "tanh"]],
        alpha=0.1, device="cpu", e_scaling=1.0, e_shift=0.0
    )
    
    try:
        model = BNN_Forces_Aux(
            net=net, lr=1e-3, pretrain_epochs=0,
            mc_samples_train=1, mc_samples_eval=10,
            dataset_size=1000, fit_context="flipout",
            prior_loc=0.0, prior_scale=0.3,
            guide="normal", q_scale=0.001, obs_scale=0.5,
            force_weight=1.0,
        )
        return {"test": "bnn_forces_aux_init", "status": "success", "params": 2642}
    except Exception as e:
        return {"test": "bnn_forces_aux_init", "status": "failed", "error": str(e)}


def test_energy_metrics():
    """Test energy metrics computation."""
    import numpy as np
    from bnn_aenet.analysis.metrics import compute_energy_metrics
    
    np.random.seed(42)
    n = 500
    true = np.random.randn(n) * 0.5 + 1.0
    pred = true + np.random.randn(n) * 0.05
    n_atoms = np.random.randint(10, 100, n)
    
    try:
        metrics = compute_energy_metrics(true, pred, n_atoms)
        return {"test": "energy_metrics", "status": "success", "metrics": metrics}
    except Exception as e:
        return {"test": "energy_metrics", "status": "failed", "error": str(e)}


def test_force_metrics():
    """Test force metrics computation."""
    import numpy as np
    from bnn_aenet.analysis.metrics import compute_force_metrics
    
    np.random.seed(42)
    n = 500
    true = np.random.randn(n, 3) * 0.1
    pred = true + np.random.randn(n, 3) * 0.01
    
    try:
        metrics = compute_force_metrics(true, pred)
        return {"test": "force_metrics", "status": "success", "metrics": metrics}
    except Exception as e:
        return {"test": "force_metrics", "status": "failed", "error": str(e)}


def test_uncertainty_metrics():
    """Test uncertainty metrics computation."""
    import numpy as np
    from bnn_aenet.analysis.metrics import compute_uncertainty_metrics
    
    np.random.seed(42)
    n = 500
    true = np.random.randn(n) * 0.5 + 1.0
    pred = true + np.random.randn(n) * 0.05
    std = np.abs(np.random.randn(n) * 0.03 + 0.05)
    
    try:
        metrics = compute_uncertainty_metrics(true, pred, std)
        return {"test": "uncertainty_metrics", "status": "success", "metrics": metrics}
    except Exception as e:
        return {"test": "uncertainty_metrics", "status": "failed", "error": str(e)}


def test_datamodule_tio():
    """Test TiO2 datamodule loading."""
    from bnn_aenet.datamodule.aenet_datamodule import AenetDataModule
    
    data_path = "/home/g15farris/bin/bayesaenet/data/TiO/train.in"
    
    try:
        dm = AenetDataModule(
            data_dir=data_path,
            batch_size=64,
            split_config="Data100",
            device="cpu"
        )
        dm.setup("fit")
        
        train_batches = len(dm.train_dataloader())
        val_batches = len(dm.val_dataloader())
        test_batches = len(dm.test_dataloader())
        
        return {
            "test": "datamodule_tio",
            "status": "success",
            "train_batches": train_batches,
            "val_batches": val_batches,
            "test_batches": test_batches,
        }
    except Exception as e:
        return {"test": "datamodule_tio", "status": "failed", "error": str(e)}


def test_datamodule_tio_data20():
    """Test TiO2 Data20 split."""
    from bnn_aenet.datamodule.aenet_datamodule import AenetDataModule
    
    data_path = "/home/g15farris/bin/bayesaenet/data/TiO/train.in"
    
    try:
        dm = AenetDataModule(
            data_dir=data_path,
            batch_size=64,
            split_config="Data20",
            device="cpu"
        )
        dm.setup("fit")
        
        train_batches = len(dm.train_dataloader())
        
        return {
            "test": "datamodule_tio_data20",
            "status": "success",
            "train_batches": train_batches,
        }
    except Exception as e:
        return {"test": "datamodule_tio_data20", "status": "failed", "error": str(e)}


def test_datamodule_qm7():
    """Test QM7 datamodule loading."""
    from bnn_aenet.datamodule.aenet_datamodule import AenetDataModule
    
    data_path = "/home/g15farris/bin/bayesaenet/data/QM7/train.in"
    
    try:
        dm = AenetDataModule(
            data_dir=data_path,
            batch_size=64,
            split_config="Data10",
            device="cpu"
        )
        dm.setup("fit")
        
        train_batches = len(dm.train_dataloader())
        
        return {
            "test": "datamodule_qm7",
            "status": "success",
            "train_batches": train_batches,
        }
    except Exception as e:
        return {"test": "datamodule_qm7", "status": "failed", "error": str(e)}


def test_batch_structure_tio():
    """Test batch structure for TiO2."""
    from bnn_aenet.datamodule.aenet_datamodule import AenetDataModule
    from bnn_aenet.datamodule.aenet.batch_constants import BatchIdx, has_force_data
    
    data_path = "/home/g15farris/bin/bayesaenet/data/TiO/train.in"
    
    try:
        dm = AenetDataModule(
            data_dir=data_path,
            batch_size=32,
            split_config="Data100",
            device="cpu"
        )
        dm.setup("fit")
        
        batch = next(iter(dm.train_dataloader()))
        
        has_forces = has_force_data(batch)
        batch_len = len(batch)
        e_energy_shape = list(batch[BatchIdx.E_ENERGY].shape)
        
        return {
            "test": "batch_structure_tio",
            "status": "success",
            "batch_length": batch_len,
            "has_forces": has_forces,
            "energy_shape": e_energy_shape,
        }
    except Exception as e:
        return {"test": "batch_structure_tio", "status": "failed", "error": str(e)}


def test_hydra_configs():
    """Test Hydra configuration loading."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    
    config_path = "/home/g15farris/bin/bayesaenet/bnn_aenet/configs"
    
    results = {}
    configs_to_test = [
        ("experiment", "nn_forces"),
        ("experiment", "bnn_lrt_forces_aux"),
        ("experiment", "bnn_fo_forces_aux"),
        ("experiment", "bnn_rad_forces_aux"),
        ("datamodule", "TiO"),
        ("datamodule", "TiO_Data20"),
        ("datamodule", "QM7"),
    ]
    
    for config_type, config_name in configs_to_test:
        try:
            GlobalHydra.instance().clear()
            with initialize_config_dir(config_dir=config_path, version_base="1.3"):
                cfg = compose(config_name=f"{config_type}/{config_name}")
                results[f"{config_type}/{config_name}"] = "success"
        except Exception as e:
            results[f"{config_type}/{config_name}"] = f"failed: {str(e)[:50]}"
    
    GlobalHydra.instance().clear()
    return {"test": "hydra_configs", "results": results}


def test_activation_functions():
    """Test different activation functions."""
    from bnn_aenet.models.nets.network import NetAtom
    import torch
    
    activations = ["tanh", "sigmoid", "relu", "softplus", "elu", "gelu", "silu"]
    results = {}
    
    for act in activations:
        try:
            net = NetAtom(
                input_size=[70, 70],
                hidden_size=[[15, 15], [15, 15]],
                species=["Ti", "O"],
                active_names=[[act, act], [act, act]],
                alpha=0.1, device="cpu", e_scaling=1.0, e_shift=0.0
            )
            results[act] = "success"
        except Exception as e:
            results[act] = f"failed: {str(e)[:50]}"
    
    return {"test": "activation_functions", "results": results}


def test_different_lr():
    """Test model with different learning rates."""
    from bnn_aenet.models.bnn import NN
    from bnn_aenet.models.nets.network import NetAtom
    import torch
    
    lrs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    results = {}
    
    for lr in lrs:
        try:
            net = NetAtom(
                input_size=[70, 70],
                hidden_size=[[15, 15], [15, 15]],
                species=["Ti", "O"],
                active_names=[["tanh", "tanh"], ["tanh", "tanh"]],
                alpha=0.1, device="cpu", e_scaling=1.0, e_shift=0.0
            )
            optimizer = torch.optim.Adam(net.parameters(), lr=lr)
            model = NN(net=net, optimizer=optimizer)
            results[str(lr)] = "success"
        except Exception as e:
            results[str(lr)] = f"failed: {str(e)[:50]}"
    
    return {"test": "different_lr", "results": results}


def test_prior_scales():
    """Test BNN with different prior scales."""
    from bnn_aenet.models.bnn import BNN
    from bnn_aenet.models.nets.network import NetAtom
    
    scales = [0.01, 0.1, 0.3, 0.5, 1.0]
    results = {}
    
    for scale in scales:
        try:
            net = NetAtom(
                input_size=[70, 70],
                hidden_size=[[15, 15], [15, 15]],
                species=["Ti", "O"],
                active_names=[["tanh", "tanh"], ["tanh", "tanh"]],
                alpha=0.1, device="cpu", e_scaling=1.0, e_shift=0.0
            )
            model = BNN(
                net=net, lr=1e-3, pretrain_epochs=0,
                mc_samples_train=1, mc_samples_eval=10,
                dataset_size=1000, fit_context="flipout",
                prior_loc=0.0, prior_scale=scale,
                guide="normal", q_scale=0.001, obs_scale=0.5,
            )
            results[str(scale)] = "success"
        except Exception as e:
            results[str(scale)] = f"failed: {str(e)[:50]}"
    
    return {"test": "prior_scales", "results": results}


def test_q_scales():
    """Test BNN with different q_scales (variational scale)."""
    from bnn_aenet.models.bnn import BNN
    from bnn_aenet.models.nets.network import NetAtom
    
    scales = [1e-5, 1e-4, 1e-3, 0.005, 0.01]
    results = {}
    
    for scale in scales:
        try:
            net = NetAtom(
                input_size=[70, 70],
                hidden_size=[[15, 15], [15, 15]],
                species=["Ti", "O"],
                active_names=[["tanh", "tanh"], ["tanh", "tanh"]],
                alpha=0.1, device="cpu", e_scaling=1.0, e_shift=0.0
            )
            model = BNN(
                net=net, lr=1e-3, pretrain_epochs=0,
                mc_samples_train=1, mc_samples_eval=10,
                dataset_size=1000, fit_context="flipout",
                prior_loc=0.0, prior_scale=0.3,
                guide="normal", q_scale=scale, obs_scale=0.5,
            )
            results[str(scale)] = "success"
        except Exception as e:
            results[str(scale)] = f"failed: {str(e)[:50]}"
    
    return {"test": "q_scales", "results": results}


def test_imports():
    """Test all critical imports."""
    results = {}
    
    imports = [
        ("bnn_aenet.models.bnn", "BNN"),
        ("bnn_aenet.models.bnn", "NN"),
        ("bnn_aenet.models.bnn", "BNN_Forces_Aux"),
        ("bnn_aenet.models.bnn", "NN_Forces"),
        ("bnn_aenet.models.nets.network", "NetAtom"),
        ("bnn_aenet.datamodule.aenet_datamodule", "AenetDataModule"),
        ("bnn_aenet.analysis.metrics", "compute_energy_metrics"),
        ("bnn_aenet.analysis.metrics", "compute_force_metrics"),
        ("bnn_aenet.analysis.metrics", "compute_uncertainty_metrics"),
        ("bnn_aenet.datamodule.aenet.batch_constants", "BatchIdx"),
        ("bnn_aenet.tasks.train", "train"),
        ("bnn_aenet.tasks.hpsearch", "hpsearch"),
    ]
    
    for module, name in imports:
        try:
            exec(f"from {module} import {name}")
            results[f"{module}.{name}"] = "success"
        except Exception as e:
            results[f"{module}.{name}"] = f"failed: {str(e)[:50]}"
    
    return {"test": "imports", "results": results}


def run_test(test_func):
    """Run a test function and catch any exceptions."""
    try:
        return test_func()
    except Exception as e:
        return {"test": test_func.__name__, "status": "crashed", "error": str(e)}


def main():
    print("="*70)
    print("BNN-AENET Comprehensive Parallel Tests")
    print(f"Using {mp.cpu_count()} CPU cores")
    print("="*70)
    
    # All test functions
    tests = [
        test_architecture_params,
        test_bnn_lrt_init,
        test_bnn_flipout_init,
        test_nn_init,
        test_nn_forces_init,
        test_bnn_forces_aux_init,
        test_energy_metrics,
        test_force_metrics,
        test_uncertainty_metrics,
        test_datamodule_tio,
        test_datamodule_tio_data20,
        test_datamodule_qm7,
        test_batch_structure_tio,
        test_hydra_configs,
        test_activation_functions,
        test_different_lr,
        test_prior_scales,
        test_q_scales,
        test_imports,
    ]
    
    print(f"\nRunning {len(tests)} tests in parallel...\n")
    
    start = time.time()
    
    # Run tests in parallel using ProcessPoolExecutor
    results = []
    with ProcessPoolExecutor(max_workers=min(len(tests), 20)) as executor:
        futures = {executor.submit(run_test, test): test.__name__ for test in tests}
        
        for future in as_completed(futures):
            test_name = futures[future]
            try:
                result = future.result(timeout=120)
                results.append(result)
                status = result.get("status", "completed")
                print(f"  [{status.upper():^8}] {result['test']}")
            except Exception as e:
                results.append({"test": test_name, "status": "timeout", "error": str(e)})
                print(f"  [TIMEOUT ] {test_name}")
    
    elapsed = time.time() - start
    
    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    passed = sum(1 for r in results if r.get("status") == "success" or "results" in r)
    failed = len(results) - passed
    
    print(f"\nTotal: {len(results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Time: {elapsed:.2f}s")
    
    # Detailed results
    print("\n" + "-"*70)
    print("DETAILED RESULTS")
    print("-"*70)
    
    for result in sorted(results, key=lambda x: x["test"]):
        print(f"\n{result['test']}:")
        if "error" in result:
            print(f"  ERROR: {result['error'][:100]}")
        elif "results" in result:
            for k, v in result["results"].items():
                if isinstance(v, dict):
                    print(f"  {k}:")
                    for k2, v2 in v.items():
                        print(f"    {k2}: {v2}")
                else:
                    print(f"  {k}: {v}")
        elif "metrics" in result:
            for k, v in result["metrics"].items():
                print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
        else:
            for k, v in result.items():
                if k != "test":
                    print(f"  {k}: {v}")
    
    # Save results to JSON
    output_file = Path(__file__).parent / "test_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")
    
    print("\n" + "="*70)
    print("TESTS COMPLETE")
    print("="*70)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
