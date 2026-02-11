"""Prediction script for force-trained models.

Runs predictions on all checkpoints for a given model type (nn/lrt/fo/rad),
saving energy and force predictions separately to handle different array lengths.

Usage:
    python -m bnn_aenet.tasks.predict_forces \
        --model-type nn --runs-dir bnn_aenet/logs/TiO2_big/nn_train \
        --output-dir bnn_aenet/logs/TiO2_big/nn_pred --device gpu \
        --data-dir data/TiO
"""

import argparse
import os
import sys
from pathlib import Path

# IMPORTANT: import torch BEFORE numpy/pandas to avoid segfault on iqtc10 nodes
# where CUDA/MKL library initialization order matters
import torch
import lightning.pytorch as L
from lightning.pytorch import Trainer
import numpy as np
import pandas as pd
import yaml

# ALL bnn_aenet imports are lazy to avoid triggering bnn_aenet/models/__init__.py
# which imports pyro/tyxe and segfaults on iqtc10 nodes when combined with
# torch+lightning in the same process.
AenetDataModule = None
NetAtom = None
NN_Forces = None
BNN_Forces_Aux = None


def _ensure_datamodule():
    global AenetDataModule
    if AenetDataModule is None:
        from bnn_aenet.datamodule.aenet_datamodule import AenetDataModule as _cls
        AenetDataModule = _cls


def _ensure_net_atom():
    global NetAtom
    if NetAtom is None:
        from bnn_aenet.models.nets.network import NetAtom as _cls
        NetAtom = _cls


def _ensure_nn_forces():
    global NN_Forces
    if NN_Forces is None:
        from bnn_aenet.models.nn import NN_Forces as _cls
        NN_Forces = _cls


def _ensure_bnn_forces_aux():
    global BNN_Forces_Aux
    if BNN_Forces_Aux is None:
        from bnn_aenet.models.bnn_forces import BNN_Forces_Aux as _cls
        BNN_Forces_Aux = _cls


def load_overrides(run_dir: Path) -> dict:
    """Load Hydra overrides from a training run and parse into a dict."""
    override_file = run_dir / ".hydra" / "overrides.yaml"
    if not override_file.exists():
        raise FileNotFoundError(f"No overrides found at {override_file}")
    
    with open(override_file) as f:
        lines = yaml.safe_load(f)
    
    overrides = {}
    for line in lines:
        line = line.lstrip("+-")
        if "=" in line:
            key, val = line.split("=", 1)
            overrides[key] = val
    return overrides


def instantiate_datamodule(data_dir: str, batch_size: int = 32):
    """Instantiate the TiO_Forces datamodule."""
    _ensure_datamodule()
    return AenetDataModule(
        data_dir=data_dir,
        device="cpu",
        batch_size=batch_size,
        test_split=0.1,
        valid_split=0.1,
        name="TiO2_Forces",
    )


def build_net(dm, alpha: float = 0.1):
    """Build the network from datamodule parameters."""
    _ensure_net_atom()
    return NetAtom(
        input_size=dm.input_size,
        hidden_size=dm.hidden_size,
        species=dm.species,
        active_names=dm.active_names,
        alpha=alpha,
        device="cpu",
        e_scaling=dm.e_scaling,
        e_shift=dm.e_shift,
    )


def load_nn_model(ckpt_path: Path, dm):
    """Load NN_Forces model from checkpoint."""
    _ensure_nn_forces()
    net = build_net(dm, alpha=0.1)
    model = NN_Forces.load_from_checkpoint(
        str(ckpt_path),
        net=net,
        optimizer=torch.optim.Adam,
        alpha=0.1,
        strict=False,
    )
    model.eval()
    return model


def load_bnn_model(ckpt_path: Path, dm, run_dir: Path, mc_eval: int = 20):
    """Load BNN_Forces_Aux model from checkpoint."""
    _ensure_bnn_forces_aux()
    overrides = load_overrides(run_dir)
    
    net = build_net(dm, alpha=0.1)
    
    # Parse BNN hyperparameters from overrides
    lr = float(overrides.get("model.lr", "0.001"))
    mc_train = int(overrides.get("model.mc_samples_train", "2"))
    prior_scale = float(overrides.get("model.prior_scale", "0.1"))
    q_scale = float(overrides.get("model.q_scale", "0.001"))
    obs_scale = float(overrides.get("model.obs_scale", "0.5"))
    pretrain_epochs = int(overrides.get("model.pretrain_epochs", "0"))
    
    # Determine fit_context and guide from experiment name
    experiment = overrides.get("experiment", "")
    if "lrt" in experiment:
        fit_context = "lrt"
        guide = "normal"
    elif "fo" in experiment:
        fit_context = "flipout"
        guide = "normal"
    elif "rad" in experiment:
        fit_context = None
        guide = "radial"
    else:
        fit_context = "lrt"
        guide = "normal"
    
    model = BNN_Forces_Aux.load_from_checkpoint(
        str(ckpt_path),
        net=net,
        lr=lr,
        mc_samples_train=mc_train,
        mc_samples_eval=mc_eval,
        dataset_size=dm.train_size,
        fit_context=fit_context,
        guide=guide,
        prior_loc=0,
        prior_scale=prior_scale,
        q_scale=q_scale,
        obs_scale=obs_scale,
        force_lr_scale=0.1,
        scale_lr_factor=0.5,
        grad_clip_val=1.0,
        pretrain_epochs=pretrain_epochs,
        strict=False,
    )
    model.eval()
    return model


def run_predictions(model, datamodule, subsets, device):
    """Run predictions on specified data subsets."""
    trainer = Trainer(
        accelerator=device,
        devices=1,
        logger=False,
        enable_progress_bar=True,
    )

    dataloader_map = {
        "train": datamodule.train_dataloader(),
        "val": datamodule.val_dataloader(),
        "test": datamodule.test_dataloader(),
    }

    results = {}
    for subset in subsets:
        dl = dataloader_map[subset]
        print(f"  Predicting on {subset} set...")
        batch_results = trainer.predict(model=model, dataloaders=dl)
        results[subset] = batch_results

    return results


def save_predictions(batch_results, output_path, run_name, subset):
    """Save predictions: energy to CSV, forces to npz."""
    output_path.mkdir(parents=True, exist_ok=True)

    all_true, all_preds, all_stds, all_n_atoms = [], [], [], []
    all_true_forces, all_pred_forces, all_std_forces = [], [], []

    for batch in batch_results:
        all_true.append(np.asarray(batch["true"]).flatten())
        all_preds.append(np.asarray(batch["preds"]).flatten())
        all_stds.append(np.asarray(batch["stds"]).flatten())
        all_n_atoms.append(np.asarray(batch["n_atoms"]).flatten())

        if batch.get("true_forces") is not None:
            all_true_forces.append(np.asarray(batch["true_forces"]).flatten())
            all_pred_forces.append(np.asarray(batch["pred_forces"]).flatten())
            all_std_forces.append(np.asarray(batch["std_forces"]).flatten())

    # Energy DataFrame
    energy_df = pd.DataFrame({
        "true": np.concatenate(all_true),
        "preds": np.concatenate(all_preds),
        "stds": np.concatenate(all_stds),
        "n_atoms": np.concatenate(all_n_atoms),
    })

    energy_file = output_path / f"{run_name}_{subset}_energy.csv"
    energy_df.to_csv(energy_file, index=False)
    e_rmse = np.sqrt(np.mean((energy_df["true"] - energy_df["preds"]) ** 2))
    e_mae = np.mean(np.abs(energy_df["true"] - energy_df["preds"]))
    print(f"    Energy: {len(energy_df)} structures, RMSE={e_rmse:.4f}, MAE={e_mae:.4f}")

    # Force data
    if len(all_true_forces) > 0:
        ft = np.concatenate(all_true_forces)
        fp = np.concatenate(all_pred_forces)
        fs = np.concatenate(all_std_forces)

        force_file = output_path / f"{run_name}_{subset}_forces.npz"
        np.savez_compressed(force_file, true_forces=ft, pred_forces=fp, std_forces=fs)

        f_rmse = np.sqrt(np.mean((ft - fp) ** 2))
        f_mae = np.mean(np.abs(ft - fp))
        print(f"    Forces: {len(ft)} components, RMSE={f_rmse:.4f}, MAE={f_mae:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Run predictions for force-trained models")
    parser.add_argument("--model-type", type=str, required=True, choices=["nn", "lrt", "fo", "rad"])
    parser.add_argument("--runs-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--subsets", type=str, nargs="+", default=["train", "val", "test"])
    parser.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mc-samples", type=int, default=20)

    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find run directories
    run_dirs = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir() and (d / "checkpoints").exists()]
    )
    
    if not run_dirs:
        print(f"No run directories found in {runs_dir}")
        return

    print(f"Found {len(run_dirs)} runs")
    print(f"Model type: {args.model_type}")
    print(f"Data: {args.data_dir}")
    print(f"Device: {args.device}")
    print()

    # Load datamodule once
    print("Loading datamodule...")
    dm = instantiate_datamodule(args.data_dir, batch_size=args.batch_size)
    print(f"  Train size: {dm.train_size}, Species: {dm.species}")
    print()

    for run_dir in run_dirs:
        run_name = run_dir.name
        print(f"{'='*60}")
        print(f"Run: {run_name}")

        # Find best checkpoint
        ckpt_files = sorted(run_dir.glob("checkpoints/epoch_*.ckpt"))
        if not ckpt_files:
            print("  No checkpoint found, skipping")
            continue

        ckpt_path = ckpt_files[-1]
        print(f"  Checkpoint: {ckpt_path.name}")

        try:
            if args.model_type == "nn":
                model = load_nn_model(ckpt_path, dm)
            else:
                model = load_bnn_model(ckpt_path, dm, run_dir, mc_eval=args.mc_samples)

            results = run_predictions(model, dm, args.subsets, args.device)

            for subset, batch_results in results.items():
                save_predictions(batch_results, output_dir, run_name, subset)

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

        print()

    print("All predictions complete!")


if __name__ == "__main__":
    main()
