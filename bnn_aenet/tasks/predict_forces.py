"""Prediction script for force-trained models.

Runs predictions on all checkpoints for a given model type (nn/lrt/rad),
saving energy and force predictions separately to handle different array lengths.

Usage:
    python -m bnn_aenet.tasks.predict_forces \
        --model-type nn --runs-dir bnn_aenet/logs/TiO2_small/train/runs/nn \
        --output-dir bnn_aenet/logs/TiO2_small/pred/nn \
        --data-dir data/TiO/train_forces.in \
        --use-run-config   # use same splits as training
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
BNN_Forces = None
PartialBNN_Forces = None
BNN_Forces_Hetero = None


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


def _ensure_bnn():
    global BNN_Forces
    if BNN_Forces is None:
        from bnn_aenet.models.bnn_forces import BNN_Forces as _cls
        BNN_Forces = _cls


def _ensure_partial_bnn():
    global PartialBNN_Forces
    if PartialBNN_Forces is None:
        from bnn_aenet.models.bnn_forces import PartialBNN_Forces as _cls
        PartialBNN_Forces = _cls


def _ensure_bnn_hetero():
    global BNN_Forces_Hetero
    if BNN_Forces_Hetero is None:
        from bnn_aenet.models.bnn_forces_hetero import BNN_Forces_Hetero as _cls
        BNN_Forces_Hetero = _cls


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


def instantiate_datamodule(data_dir: str, batch_size: int = 32, split_config: str = None):
    """Instantiate the TiO_Forces datamodule.

    split_config must match training (e.g. Data20 for TiO2_small, Data100 for TiO2_big).
    If omitted, uses data_dir/splits/ which may differ from training splits.
    """
    _ensure_datamodule()
    return AenetDataModule(
        data_dir=data_dir,
        device="cpu",
        batch_size=batch_size,
        test_split=0.1,
        valid_split=0.1,
        name="TiO2_Forces",
        split_config=split_config,
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
    """Load NN model from checkpoint."""
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
    """Load BNN_Forces or PartialBNN_Forces model from checkpoint."""
    overrides = load_overrides(run_dir)
    experiment = overrides.get("experiment", "")

    use_partial = "partial" in experiment
    use_hetero = "hetero" in experiment
    if use_partial:
        _ensure_partial_bnn()
        cls = PartialBNN_Forces
    elif use_hetero:
        _ensure_bnn_hetero()
        cls = BNN_Forces_Hetero
    else:
        _ensure_bnn()
        cls = BNN_Forces

    net = build_net(dm, alpha=0.1)

    # Parse BNN hyperparameters from overrides
    lr = float(overrides.get("model.lr", "0.001"))
    mc_train = int(overrides.get("model.mc_samples_train", "2"))
    prior_scale = float(overrides.get("model.prior_scale", "0.1"))
    q_scale = float(overrides.get("model.q_scale", "0.001"))
    obs_scale = float(overrides.get("model.obs_scale", "0.5"))
    scale_force = float(overrides.get("model.scale_force", "0.1"))
    learn_noise_raw = overrides.get("model.learn_noise", "false")
    learn_noise = learn_noise_raw.lower() in ("true", "1", "yes")
    pretrain_epochs = int(overrides.get("model.pretrain_epochs", "0"))

    # Bayesian layers for partial (default last)
    raw = overrides.get("model.bayesian_layers", "last")
    if raw in ("null", "None"):
        bayesian_layers = "last"
    elif raw in ("last", "first", "first_last", "all"):
        bayesian_layers = raw
    else:
        try:
            bayesian_layers = yaml.safe_load(raw)
        except Exception:
            bayesian_layers = "last"

    # Determine fit_context and guide from experiment name
    if "lrt" in experiment:
        fit_context = "lrt"
        guide = "normal"
    elif "rad" in experiment:
        fit_context = None
        guide = "radial"
    else:
        fit_context = "lrt"
        guide = "normal"

    kwargs = dict(
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
        scale_force=scale_force,
        grad_clip_val=1.0,
        learn_noise=learn_noise,
        pretrain_epochs=pretrain_epochs,
        strict=False,
    )
    if use_partial:
        kwargs["bayesian_layers"] = bayesian_layers
    if use_hetero:
        kwargs["noise_hidden_size"] = int(overrides.get("model.noise_hidden_size", "15"))
        kwargs["noise_min"] = float(overrides.get("model.noise_min", "0.01"))
    model = cls.load_from_checkpoint(str(ckpt_path), **kwargs)
    model.eval()
    return model


def run_predictions(model, datamodule, subsets, device, model_type=None, ckpt_path=None, run_dir=None, mc_eval=20):
    """Run predictions on specified data subsets."""
    trainer = Trainer(
        accelerator=device,
        devices=1,
        logger=False,
        enable_progress_bar=True,
        inference_mode=False,
    )

    dataloader_map = {
        "train": datamodule.train_dataloader(),
        "val": datamodule.val_dataloader(),
        "test": datamodule.test_dataloader(),
    }

    results = {}
    for i, subset in enumerate(subsets):
        dl = dataloader_map[subset]
        print(f"  Predicting on {subset} set...")
        # BNNs need fresh model per subset (Pyro "executed outside supermodule" when reusing)
        if model_type in ("lrt", "rad") and i > 0:
            model = load_bnn_model(ckpt_path, datamodule, run_dir, mc_eval=mc_eval)
        batch_results = trainer.predict(model=model, dataloaders=dl)
        results[subset] = batch_results

    return results


def save_predictions(batch_results, output_path, run_name, subset, e_scaling):
    """Save predictions: energy to CSV, forces to npz. Metrics in meV/atom and meV/Å."""
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

    # Energy DataFrame (values in normalized units)
    energy_df = pd.DataFrame({
        "true": np.concatenate(all_true),
        "preds": np.concatenate(all_preds),
        "stds": np.concatenate(all_stds),
        "n_atoms": np.concatenate(all_n_atoms),
    })

    energy_file = output_path / f"{run_name}_{subset}_energy.csv"
    energy_df.to_csv(energy_file, index=False)
    # Per-atom energy RMSE in meV/atom
    per_atom_err = (energy_df["true"] - energy_df["preds"]) / energy_df["n_atoms"]
    e_rmse = np.sqrt(np.mean(per_atom_err**2)) / e_scaling * 1000
    e_mae = np.mean(np.abs(per_atom_err)) / e_scaling * 1000
    print(f"    Energy: {len(energy_df)} structures, RMSE={e_rmse:.4f} meV/atom, MAE={e_mae:.4f} meV/atom")

    # Force data (values in normalized units)
    if len(all_true_forces) > 0:
        ft = np.concatenate(all_true_forces)
        fp = np.concatenate(all_pred_forces)
        fs = np.concatenate(all_std_forces)

        force_file = output_path / f"{run_name}_{subset}_forces.npz"
        np.savez_compressed(
            force_file,
            true_forces=ft,
            pred_forces=fp,
            std_forces=fs,
            e_scaling=np.array([e_scaling]),
        )

        f_rmse = np.sqrt(np.mean((ft - fp) ** 2)) / e_scaling * 1000  # meV/Å
        f_mae = np.mean(np.abs(ft - fp)) / e_scaling * 1000
        print(f"    Forces: {len(ft)} components, RMSE={f_rmse:.4f} meV/Å, MAE={f_mae:.4f} meV/Å")


def main():
    parser = argparse.ArgumentParser(description="Run predictions for force-trained models")
    parser.add_argument("--model-type", type=str, required=True, choices=["nn", "lrt", "rad", "lrt_hetero", "rad_hetero"])
    parser.add_argument("--runs-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--split-config", type=str, default=None,
                        help="Split config to match training (e.g. Data20 for TiO2_small). Must match dataset size.")
    parser.add_argument("--use-run-config", action="store_true",
                        help="Load datamodule config (data_dir, split_config) from first run's .hydra/config.yaml")
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
    print(f"Split config: {args.split_config or 'default (data_dir/splits/)'}")
    print(f"Device: {args.device}")
    print()

    # Load datamodule once (split_config must match training!)
    data_dir = args.data_dir
    split_config = args.split_config
    if args.use_run_config and run_dirs:
        run_config = run_dirs[0] / ".hydra" / "config.yaml"
        if run_config.exists():
            with open(run_config) as f:
                cfg = yaml.safe_load(f)
            dm_cfg = cfg.get("datamodule", {})
            if "data_dir" in dm_cfg:
                root = Path(os.environ.get("PROJECT_ROOT", os.getcwd()))
                data_dir = str(dm_cfg["data_dir"]).replace("${paths.data_dir}", str(root / "data"))
            if "split_config" in dm_cfg:
                split_config = dm_cfg["split_config"]
            print(f"  Using config from {run_dirs[0].name}")

    print("Loading datamodule...")
    dm = instantiate_datamodule(
        data_dir,
        batch_size=args.batch_size,
        split_config=split_config,
    )
    print(f"  Train size: {dm.train_size}, Species: {dm.species}")
    print()

    for run_dir in run_dirs:
        run_name = run_dir.name
        print(f"{'='*60}")
        print(f"Run: {run_name}")

        # Find best checkpoint (save_top_k=1 keeps best; prefer epoch_* over last.ckpt)
        ckpt_files = sorted(run_dir.glob("checkpoints/epoch_*.ckpt"))
        if not ckpt_files:
            last_ckpt = run_dir / "checkpoints" / "last.ckpt"
            if last_ckpt.exists():
                ckpt_path = last_ckpt
            else:
                print("  No checkpoint found, skipping")
                continue
        else:
            ckpt_path = ckpt_files[-1]
        print(f"  Checkpoint: {ckpt_path.name}")

        try:
            if args.model_type == "nn":
                model = load_nn_model(ckpt_path, dm)
            else:
                model = load_bnn_model(ckpt_path, dm, run_dir, mc_eval=args.mc_samples)

            results = run_predictions(
                model, dm, args.subsets, args.device,
                model_type=args.model_type,
                ckpt_path=ckpt_path,
                run_dir=run_dir,
                mc_eval=args.mc_samples,
            )

            for subset, batch_results in results.items():
                save_predictions(batch_results, output_dir, run_name, subset, dm.e_scaling)

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

        print()

    print("All predictions complete!")


if __name__ == "__main__":
    main()
