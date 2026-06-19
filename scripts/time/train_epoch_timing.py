#!/usr/bin/env python
"""Per-epoch and total training-time benchmark for LRT / RAD / DE on TiO2.

Measures per-epoch wall-clock for each (model, regime) using each regime's
*actual* Optuna batch size and MC-sample count (Tables 1-2 of the paper) and the
matching data fraction, so the resulting Total Time is a faithful "how long does
a full training take" number. Per-epoch is the average wall-clock between
consecutive training-epoch starts (full epoch incl. validation), discarding
warmup epochs.

Total Time = (epochs to converge) x (per-epoch), x10 for DE (full ensemble).

Usage:
    python scripts/time/train_epoch_timing.py --device gpu
"""

import argparse
import csv
import time
from pathlib import Path

# isort: off
import torch  # import first (CUDA/MKL init order)

import numpy as np
import pyrootutils

# isort: on

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from lightning.pytorch import Trainer  # noqa: E402
from lightning.pytorch.callbacks import Callback  # noqa: E402

# (model, regime) -> training config. Batch size and MC samples are the Optuna
# values (paper Tables 1-2); lr/prior/q-scale do not affect per-epoch timing.
CONFIGS = {
    ("lrt", "low"): dict(
        batch_size=256, mc=1, fit_context="lrt", guide="normal", train_fraction=0.2
    ),
    ("rad", "low"): dict(
        batch_size=512, mc=2, fit_context=None, guide="radial", train_fraction=0.2
    ),
    ("de", "low"): dict(batch_size=128, train_fraction=0.2),
    ("lrt", "high"): dict(
        batch_size=128, mc=1, fit_context="lrt", guide="normal", train_fraction=1.0
    ),
    ("rad", "high"): dict(
        batch_size=256, mc=1, fit_context=None, guide="radial", train_fraction=1.0
    ),
    ("de", "high"): dict(batch_size=256, train_fraction=1.0),
}

# Epochs to convergence (mean, std) over 10 runs, from the paper.
EPOCHS = {
    ("lrt", "low"): (6994, 1128),
    ("rad", "low"): (2533, 385),
    ("de", "low"): (1777, 237),
    ("lrt", "high"): (2641, 156),
    ("rad", "high"): (4254, 263),
    ("de", "high"): (4185, 149),
}

DE_MEMBERS = 5  # reported DE is a 5-member ensemble (pool of 10 trained, 5 used)
ORDER = [
    ("lrt", "low"),
    ("rad", "low"),
    ("de", "low"),
    ("lrt", "high"),
    ("rad", "high"),
    ("de", "high"),
]


def build_datamodule(data_dir, batch_size, train_fraction):
    from bnn_aenet.datamodule.aenet_datamodule import AenetDataModule

    return AenetDataModule(
        data_dir=str(data_dir),
        device="cpu",
        batch_size=batch_size,
        test_split=0.1,
        valid_split=0.1,
        train_fraction=train_fraction,
        name="TiO2_Forces",
    )


def build_net(dm):
    from bnn_aenet.models.nets.network import NetAtom

    return NetAtom(
        input_size=dm.input_size,
        hidden_size=dm.hidden_size,
        species=dm.species,
        active_names=dm.active_names,
        alpha=0.1,
        device="cpu",
        e_scaling=dm.e_scaling,
        e_shift=dm.e_shift,
    )


def build_model(model, dm, cfg):
    net = build_net(dm)
    if model == "de":
        from bnn_aenet.models.nn import NN_Forces

        return NN_Forces(net=net, optimizer=torch.optim.Adam, alpha=0.1)

    import pyro

    from bnn_aenet.models.bnn_forces import BNN_Forces

    pyro.clear_param_store()
    return BNN_Forces(
        net=net,
        lr=1e-4,
        pretrain_epochs=0,
        mc_samples_train=cfg["mc"],
        mc_samples_eval=20,
        dataset_size=dm.train_size,
        fit_context=cfg["fit_context"],
        prior_loc=0,
        prior_scale=0.3,
        guide=cfg["guide"],
        q_scale=2e-4,
        obs_scale=0.5,
        scale_force=0.1,
        learn_noise=True,
    )


class EpochTimer(Callback):
    def __init__(self, use_cuda):
        self.use_cuda = use_cuda
        self.starts = []

    def on_train_epoch_start(self, trainer, pl_module):
        if self.use_cuda:
            torch.cuda.synchronize()
        self.starts.append(time.perf_counter())

    def per_epoch(self, warmup):
        deltas = np.diff(np.array(self.starts))
        timed = deltas[warmup:]
        return float(timed.mean()), float(timed.std())


def time_config(model, regime, data_dir, device, epochs, warmup):
    cfg = CONFIGS[(model, regime)]
    dm = build_datamodule(data_dir, cfg["batch_size"], cfg["train_fraction"])
    mdl = build_model(model, dm, cfg)
    timer = EpochTimer(use_cuda=(device == "gpu"))
    trainer = Trainer(
        max_epochs=epochs,
        accelerator=device,
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        precision=32,
        callbacks=[timer],
    )
    trainer.fit(mdl, datamodule=dm)
    return timer.per_epoch(warmup)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    p.add_argument("--data-dir", default=str(root / "data" / "TiO" / "train_forces_local.in"))
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--output", default=str(Path(__file__).parent / "train_time_5090_byregime.csv"))
    args = p.parse_args()

    if args.device == "gpu" and not torch.cuda.is_available():
        raise SystemExit("--device gpu but CUDA unavailable.")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    print("=" * 72)
    print(f"Faithful training-time benchmark | device={gpu}")
    print(f"  per-regime Optuna batch/MC, {args.epochs} epochs ({args.warmup} warmup)")
    print("=" * 72)

    rows = []
    for model, regime in ORDER:
        cfg = CONFIGS[(model, regime)]
        mean_s, std_s = time_config(
            model, regime, args.data_dir, args.device, args.epochs, args.warmup
        )
        ep_mean, ep_std = EPOCHS[(model, regime)]
        scale = DE_MEMBERS if model == "de" else 1
        total_h = ep_mean * mean_s * scale / 3600.0
        total_h_std = ep_std * mean_s * scale / 3600.0
        print(
            f"{regime:>4} {model.upper():<4} batch={cfg['batch_size']:>4} "
            f"mc={cfg.get('mc','-'):<2} frac={cfg['train_fraction']}  "
            f"{mean_s:7.3f} s/epoch  ->  Total {total_h:5.2f} +/- {total_h_std:.2f} h"
        )
        rows.append(
            {
                "regime": regime,
                "model": model.upper(),
                "batch_size": cfg["batch_size"],
                "mc": cfg.get("mc", "-"),
                "train_fraction": cfg["train_fraction"],
                "per_epoch_s": round(mean_s, 4),
                "per_epoch_s_std": round(std_s, 4),
                "epochs": ep_mean,
                "epochs_std": ep_std,
                "total_h": round(total_h, 3),
                "total_h_std": round(total_h_std, 3),
            }
        )

    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
