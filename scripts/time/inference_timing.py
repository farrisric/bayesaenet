#!/usr/bin/env python
"""Inference-cost benchmark for force-trained TiO2 models (LRT, RAD, DE).

Measures the wall-clock time to run the real ``predict_step`` (energy + forces)
over the fixed TiO2 test set, at the paper's evaluation settings
(``mc_samples_eval=20`` for the BNNs; DE = one NN member x5). The number is a
companion to the training-time table (Table 9).

Notes
-----
* Inference cost is **regime-independent**: the test set, the network
  architecture, and ``mc_samples_eval`` are identical in the high- and low-data
  regimes, so there is a single row per model (not one per regime).
* **No trained checkpoints are needed**: forward/backward cost does not depend
  on the weight *values*, and the BNN variational guide is built lazily by
  ``on_predict_start``. Representative hyperparameters are therefore fine.
* Run this on the same GPU as the training benchmark (RTX 4090) for a
  comparable number.

Usage
-----
    python scripts/time/inference_timing.py --device gpu
    python scripts/time/inference_timing.py --device cpu --mc-samples 5 --repeats 2  # quick smoke test
"""

# Import torch BEFORE numpy/pandas (CUDA/MKL init order; see predict_forces.py).
import torch

import argparse
import csv
from pathlib import Path

import numpy as np
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

# bnn_aenet model imports are lazy (avoid importing pyro/tyxe at module load,
# which segfaults on some iqtc nodes when combined with torch+lightning).
from bnn_aenet.datamodule.aenet.batch_constants import BatchIdx  # noqa: E402

MODELS = ["lrt", "rad", "de"]
DE_MEMBERS = 5  # DE total = single NN member x5 (reported DE is a 5-member ensemble)


def build_datamodule(data_dir, batch_size):
    """Instantiate the TiO2 force datamodule on CPU (batches moved later)."""
    from bnn_aenet.datamodule.aenet_datamodule import AenetDataModule

    return AenetDataModule(
        data_dir=str(data_dir),
        device="cpu",
        batch_size=batch_size,
        test_split=0.1,
        valid_split=0.1,
        train_fraction=1.0,  # test set is identical across fractions
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


def build_model(model_type, dm, mc_eval):
    """Build a fresh (untrained) model. Hyperparameter values do not affect
    timing; only the architecture and ``mc_samples_eval`` matter."""
    net = build_net(dm)

    if model_type == "de":
        from bnn_aenet.models.nn import NN_Forces

        return NN_Forces(net=net, optimizer=torch.optim.Adam, alpha=0.1)

    import pyro
    from bnn_aenet.models.bnn_forces import BNN_Forces

    pyro.clear_param_store()  # avoid state bleed between BNN builds

    if model_type == "lrt":
        fit_context, guide = "lrt", "normal"
    elif model_type == "rad":
        fit_context, guide = None, "radial"
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return BNN_Forces(
        net=net,
        lr=1e-4,
        pretrain_epochs=0,
        mc_samples_train=1,
        mc_samples_eval=mc_eval,
        dataset_size=dm.train_size,
        fit_context=fit_context,
        prior_loc=0,
        prior_scale=0.3,
        guide=guide,
        q_scale=2e-4,
        obs_scale=0.2,
        scale_force=0.1,
    )


def move_batch(obj, device):
    """Recursively move a batch (list of tensors / lists / None) to device."""
    if obj is None:
        return None
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        return type(obj)(move_batch(x, device) for x in obj)
    return obj


def count_structures(batches):
    return int(sum(len(b[BatchIdx.E_ENERGY]) for b in batches))


def run_once(model, batches):
    """One full pass over the test set; returns nothing (timed by caller)."""
    for i, batch in enumerate(batches):
        model.predict_step(batch, i)


def time_model(model_type, dm, batches, device, mc_eval, n_warmup, n_repeats):
    model = build_model(model_type, dm, mc_eval)
    model.eval()
    model.to(device)
    # Builds the variational guide for BNNs; Lightning no-op for NN.
    model.on_predict_start()

    use_cuda = device.type == "cuda"

    for _ in range(n_warmup):
        run_once(model, batches)
    if use_cuda:
        torch.cuda.synchronize()

    times = []
    for _ in range(n_repeats):
        if use_cuda:
            torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True) if use_cuda else None
        if use_cuda:
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
            run_once(model, batches)
            t1.record()
            torch.cuda.synchronize()
            times.append(t0.elapsed_time(t1) / 1000.0)  # ms -> s
        else:
            import time as _time

            start = _time.perf_counter()
            run_once(model, batches)
            times.append(_time.perf_counter() - start)

    return float(np.mean(times)), float(np.std(times))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    p.add_argument(
        "--data-dir",
        default=str(root / "data" / "TiO" / "train_forces.in"),
        help="Path to TiO2 train_forces.in",
    )
    p.add_argument("--batch-size", type=int, default=128,
                   help="Eval batch size (same for all models, for fairness).")
    p.add_argument("--mc-samples", type=int, default=20,
                   help="mc_samples_eval for the BNNs (paper default 20).")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument(
        "--output",
        default=str(Path(__file__).parent / "inference_timing_results.csv"),
    )
    args = p.parse_args()

    device = torch.device("cuda" if args.device == "gpu" else "cpu")
    if args.device == "gpu" and not torch.cuda.is_available():
        raise SystemExit("--device gpu requested but CUDA is not available.")

    print("=" * 60)
    print("Inference-cost benchmark (energy + forces, TiO2 test set)")
    print(f"  device={device}  batch_size={args.batch_size}  "
          f"mc_samples_eval={args.mc_samples}")
    print(f"  warmup={args.warmup}  repeats={args.repeats}")
    print("=" * 60)

    dm = build_datamodule(args.data_dir, args.batch_size)
    cpu_batches = list(dm.test_dataloader())
    batches = [move_batch(b, device) for b in cpu_batches]
    n_struct = count_structures(batches)
    print(f"Test set: {n_struct} structures in {len(batches)} batches\n")

    rows = []
    for mt in MODELS:
        mean_s, std_s = time_model(
            mt, dm, batches, device, args.mc_samples, args.warmup, args.repeats
        )
        scale = DE_MEMBERS if mt == "de" else 1
        total_mean, total_std = mean_s * scale, std_s * scale
        per_struct_ms = total_mean / n_struct * 1000.0
        label = "DE (x5 members)" if mt == "de" else mt.upper()
        print(f"{label:<18} total {total_mean:8.3f} +/- {total_std:.3f} s "
              f"| {per_struct_ms:7.2f} ms/structure")
        rows.append({
            "model": mt.upper(),
            "n_structures": n_struct,
            "mc_samples": args.mc_samples if mt != "de" else 1,
            "batch_size": args.batch_size,
            "total_time_s_mean": round(total_mean, 4),
            "total_time_s_std": round(total_std, 4),
            "per_structure_ms": round(per_struct_ms, 4),
        })

    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nResults written to {args.output}")

    print("\nLaTeX 'Inference Time (s)' column for Table 9:")
    for r in rows:
        print(f"  {r['model']:<5} & {r['total_time_s_mean']:.2f} "
              f"$\\pm$ {r['total_time_s_std']:.2f} \\\\")


if __name__ == "__main__":
    main()
