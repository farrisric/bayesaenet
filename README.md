# bayesian-aenet

**bayesian-aenet** is a research-focused extension of the [aenet-PyTorch](https://pubs.aip.org/aip/jcp/article/158/16/164105/2885330/anet-PyTorch-A-GPU-supported-implementation-for) framework for benchmarking **uncertainty quantification (UQ)** methods in **machine learning interatomic potentials (MLIPs)**. It enables the systematic comparison of two widely used strategies:

- **Deep Ensembles (DE)** — multiple independently trained deterministic networks
- **Variational Bayesian Neural Networks (VBNNs)** — Pyro/TyXe ELBO-based inference

The code accompanies the paper *Bayesian Neural Networks versus deep ensembles for uncertainty quantification in machine learning interatomic potentials* (see [Citation](#citation)).

---

## Purpose

Traditional ML interatomic potentials provide point estimates but lack a principled assessment of uncertainty, limiting their reliability in out-of-distribution scenarios and their utility in active learning. This library addresses that by:

- Implementing **Bayesian neural networks** via **variational inference**, leveraging [Pyro](https://pyro.ai/) and [TyXe](https://github.com/TyXe-BDL/TyXe)
- Benchmarking against **deep ensembles**, a widely used method for epistemic UQ
- Evaluating **predictive accuracy**, **uncertainty calibration**, and **robustness** across data regimes

Models predict DFT energies and forces for atomic structures. The primary benchmark is a dataset of [7,815 DFT-computed TiO₂ structures](https://www.sciencedirect.com/science/article/abs/pii/S0927025615007806?via%3Dihub); QM7 is also supported.

## Installation

Requires **Python ≥ 3.9** and a PyTorch ≥ 2.0 installation.

```bash
# 1. Create an environment
conda create -n bayesaenet python=3.10
conda activate bayesaenet

# 2. Install bayesian-aenet (with dev + UQ extras)
pip install -e ".[dev,uq]"

# 3. Install TyXe separately — it is a git-only dependency, not on PyPI
pip install git+https://github.com/TyXe-BDL/TyXe.git
```

Extras: `[dev]` (pytest, black, isort, flake8, pre-commit), `[uq]` (uncertainty-toolbox), `[all]` (both).

## Datasets

Dataset definitions live under `bnn_aenet/configs/datamodule/` and the AENET-format inputs under `data/`. Selectable datasets include:

| Config | Dataset | Notes |
|--------|---------|-------|
| `TiO` | TiO₂ (full, 7,815 structures) | Primary benchmark |
| `TiO_Data20` / `TiO_Data100` | TiO₂ at 20% / 100% | Data-regime studies |
| `TiO_Forces*` | TiO₂ with force training | Joint energy+force likelihood |
| `QM7` / `QM7_Data10/20/100` | QM7 organic molecules | |
| `H2O`, `IrO`, `PdO` | Additional systems | |

The raw TiO₂ structures originate from the [Artrith & Urban dataset](https://www.sciencedirect.com/science/article/abs/pii/S0927025615007806?via%3Dihub); see that reference to obtain the full structure data if it is not already present under `data/`.

## Quick start

The entry points are Hydra applications. Config resolution is `configs/train.yaml` → `experiment=<name>` → CLI overrides.

```bash
# Train a variational BNN (local reparameterization guide) on TiO₂
python bnn_aenet/tasks/train.py experiment=bnn_lrt datamodule=TiO trainer=gpu

# Train a deterministic network (one deep-ensemble member)
python bnn_aenet/tasks/train.py experiment=nn datamodule=TiO trainer=gpu

# Hyperparameter search (Optuna)
python bnn_aenet/tasks/hpsearch.py experiment=bnn_lrt datamodule=TiO trainer=gpu

# Predict with Monte-Carlo uncertainty → Parquet output
python bnn_aenet/tasks/predict.py experiment=bnn_lrt datamodule=TiO
```

Key experiment presets: `bnn_lrt`, `bnn_rad` (variational, radial guide), `nn` (deterministic), plus `*_hetero` (heteroscedastic) and `*_partial` variants.

> **Note:** the LRT guide (`bnn_lrt`) is incompatible with mixed precision — always run it with `precision=32`.

## Reproducing the paper

- **Deep Ensemble** results use 5 independently trained `nn` members (train with 5 seeds, aggregate at prediction time).
- **VBNNs** use `bnn_lrt` / `bnn_rad` with `mc_samples_eval=20` at prediction.
- The energy/force trade-off `alpha` is fixed at `0.1` and is **not** optimized by Optuna.

Optuna study databases and logs follow the layout in [Logs and results](#logs-and-results), so each experiment is identifiable by dataset, task, model, and run id.

## Project structure

```
bnn_aenet/
├── models/        # NN (deterministic), BNN / BNN_Forces (variational), guides, likelihoods
├── datamodule/    # Lightning DataModule + AENET-format readers
├── tasks/         # train.py, hpsearch.py, predict.py entry points
├── utils/         # metrics (per-atom RMSE, NLL, calibration), paths
└── configs/       # Hydra YAML (model, experiment, datamodule, trainer)
tests/             # pytest suite
data/              # dataset inputs (AENET format)
```

## Testing

```bash
pytest tests/ -v
```

Some integration tests require the local TiO₂ dataset and are skipped automatically when it is absent.

## Logs and results

New runs follow a unified layout for Optuna results and logs:

- **Optuna result databases**: `bnn_aenet/results/<dataset>/<model>.db`
  - Examples: `bnn_aenet/results/tio2_small/lrt.db`, `bnn_aenet/results/qm7/nn.db`
- **Logs** (hyperparameter search, training, predictions): `bnn_aenet/logs/<dataset>/<task>/<model>/run_<N>/`
  - `task` is one of `hps`, `train`, `pred`; `N` is a zero-padded run index.

Analysis and plotting utilities read from these locations.

## Citation

If you use this software, please cite the accompanying paper:

```bibtex
@article{farris2025bnn,
  title   = {Bayesian Neural Networks versus deep ensembles for uncertainty
             quantification in machine learning interatomic potentials},
  author  = {Farris, Riccardo and Telari, Emanuele and Artrith, Nongnuch and
             Neyman, Konstantin M. and Bruix, Albert},
  journal = {arXiv preprint},
  year    = {2025},
  url     = {https://github.com/farrisric/bayesaenet}
}
```

Machine-readable metadata is provided in [`CITATION.cff`](CITATION.cff).

## License

Released under the [MIT License](LICENSE).
