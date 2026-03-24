# bayesian-aenet

**bayesian-aenet** is a research-focused extension of the [aenet-PyTorch](https://pubs.aip.org/aip/jcp/article/158/16/164105/2885330/anet-PyTorch-A-GPU-supported-implementation-for) framework, designed for benchmarking **uncertainty quantification (UQ)** methods in **machine learning interatomic potentials (MLIPs)**. It enables the systematic comparison of two widely used strategies:

- **Deep Ensembles (DE)**
- **Variational Bayesian Neural Networks (VBNNs)**

---

## 🎯 Purpose

Traditional ML interatomic potentials provide point estimates but lack a principled assessment of uncertainty, limiting their reliability in out-of-distribution scenarios and their utility in active learning pipelines. This library addresses these challenges by:

- Implementing **Bayesian neural networks** using **variational inference**, leveraging [Pyro](https://pyro.ai/) and [TyXe](https://github.com/TyXe-BDL/TyXe.git)
- Benchmarking against **deep ensembles**, a widely used method for epistemic UQ
- Evaluating **predictive accuracy**, **uncertainty calibration**, and **robustness** across data regimes

The models are tested on a dataset of [7,815 DFT-computed TiO₂ structures](https://www.sciencedirect.com/science/article/abs/pii/S0927025615007806?via%3Dihub), originally developed for validating atom-centered neural network potentials.

For a full methodological description, implementation details, and performance results, refer to the accompanying preprint:

```bibtex
@article{farris2025uncertainty,
  title={To be Published: Uncertainty Quantification for Machine Learning Interatomic Potentials},
  author={Farris, Riccardo and Telari, Emanuele and Bruix, Albert and Artrith, Nongnuch},
  journal={arXiv preprint},
  year={2025},
  url={https://github.com/farrisric/bayesaenet}
}
```

## Logs and results

New runs follow a unified layout for Optuna results and logs:

- **Optuna result databases**: `bnn_aenet/results/<dataset>/<model>.db`  
  - Examples: `bnn_aenet/results/tio2_small/lrt.db`, `bnn_aenet/results/tio2_big/rad.db`, `bnn_aenet/results/qm7/nn.db`
- **Logs (hyperparameter search, training, predictions)**:  
  - Root: `bnn_aenet/logs`  
  - Per run: `bnn_aenet/logs/<dataset>/<task>/<model>/run_<N>/...`  
    - `dataset`: e.g. `tio2_small`, `tio2_big`, `qm7`  
    - `task`: one of `hps`, `train`, `pred`  
    - `model`: e.g. `lrt`, `rad`, `nn`  
    - `N`: zero-padded run index, e.g. `001`

Analysis scripts and plotting utilities read from these locations, so you can identify experiments by dataset, task, model, and run id in a consistent way.

