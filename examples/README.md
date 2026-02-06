# BNN-AENET Examples

This directory contains example scripts demonstrating how to use the BNN-AENET library for training Bayesian Neural Networks for atomic energy prediction.

## Quick Start Examples

### 1. Training a Single Model

Train a Local Reparameterization Trick (LRT) BNN on QM7:

```bash
# From the project root
python bnn_aenet/tasks/train.py experiment=final/lrt_qm7 seed=42
```

Train a Flipout BNN on TiO2:

```bash
python bnn_aenet/tasks/train.py experiment=final/fo_tio2 seed=42
```

### 2. Training a Deep Ensemble

Train 5 ensemble members with different seeds:

```bash
for seed in 42 123 456 789 1024; do
    python bnn_aenet/tasks/train.py experiment=final/de_qm7 seed=$seed run_name=de_qm7_seed_$seed
done
```

### 3. Making Predictions

```bash
python bnn_aenet/tasks/predict.py checkpoint=path/to/checkpoint.ckpt
```

### 4. Hyperparameter Search

```bash
python bnn_aenet/tasks/hpsearch.py hpsearch=bnn_lrt datamodule.data_dir=data/QM7/train.in
```

## Directory Structure

```
examples/
├── notebooks/           # Jupyter notebooks for exploration
│   ├── hparam_opt.ipynb    # Hyperparameter optimization demo
│   └── loadbnn.ipynb       # Loading and using trained models
├── plot/                # Visualization examples
│   └── *.py
└── train/               # Training scripts
    └── *.py
```

## Configuration

All experiments use Hydra for configuration. Key config files:

- `bnn_aenet/configs/experiment/final/` - Final training configurations
- `bnn_aenet/configs/model/` - Model architectures (bnn_lrt, bnn_fo, bnn_rad, nn)
- `bnn_aenet/configs/datamodule/` - Dataset configurations

## GPU Training

To train on GPU, add these flags:

```bash
python bnn_aenet/tasks/train.py \
    experiment=final/lrt_qm7 \
    trainer.accelerator=gpu \
    trainer.devices=1
```

For cluster submission (SGE):

```bash
qsub scripts/final/submit_all_final.sh
```

## Analysis

After training, analyze results:

```bash
python scripts/final/analyze_results.py --dataset qm7
```

This creates:
- Summary statistics table (`summary.csv`, `summary.tex`)
- Parity plots for all methods
- Uncertainty calibration curves
- Method comparison bar charts
