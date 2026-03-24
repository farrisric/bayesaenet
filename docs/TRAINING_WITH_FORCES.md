# Training with Forces in BNN-AENET

This document describes the implementation of force training and hyperparameter optimization in the BNN-AENET library.

## Table of Contents

1. [Overview](#overview)
2. [Force Training Implementation](#force-training-implementation)
3. [Models](#models)
4. [Loss Functions](#loss-functions)
5. [Hyperparameter Optimization](#hyperparameter-optimization)
6. [Configuration System](#configuration-system)
7. [Running Experiments](#running-experiments)

---

## Overview

BNN-AENET extends atomic energy neural network potentials with:
- **Bayesian Neural Networks (BNNs)** for uncertainty quantification
- **Force training** for improved accuracy and physical consistency
- **Hyperparameter optimization** using Optuna

The library supports four model types:
| Model | Type | Force Support | Uncertainty |
|-------|------|---------------|-------------|
| NN_Forces | Deterministic | Yes | No (ensemble for UQ) |
| BNN_LRT | Bayesian (Local Reparameterization) | Yes | Yes |
| BNN_Flipout | Bayesian (Flipout) | Yes | Yes |
| BNN_Radial | Bayesian (Radial Guide) | Yes | Yes |

---

## Force Training Implementation

### Computing Forces from Energy

Forces are computed as the negative gradient of atomic energy with respect to atomic positions:

```
F = -∂E/∂r
```

In practice, this is implemented using automatic differentiation through the symmetry function descriptors:

```python
# In NetAtom.forward_F()
def forward_F(self, group_descrp, sfderiv_i, sfderiv_j, ...):
    # Forward pass for energy
    E_atom = self.forward(group_descrp, logic_reduce, input_size)
    
    # Compute gradient of energy w.r.t. descriptors
    dE_dG = torch.autograd.grad(E_atom, group_descrp, ...)
    
    # Chain rule: F = -dE/dr = -dE/dG * dG/dr
    # where dG/dr are the symmetry function derivatives (sfderiv)
    F_pred = compute_forces_from_derivatives(dE_dG, sfderiv_i, sfderiv_j, ...)
    
    return E_atom, F_pred
```

### Data Loading for Forces

Force data is loaded from AENET format files specified in `train.in`:

```
FORCES
TRAININGSET data.train.forces
alpha 0.1
```

The `AenetDataModule` handles loading both energy and force data, creating separate batches that are synchronized during training.

---

## Models

### NN_Forces (Deep Ensemble with Forces)

A deterministic neural network trained with both energy and force losses.

```python
class NN_Forces(NN):
    def __init__(self, net, optimizer, force_weight=1.0, alpha=0.1):
        self.force_weight = force_weight  # Additional scaling for force loss
        self.alpha = alpha  # Energy/force balance (FIXED, not optimized)
    
    def training_step(self, batch):
        energy_rmse = self.compute_energy_loss(batch)
        force_rmse = self.compute_force_loss(batch)
        
        # Training loss includes force_weight
        total_loss = (1 - alpha) * energy_rmse + alpha * force_weight * force_rmse
        return total_loss
    
    def validation_step(self, batch):
        energy_rmse = self.compute_energy_loss(batch)
        force_rmse = self.compute_force_loss(batch)
        
        # Validation metric does NOT include force_weight (fair comparison)
        total_rmse = (1 - alpha) * energy_rmse + alpha * force_rmse
        self.log("total_rmse/val", total_rmse)
```

### BNN_Forces_Aux (Bayesian NN with Auxiliary Force Loss)

A Bayesian neural network using TyXe/Pyro for variational inference, with forces added as an auxiliary loss term.

**Why Auxiliary Loss?**

TyXe's standard ELBO formulation doesn't directly support multi-output losses (energy + forces). Our solution:
1. Energy training uses standard TyXe ELBO (variational inference)
2. Force loss is computed separately and gradients are applied to the variational parameters

```python
class BNN_Forces_Aux(BNN):
    def __init__(self, ..., force_weight=1.0, force_lr_scale=0.1, scale_lr_factor=0.5):
        self.force_weight = force_weight      # Force loss multiplier
        self.force_lr_scale = force_lr_scale  # LR scale for force updates
        self.scale_lr_factor = scale_lr_factor  # LR factor for uncertainty params
    
    def training_step(self, batch):
        # 1. Standard energy ELBO update (TyXe handles this)
        energy_loss = self.svi.step(x, y)
        
        # 2. Auxiliary force loss with manual gradient application
        force_rmse = self.compute_force_loss(batch)
        weighted_force_loss = alpha * force_weight * force_rmse
        
        # Apply force gradients to variational parameters
        self.apply_force_gradients(weighted_force_loss)
        
        return energy_loss
```

**Manual Gradient Application for Forces:**

Since TyXe manages the Pyro parameter store, we manually apply force gradients:

```python
def compute_force_loss_and_update(self, batch):
    force_rmse = self.compute_force_loss(batch)
    weighted_loss = alpha * force_rmse
    weighted_loss.backward()
    
    # Apply gradients to Pyro guide parameters
    for name, param in pyro.get_param_store().items():
        if param.grad is not None:
            lr = self.hparams.lr
            # Scale learning rate for different parameter types
            if 'scale' in name:
                lr *= self.scale_lr_factor
            lr *= self.force_lr_scale
            
            param.data -= lr * param.grad
            param.grad.zero_()
```

---

## Loss Functions

### Training Loss

```
L_train = (1 - α) × E_RMSE + α × force_weight × F_RMSE
```

Where:
- `α` = 0.1 (fixed) - balance between energy and forces
- `force_weight` = tunable hyperparameter (0.1 - 10.0)
- `E_RMSE` = Root Mean Square Error of atomic energies
- `F_RMSE` = Root Mean Square Error of force components

### Validation/Monitoring Metric

```
total_rmse = (1 - α) × E_RMSE + α × F_RMSE
```

**Important:** The validation metric does NOT include `force_weight`. This ensures:
1. Fair comparison across different `force_weight` values
2. No bias in hyperparameter optimization (Optuna)

### Why α is Fixed at 0.1

If α were optimized by Optuna, lower values would always appear better because:
- Energy RMSE is typically smaller in magnitude than Force RMSE
- Lower α = more weight on energy = lower total_rmse
- This creates bias toward α → 0 (ignoring forces)

By fixing α = 0.1, we ensure:
- Forces always contribute 10% to the metric
- Optuna finds hyperparameters that genuinely improve both energy and force predictions

---

## Hyperparameter Optimization

### Framework: Optuna with Hydra

The library uses Optuna for hyperparameter search, integrated with Hydra configuration management.

```
bnn_aenet/
├── configs/
│   ├── hpsearch/
│   │   ├── nn_forces.yaml      # NN HPS config
│   │   ├── bnn_lrt_forces.yaml # LRT BNN HPS config
│   │   ├── bnn_fo_forces.yaml  # Flipout BNN HPS config
│   │   └── bnn_rad_forces.yaml # Radial BNN HPS config
│   └── experiment/
│       ├── nn_forces.yaml
│       └── bnn_*_forces_aux.yaml
└── tasks/
    └── hpsearch.py             # Objective functions
```

### Hyperparameters by Model Type

#### NN_Forces (Deep Ensemble)

| Hyperparameter | Range | Description |
|----------------|-------|-------------|
| `lr` | 1e-5 - 1e-2 | Learning rate |
| `weight_decay` | 1e-6 - 1e-2 | L2 regularization |
| `force_weight` | 0.1 - 10.0 | Force loss multiplier |
| `batch_size` | [64, 128, 256, 512] | Training batch size |

**Fixed:** `alpha = 0.1`

#### BNN_Forces_Aux (Bayesian Models)

| Hyperparameter | Range | Description |
|----------------|-------|-------------|
| `pretrain_epochs` | [0, 5] | Epochs of deterministic pretraining |
| `lr` | 1e-5 - 1e-3 | Learning rate |
| `mc_samples_train` | [1, 2] | Monte Carlo samples during training |
| `prior_scale` | 0.05 - 1.0 | Prior distribution scale |
| `q_scale` | 1e-4 - 0.05 | Initial variational scale |
| `obs_scale` | 0.1 - 2.0 | Observation noise scale |
| `force_weight` | 0.1 - 10.0 | Force loss multiplier |
| `force_lr_scale` | 0.01 - 1.0 | LR scale for force updates |
| `scale_lr_factor` | 0.1 - 2.0 | LR factor for uncertainty params |

**Fixed:** `alpha = 0.1` (in network config)

### Objective Functions

```python
# bnn_aenet/tasks/hpsearch.py

def objective_nn_forces(trial, cfg, output_dir):
    """NN with forces - no pretraining needed."""
    cfg.model.optimizer.lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    cfg.model.optimizer.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    cfg.model.force_weight = trial.suggest_float("force_weight", 0.1, 10.0, log=True)
    # alpha is FIXED at 0.1 (from config)
    return objective(trial, cfg, output_dir)

def objective_bnn_forces(trial, cfg, output_dir):
    """BNN with forces - includes pretraining option."""
    cfg.model.pretrain_epochs = trial.suggest_categorical("pretrain_epochs", [0, 5])
    cfg.model.lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    cfg.model.prior_scale = trial.suggest_float("prior_scale", 0.05, 1.0, log=True)
    cfg.model.q_scale = trial.suggest_float("q_scale", 1e-4, 0.05, log=True)
    cfg.model.obs_scale = trial.suggest_float("obs_scale", 0.1, 2.0, log=True)
    cfg.model.force_weight = trial.suggest_float("force_weight", 0.1, 10.0, log=True)
    cfg.model.force_lr_scale = trial.suggest_float("force_lr_scale", 0.01, 1.0, log=True)
    cfg.model.scale_lr_factor = trial.suggest_float("scale_lr_factor", 0.1, 2.0, log=True)
    # alpha is FIXED at 0.1 (in net config)
    return objective(trial, cfg, output_dir)
```

### Monitoring Metric

All HPS configurations use:
```yaml
hpsearch:
  monitor: total_rmse/val  # Combined energy + force RMSE

callbacks:
  early_stopping:
    monitor: total_rmse/val
    patience: 50
```

---

## Configuration System

### Directory Structure for Logs

Logs are organized by model and task:
```
bnn_aenet/logs/
├── nn_forces/
│   ├── hpsearch/
│   │   └── 2026-02-05_15-45-56/
│   │       ├── .hydra/
│   │       ├── 000/  # Trial 0
│   │       ├── 001/  # Trial 1
│   │       └── ...
│   └── train/
│       └── 2026-02-05_16-08-20/
├── lrt_forces/
├── fo_forces/
└── rad_forces/
```

### Optuna Database

Results are stored per dataset in SQLite databases:
```
bnn_aenet/results/
├── TiO2_big/          # 100% data
│   ├── nn.db
│   ├── lrt.db
│   ├── fo.db
│   └── rad.db
├── TiO2_small/        # 20% data
│   ├── nn_small.db
│   └── ...
└── QM7/
    └── ...
```
See `bnn_aenet/results/README.md` for details.

---

## Running Experiments

### Hyperparameter Search

```bash
# NN with forces
python bnn_aenet/tasks/hpsearch.py \
    hpsearch=nn_forces \
    datamodule=TiO \
    datamodule.data_dir=data/TiO/train_forces.in \
    trainer.accelerator=gpu

# BNN (LRT) with forces
python bnn_aenet/tasks/hpsearch.py \
    hpsearch=bnn_lrt_forces \
    datamodule=TiO \
    datamodule.data_dir=data/TiO/train_forces.in \
    trainer.accelerator=gpu
```

### Training with Best Hyperparameters

```bash
python bnn_aenet/tasks/train.py \
    experiment=nn_forces \
    datamodule=TiO \
    datamodule.data_dir=data/TiO/train_forces.in \
    model.alpha=0.1 \
    model.force_weight=<best_value> \
    trainer.max_epochs=50000 \
    trainer.accelerator=gpu
```

### Monitoring Training

```bash
# TensorBoard
tensorboard --logdir bnn_aenet/logs/

# Check Optuna study
python -c "
import optuna
study = optuna.load_study(
    study_name='nn_forces',
    storage='sqlite:///bnn_aenet/results/nn/nn_forces.db'
)
print(study.best_trial)
"
```

---

## Key Design Decisions

### 1. Alpha Fixed at 0.1

**Problem:** If alpha is optimized, Optuna prefers lower values (biases toward energy-only).

**Solution:** Fix alpha = 0.1 in configuration, not in HPS search space.

### 2. force_weight in Training Only

**Problem:** Need a way to tune force emphasis without biasing validation metric.

**Solution:** 
- Training: `loss = (1-α)*E + α*force_weight*F`
- Validation: `metric = (1-α)*E + α*F` (no force_weight)

### 3. Auxiliary Loss for BNN Forces

**Problem:** TyXe/Pyro ELBO doesn't support multi-output losses.

**Solution:** 
- Energy uses standard ELBO (TyXe handles optimization)
- Forces use auxiliary loss with manual gradient application to Pyro parameters

### 4. Pretraining for BNNs

**Problem:** BNN training can be unstable with random initialization.

**Solution:** Optional pretraining (0 or 5 epochs) of deterministic NN, then load weights into BNN.

---

## References

- TyXe: https://github.com/TyXe-BDL/TyXe
- Pyro: https://pyro.ai/
- Optuna: https://optuna.org/
- AENET: https://github.com/atomisticnet/aenet
