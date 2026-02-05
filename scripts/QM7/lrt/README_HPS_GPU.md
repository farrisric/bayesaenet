# LRT Hyperparameter Search on GPU

## Purpose

Search for optimal LRT hyperparameters for GPU training to avoid NaN issues in variational parameters.

## Problem

Current hyperparameters cause NaN in Pyro's variational distribution initialization:
```
ValueError: Expected parameter scale (Tensor of shape (1,)) of distribution Normal...
but found invalid values: tensor([nan], device='cuda:0')
```

## Solution

Use Optuna to search hyperparameter space for:
- `model.lr`: Learning rate
- `model.prior_scale`: Prior distribution scale
- `model.q_scale`: Variational posterior scale  
- `model.obs_scale`: Observation noise scale
- `model.mc_samples_train`: Monte Carlo samples during training

## Usage

```bash
cd /home/g15farris/bin/bayesaenet/scripts/QM7/lrt
qsub lrt_hps_gpu.sh
```

## Monitor

```bash
# Watch output
tail -f lrt_hps_gpu.out

# Check errors
tail -f lrt_hps_gpu.err

# Check GPU
ssh merry04 nvidia-smi
```

## Configuration

- **Trials**: Defined in `bnn_aenet/configs/hparams_search/bnn_lrt_optuna.yaml`
- **Epochs**: 1000-5000 (short for fast search)
- **Batch size**: 32
- **GPU**: Single RTX 4090
- **Objective**: Minimize validation MSE

## Output

Results saved to:
```
/home/g15farris/bin/bayesaenet/bnn_aenet/logs/lrt_hps_gpu/multiruns/YYYY-MM-DD_HH-MM-SS/
```

Optuna study database:
```
/home/g15farris/bin/bayesaenet/bnn_aenet/logs/lrt_hps_gpu/optuna_study.db
```

## Expected Runtime

- Per trial: ~5-10 minutes (depends on early stopping)
- Total: ~2-4 hours for 20-50 trials

## After Completion

1. Check best hyperparameters in logs
2. Update `lrt_train_gpu_multirun.sh` with best values
3. Re-run full training with optimized hyperparameters
