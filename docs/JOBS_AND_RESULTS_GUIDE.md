# Jobs and Results Guide

Where to find results, how to monitor jobs, and how the directory structure works.

## Directory Structure: log / db / task / runs

```
bayesaenet/
├── log/                      # SGE job stdout/stderr (qsub output)
│   ├── multirun/            # Training job logs
│   │   └── TiO2_small_{nn,rad,lrt}.{out,err}
│   ├── hps/                 # Hyperparameter search job logs
│   │   └── TiO2_small_hps_{model}.{out,err}
│   └── predict/             # Prediction job logs
│
├── db/                      # Optuna HPS databases
│   ├── TiO2_small/
│   │   ├── nn_small.db
│   │   ├── bnn_lrt_forces_likelihood.db
│   │   └── bnn_rad_forces_likelihood.db
│   └── TiO2_big/
│       └── ...
│
└── task/                    # Train/HPS/Predict outputs (checkpoints, tensorboard)
    ├── train/
    │   └── runs/
    │       ├── nn/
    │       │   ├── nn_train_0/
    │       │   │   ├── checkpoints/
    │       │   │   ├── .hydra/
    │       │   │   └── tensorboard/
    │       │   └── ...
    │       ├── lrt/
    │       └── rad/
    ├── hpsearch/
    │   └── runs/
    │       └── {model}/
    └── predict/
        └── runs/
            └── TiO2_small/
                ├── nn/
                ├── lrt/
                └── rad/
```

## Migration from legacy layout

If you have existing DBs in `bnn_aenet/results/`:

```bash
mkdir -p db
cp -r bnn_aenet/results/TiO2_small db/ 2>/dev/null || true
cp -r bnn_aenet/results/TiO2_big db/ 2>/dev/null || true
```

Or use `--results-dir bnn_aenet/results` when running `generate_train_scripts`.

## Where Your Results Are

| What | Location |
|------|----------|
| **Training checkpoints** | `task/train/runs/{model}/{run_name}/checkpoints/` |
| **SGE stdout/stderr** | `log/multirun/`, `log/hps/`, `log/predict/` |
| **Optuna HPS databases** | `db/TiO2_small/*.db`, `db/TiO2_big/*.db` |
| **Predictions** | `task/predict/runs/TiO2_small/{model}/` |

## Monitoring Jobs

```bash
# List your jobs
qstat -u $USER

# Job details (when running)
qstat -j <job_id>

# After job finishes: check SGE logs
tail -100 log/multirun/TiO2_small_lrt.err
tail -100 log/multirun/TiO2_small_lrt.out
```

## Common Issues

### 1. "TiO2_small is empty"

- New structure: outputs go to `task/train/runs/{model}/`, predictions to `task/predict/runs/TiO2_small/{model}/`.
- For prediction, `--runs-dir` should be `task/train/runs/{model}` (e.g. `task/train/runs/lrt`).
- Legacy: old runs may be in `bnn_aenet/logs/{model_name}/`. Use `--results-dir bnn_aenet/results` for old DBs.

### 2. multirun script uses old experiment name

- If `multirun_lrt.sh` still has `experiment=bnn_lrt_forces_likelihood`, update to `experiment=bnn_lrt`.
- Regenerate scripts: `python -m bnn_aenet.tasks.generate_train_scripts --dataset TiO2_small --output-dir scripts/final/TiO2_small/train`

### 3. PROJECT_ROOT not set

- Paths use `${oc.env:PROJECT_ROOT}`. If unset, outputs can go to wrong places.
- The train/hpsearch tasks set it via `pyrootutils`. For custom scripts, run from project root and ensure `PROJECT_ROOT` is set or let pyrootutils detect it.

### 4. Predict script paths

- **New structure**: `task/train/runs/nn`, `task/train/runs/lrt`, `task/train/runs/rad`
- **Legacy**: `bnn_aenet/logs/nn_forces/`, `bnn_aenet/logs/lrt_forces_likelihood/`, etc.
- Regenerate scripts: `python -m bnn_aenet.tasks.generate_train_scripts --dataset TiO2_small --output-dir scripts/final/TiO2_small/train`
- For old DBs: `--results-dir bnn_aenet/results`
