# Optuna HPS Results

## Directory structure

Optuna study databases are stored per dataset:

```
bnn_aenet/results/
├── TiO2_big/          # 100% data (Data100)
│   ├── nn.db
│   ├── lrt.db
│   ├── fo.db
│   ├── rad.db
│   ├── bnn_lrt_forces_likelihood.db
│   └── bnn_rad_forces_likelihood.db
├── TiO2_small/        # 20% data (Data20)
│   ├── nn_small.db
│   ├── lrt_small.db
│   ├── fo_small.db
│   ├── rad_small.db
│   ├── bnn_lrt_forces_likelihood.db
│   └── bnn_rad_forces_likelihood.db
├── QM7/               # QM7 dataset (when applicable)
│   └── ...
└── bayesian/          # DEPRECATED - legacy location
    └── README_DEPRECATED.md
```

## Configuring the DB path

Submission scripts must set `hpsearch.results_subdir` to the dataset name:

- `TiO2_big` for 100% TiO2 data
- `TiO2_small` for 20% TiO2 data
- `QM7` for QM7

Example:
```bash
hpsearch.results_subdir=TiO2_big
```

If not set, `tags[0]` is used (tags should start with the dataset name).

## Legacy (bayesian/)

The `bayesian/` subdirectory was used by older likelihood HPS runs with
`tags=["bayesian", ...]`. Those runs used `datamodule=TiO_Forces` with
`split_config=null` (default splits from data/TiO/splits/), which is
**not** the same as TiO2 Data20 or Data100.

For proper TiO2_big and TiO2_small HPS, use the scripts in:
- `scripts/final/TiO2_big/hps/submit_hps_likelihood.sh`
- `scripts/final/TiO2_small/hps/submit_hps_likelihood.sh`
