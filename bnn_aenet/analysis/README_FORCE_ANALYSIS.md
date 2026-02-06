# Force Prediction Analysis

## Overview
Tools for analyzing BNN predictions that include force data, computing comprehensive metrics for both energies and forces.

## Quick Start

### 1. Train Model with Forces
```bash
python bnn_aenet/tasks/train.py \
    experiment=bnn_lrt_forces_aux \
    trainer.accelerator=gpu \
    datamodule=QM7 \
    model.force_weight=1.0
```

### 2. Run Predictions
```bash
python bnn_aenet/tasks/predict.py \
    method=bnn_forces_aux \
    runs_dir=bnn_aenet/logs/lrt_forces_train/runs/run_0 \
    ckpt_path=all \
    datamodule=QM7
```

### 3. Analyze Results

#### Single File
```bash
python bnn_aenet/analysis/analyze_force_predictions.py \
    bnn_aenet/logs/lrt_forces_train/runs/pred_0/bnn_forces_aux_000_val.parquet \
    --output metrics.json
```

#### Multiple Runs (Ensemble)
```bash
python bnn_aenet/analysis/analyze_force_predictions.py \
    bnn_aenet/logs/lrt_forces_train/runs/ \
    --multiple \
    --output ensemble_summary.csv
```

## Output Metrics

### Energy Metrics
- `mae`: Mean absolute error
- `rmse`: Root mean squared error
- `maxerr`: Maximum error
- `r2score`: R² coefficient
- `sharp`: Average uncertainty (std)
- `overlap`: Uncertainty-error alignment (%)
- `nll`: Negative log-likelihood

### Force Metrics
- `force_mae`: Mean absolute error (all components)
- `force_rmse`: Root mean squared error
- `force_maxerr`: Maximum component error
- `force_r2`: R² for force prediction

### Force Vector Metrics
- `force_mag_mae`: MAE of force magnitudes
- `force_mag_rmse`: RMSE of force magnitudes
- `force_angular_mae`: Mean angular error (degrees)

### Force Uncertainty Metrics
- `force_sharp`: Average force uncertainty
- `force_overlap`: UQ-error alignment
- `force_nll`: Force negative log-likelihood

## Python API

```python
from bnn_aenet.analysis.analyze_force_predictions import (
    analyze_prediction_file,
    analyze_multiple_runs,
    compute_metrics_from_predictions
)

# Single file
metrics = analyze_prediction_file('predictions.parquet', verbose=True)

# Multiple files
df_summary = analyze_multiple_runs(
    'logs/runs/',
    pattern='*_val.parquet',
    output_file='summary.csv'
)

# From DataFrame
import pandas as pd
df = pd.read_parquet('predictions.parquet')
metrics = compute_metrics_from_predictions(df, has_forces=True)
```

## Prediction Output Format

Parquet files from `BNN_Forces_Aux.predict_step()` contain:

| Column | Type | Description |
|--------|------|-------------|
| `true` | float | True energy (normalized) |
| `preds` | float | Predicted energy mean |
| `stds` | float | Energy uncertainty (std) |
| `n_atoms` | int | Number of atoms in structure |
| `true_forces` | array | True force components (flattened N×3) |
| `pred_forces` | array | Predicted force components (mean over MC) |
| `std_forces` | array | Force uncertainties (std over MC) |
| `force_rmse` | float | Per-structure force RMSE |
| `force_mae` | float | Per-structure force MAE |

## Example: Compare Energy-Only vs Force-Trained

```python
from pathlib import Path
from bnn_aenet.analysis.analyze_force_predictions import analyze_prediction_file

# Analyze both models
metrics_energy_only = analyze_prediction_file(
    Path('logs/bnn_lrt/pred_0/bnn_lrt_000_val.parquet')
)

metrics_with_forces = analyze_prediction_file(
    Path('logs/bnn_lrt_forces/pred_0/bnn_forces_aux_000_val.parquet')
)

# Compare
print(f"Energy RMSE - Energy-only: {metrics_energy_only['rmse']:.4f}")
print(f"Energy RMSE - With forces: {metrics_with_forces['rmse']:.4f}")
print(f"Force RMSE: {metrics_with_forces['force_rmse']:.4f}")
```

## Notes

- Force data must be present in training set (controlled by `train.in`)
- If no force data available, force metrics return NaN/None
- Force components stored as flattened arrays (atom1_x, atom1_y, atom1_z, atom2_x, ...)
- All force units in mHa/Bohr (milli-Hartree per Bohr)
- Angular errors use cosine similarity (0° = perfect alignment)

## Integration with Existing Analysis

The force analysis tools integrate with the existing `bnn_aenet/analysis/` framework:
- Uses same metrics API
- Compatible with plotting workflow
- Works with existing `ResultSaver` and data loading utilities
