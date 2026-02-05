# QM7 Prediction and Plotting Scripts

This directory contains experimental scripts for predicting and plotting results from QM7 trained models (both LRT and NN).

## Directory Structure

```
cursor/
├── lrt/
│   ├── pred/
│   │   └── create_script.py  # Generate LRT prediction job scripts
│   └── plot/
│       └── plot.py            # Plot LRT results with uncertainty
├── nn/
│   ├── pred/
│   │   └── create_script.py  # Generate NN prediction job scripts
│   └── plot/
│       └── plot.py            # Plot NN results (no uncertainty)
└── README.md
```

## Parameters Used

From QM7 training logs:
- **e_scaling**: 0.9754923797786934
- **e_shift**: -4.652443333333333
- **Data splits**: 80% train, 10% validation, 10% test

## Usage

### Step 1: Generate Prediction Scripts

**For LRT models:**
```bash
cd /home/g15farris/bin/bayesaenet/scripts/QM7/cursor/lrt/pred
python create_script.py
```

This will create prediction job scripts for all LRT training runs found in:
- `bnn_aenet/logs/lrt_train/runs/lrt_train_0` through `lrt_train_9`

**For NN models:**
```bash
cd /home/g15farris/bin/bayesaenet/scripts/QM7/cursor/nn/pred
python create_script.py
```

This will create prediction job scripts for NN training runs in:
- `bnn_aenet/logs/de_train/runs/de_0`

### Step 2: Submit Prediction Jobs (Optional)

To automatically submit jobs, edit the `create_script.py` files and uncomment the line:
```python
# os.system(f'qsub {predict_filename}')
```

Or manually submit the generated `.sh` scripts:
```bash
qsub pred_lrt_train_0.sh
qsub pred_lrt_train_1.sh
# ... etc
```

### Step 3: Wait for Predictions to Complete

Predictions will be saved as `.parquet` files in:
- `bnn_aenet/logs/lrt_pred/runs/pred_lrt_train_*/LRT_0_val.parquet`
- `bnn_aenet/logs/de_pred/runs/pred_de_0/NN_0_val.parquet`

### Step 4: Plot Results

**For LRT models:**
```bash
cd /home/g15farris/bin/bayesaenet/scripts/QM7/cursor/lrt/plot
python plot.py
```

This will generate:
- Residuals vs Uncertainty plots (with color coding by number of atoms)
- Uncertainty quantification plots using uncertainty-toolbox
- Plots for Train, Validation, and Test sets

**For NN models:**
```bash
cd /home/g15farris/bin/bayesaenet/scripts/QM7/cursor/nn/plot
python plot.py
```

This will generate:
- Prediction vs True value scatter plots
- Residual plots
- Summary statistics (RMSE, MAE)

### Output

Figures will be saved in subdirectories:
- `lrt/plot/figs_pred_lrt_train_*/`
- `nn/plot/figs_pred_de_0/`

## Key Differences from TiO2 Scripts

1. **Dataset**: QM7 instead of TiO2
2. **Scaling parameters**: Different e_scaling and e_shift values
3. **Data indices**: Currently uses approximate 80/10/10 splits. You may need to adjust if your training used different indices.
4. **Queue**: Uses `iqtc09.q` instead of `iqtc12.q`

## Notes

- The plotting scripts assume sequential splitting of data (first 80% train, next 10% valid, last 10% test)
- If your training used random shuffling with a specific seed, you may need to adjust the index computation
- LRT plots include uncertainty quantification metrics
- NN plots show deterministic predictions only (no uncertainty)

## Troubleshooting

**No prediction runs found:**
- Make sure you've run the prediction scripts first
- Check that predictions completed successfully

**Wrong data splits:**
- If the plots don't look right, you may need to extract the actual train/val/test indices from your training logs
- Check the training logs for "Valid indices:" and "Test indices:" outputs

**Missing dependencies:**
```bash
conda activate bnn
pip install uncertainty-toolbox matplotlib seaborn pandas
```
