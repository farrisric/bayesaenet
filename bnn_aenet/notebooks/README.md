# Analysis Notebooks

## Quick Start - TiO2

### Option 1: Run Python Script (Recommended)

```bash
conda activate bnn
cd /home/g15farris/bin/bayesaenet/bnn_aenet/notebooks/TiO2
python run_analysis.py
```

Or use the shell script:
```bash
./RUN.sh
```

### Option 2: Use Jupyter Notebooks

```bash
conda activate bnn
cd /home/g15farris/bin/bayesaenet/bnn_aenet/notebooks/TiO2
jupyter notebook
```

Then open and run:
1. `01_compute_metrics.ipynb` - Computes all metrics
2. `02_create_plots.ipynb` - Creates publication plots

## Output

- **Metrics**: `TiO2/results/uq_metrics_Test.csv`
- **Figures**: `TiO2/figures/*.png`
- **Failed runs**: `TiO2/results/failed_experiments.txt` (if any)

## Requirements

Already installed in `bnn` conda environment:
- pandas, numpy, matplotlib, seaborn, torch, scikit-learn

Optional (for advanced calibration plots):
```bash
pip install uncertainty-toolbox
```
