#!/bin/bash
#$ -N pred_plot_f
#$ -pe smp 1
#$ -l iqtcgpu=1
#$ -q iqtc10.q
#$ -S /bin/bash
#$ -cwd
#$ -o predict_plot_forces.out
#$ -e predict_plot_forces.err
. /etc/profile
__conda_setup="$('/aplic/anaconda/2020.02/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup" 
else
    if [ -f "/aplic/anaconda/2024.10/etc/profile.d/conda.sh" ]; then
        . "/aplic/anaconda/2024.10/etc/profile.d/conda.sh"
    else
        export PATH="/aplic/anaconda/2024.10/bin:$PATH"
    fi
fi
unset __conda_setup
conda activate bnn
export HYDRA_FULL_ERROR=1
export PYTHONPATH="${PYTHONPATH}:/home/g15farris/bin/bayesaenet/bnn_aenet"
export OMP_NUM_THREADS=1
cd /home/g15farris/bin/bayesaenet

# Run predictions using trained force model
echo "=== Running Predictions ==="
python bnn_aenet/tasks/predict.py \
    method=bnn_forces_aux \
    runs_dir=bnn_aenet/logs/train/runs/default \
    ckpt_path=all \
    datamodule=QM7 \
    datamodule.device=cuda \
    trainer.accelerator=gpu \
    trainer.devices=1

# Generate plots
echo "=== Generating Plots ==="
python -c "
from pathlib import Path
from bnn_aenet.analysis.plot_force_predictions import plot_comprehensive_force_analysis
from bnn_aenet.analysis.analyze_force_predictions import analyze_prediction_file

# Find prediction files
pred_dir = Path('bnn_aenet/logs/train/runs/default')
pred_files = list(pred_dir.glob('**/bnn_forces_aux*val.parquet'))

if not pred_files:
    pred_files = list(pred_dir.glob('**/*.parquet'))

print(f'Found {len(pred_files)} prediction files')

for pred_file in pred_files:
    print(f'Processing: {pred_file}')
    
    # Compute metrics
    metrics = analyze_prediction_file(pred_file, verbose=True)
    
    # Generate plots
    output_dir = pred_file.parent / 'plots'
    figures = plot_comprehensive_force_analysis(
        pred_file,
        output_dir=output_dir,
        show=False,
        prefix=pred_file.stem + '_'
    )
    print(f'Generated {len(figures)} plots in {output_dir}')
"

echo "=== Done ==="
