#!/bin/bash
#$ -N tio_pred
#$ -pe smp 1
#$ -l iqtcgpu=1
#$ -q iqtc10.q
#$ -S /bin/bash
#$ -cwd
#$ -o predict_forces_test.out
#$ -e predict_forces_test.err

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
export PROJECT_ROOT=/home/g15farris/bin/bayesaenet

cd /home/g15farris/bin/bayesaenet

CKPT="/home/g15farris/bin/bayesaenet/bnn_aenet/logs/train/runs/tio_forces_aux_test/checkpoints/epoch_99-step_2500.ckpt"

echo "=== Running Force Predictions & Plotting ==="
echo "Checkpoint: $CKPT"
echo "Start time: $(date)"

python scripts/TiO2/lrt/run_force_prediction.py \
    --ckpt "$CKPT" \
    --train_in data/TiO/train_forces.in \
    --mc_samples 50 \
    --device cuda

echo "End time: $(date)"
