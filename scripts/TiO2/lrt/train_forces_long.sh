#!/bin/bash
#$ -N tio_f_long
#$ -pe smp 1
#$ -l iqtcgpu=1
#$ -q iqtc10.q
#$ -S /bin/bash
#$ -cwd
#$ -o train_forces_long.out
#$ -e train_forces_long.err

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

echo "=== Training BNN_Forces_Aux on TiO2 - Long Run ==="
echo "Start time: $(date)"

python bnn_aenet/tasks/train.py \
    experiment=bnn_lrt_forces_aux \
    datamodule.data_dir=data/TiO/train_forces.in \
    datamodule.batch_size=32 \
    datamodule.device=cuda \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    trainer.max_epochs=10000 \
    trainer.min_epochs=1000 \
    model.force_weight=1.0 \
    model.mc_samples_train=2 \
    model.mc_samples_eval=20 \
    callbacks.early_stopping.patience=200 \
    callbacks.early_stopping.monitor=rmse/val \
    run_name=tio_forces_long \
    seed=42

echo "End time: $(date)"
echo "=== Training Done ==="

# After training, run predictions
echo "=== Starting Predictions ==="

# Find the best checkpoint
CKPT_DIR="/home/g15farris/bin/bayesaenet/bnn_aenet/logs/train/runs/tio_forces_long/checkpoints"
BEST_CKPT=$(ls -t ${CKPT_DIR}/epoch*.ckpt 2>/dev/null | head -1)

if [ -z "$BEST_CKPT" ]; then
    echo "No checkpoint found, using last.ckpt"
    BEST_CKPT="${CKPT_DIR}/last.ckpt"
fi

echo "Using checkpoint: $BEST_CKPT"

python bnn_aenet/tasks/predict.py \
    experiment=bnn_lrt_forces_aux \
    datamodule.data_dir=data/TiO/train_forces.in \
    datamodule.batch_size=32 \
    datamodule.device=cuda \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    model.mc_samples_eval=50 \
    ckpt_path="$BEST_CKPT" \
    run_name=tio_forces_long_pred

echo "=== Predictions Done ==="
echo "End time: $(date)"
