#!/bin/bash
#$ -N train_nn_forces
#$ -q iqtc13.q
#$ -l iqtcgpu=1
#$ -cwd
#$ -j y
#$ -o train_nn_forces.out
#$ -e train_nn_forces.err

# ==============================================================================
# Long Training: NN (Deep Ensemble) with Forces on TiO2
# ==============================================================================

echo "=========================================="
echo "Training: NN with Forces on TiO2"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "=========================================="

# Environment setup
source /etc/profile
module load anaconda
source activate bnn

export PYTHONPATH=/home/g15farris/bin/bayesaenet/bnn_aenet:$PYTHONPATH
export PROJECT_ROOT=/home/g15farris/bin/bayesaenet
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

cd /home/g15farris/bin/bayesaenet

echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

echo ""
echo "Starting long training..."
python bnn_aenet/tasks/train.py \
    experiment=nn_forces \
    datamodule=TiO \
    datamodule.data_dir=data/TiO/train_forces.in \
    trainer.max_epochs=50000 \
    trainer.min_epochs=5000 \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    callbacks.early_stopping.patience=200 \
    callbacks.early_stopping.monitor=total_rmse/val \
    callbacks.model_checkpoint.monitor=total_rmse/val \
    model.alpha=0.1 \
    model.force_weight=1.0 \
    seed=42 \
    tags="[nn,forces,long_train]"

echo ""
echo "Training complete at: $(date)"
