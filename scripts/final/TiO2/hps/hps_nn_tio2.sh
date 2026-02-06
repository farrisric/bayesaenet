#!/bin/bash
#$ -N hps_nn_tio2
#$ -q iqtc10.q
#$ -l iqtcgpu=1
#$ -cwd
#$ -j y
#$ -o hps_nn_tio2.out
#$ -e hps_nn_tio2.err

# ==============================================================================
# HPS: NN (Deep Ensemble) with Forces on TiO2 - iqtc10 GPU 0
# ==============================================================================

echo "=========================================="
echo "HPS: NN with Forces on TiO2"
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
echo "Starting HPS..."
python bnn_aenet/tasks/hpsearch.py \
    hpsearch=nn_forces \
    datamodule=TiO \
    datamodule.data_dir=data/TiO/train_forces.in \
    hpsearch.n_trials=30 \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    seed=42

echo ""
echo "HPS complete at: $(date)"
