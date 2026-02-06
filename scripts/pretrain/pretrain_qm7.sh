#!/bin/bash
#$ -N pretrain_qm7
#$ -q iqtc10.q
#$ -l iqtcgpu=1
#$ -cwd
#$ -j y
#$ -o pretrain_qm7.out
#$ -e pretrain_qm7.err

# ==============================================================================
# Pretrain deterministic NN for QM7 - to be used for BNN initialization
# ==============================================================================

echo "=========================================="
echo "Pretraining NN for QM7"
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
echo "Starting pretraining for 5 epochs..."
python scripts/pretrain/pretrain_nn.py \
    datamodule=QM7 \
    model=nn \
    trainer.max_epochs=5 \
    tags="[bayesian]" \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    seed=42

echo ""
echo "Pretraining complete at: $(date)"
