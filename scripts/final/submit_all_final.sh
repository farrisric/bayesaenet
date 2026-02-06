#!/bin/bash
#$ -N final_all
#$ -q iqtc10.q
#$ -l iqtcgpu=1
#$ -cwd
#$ -j y
#$ -o final_all.out
#$ -e final_all.err
#$ -pe omp 4

# ==============================================================================
# Final Training Job Script: All Methods x All Datasets
# ==============================================================================
# Submits sequential training of:
#   - LRT, Flipout, Radial (BNN methods) on QM7 and TiO2
#   - Deep Ensemble (5 members) on QM7 and TiO2
#
# Usage:
#   qsub scripts/final/submit_all_final.sh
# ==============================================================================

echo "=========================================="
echo "BNN-AENET Final Training"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "=========================================="

# Environment setup
source /etc/profile
module load anaconda
source activate bnn

# Set paths
export PYTHONPATH=/home/g15farris/bin/bayesaenet/bnn_aenet:$PYTHONPATH
export HYDRA_FULL_ERROR=1

# Navigate to project root
cd /home/g15farris/bin/bayesaenet

# Check GPU availability
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

# Run all final training
echo ""
echo "Starting final training..."
python scripts/final/run_final_training.py --all --gpu 0

echo ""
echo "Training complete at: $(date)"
