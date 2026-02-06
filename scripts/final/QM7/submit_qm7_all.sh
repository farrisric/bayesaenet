#!/bin/bash
#$ -N qm7_final
#$ -q iqtc10.q
#$ -l iqtcgpu=1
#$ -cwd
#$ -j y
#$ -o qm7_final.out
#$ -e qm7_final.err
#$ -pe omp 4

# ==============================================================================
# Final Training Job Script: All Methods on QM7
# ==============================================================================

echo "=========================================="
echo "QM7 Final Training"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "=========================================="

# Environment setup
source /etc/profile
module load anaconda
source activate bnn

export PYTHONPATH=/home/g15farris/bin/bayesaenet/bnn_aenet:$PYTHONPATH
export HYDRA_FULL_ERROR=1

cd /home/g15farris/bin/bayesaenet

echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

# Train LRT
echo ""
echo "Training LRT on QM7..."
python scripts/final/run_final_training.py --method lrt --dataset qm7 --gpu 0

# Train Flipout
echo ""
echo "Training Flipout on QM7..."
python scripts/final/run_final_training.py --method fo --dataset qm7 --gpu 0

# Train Radial
echo ""
echo "Training Radial on QM7..."
python scripts/final/run_final_training.py --method rad --dataset qm7 --gpu 0

# Train Deep Ensemble (5 members)
echo ""
echo "Training Deep Ensemble (5 members) on QM7..."
python scripts/final/run_final_training.py --method de --dataset qm7 --n-ensemble 5 --gpu 0

echo ""
echo "QM7 training complete at: $(date)"
