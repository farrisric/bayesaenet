#!/bin/bash
#$ -N arch_comparison
#$ -q iqtc12.q
#$ -pe smp 8
#$ -cwd
#$ -o /home/g15farris/bin/bayesaenet/logs/arch_comparison.out
#$ -e /home/g15farris/bin/bayesaenet/logs/arch_comparison.err
#$ -j n

echo "============================================================"
echo "Architecture Comparison Test: 15:15 vs 25:25 vs 15:15:15"
echo "============================================================"
echo "Job ID: $JOB_ID"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Queue: iqtc12.q (CPU only)"
echo "Cores: $NSLOTS"
echo "============================================================"

# Set up environment - try multiple conda paths for compatibility across nodes
if [ -f "/home/g15farris/miniconda3/etc/profile.d/conda.sh" ]; then
    source /home/g15farris/miniconda3/etc/profile.d/conda.sh
elif [ -f "/aplic/anaconda/2020.02/etc/profile.d/conda.sh" ]; then
    source /aplic/anaconda/2020.02/etc/profile.d/conda.sh
else
    echo "ERROR: Could not find conda installation"
    exit 1
fi
conda activate bnn

# Set number of threads for PyTorch
export OMP_NUM_THREADS=$NSLOTS
export MKL_NUM_THREADS=$NSLOTS

# Add project to path
export PYTHONPATH=/home/g15farris/bin/bayesaenet:$PYTHONPATH

echo ""
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo ""

# Run the architecture comparison
python /home/g15farris/bin/bayesaenet/scripts/tests/test_architecture_comparison.py

echo ""
echo "============================================================"
echo "Job completed at: $(date)"
echo "============================================================"
