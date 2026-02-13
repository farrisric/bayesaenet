#!/bin/bash
# Submit TiO2_small training (NN, RAD, LRT) with best HPS values
#
# Queue layout:
#   iqtc13 (3 GPUs): NN, RAD (each 1 GPU)
#   iqtc10 (1 GPU):  LRT (no mixed precision)
#
# Usage: bash scripts/final/TiO2_small/train/submit_train.sh

BASEDIR="/home/g15farris/bin/bayesaenet"
cd ${BASEDIR}
mkdir -p log/multirun

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== TiO2_small Training ==="
echo "NN (iqtc13), RAD (iqtc13), LRT (iqtc10)"
echo ""

J1=$(qsub "${SCRIPT_DIR}/multirun_nn.sh")
echo "  NN:  $J1 (iqtc13)"

J2=$(qsub "${SCRIPT_DIR}/multirun_rad.sh")
echo "  RAD: $J2 (iqtc13)"

J3=$(qsub "${SCRIPT_DIR}/multirun_lrt.sh")
echo "  LRT: $J3 (iqtc10)"

echo ""
echo "All jobs submitted. Logs: log/multirun/TiO2_small_{nn,rad,lrt}.{out,err}"
