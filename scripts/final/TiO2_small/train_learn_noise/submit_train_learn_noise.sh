#!/bin/bash
# Submit TiO2_small learn_noise training (RAD, LRT) on iqtc13
#
# Both use iqtc13 (3 GPUs available); each job takes 1 GPU.
# Note: LRT uses NO mixed precision (incompatible with LRT).
#
# Usage: bash scripts/final/TiO2_small/train_learn_noise/submit_train_learn_noise.sh

BASEDIR="/home/g15farris/bin/bayesaenet"
cd ${BASEDIR}
mkdir -p log/multirun

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== TiO2_small Learn Noise Training ==="
echo "RAD (iqtc13), LRT (iqtc13)"
echo ""

J1=$(qsub "${SCRIPT_DIR}/multirun_rad_learn_noise.sh")
echo "  RAD: $J1 (iqtc13)"

J2=$(qsub "${SCRIPT_DIR}/multirun_lrt_learn_noise.sh")
echo "  LRT: $J2 (iqtc13)"

echo ""
echo "All jobs submitted."
echo "Logs: log/multirun/TiO2_small_{rad,lrt}_learn_noise.{out,err}"
echo ""
echo "Run names: rad_ln_train_0..9, lrt_ln_train_0..9"
echo "Output: bnn_aenet/logs/TiO2_small/train/runs/{rad,lrt}/"
