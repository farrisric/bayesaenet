#!/bin/bash
#$ -N inference_timing
#$ -q iqtc13.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o /home/g15farris/bin/bayesaenet/log/time/inference_timing.out
#$ -e /home/g15farris/bin/bayesaenet/log/time/inference_timing.err

# Inference-cost benchmark (energy + forces) on the TiO2 test set, for the
# three force-trained models (LRT, RAD, DE). Companion to the training-time
# table (Table 9). Run on the same RTX 4090 used for the training benchmark.

. /etc/profile
__conda_setup="$('/aplic/anaconda/2020.02/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then eval "$__conda_setup"; else
  if [ -f "/aplic/anaconda/2024.10/etc/profile.d/conda.sh" ]; then . "/aplic/anaconda/2024.10/etc/profile.d/conda.sh"; else export PATH="/aplic/anaconda/2024.10/bin:$PATH"; fi
fi
unset __conda_setup
module load cuda/12.4
conda activate bnn
export OMP_NUM_THREADS=4
export TMPDIR=/tmp/g15farris
export PYTHONPATH=/home/g15farris/bin/bayesaenet:$PYTHONPATH

cd /home/g15farris/bin/bayesaenet
mkdir -p log/time

python scripts/time/inference_timing.py \
  --device gpu \
  --batch-size 128 \
  --mc-samples 20 \
  --warmup 2 \
  --repeats 5 \
  --output scripts/time/inference_timing_results.csv
