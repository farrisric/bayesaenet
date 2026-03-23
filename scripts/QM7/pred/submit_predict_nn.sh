#!/bin/bash
#$ -N pred_nn_qm7
#$ -q iqtc13.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o /home/g15farris/bin/bayesaenet/log/predict/QM7_pred_nn.out
#$ -e /home/g15farris/bin/bayesaenet/log/predict/QM7_pred_nn.err

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

module load cuda/12.4
conda activate bnn

export OMP_NUM_THREADS=4
export TMPDIR=/tmp/g15farris
export PYTHONPATH=/home/g15farris/bin/bayesaenet:$PYTHONPATH
cd /home/g15farris/bin/bayesaenet
mkdir -p log/predict
mkdir -p bnn_aenet/logs/QM7/pred/runs/nn

python -m bnn_aenet.tasks.predict_forces \
    --model-type nn \
    --runs-dir bnn_aenet/logs/QM7/train/runs/nn \
    --output-dir bnn_aenet/logs/QM7/pred/runs/nn \
    --data-dir data/QM7/train.in \
    --subsets train val test \
    --device gpu \
    --batch-size 256
