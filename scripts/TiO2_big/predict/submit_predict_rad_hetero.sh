#!/bin/bash
#$ -N pred_rad_het_bg
#$ -q iqtc13.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o /home/g15farris/bin/bayesaenet/log/predict/TiO2_big_pred_rad_hetero.out
#$ -e /home/g15farris/bin/bayesaenet/log/predict/TiO2_big_pred_rad_hetero.err

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
mkdir -p bnn_aenet/logs/TiO2_big/pred/runs/rad_hetero

python -m bnn_aenet.tasks.predict_forces \
    --model-type rad_hetero \
    --runs-dir bnn_aenet/logs/TiO2_big/train/runs/rad_hetero \
    --output-dir bnn_aenet/logs/TiO2_big/pred/runs/rad_hetero \
    --data-dir data/TiO/train_forces.in \
    --split-config Data100 \
    --subsets train val test \
    --device gpu \
    --batch-size 128 \
    --mc-samples 20
