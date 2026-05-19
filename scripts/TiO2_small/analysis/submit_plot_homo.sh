#!/bin/bash
#$ -N plot_sm_homo
#$ -q iqtc13.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o /home/g15farris/bin/bayesaenet/log/analysis/TiO2_small_plot_homo.out
#$ -e /home/g15farris/bin/bayesaenet/log/analysis/TiO2_small_plot_homo.err

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

mkdir -p log/analysis
mkdir -p plots/TiO2_small/homo_metrics

python -m bnn_aenet.tasks.plot \
    --pred-dir bnn_aenet/logs/TiO2_small/pred/runs \
    --output-dir plots/TiO2_small/homo_metrics \
    --train-dir bnn_aenet/logs/TiO2_small/train \
    --subsets train val test \
    --models DE lrt rad
