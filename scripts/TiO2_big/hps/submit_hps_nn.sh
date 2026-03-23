#!/bin/bash
#$ -N hps_nn_bg
#$ -q iqtc13.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o /home/g15farris/bin/bayesaenet/log/hps/TiO2_big_hps_nn.out
#$ -e /home/g15farris/bin/bayesaenet/log/hps/TiO2_big_hps_nn.err

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

python -m bnn_aenet.tasks.hpsearch \
    hpsearch=nn \
    datamodule=TiO_Forces_Data100 \
    dataset=TiO2_big \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    hpsearch.results_subdir=TiO2_big \
    hpsearch.study.study_name=nn_big \
    'tags=["TiO2_big", "nn", "hps"]'
