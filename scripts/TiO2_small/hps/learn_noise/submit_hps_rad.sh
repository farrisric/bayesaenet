#!/bin/bash
#$ -N hps_rad_ln_sm
#$ -q iqtc13.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o /home/g15farris/bin/bayesaenet/log/hps/TiO2_small_hps_rad_learn_noise.out
#$ -e /home/g15farris/bin/bayesaenet/log/hps/TiO2_small_hps_rad_learn_noise.err

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
    hpsearch=bnn_rad \
    datamodule=TiO_Forces_Data20 \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    +trainer.precision=16-mixed \
    trainer.max_epochs=10000 \
    trainer.min_epochs=1000 \
    callbacks.early_stopping.patience=800 \
    hpsearch.results_subdir=TiO2_small \
    hpsearch.study.study_name=rad_small_learn_noise \
    model.learn_noise=true \
    'tags=["TiO2_small", "rad", "learn_noise", "hps"]'
