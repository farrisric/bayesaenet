#!/bin/bash
#$ -N hps_lrt_ln_sm
#$ -q iqtc13.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o /home/g15farris/bin/bayesaenet/log/hps/TiO2_small_hps_lrt_learn_noise.out
#$ -e /home/g15farris/bin/bayesaenet/log/hps/TiO2_small_hps_lrt_learn_noise.err

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

# Choose HPS optimization metric:
#   MONITOR=total_rmse/val  (default, accuracy-focused)
#   MONITOR=elbo/val        (variational objective)
MONITOR="${MONITOR:-total_rmse/val}"
MODE="${MODE:-min}"

python -m bnn_aenet.tasks.hpsearch \
    hpsearch=bnn_lrt \
    datamodule=TiO_Forces_Data20 \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    trainer.max_epochs=10000 \
    trainer.min_epochs=1000 \
    hpsearch.monitor="${MONITOR}" \
    callbacks.early_stopping.patience=800 \
    callbacks.early_stopping.monitor="${MONITOR}" \
    callbacks.early_stopping.mode="${MODE}" \
    callbacks.model_checkpoint.monitor="${MONITOR}" \
    callbacks.model_checkpoint.mode="${MODE}" \
    hpsearch.results_subdir=TiO2_small \
    hpsearch.study.study_name=lrt_small_learn_noise \
    model.learn_noise=true \
    'tags=["TiO2_small", "lrt", "learn_noise", "hps"]'
