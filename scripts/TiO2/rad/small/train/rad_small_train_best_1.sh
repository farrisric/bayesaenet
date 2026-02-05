#!/bin/bash
#$ -N rad_train
#$ -pe smp* 1
#$ -q iqtc12.q
#$ -S /bin/bash
#$ -cwd
#$ -o train_rad.out
#$ -e train_rad.err
. /etc/profile
__conda_setup="$('/aplic/anaconda/2020.02/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup" 
else
    if [ -f "/aplic/anaconda/2020.02/etc/profile.d/conda.sh" ]; then
        . "/aplic/anaconda/2020.02/etc/profile.d/conda.sh"
    else
        export PATH="/aplic/anaconda/2020.02/bin:$PATH"
    fi
fi
unset __conda_setup
export CUDA_VISIBLE_DEVICES=`cat $TMPDIR/.gpus`
conda activate bnn
export PYTHONPATH="${PYTHONPATH}:/home/g15farris/bin/bayesaenet/bnn_aenet"
export OMP_NUM_THREADS=1
cd /home/g15farris/bin/bayesaenet
python bnn_aenet/tasks/train.py \
    trainer.min_epochs=100000 \
    trainer.max_epochs=100000 \
    experiment=bnn_rad \
    trainer.deterministic=False \
    task_name=rad_train \
    run_name=rad_small_train_best_1 \
    datamodule=TiO \
    datamodule.valid_split=20 \
    datamodule.batch_size=32 \
    model.lr=0.00013675828334445094 \
    model.mc_samples_train=2 \
    model.prior_scale=0.11481524444729974 \
    model.q_scale=0.00017214030652976605 \
    model.obs_scale=0.7932033798886934\
    seed=321439
