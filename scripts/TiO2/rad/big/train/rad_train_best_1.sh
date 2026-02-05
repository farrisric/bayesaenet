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
    run_name=rad_train_best_1 \
    datamodule=TiO \
    datamodule.valid_split=100 \
    datamodule.batch_size=64 \
    model.lr=0.0005362126384507363 \
    model.mc_samples_train=2 \
    model.prior_scale=0.10785726105500282 \
    model.q_scale=0.0008004456799575063 \
    model.obs_scale=0.29404969365158845 \
    seed=321439
