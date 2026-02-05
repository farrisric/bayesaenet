#!/bin/bash
#$ -N lrt_hps
#$ -pe smp* 1
#$ -q iqtc09.q
#$ -S /bin/bash
#$ -cwd
#$ -o out
#$ -e err
#$ -m e
#$ -M farrisric@outlook.com
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
conda activate bnn
export PYTHONPATH="${PYTHONPATH}:/home/g15farris/bin/bayesaenet/bnn_aenet"
export OMP_NUM_THREADS=1
cd /home/g15farris/bin/bayesaenet
python bnn_aenet/tasks/train.py \
    trainer.min_epochs=100000 \
    trainer.max_epochs=100000 \
    experiment=bnn_lrt \
    trainer.deterministic=False \
    task_name=lrt_train \
    run_name=lrt_train_8 \
    datamodule=QM7 \
    datamodule.valid_split=100 \
    datamodule.batch_size=32 \
    model.lr=0.0001834577852050578 \
    model.mc_samples_train=2 \
    model.prior_scale=0.12457376609465613 \
    model.q_scale=0.0007539921163280931 \
    model.obs_scale=0.48983719064601916 \
    seed=217693
