#!/bin/bash
#$ -N fo_train
#$ -pe smp* 1
#$ -q iqtc12.q
#$ -S /bin/bash
#$ -cwd
#$ -o t.out
#$ -e .err
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
seed=$1
python bnn_aenet/tasks/train.py \
    trainer.min_epochs=50000 \
    trainer.max_epochs=50000 \
    experiment=bnn_fo \
    trainer.deterministic=False \
    task_name=fo_train \
    run_name=training_0.00178 \
    datamodule=TiO \
    datamodule.valid_split=100 \
    datamodule.batch_size=64 \
    model.lr=9.817696733087544e-05 \
    model.mc_samples_train=2 \
    model.prior_scale=0.2791414697766688 \
    model.q_scale=0.0005423385492206237 \
    model.obs_scale=0.7121278289554771
