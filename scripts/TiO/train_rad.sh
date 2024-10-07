#!/bin/bash
#$ -N train_rad_TiO
#$ -pe smp* 1
#$ -q iqtc09.q
#$ -S /bin/bash
#$ -cwd
#$ -o train_rad.out
#$ -e train_rad.err
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
export CUDA_VISIBLE_DEVICES=`cat $TMPDIR/.gpus`
conda activate bnn
export PYTHONPATH="${PYTHONPATH}:/home/g15farris/bin/bayesaenet/src"
export OMP_NUM_THREADS=1
cd /home/g15farris/bin/forks/bayesaenet
python src/tasks/train.py \
    experiment=bnn_rad \
    trainer.deterministic=False \
    trainer.min_epochs=10000 \
    task_name=train_rad \
    datamodule=TiO\
    model.pretrain_epochs=0\
    model.lr=0.0005964703061220273\
    model.mc_samples_train=1\
    model.prior_scale=0.24847314880231502\
    model.q_scale=0.0010668894650671856\
    model.obs_scale=0.1420654727071492\
    datamodule.batch_size=32\
    tags=["TiO"]\