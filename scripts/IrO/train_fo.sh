#!/bin/bash
#$ -N train_fo_IrO
#$ -pe smp* 1
#$ -q iqtc09.q
#$ -S /bin/bash
#$ -cwd
#$ -o train_fo.out
#$ -e train_fo.err
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
    trainer.min_epochs=10000 \
    experiment=bnn_fo \
    seed=15 \
    trainer.deterministic=False \
    task_name=train_fo \
    datamodule=IrO\
    model.pretrain_epochs=0\
    model.lr=0.000794772269279481\
    model.mc_samples_train=2\
    model.prior_scale=0.05232863499652436\
    model.q_scale=0.002927082968176485\
    model.obs_scale=0.3657322844763382\
    datamodule.batch_size=32\
    tags=["IrO"]
    