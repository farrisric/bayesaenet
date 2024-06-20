#!/bin/bash
#$ -N train_rad_IrO
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
    datamodule=IrO\
    model.pretrain_epochs=0\
    model.lr=0.00048357862190424346\
    model.mc_samples_train=2\
    model.prior_scale=0.32525978096701874\
    model.q_scale=0.007456841427543389\
    model.obs_scale=0.3624831404788194\
    datamodule.batch_size=32\
    tags=["IrO"]\
    ckpt_path=/home/g15farris/bin/forks/bayesaenet/src/logs/train_rad/runs/2024-05-17_09-23-04/checkpoints/epoch_237-step_68544.ckpt