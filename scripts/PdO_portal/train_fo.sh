#!/bin/bash
#$ -N train_fo
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
conda activate bayesian
export PYTHONPATH="${PYTHONPATH}:/home/g15farris/bin/bayesaenet/src"
export OMP_NUM_THREADS=1
cd /home/g15farris/bin/bayesaenet
python src/tasks/train.py \
    trainer.min_epochs=10000 \
    experiment=bnn_fo \
    seed=15 \
    trainer.deterministic=False \
    task_name=train_fo \
    datamodule=PdO \