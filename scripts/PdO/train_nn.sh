#!/bin/bash
#$ -N multi_train_nn
#$ -pe smp* 1
#$ -q iqtc09.q
#$ -S /bin/bash
#$ -cwd
#$ -o multi_train_nn.out
#$ -e multi_train_nn.out
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
export PYTHONPATH="${PYTHONPATH}:/home/g15farris/bin/bayesaenet/src"
export OMP_NUM_THREADS=1
cd /home/g15farris/bin/forks/bayesaenet
python src/tasks/train.py \
    task_name=train_nn \
    experiment=nn \
    datamodule=PdO \
    trainer.min_epochs=10000\
    tags=["PdO"]\
    seed=1,2,3,4,5,6,7,8,9,10\
    --multirun
