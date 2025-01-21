#!/bin/bash
#$ -N commitie_20perc
#$ -pe smp* 1
#$ -q iqtc12.q
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
python bnn_aenet/tasks/train.py task_name=TiO_train_deepens_20perc experiment=nn datamodule=TiO trainer.min_epochs=10000 trainer.max_epochs=50000 datamodule.test_split=0.7 datamodule.valid_split=0.1 datamodule.batch_size=32 model.optimizer.lr=0.0005640905763748237 seed=1,2,3,4,5,6,7,8,9,10 --multirun