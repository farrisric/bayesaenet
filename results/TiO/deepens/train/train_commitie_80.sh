#!/bin/bash
#$ -N commitie_80perc
#$ -pe smp* 1
#$ -q iqtc09.q
#$ -S /bin/bash
#$ -cwd
#$ -o out
#$ -e err
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
conda activate bayesian
export PYTHONPATH="${PYTHONPATH}:/home/g15telari/TiO/bayesaenet/bnn_aenet"
export OMP_NUM_THREADS=1
cd /home/g15telari/TiO/bayesaenet
python bnn_aenet/tasks/train.py task_name=TiO_train_deepens_80perc experiment=nn datamodule=TiO trainer.min_epochs=10000 trainer.max_epochs=50000 datamodule.test_split=0.1 datamodule.valid_split=0.1 datamodule.batch_size=64 model.optimizer.lr=0.0008779137387293351 seed=4493482,14124323,31343,89794,4323525,686476,543427,52342418,98656749,13424100 --multirun
