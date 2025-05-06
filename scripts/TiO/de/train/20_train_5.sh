#!/bin/bash
#$ -N de_train20perc
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
seed=$1
python bnn_aenet/tasks/train.py \
	task_name=de_train_20perc_5 \
	experiment=nn \
	datamodule=TiO \
	trainer.min_epochs=10000 \
	trainer.max_epochs=50000 \
	datamodule.valid_split=20 \
	datamodule.batch_size=32 \
	model.optimizer.lr=0.0014493825421745437 \
	seed=59063207,84052707,28179010,12135979,89044521,15565002,53347157,34415367,47887623,54561127 \
	--multirun
