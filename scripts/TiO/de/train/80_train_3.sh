#!/bin/bash
#$ -N de_train80perc
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
python bnn_aenet/tasks/train.py \
	task_name=de_train_80perc_3 \
	experiment=nn \
	datamodule=TiO \
	trainer.min_epochs=10000 \
	trainer.max_epochs=50000 \
	datamodule.valid_split=100 \
	datamodule.batch_size=32 \
	model.optimizer.lr=0.0005486825664406678 \
	seed=27317724,41007174,88074495,99888956,81052829,56807324,62341859,60545057,78079452,15857814 \
	--multirun
