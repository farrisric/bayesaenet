#!/bin/bash
#$ -N pred_fo_5perc
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

for perc in 5 10 20 30 40 50 60 70 80;
do
    python bnn_aenet/tasks/predict.py \
           task_name=predict_fo_${perc}perc \
           prediction=TiO \
           ckpt_path=all \
           +method=FO \
           runs_dir=bnn_aenet/logs/TiO_train_fo_${perc}
done