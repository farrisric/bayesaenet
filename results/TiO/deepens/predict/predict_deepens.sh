#!/bin/bash
#$ -N pred_deepens
#$ -pe smp* 1
#$ -q iqtc08.q
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
export HYDRA_FULL_ERROR=1
export PYTHONPATH="${PYTHONPATH}:/home/g15telari/TiO/bayesaenet/bnn_aenet"
export OMP_NUM_THREADS=1
cd /home/g15telari/TiO/bayesaenet

for perc in 80;
do
    python bnn_aenet/tasks/predict.py \
           task_name=TiO_pred_deepens_${perc}perc_final \
           prediction=TiO \
           ckpt_path=all \
           +method=NN \
           runs_dir=bnn_aenet/logs/TiO_train_deepens_${perc}perc
done
