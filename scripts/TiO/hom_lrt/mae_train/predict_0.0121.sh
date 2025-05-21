#!/bin/bash
#$ -N lrt_predict
#$ -pe smp* 1
#$ -q iqtc09.q
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
export HYDRA_FULL_ERROR=1
export PYTHONPATH="${PYTHONPATH}:/home/g15farris/bin/bayesaenet/bnn_aenet"
export OMP_NUM_THREADS=1
cd /home/g15farris/bin/bayesaenet
python bnn_aenet/tasks/predict.py \
    task_name=hom_lrt_pred_0.012 \
    prediction=TiO \
    ckpt_path=/home/g15farris/bin/bayesaenet/bnn_aenet/logs/lrt_train_0.012/runs/2025-05-20_16-41-26/checkpoints/epoch_6949-step_1376100.ckpt  \
    datamodule.valid_split=100 \
    +method=HLRT \
    runs_dir=bnn_aenet/logs/lrt_train_0.012
