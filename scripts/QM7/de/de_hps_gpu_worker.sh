#!/bin/bash
#$ -N de_hps_w
#$ -pe smp 1
#$ -l iqtcgpu=1
#$ -q iqtc13.q
#$ -S /bin/bash
#$ -cwd
#$ -o de_hps_gpu_$JOB_ID.out
#$ -e de_hps_gpu_$JOB_ID.err
. /etc/profile
__conda_setup="$('/aplic/anaconda/2020.02/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup" 
else
    if [ -f "/aplic/anaconda/2024.10/etc/profile.d/conda.sh" ]; then
        . "/aplic/anaconda/2024.10/etc/profile.d/conda.sh"
    else
        export PATH="/aplic/anaconda/2024.10/bin:$PATH"
    fi
fi
unset __conda_setup
module load cuda/12.4
conda activate bnn
export HYDRA_FULL_ERROR=1
export PYTHONPATH="${PYTHONPATH}:/home/g15farris/bin/bayesaenet/bnn_aenet"
export OMP_NUM_THREADS=1

# Use GPU 0 for DE HPS
export CUDA_VISIBLE_DEVICES=0
cd /home/g15farris/bin/bayesaenet

# DE HPS worker - connects to shared Optuna study
# Multiple workers can run in parallel, each picking up unfinished trials
python bnn_aenet/tasks/hpsearch.py \
    hpsearch=nn \
    hpsearch.n_trials=30 \
    trainer.min_epochs=500 \
    trainer.max_epochs=3000 \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    trainer.deterministic=False \
    task_name=de_hps_gpu \
    datamodule=QM7 \
    datamodule.device=cuda \
    datamodule.valid_split=100 \
    seed=$RANDOM
