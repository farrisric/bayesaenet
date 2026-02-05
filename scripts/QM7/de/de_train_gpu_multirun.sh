#!/bin/bash
#$ -N de_train_gpu_multirun
#$ -pe smp 1
#$ -l iqtcgpu=1
#$ -q iqtc13.q
#$ -S /bin/bash
#$ -cwd
#$ -o train_de_gpu_multirun.out
#$ -e train_de_gpu_multirun.err
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
#export CUDA_VISIBLE_DEVICES=`cat $TMPDIR/.gpus`
module load cuda/12.4
conda activate bnn
export HYDRA_FULL_ERROR=1
export PYTHONPATH="${PYTHONPATH}:/home/g15farris/bin/bayesaenet/bnn_aenet"
export OMP_NUM_THREADS=1
cd /home/g15farris/bin/bayesaenet

# Run all 10 ensemble members sequentially using Hydra multirun
# Optimized epochs: model converges ~6500 epochs, early stopping enabled
python bnn_aenet/tasks/train.py \
    --multirun \
    trainer.min_epochs=5000 \
    trainer.max_epochs=30000 \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    +trainer.precision=16 \
    experiment=nn \
    trainer.deterministic=False \
    callbacks.early_stopping.patience=200 \
    task_name=de_train_gpu \
    run_name=de_gpu_0,de_gpu_1,de_gpu_2,de_gpu_3,de_gpu_4,de_gpu_5,de_gpu_6,de_gpu_7,de_gpu_8,de_gpu_9 \
    datamodule=QM7 \
    datamodule.device=cuda \
    datamodule.valid_split=100 \
    datamodule.batch_size=256 \
    seed=130119,635059,334166,478577,923204,462598,596067,103161,648393,737324
