#!/bin/bash
#$ -N lrt_train_gpu_multirun
#$ -pe smp 1
#$ -l iqtcgpu=1
#$ -q iqtc13.q
#$ -S /bin/bash
#$ -cwd
#$ -o lrt_train_gpu_multirun.out
#$ -e lrt_train_gpu_multirun.err
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

# Run all 10 LRT models sequentially using Hydra multirun
# Optimized epochs: BNNs need more epochs, early stopping enabled
# Note: Update hyperparameters after HPS completes with GPU-optimized values
python bnn_aenet/tasks/train.py --multirun \
    trainer.min_epochs=5000 \
    trainer.max_epochs=50000 \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    experiment=bnn_lrt \
    trainer.deterministic=False \
    callbacks.early_stopping.patience=100 \
    task_name=lrt_train_gpu \
    run_name=lrt_gpu_0,lrt_gpu_1,lrt_gpu_2,lrt_gpu_3,lrt_gpu_4,lrt_gpu_5,lrt_gpu_6,lrt_gpu_7,lrt_gpu_8,lrt_gpu_9 \
    datamodule=QM7 \
    datamodule.device=cuda \
    datamodule.valid_split=100 \
    datamodule.batch_size=32 \
    model.lr=0.0001834577852050578 \
    model.mc_samples_train=2 \
    model.prior_scale=0.12457376609465613 \
    model.q_scale=0.0007539921163280931 \
    model.obs_scale=0.48983719064601916 \
    seed=130119,635059,334166,478577,923204,462598,596067,103161,648393,737324
