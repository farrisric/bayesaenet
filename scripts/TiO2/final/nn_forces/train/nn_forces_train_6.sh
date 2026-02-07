#!/bin/bash
#$ -N nn_forces_train_6
#$ -q iqtc13.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o /home/g15farris/bin/bayesaenet/logs/nn_forces/nn_forces_train_6.out
#$ -e /home/g15farris/bin/bayesaenet/logs/nn_forces/nn_forces_train_6.err

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

export OMP_NUM_THREADS=4
cd /home/g15farris/bin/bayesaenet

python bnn_aenet/tasks/train.py \
    experiment=nn_forces \
    datamodule=TiO \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    trainer.min_epochs=50000 \
    trainer.max_epochs=50000 \
    trainer.deterministic=False \
    task_name=nn_forces_train \
    run_name=nn_forces_train_6 \
    datamodule.batch_size=128 \
    model.optimizer.lr=0.0002497627476497723 \
    model.optimizer.weight_decay=0.0024552339549495416 \
    model.force_weight=0.5081669394595789 \
    callbacks.model_checkpoint.monitor=total_rmse/val \
    callbacks.early_stopping.monitor=total_rmse/val \
    callbacks.early_stopping.patience=500 \
    seed=110268
