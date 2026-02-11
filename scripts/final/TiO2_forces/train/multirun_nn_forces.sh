#!/bin/bash
#$ -N multi_nn_f
#$ -q iqtc10.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o /home/g15farris/bin/bayesaenet/logs/multirun/nn_forces.out
#$ -e /home/g15farris/bin/bayesaenet/logs/multirun/nn_forces.err

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

conda activate bnn

export OMP_NUM_THREADS=4
export TMPDIR=/tmp/g15farris
export PYTHONPATH=/home/g15farris/bin/bayesaenet:$PYTHONPATH
cd /home/g15farris/bin/bayesaenet

# Best NN Forces HPS parameters (trial 42, total_rmse/val = 41.75)
LR=0.0008682741606253118
BS=128

SEEDS=(121958 671155 131932 365838 259178 644167 110268 732180 54886 137337)

for i in $(seq 0 9); do
    echo "=== Starting NN Forces run $i with seed ${SEEDS[$i]} at $(date) ==="
    python -m bnn_aenet.tasks.train \
        experiment=nn_forces \
        datamodule=TiO_Forces \
        trainer.accelerator=gpu \
        trainer.devices=1 \
        +trainer.precision=16-mixed \
        trainer.max_epochs=50000 \
        task_name=nn_forces_train \
        run_name=nn_forces_train_${i} \
        datamodule.batch_size=${BS} \
        model.optimizer.lr=${LR} \
        callbacks.model_checkpoint.monitor=total_rmse/val \
        callbacks.early_stopping.monitor=total_rmse/val \
        callbacks.early_stopping.patience=500 \
        seed=${SEEDS[$i]}
    echo "=== Finished NN Forces run $i at $(date) ==="
done
