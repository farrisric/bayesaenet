#!/bin/bash
#$ -N nn_big
#$ -q iqtc13.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o /home/g15farris/bin/bayesaenet/log/multirun/TiO2_big_nn.out
#$ -e /home/g15farris/bin/bayesaenet/log/multirun/TiO2_big_nn.err

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
export TMPDIR=/tmp/g15farris
export PYTHONPATH=/home/g15farris/bin/bayesaenet:$PYTHONPATH
cd /home/g15farris/bin/bayesaenet

# Best params from latest hps_nn_bg run window (job 3524374), study nn_big, trial 40
LR=0.001023744343893084
BS=256

SEEDS=(121958 671155 131932 365838 259178 644167 110268 732180 54886 137337)

for i in $(seq 0 9); do
    echo "=== Starting NN run $i with seed ${SEEDS[$i]} at $(date) ==="
    python -m bnn_aenet.tasks.train \
        experiment=nn \
        datamodule=TiO_Forces_Data100 \
        trainer.accelerator=gpu \
        trainer.devices=1 \
        trainer.max_epochs=50000 \
        dataset=TiO2_big \
        task_name=train \
        run_name=nn_train_${i} \
        datamodule.batch_size=${BS} \
        model.optimizer.lr=${LR} \
        callbacks.model_checkpoint.monitor=total_rmse/val \
        callbacks.early_stopping.monitor=total_rmse/val \
        callbacks.early_stopping.patience=500 \
        seed=${SEEDS[$i]} \
        'tags=["TiO2_big", "nn", "train"]'
    echo "=== Finished NN run $i at $(date) ==="
done
