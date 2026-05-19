#!/bin/bash
#SBATCH --job-name=relax
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --partition=iqtc13.q
#SBATCH --error=QM7_nn.out
#SBATCH --output=QM7_nn.err


ulimit -l unlimited
ulimit -s unlimited

. /etc/profile
__conda_setup="$('/aplic/anaconda/2024.10/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
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
source activate /home/g15farris/.conda/envs/bnn

export OMP_NUM_THREADS=4
export TMPDIR=/tmp/g15farris
export PYTHONPATH=/home/g15farris/bin/bayesaenet:$PYTHONPATH
cd /home/g15farris/bin/bayesaenet

mkdir -p log/multirun

# Best params from latest hps_nn_qm7 run window (job 3524375), study nn_qm7, trial 112
LR=0.008492113538781404
BS=128

# SEEDS=(121958 671155 131932 365838 259178 644167 110268 732180 54886 137337)
SEEDS=($(for i in {1..100}; do od -An -N2 -tu2 < /dev/urandom; done))
for i in $(seq 40 49); do
    echo "=== Starting QM7 NN run $i with seed ${SEEDS[$i]} at $(date) ==="
    python -m bnn_aenet.tasks.train \
        experiment=nn \
        datamodule=QM7_Data100 \
        trainer.accelerator=gpu \
        trainer.devices=1 \
        trainer.max_epochs=50000 \
        dataset=QM7 \
        task_name=train \
        run_name=nn_qm7_train_${i} \
        datamodule.batch_size=${BS} \
        model.optimizer.lr=${LR} \
        callbacks.model_checkpoint.monitor=total_rmse/val \
        callbacks.early_stopping.monitor=total_rmse/val \
        callbacks.early_stopping.patience=500 \
        seed=${SEEDS[$i]} \
        'tags=["QM7", "nn", "train"]'
    echo "=== Finished QM7 NN run $i at $(date) ==="
done
