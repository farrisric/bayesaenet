#!/bin/bash
#SBATCH --job-name=multi_rad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --partition=iqtc13.q
#SBATCH --error=tio2_nn.out
#SBATCH --output=tio2_nn.err
#SBATCH --output=/home/g15farris/bin/bayesaenet/log/multirun/TiO2_small_rad.out
#SBATCH --error=/home/g15farris/bin/bayesaenet/log/multirun/TiO2_small_rad.err


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

# Best RAD HPS parameters
LR=0.00035722967627023364
BS=512
MC=2
PRIOR_SCALE=0.15593351063684593
Q_SCALE=2.258285167134673e-03
OBS_SCALE=1.0262810235808921
SCALE_FORCE=0.887890491346372

SEEDS=($(for i in {1..50}; do od -An -N2 -tu2 < /dev/urandom; done))
for i in $(seq 25 29); do
    echo "=== Starting RAD run $i with seed ${SEEDS[$i]} at $(date) ==="
    python -m bnn_aenet.tasks.train \
        experiment=bnn_rad \
        datamodule=TiO_Forces_Data20 \
        trainer.accelerator=gpu \
        trainer.devices=1 \
        +trainer.precision=16-mixed \
        trainer.max_epochs=100000 \
        dataset=TiO2_small \
        task_name=train \
        run_name=rad_train_${i} \
        datamodule.batch_size=${BS} \
        model.lr=${LR} \
        model.mc_samples_train=${MC} \
        model.prior_scale=${PRIOR_SCALE} \
        model.q_scale=${Q_SCALE} \
        model.obs_scale=${OBS_SCALE} \
        model.pretrain_epochs=0 \
        model.scale_force=${SCALE_FORCE} \
        callbacks.model_checkpoint.monitor=total_rmse/val \
        callbacks.early_stopping.monitor=total_rmse/val \
        callbacks.early_stopping.patience=100 \
        seed=${SEEDS[$i]} \
        'tags=["TiO2_small", "rad", "train"]'
    echo "=== Finished RAD run $i at $(date) ==="
done
