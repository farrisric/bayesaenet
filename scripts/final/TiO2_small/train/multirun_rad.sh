#!/bin/bash
#$ -N multi_rad
#$ -q iqtc13.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o /home/g15farris/bin/bayesaenet/log/multirun/TiO2_small_rad.out
#$ -e /home/g15farris/bin/bayesaenet/log/multirun/TiO2_small_rad.err

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

# Best RAD HPS parameters
LR=0.0007707725310702081
BS=256
MC=2
PRIOR_SCALE=0.21380042973105887
Q_SCALE=1.4513837427658247e-05
OBS_SCALE=0.12019856849128742
SCALE_FORCE=0.16723055999153458

SEEDS=(121958 671155 131932 365838 259178 644167 110268 732180 54886 137337)

for i in $(seq 0 9); do
    echo "=== Starting RAD run $i with seed ${SEEDS[$i]} at $(date) ==="
    python -m bnn_aenet.tasks.train \
        experiment=bnn_rad \
        datamodule=TiO_Forces_Data20 \
        trainer.accelerator=gpu \
        trainer.devices=1 \
        +trainer.precision=16-mixed \
        trainer.max_epochs=50000 \
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
        callbacks.early_stopping.patience=500 \
        seed=${SEEDS[$i]} \
        'tags=["TiO2_small", "rad", "train"]'
    echo "=== Finished RAD run $i at $(date) ==="
done
