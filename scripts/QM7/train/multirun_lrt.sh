#!/bin/bash
#$ -N lrt_qm7
#$ -q iqtc13.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o /home/g15farris/bin/bayesaenet/log/multirun/QM7_lrt.out
#$ -e /home/g15farris/bin/bayesaenet/log/multirun/QM7_lrt.err

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

mkdir -p log/multirun

# Best params from latest hps_lrt_qm7 run window (job 3525111), study bnn_lrt_qm7, trial 119
LR=4.6471405980581316e-05
BS=256
MC=1
PRIOR_SCALE=0.1693797864942707
Q_SCALE=2.490874683534868e-05
OBS_SCALE=0.11300771623660232
SCALE_FORCE=0.553903416100093

SEEDS=(121958 671155 131932 365838 259178 644167 110268 732180 54886 137337)

for i in $(seq 0 9); do
    echo "=== Starting QM7 LRT run $i with seed ${SEEDS[$i]} at $(date) ==="
    python -m bnn_aenet.tasks.train \
        experiment=bnn_lrt \
        datamodule=QM7_Data100 \
        trainer.accelerator=gpu \
        trainer.devices=1 \
        trainer.max_epochs=50000 \
        dataset=QM7 \
        task_name=train \
        run_name=lrt_qm7_train_${i} \
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
        'tags=["QM7", "lrt", "train"]'
    echo "=== Finished QM7 LRT run $i at $(date) ==="
done
