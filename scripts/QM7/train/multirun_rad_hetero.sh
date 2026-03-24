#!/bin/bash
#$ -N rad_het_qm7
#$ -q iqtc13.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o /home/g15farris/bin/bayesaenet/log/multirun/QM7_rad_hetero.out
#$ -e /home/g15farris/bin/bayesaenet/log/multirun/QM7_rad_hetero.err

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

# Latest hps_rad_het_qm7 run window (job 3525100) has no COMPLETE trials yet.
# Keep provisional params until a COMPLETE trial appears in that latest window.
LR=2.4156772025296228e-05
BS=512
MC=1
PRIOR_SCALE=0.3539501434391102
Q_SCALE=0.0012738881866537178
NOISE_MIN=0.003509014885796853

SEEDS=(121958 671155 131932 365838 259178 644167 110268 732180 54886 137337)

for i in $(seq 0 9); do
    echo "=== Starting QM7 RAD hetero run $i with seed ${SEEDS[$i]} at $(date) ==="
    python -m bnn_aenet.tasks.train \
        experiment=bnn_rad_hetero \
        datamodule=QM7_Data100 \
        trainer.accelerator=gpu \
        trainer.devices=1 \
        +trainer.precision=16-mixed \
        trainer.max_epochs=50000 \
        dataset=QM7 \
        task_name=train \
        run_name=rad_hetero_qm7_train_${i} \
        datamodule.batch_size=${BS} \
        model.lr=${LR} \
        model.mc_samples_train=${MC} \
        model.prior_scale=${PRIOR_SCALE} \
        model.q_scale=${Q_SCALE} \
        model.pretrain_epochs=0 \
        model.noise_hidden_size=15 \
        model.noise_min=${NOISE_MIN} \
        callbacks.model_checkpoint.monitor=total_rmse/val \
        callbacks.early_stopping.monitor=total_rmse/val \
        callbacks.early_stopping.patience=500 \
        seed=${SEEDS[$i]} \
        'tags=["QM7", "rad", "heteroscedastic", "train"]'
    echo "=== Finished QM7 RAD hetero run $i at $(date) ==="
done
