#!/bin/bash
#$ -N rad_het_sm
#$ -q iqtc10.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o /home/g15farris/bin/bayesaenet/log/multirun/TiO2_small_rad_hetero.out
#$ -e /home/g15farris/bin/bayesaenet/log/multirun/TiO2_small_rad_hetero.err

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

# Best RAD hetero HPS parameters for TiO2_small (study bnn_rad_hetero, trial 14)
LR=0.0001338065849033924
BS=128
MC=2
PRIOR_SCALE=0.2259815570093957
Q_SCALE=3.5911832821240514e-05
NOISE_MIN=0.03613364569649651

SEEDS=(121958 671155 131932 365838 259178 644167 110268 732180 54886 137337)

for i in $(seq 0 9); do
    echo "=== Starting RAD hetero run $i with seed ${SEEDS[$i]} at $(date) ==="
    python -m bnn_aenet.tasks.train \
        experiment=bnn_rad_hetero \
        datamodule=TiO_Forces_Data20 \
        trainer.accelerator=gpu \
        trainer.devices=1 \
        +trainer.precision=16-mixed \
        trainer.max_epochs=50000 \
        dataset=TiO2_small \
        task_name=train \
        run_name=rad_hetero_train_${i} \
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
        'tags=["TiO2_small", "rad", "heteroscedastic", "train"]'
    echo "=== Finished RAD hetero run $i at $(date) ==="
done
