#!/bin/bash
#$ -N lrt_het_bg
#$ -q iqtc13.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o /home/g15farris/bin/bayesaenet/log/multirun/TiO2_big_lrt_hetero.out
#$ -e /home/g15farris/bin/bayesaenet/log/multirun/TiO2_big_lrt_hetero.err

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

# Best params from latest hps_lrt_het_bg run window (job 3525103), study bnn_lrt_hetero, trial 42
LR=6.917766954348385e-05
BS=512
MC=2
PRIOR_SCALE=0.42903923007274075
Q_SCALE=1.5089772306100152e-05
NOISE_MIN=0.09761279526401208

SEEDS=(121958 671155 131932 365838 259178 644167 110268 732180 54886 137337)

for i in $(seq 0 9); do
    echo "=== Starting LRT hetero run $i with seed ${SEEDS[$i]} at $(date) ==="
    python -m bnn_aenet.tasks.train \
        experiment=bnn_lrt_hetero \
        datamodule=TiO_Forces_Data100 \
        trainer.accelerator=gpu \
        trainer.devices=1 \
        trainer.max_epochs=50000 \
        dataset=TiO2_big \
        task_name=train \
        run_name=lrt_hetero_train_${i} \
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
        'tags=["TiO2_big", "lrt", "heteroscedastic", "train"]'
    echo "=== Finished LRT hetero run $i at $(date) ==="
done
