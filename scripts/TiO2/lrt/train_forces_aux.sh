#!/bin/bash
#$ -N tio_f_aux
#$ -pe smp 1
#$ -l iqtcgpu=1
#$ -q iqtc10.q
#$ -S /bin/bash
#$ -cwd
#$ -o train_forces_aux.out
#$ -e train_forces_aux.err

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

export HYDRA_FULL_ERROR=1
export PYTHONPATH="${PYTHONPATH}:/home/g15farris/bin/bayesaenet/bnn_aenet"
export OMP_NUM_THREADS=1
export PROJECT_ROOT=/home/g15farris/bin/bayesaenet

cd /home/g15farris/bin/bayesaenet

echo "=== Training BNN_Forces_Aux on TiO2 with forces ==="
echo "Start time: $(date)"

python bnn_aenet/tasks/train.py \
    experiment=bnn_lrt_forces_aux \
    datamodule.data_dir=data/TiO/train_forces.in \
    datamodule.batch_size=32 \
    datamodule.device=cuda \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    trainer.max_epochs=100 \
    trainer.min_epochs=10 \
    model.force_weight=1.0 \
    model.mc_samples_train=2 \
    model.mc_samples_eval=20 \
    callbacks.early_stopping.patience=20 \
    run_name=tio_forces_aux_test \
    seed=42

echo "End time: $(date)"
echo "=== Done ==="
