#!/bin/bash
#$ -N bnn_f_hps
#$ -pe smp 1
#$ -l iqtcgpu=1
#$ -q iqtc10.q
#$ -S /bin/bash
#$ -cwd
#$ -o hps_forces_aux.out
#$ -e hps_forces_aux.err

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

echo "=== BNN_Forces_Aux Hyperparameter Search ==="
echo "Start time: $(date)"
echo "Worker: HPS Worker 0"

# Create results directory if it doesn't exist
mkdir -p bnn_aenet/results/bayesian

python bnn_aenet/tasks/hpsearch.py \
    hpsearch=bnn_forces_aux \
    datamodule.data_dir=data/TiO/train_forces.in \
    datamodule.device=cuda \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    hpsearch.n_trials=20 \
    seed=42

echo "End time: $(date)"
echo "=== HPS Done ==="
