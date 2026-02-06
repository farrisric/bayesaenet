#!/bin/bash
#$ -N lrt_f_aux_test
#$ -pe smp 1
#$ -l iqtcgpu=1
#$ -q iqtc10.q
#$ -S /bin/bash
#$ -cwd
#$ -o lrt_forces_aux_test.out
#$ -e lrt_forces_aux_test.err
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
# Skip CUDA module load for iqtc10 - conda env has CUDA
conda activate bnn
export HYDRA_FULL_ERROR=1
export PYTHONPATH="${PYTHONPATH}:/home/g15farris/bin/bayesaenet/bnn_aenet"
export OMP_NUM_THREADS=1
cd /home/g15farris/bin/bayesaenet

# Test BNN with auxiliary force loss on QM7
# Short run to verify implementation works
python bnn_aenet/tasks/train.py \
    experiment=bnn_lrt_forces_aux \
    trainer.min_epochs=10 \
    trainer.max_epochs=100 \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    trainer.deterministic=False \
    datamodule=QM7 \
    datamodule.device=cuda \
    datamodule.valid_split=100 \
    datamodule.batch_size=32 \
    model.force_weight=1.0 \
    seed=42
