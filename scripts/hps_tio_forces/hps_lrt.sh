#!/bin/bash
#$ -N hps_lrt_forces
#$ -q iqtc13.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o /home/g15farris/bin/bayesaenet/logs/hps_lrt_forces.out
#$ -e /home/g15farris/bin/bayesaenet/logs/hps_lrt_forces.err

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

cd /home/g15farris/bin/bayesaenet

export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4

# Disabled mixed precision for LRT - causes NaN in variational parameters
python -m bnn_aenet.tasks.hpsearch \
    hpsearch=bnn_lrt_forces \
    datamodule=TiO_Forces \
    trainer.accelerator=gpu \
    trainer.devices=1
