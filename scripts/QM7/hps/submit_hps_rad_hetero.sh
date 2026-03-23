#!/bin/bash
#$ -N hps_rad_het_qm7
#$ -q iqtc13.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o /home/g15farris/bin/bayesaenet/log/hps/QM7_hps_rad_hetero.out
#$ -e /home/g15farris/bin/bayesaenet/log/hps/QM7_hps_rad_hetero.err

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

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export OMP_NUM_THREADS=4
export TMPDIR=/tmp/g15farris
export PYTHONPATH=/home/g15farris/bin/bayesaenet:$PYTHONPATH
cd /home/g15farris/bin/bayesaenet

python -m bnn_aenet.tasks.hpsearch \
    hpsearch=bnn_rad_hetero \
    datamodule=QM7_Data100 \
    dataset=QM7 \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    +trainer.benchmark=True \
    +trainer.log_every_n_steps=10 \
    +trainer.precision=16-mixed \
    hpsearch.results_subdir=QM7 \
    hpsearch.study.study_name=bnn_rad_hetero_qm7 \
    'tags=["QM7", "rad", "hetero", "hps"]'
