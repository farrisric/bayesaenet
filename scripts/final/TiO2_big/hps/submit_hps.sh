#!/bin/bash
# Submit HPS jobs for TiO2_big (100% dataset, force-trained models)
# Each job runs 30 Optuna trials on GPU
#
# Directory convention:
#   Optuna DBs: bnn_aenet/results/TiO2_big/{method}.db
#   SGE logs:   logs/hps/TiO2_big_hps_{model}.{out,err}

BASEDIR="/home/g15farris/bin/bayesaenet"
cd ${BASEDIR}
mkdir -p logs/hps

echo "=== TiO2_big HPS Jobs ==="

# NN HPS - iqtc13
qsub << 'HEREDOC'
#!/bin/bash
#$ -N hps_nn_bg
#$ -q iqtc13.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o /home/g15farris/bin/bayesaenet/logs/hps/TiO2_big_hps_nn.out
#$ -e /home/g15farris/bin/bayesaenet/logs/hps/TiO2_big_hps_nn.err

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

export OMP_NUM_THREADS=4
export TMPDIR=/tmp/g15farris
export PYTHONPATH=/home/g15farris/bin/bayesaenet:$PYTHONPATH
cd /home/g15farris/bin/bayesaenet

python -m bnn_aenet.tasks.hpsearch \
    hpsearch=nn_forces \
    datamodule=TiO_Forces \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    +trainer.precision=16-mixed \
    hpsearch.study.study_name=nn \
    'tags=["TiO2_big", "nn", "hps"]'
HEREDOC

echo "  NN HPS submitted (iqtc13)"

# LRT HPS (no mixed precision) - iqtc10
qsub << 'HEREDOC'
#!/bin/bash
#$ -N hps_lrt_bg
#$ -q iqtc10.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o /home/g15farris/bin/bayesaenet/logs/hps/TiO2_big_hps_lrt.out
#$ -e /home/g15farris/bin/bayesaenet/logs/hps/TiO2_big_hps_lrt.err

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

python -m bnn_aenet.tasks.hpsearch \
    hpsearch=bnn_lrt_forces \
    datamodule=TiO_Forces \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    hpsearch.study.study_name=lrt \
    'tags=["TiO2_big", "lrt", "hps"]'
HEREDOC

echo "  LRT HPS submitted (iqtc10)"

# FO HPS - iqtc13
qsub << 'HEREDOC'
#!/bin/bash
#$ -N hps_fo_bg
#$ -q iqtc13.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o /home/g15farris/bin/bayesaenet/logs/hps/TiO2_big_hps_fo.out
#$ -e /home/g15farris/bin/bayesaenet/logs/hps/TiO2_big_hps_fo.err

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

python -m bnn_aenet.tasks.hpsearch \
    hpsearch=bnn_fo_forces \
    datamodule=TiO_Forces \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    +trainer.precision=16-mixed \
    hpsearch.study.study_name=fo \
    'tags=["TiO2_big", "fo", "hps"]'
HEREDOC

echo "  FO HPS submitted (iqtc13)"

# RAD HPS - iqtc13
qsub << 'HEREDOC'
#!/bin/bash
#$ -N hps_rad_bg
#$ -q iqtc13.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o /home/g15farris/bin/bayesaenet/logs/hps/TiO2_big_hps_rad.out
#$ -e /home/g15farris/bin/bayesaenet/logs/hps/TiO2_big_hps_rad.err

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

python -m bnn_aenet.tasks.hpsearch \
    hpsearch=bnn_rad_forces \
    datamodule=TiO_Forces \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    +trainer.precision=16-mixed \
    hpsearch.study.study_name=rad \
    'tags=["TiO2_big", "rad", "hps"]'
HEREDOC

echo "  RAD HPS submitted (iqtc13)"
echo "=== All TiO2_big HPS jobs submitted ==="
