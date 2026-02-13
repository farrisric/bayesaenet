#!/bin/bash
# Submit HPS jobs for TiO2_small (20% dataset, force-trained models)
# Each job runs 30 Optuna trials on GPU
# NN on iqtc13 (no module load cuda needed), BNNs split across iqtc10 + iqtc13
#
# Directory convention:
#   Optuna DBs: bnn_aenet/results/TiO2_small/{study_name}.db
#   SGE logs:   logs/hps/TiO2_small_hps_{model}.{out,err}

BASEDIR="/home/g15farris/bin/bayesaenet"
cd ${BASEDIR}
mkdir -p log/hps

echo "=== TiO2_small HPS Jobs ==="

# NN HPS - iqtc13 (no module load cuda needed)
qsub << 'HEREDOC'
#!/bin/bash
#$ -N hps_nn_sm
#$ -q iqtc13.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o /home/g15farris/bin/bayesaenet/log/hps/TiO2_small_hps_nn.out
#$ -e /home/g15farris/bin/bayesaenet/log/hps/TiO2_small_hps_nn.err

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
    hpsearch=nn \
    datamodule=TiO_Forces_Data20 \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    +trainer.precision=16-mixed \
    hpsearch.results_subdir=TiO2_small \
    hpsearch.study.study_name=nn_small \
    'tags=["TiO2_small", "nn", "hps"]'
HEREDOC

echo "  NN HPS submitted (iqtc13)"

# LRT HPS (no mixed precision) - iqtc10
qsub << 'HEREDOC'
#!/bin/bash
#$ -N hps_lrt_sm
#$ -q iqtc10.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o /home/g15farris/bin/bayesaenet/log/hps/TiO2_small_hps_lrt.out
#$ -e /home/g15farris/bin/bayesaenet/log/hps/TiO2_small_hps_lrt.err

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
    hpsearch=bnn_lrt \
    datamodule=TiO_Forces_Data20 \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    hpsearch.results_subdir=TiO2_small \
    hpsearch.study.study_name=lrt_small \
    'tags=["TiO2_small", "lrt", "hps"]'
HEREDOC

echo "  LRT HPS submitted (iqtc10)"

# RAD HPS - iqtc13
qsub << 'HEREDOC'
#!/bin/bash
#$ -N hps_rad_sm
#$ -q iqtc13.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o /home/g15farris/bin/bayesaenet/log/hps/TiO2_small_hps_rad.out
#$ -e /home/g15farris/bin/bayesaenet/log/hps/TiO2_small_hps_rad.err

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
    hpsearch=bnn_rad \
    datamodule=TiO_Forces_Data20 \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    +trainer.precision=16-mixed \
    hpsearch.results_subdir=TiO2_small \
    hpsearch.study.study_name=rad_small \
    'tags=["TiO2_small", "rad", "hps"]'
HEREDOC

echo "  RAD HPS submitted (iqtc13)"
echo "=== All TiO2_small HPS jobs submitted ==="
