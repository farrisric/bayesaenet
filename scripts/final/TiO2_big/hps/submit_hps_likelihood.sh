#!/bin/bash
# Submit BNN_Forces_Likelihood HPS for TiO2_big (100% dataset)
# Alpha is FIXED at 0.1 (not optimized)
# Note: FO (Flipout) incompatible - input-dependent reparameterization fails with trace/replay
#
# Directory convention:
#   Optuna DBs: bnn_aenet/results/TiO2_big/{study_name}.db
#   SGE logs:   logs/hps/TiO2_big_hps_{model}_likelihood.{out,err}

BASEDIR="/home/g15farris/bin/bayesaenet"
cd ${BASEDIR}
mkdir -p logs/hps

echo "=== TiO2_big BNN_Forces_Likelihood HPS (iqtc13) ==="

# LRT
qsub << 'HEREDOC'
#!/bin/bash
#$ -N hps_lrt_lik
#$ -q iqtc13.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o /home/g15farris/bin/bayesaenet/logs/hps/TiO2_big_hps_lrt_likelihood.out
#$ -e /home/g15farris/bin/bayesaenet/logs/hps/TiO2_big_hps_lrt_likelihood.err

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
    datamodule=TiO_Forces_Data100 \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    hpsearch.results_subdir=TiO2_big \
    hpsearch.study.study_name=bnn_lrt_forces_likelihood \
    'tags=["TiO2_big", "lrt", "likelihood", "hps"]'
HEREDOC

echo "  LRT HPS submitted (iqtc13)"

# RAD (uses 16-mixed for speed)
qsub << 'HEREDOC'
#!/bin/bash
#$ -N hps_rad_lik
#$ -q iqtc13.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o /home/g15farris/bin/bayesaenet/logs/hps/TiO2_big_hps_rad_likelihood.out
#$ -e /home/g15farris/bin/bayesaenet/logs/hps/TiO2_big_hps_rad_likelihood.err

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
    datamodule=TiO_Forces_Data100 \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    +trainer.precision=16-mixed \
    hpsearch.results_subdir=TiO2_big \
    hpsearch.study.study_name=bnn_rad_forces_likelihood \
    'tags=["TiO2_big", "rad", "likelihood", "hps"]'
HEREDOC

echo "  RAD HPS submitted (iqtc13)"
echo "=== Done (LRT + RAD; FO incompatible with likelihood) ==="
