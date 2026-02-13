#!/bin/bash
# Submit BNN_Forces_Hetero HPS for TiO2_big (100% dataset)
# NoiseNet replaces obs_scale / scale_force with input-dependent noise.
# Search space: lr, mc_samples_train, prior_scale, q_scale,
#               noise_min, batch_size
#
# Directory convention:
#   Optuna DBs: bnn_aenet/logs/TiO2_big/{study_name}.db
#   SGE logs:   log/hps/TiO2_big_hps_{model}_hetero.{out,err}

BASEDIR="/home/g15farris/bin/bayesaenet"
cd ${BASEDIR}
mkdir -p log/hps

echo "=== TiO2_big BNN_Forces_Hetero HPS ==="

# LRT hetero on iqtc13 (no mixed precision)
qsub << 'HEREDOC'
#!/bin/bash
#$ -N hps_lrt_het_bg
#$ -q iqtc13.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o /home/g15farris/bin/bayesaenet/log/hps/TiO2_big_hps_lrt_hetero.out
#$ -e /home/g15farris/bin/bayesaenet/log/hps/TiO2_big_hps_lrt_hetero.err

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
    hpsearch=bnn_lrt_hetero \
    datamodule=TiO_Forces_Data100 \
    dataset=TiO2_big \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    hpsearch.results_subdir=TiO2_big \
    hpsearch.study.study_name=bnn_lrt_hetero \
    'tags=["TiO2_big", "lrt", "hetero", "hps"]'
HEREDOC

echo "  LRT hetero HPS submitted (iqtc13)"

# RAD hetero on iqtc10 (uses 16-mixed for speed)
qsub << 'HEREDOC'
#!/bin/bash
#$ -N hps_rad_het_bg
#$ -q iqtc10.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o /home/g15farris/bin/bayesaenet/log/hps/TiO2_big_hps_rad_hetero.out
#$ -e /home/g15farris/bin/bayesaenet/log/hps/TiO2_big_hps_rad_hetero.err

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
    hpsearch=bnn_rad_hetero \
    datamodule=TiO_Forces_Data100 \
    dataset=TiO2_big \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    +trainer.precision=16-mixed \
    hpsearch.results_subdir=TiO2_big \
    hpsearch.study.study_name=bnn_rad_hetero \
    'tags=["TiO2_big", "rad", "hetero", "hps"]'
HEREDOC

echo "  RAD hetero HPS submitted (iqtc10)"
echo "=== Done (LRT iqtc13 + RAD iqtc10) ==="
