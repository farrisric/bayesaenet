#!/bin/bash
# Submit Partial BNN HPS jobs for TiO2_big (100% dataset, force-trained models)
# 6 jobs: FO/LRT/RAD x first/last layer
# Each job runs 30 Optuna trials on GPU
# LRT partials on iqtc10 (no mixed precision), FO/RAD partials on iqtc13
#
# Directory convention:
#   Optuna DBs: bnn_aenet/results/TiO2_big/{study_name}.db
#   SGE logs:   logs/hps/TiO2_big_hps_partial_{method}_{layer}.{out,err}

BASEDIR="/home/g15farris/bin/bayesaenet"
cd ${BASEDIR}
mkdir -p logs/hps

echo "=== TiO2_big Partial BNN HPS Jobs ==="

# --- FO FIRST ---
qsub << 'HEREDOC'
#!/bin/bash
#$ -N hps_pfo_f
#$ -q iqtc13.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o /home/g15farris/bin/bayesaenet/logs/hps/TiO2_big_hps_partial_fo_first.out
#$ -e /home/g15farris/bin/bayesaenet/logs/hps/TiO2_big_hps_partial_fo_first.err

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
    hpsearch=partial_fo_first_forces \
    datamodule=TiO_Forces \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    +trainer.precision=16-mixed \
    hpsearch.study.study_name=partial_fo_first \
    'tags=["TiO2_big", "partial_fo_first", "hps"]'
HEREDOC

echo "  Partial FO First submitted (iqtc13)"

# --- FO LAST ---
qsub << 'HEREDOC'
#!/bin/bash
#$ -N hps_pfo_l
#$ -q iqtc13.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o /home/g15farris/bin/bayesaenet/logs/hps/TiO2_big_hps_partial_fo_last.out
#$ -e /home/g15farris/bin/bayesaenet/logs/hps/TiO2_big_hps_partial_fo_last.err

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
    hpsearch=partial_fo_last_forces \
    datamodule=TiO_Forces \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    +trainer.precision=16-mixed \
    hpsearch.study.study_name=partial_fo_last \
    'tags=["TiO2_big", "partial_fo_last", "hps"]'
HEREDOC

echo "  Partial FO Last submitted (iqtc13)"

# --- LRT FIRST (no mixed precision) ---
qsub << 'HEREDOC'
#!/bin/bash
#$ -N hps_plrt_f
#$ -q iqtc10.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o /home/g15farris/bin/bayesaenet/logs/hps/TiO2_big_hps_partial_lrt_first.out
#$ -e /home/g15farris/bin/bayesaenet/logs/hps/TiO2_big_hps_partial_lrt_first.err

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
    hpsearch=partial_lrt_first_forces \
    datamodule=TiO_Forces \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    hpsearch.study.study_name=partial_lrt_first \
    'tags=["TiO2_big", "partial_lrt_first", "hps"]'
HEREDOC

echo "  Partial LRT First submitted (iqtc10)"

# --- LRT LAST (no mixed precision) ---
qsub << 'HEREDOC'
#!/bin/bash
#$ -N hps_plrt_l
#$ -q iqtc10.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o /home/g15farris/bin/bayesaenet/logs/hps/TiO2_big_hps_partial_lrt_last.out
#$ -e /home/g15farris/bin/bayesaenet/logs/hps/TiO2_big_hps_partial_lrt_last.err

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
    hpsearch=partial_lrt_last_forces \
    datamodule=TiO_Forces \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    hpsearch.study.study_name=partial_lrt_last \
    'tags=["TiO2_big", "partial_lrt_last", "hps"]'
HEREDOC

echo "  Partial LRT Last submitted (iqtc10)"

# --- RAD FIRST ---
qsub << 'HEREDOC'
#!/bin/bash
#$ -N hps_prad_f
#$ -q iqtc13.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o /home/g15farris/bin/bayesaenet/logs/hps/TiO2_big_hps_partial_rad_first.out
#$ -e /home/g15farris/bin/bayesaenet/logs/hps/TiO2_big_hps_partial_rad_first.err

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
    hpsearch=partial_rad_first_forces \
    datamodule=TiO_Forces \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    +trainer.precision=16-mixed \
    hpsearch.study.study_name=partial_rad_first \
    'tags=["TiO2_big", "partial_rad_first", "hps"]'
HEREDOC

echo "  Partial RAD First submitted (iqtc13)"

# --- RAD LAST ---
qsub << 'HEREDOC'
#!/bin/bash
#$ -N hps_prad_l
#$ -q iqtc13.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o /home/g15farris/bin/bayesaenet/logs/hps/TiO2_big_hps_partial_rad_last.out
#$ -e /home/g15farris/bin/bayesaenet/logs/hps/TiO2_big_hps_partial_rad_last.err

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
    hpsearch=partial_rad_last_forces \
    datamodule=TiO_Forces \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    +trainer.precision=16-mixed \
    hpsearch.study.study_name=partial_rad_last \
    'tags=["TiO2_big", "partial_rad_last", "hps"]'
HEREDOC

echo "  Partial RAD Last submitted (iqtc13)"
echo "=== All TiO2_big Partial BNN HPS jobs submitted ==="
