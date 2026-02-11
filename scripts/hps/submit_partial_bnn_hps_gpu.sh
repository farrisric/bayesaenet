#!/bin/bash
# Submit HPS jobs for Partial BNN models (first + last layer) on TiO2 with forces
# GPU version - replaces CPU (iqtc12) jobs for faster execution
# Distributes across iqtc13 and iqtc10

cd /home/g15farris/bin/bayesaenet
mkdir -p logs/hps

echo "Submitting Partial BNN HPS jobs on GPU..."

# Models: {lrt, fo, rad} x {first, last} = 6 jobs
# Split across iqtc13 (4 jobs) and iqtc10 (2 jobs)

# 1. Partial LRT First (iqtc13)
qsub << 'HEREDOC'
#!/bin/bash
#$ -N hps_plrt_fi
#$ -q iqtc13.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o logs/hps/hps_partial_lrt_first_forces.out
#$ -e logs/hps/hps_partial_lrt_first_forces.err

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
module load cuda/12.4

export OMP_NUM_THREADS=4
export TMPDIR=/tmp/g15farris
export PYTHONPATH=/home/g15farris/bin/bayesaenet:$PYTHONPATH
cd /home/g15farris/bin/bayesaenet

python -m bnn_aenet.tasks.hpsearch hpsearch=partial_lrt_first_forces datamodule=TiO_Forces trainer.accelerator=gpu trainer.devices=1
HEREDOC

# 2. Partial LRT Last (iqtc13)
qsub << 'HEREDOC'
#!/bin/bash
#$ -N hps_plrt_la
#$ -q iqtc13.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o logs/hps/hps_partial_lrt_last_forces.out
#$ -e logs/hps/hps_partial_lrt_last_forces.err

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
module load cuda/12.4

export OMP_NUM_THREADS=4
export TMPDIR=/tmp/g15farris
export PYTHONPATH=/home/g15farris/bin/bayesaenet:$PYTHONPATH
cd /home/g15farris/bin/bayesaenet

python -m bnn_aenet.tasks.hpsearch hpsearch=partial_lrt_last_forces datamodule=TiO_Forces trainer.accelerator=gpu trainer.devices=1
HEREDOC

# 3. Partial FO First (iqtc13)
qsub << 'HEREDOC'
#!/bin/bash
#$ -N hps_pfo_fi
#$ -q iqtc13.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o logs/hps/hps_partial_fo_first_forces.out
#$ -e logs/hps/hps_partial_fo_first_forces.err

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
module load cuda/12.4

export OMP_NUM_THREADS=4
export TMPDIR=/tmp/g15farris
export PYTHONPATH=/home/g15farris/bin/bayesaenet:$PYTHONPATH
cd /home/g15farris/bin/bayesaenet

python -m bnn_aenet.tasks.hpsearch hpsearch=partial_fo_first_forces datamodule=TiO_Forces trainer.accelerator=gpu trainer.devices=1
HEREDOC

# 4. Partial FO Last (iqtc13)
qsub << 'HEREDOC'
#!/bin/bash
#$ -N hps_pfo_la
#$ -q iqtc13.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o logs/hps/hps_partial_fo_last_forces.out
#$ -e logs/hps/hps_partial_fo_last_forces.err

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
module load cuda/12.4

export OMP_NUM_THREADS=4
export TMPDIR=/tmp/g15farris
export PYTHONPATH=/home/g15farris/bin/bayesaenet:$PYTHONPATH
cd /home/g15farris/bin/bayesaenet

python -m bnn_aenet.tasks.hpsearch hpsearch=partial_fo_last_forces datamodule=TiO_Forces trainer.accelerator=gpu trainer.devices=1
HEREDOC

# 5. Partial RAD First (iqtc10)
qsub << 'HEREDOC'
#!/bin/bash
#$ -N hps_prad_fi
#$ -q iqtc10.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o logs/hps/hps_partial_rad_first_forces.out
#$ -e logs/hps/hps_partial_rad_first_forces.err

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

python -m bnn_aenet.tasks.hpsearch hpsearch=partial_rad_first_forces datamodule=TiO_Forces trainer.accelerator=gpu trainer.devices=1
HEREDOC

# 6. Partial RAD Last (iqtc10)
qsub << 'HEREDOC'
#!/bin/bash
#$ -N hps_prad_la
#$ -q iqtc10.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o logs/hps/hps_partial_rad_last_forces.out
#$ -e logs/hps/hps_partial_rad_last_forces.err

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

python -m bnn_aenet.tasks.hpsearch hpsearch=partial_rad_last_forces datamodule=TiO_Forces trainer.accelerator=gpu trainer.devices=1
HEREDOC

echo "All 6 Partial BNN HPS GPU jobs submitted!"
echo "  4 on iqtc13.q (LRT first/last, FO first/last)"
echo "  2 on iqtc10.q (RAD first/last)"
echo "Check status with: qstat"
