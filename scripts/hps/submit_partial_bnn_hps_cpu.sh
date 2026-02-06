#!/bin/bash
# Submit HPS jobs for all Partial BNN models on TiO2 with forces
# Runs on CPU using iqtc12 (60 cores available)

cd /home/g15farris/bin/bayesaenet

# Create log directory
mkdir -p logs/hps

echo "Submitting HPS jobs for Partial BNN models on CPU (iqtc12)..."

# Job 1: Partial LRT Last
qsub << 'EOF1'
#!/bin/bash
#$ -N hps_plrt_last
#$ -q iqtc12.q
#$ -pe smp 8
#$ -cwd
#$ -o logs/hps/hps_partial_lrt_last.out
#$ -e logs/hps/hps_partial_lrt_last.err

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

cd /home/g15farris/bin/bayesaenet
python bnn_aenet/tasks/hpsearch.py hpsearch=partial_lrt_last_forces datamodule=TiO trainer.accelerator=cpu
EOF1

# Job 2: Partial LRT First+Last
qsub << 'EOF2'
#!/bin/bash
#$ -N hps_plrt_fl
#$ -q iqtc12.q
#$ -pe smp 8
#$ -cwd
#$ -o logs/hps/hps_partial_lrt_first_last.out
#$ -e logs/hps/hps_partial_lrt_first_last.err

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

cd /home/g15farris/bin/bayesaenet
python bnn_aenet/tasks/hpsearch.py hpsearch=partial_lrt_first_last_forces datamodule=TiO trainer.accelerator=cpu
EOF2

# Job 3: Partial Flipout Last
qsub << 'EOF3'
#!/bin/bash
#$ -N hps_pfo_last
#$ -q iqtc12.q
#$ -pe smp 8
#$ -cwd
#$ -o logs/hps/hps_partial_fo_last.out
#$ -e logs/hps/hps_partial_fo_last.err

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

cd /home/g15farris/bin/bayesaenet
python bnn_aenet/tasks/hpsearch.py hpsearch=partial_fo_last_forces datamodule=TiO trainer.accelerator=cpu
EOF3

# Job 4: Partial Flipout First+Last
qsub << 'EOF4'
#!/bin/bash
#$ -N hps_pfo_fl
#$ -q iqtc12.q
#$ -pe smp 8
#$ -cwd
#$ -o logs/hps/hps_partial_fo_first_last.out
#$ -e logs/hps/hps_partial_fo_first_last.err

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

cd /home/g15farris/bin/bayesaenet
python bnn_aenet/tasks/hpsearch.py hpsearch=partial_fo_first_last_forces datamodule=TiO trainer.accelerator=cpu
EOF4

# Job 5: Partial Radial Last
qsub << 'EOF5'
#!/bin/bash
#$ -N hps_prad_last
#$ -q iqtc12.q
#$ -pe smp 8
#$ -cwd
#$ -o logs/hps/hps_partial_rad_last.out
#$ -e logs/hps/hps_partial_rad_last.err

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

cd /home/g15farris/bin/bayesaenet
python bnn_aenet/tasks/hpsearch.py hpsearch=partial_rad_last_forces datamodule=TiO trainer.accelerator=cpu
EOF5

echo "All HPS jobs submitted to iqtc12 (CPU)!"
echo "Check status with: qstat"
echo "View logs in: logs/hps/"
