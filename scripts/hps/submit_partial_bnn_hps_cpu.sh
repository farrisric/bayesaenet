#!/bin/bash
# Submit HPS jobs for Partial BNN models (first + last layer) on TiO2 100% with forces
# Runs on CPU using iqtc12

cd /home/g15farris/bin/bayesaenet
mkdir -p logs/hps

CONDA_INIT='. /etc/profile
__conda_setup="$('\''/aplic/anaconda/2020.02/bin/conda'\'' '\''shell.bash'\'' '\''hook'\'' 2> /dev/null)"
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
conda activate bnn'

echo "Submitting Partial BNN HPS jobs on CPU (iqtc12)..."

# Models: {lrt, fo, rad} x {first, last} = 6 jobs
MODELS=("partial_lrt_first" "partial_lrt_last" "partial_fo_first" "partial_fo_last" "partial_rad_first" "partial_rad_last")
SHORT=("plrt_fi" "plrt_la" "pfo_fi" "pfo_la" "prad_fi" "prad_la")

for i in "${!MODELS[@]}"; do
    MODEL=${MODELS[$i]}
    NAME=${SHORT[$i]}
    
    qsub << HEREDOC
#!/bin/bash
#\$ -N hps_${NAME}
#\$ -q iqtc12.q
#\$ -pe smp 8
#\$ -cwd
#\$ -o logs/hps/hps_${MODEL}_forces.out
#\$ -e logs/hps/hps_${MODEL}_forces.err

. /etc/profile

__conda_setup="\$('/aplic/anaconda/2020.02/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ \$? -eq 0 ]; then
    eval "\$__conda_setup"
else
    if [ -f "/aplic/anaconda/2024.10/etc/profile.d/conda.sh" ]; then
        . "/aplic/anaconda/2024.10/etc/profile.d/conda.sh"
    else
        export PATH="/aplic/anaconda/2024.10/bin:\$PATH"
    fi
fi
unset __conda_setup

conda activate bnn

cd /home/g15farris/bin/bayesaenet
python bnn_aenet/tasks/hpsearch.py hpsearch=${MODEL}_forces datamodule=TiO_Data100 trainer.accelerator=cpu
HEREDOC

done

echo "All Partial BNN HPS jobs submitted!"
echo "Check status with: qstat"
