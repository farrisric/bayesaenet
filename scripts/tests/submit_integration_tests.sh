#!/bin/bash
#$ -N integration_tests
#$ -q iqtc12.q
#$ -pe smp 16
#$ -S /bin/bash
#$ -cwd
#$ -o /home/g15farris/bin/bayesaenet/logs/integration_tests.out
#$ -e /home/g15farris/bin/bayesaenet/logs/integration_tests.err

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

export OMP_NUM_THREADS=16
export PYTHONPATH="${PYTHONPATH}:/home/g15farris/bin/bayesaenet"

python scripts/tests/test_training_integration.py
