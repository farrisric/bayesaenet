#!/bin/bash
#$ -N arch_tests_cpu
#$ -q iqtc12.q
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o /home/g15farris/bin/bayesaenet/logs/arch_tests_cpu.out
#$ -e /home/g15farris/bin/bayesaenet/logs/arch_tests_cpu.err

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

export OMP_NUM_THREADS=4

python scripts/tests/test_architecture_cpu.py
