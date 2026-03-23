#!/bin/bash
# Run analysis for TiO2_big force-trained models
# Can run on CPU (no GPU needed), but using cluster for consistency
#
# Directory convention:
#   Input:  bnn_aenet/logs/TiO2_big/forces_pred/
#   Output: plots/TiO2_big/

cd /home/g15farris/bin/bayesaenet
mkdir -p logs/analysis

echo "=== TiO2_big Analysis ==="

qsub << 'HEREDOC'
#!/bin/bash
#$ -N anal_big
#$ -q iqtc10.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o /home/g15farris/bin/bayesaenet/logs/analysis/TiO2_big_analysis.out
#$ -e /home/g15farris/bin/bayesaenet/logs/analysis/TiO2_big_analysis.err

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

python -m bnn_aenet.tasks.analyze \
    --pred-dir bnn_aenet/logs/TiO2_big/forces_pred \
    --output-dir plots/TiO2_big \
    --train-dir bnn_aenet/logs/TiO2_big
HEREDOC

echo "  Analysis submitted"
echo "=== Done ==="
