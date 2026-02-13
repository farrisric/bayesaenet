#!/bin/bash
# Run analysis for TiO2_small force-trained models
# Can run on CPU (no GPU needed), but using cluster for consistency
#
# Directory convention:
#   Input:  task/predict/runs/TiO2_small/
#   Output: plots/TiO2_small/

cd /home/g15farris/bin/bayesaenet
mkdir -p log/analysis

echo "=== TiO2_small Analysis ==="

qsub << 'HEREDOC'
#!/bin/bash
#$ -N anal_sm
#$ -q iqtc10.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o /home/g15farris/bin/bayesaenet/log/analysis/TiO2_small_analysis.out
#$ -e /home/g15farris/bin/bayesaenet/log/analysis/TiO2_small_analysis.err

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
    --pred-dir task/predict/runs/TiO2_small \
    --output-dir plots/TiO2_small \
    --train-dir task/train/runs
HEREDOC

echo "  Analysis submitted"
echo "=== Done ==="
