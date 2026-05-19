#!/bin/bash
#SBATCH --job-name=pred_rad_sm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --partition=iqtc13.q
#SBATCH --output=/home/g15farris/bin/bayesaenet/log/predict/TiO2_small_pred_rad.out
#SBATCH --error=/home/g15farris/bin/bayesaenet/log/predict/TiO2_small_pred_rad.err


ulimit -l unlimited
ulimit -s unlimited

. /etc/profile
__conda_setup="$('/aplic/anaconda/2024.10/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
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
source activate /home/g15farris/.conda/envs/bnn

export OMP_NUM_THREADS=4
export TMPDIR=/tmp/g15farris
export PYTHONPATH=/home/g15farris/bin/bayesaenet:$PYTHONPATH
cd /home/g15farris/bin/bayesaenet
mkdir -p log/predict
mkdir -p bnn_aenet/logs/TiO2_small/pred/runs/rad

python -m bnn_aenet.tasks.predict_forces \
    --model-type rad \
    --runs-dir bnn_aenet/logs/TiO2_small/train/runs/rad \
    --output-dir bnn_aenet/logs/TiO2_small/pred/runs/rad \
    --data-dir data/TiO/train_forces.in \
    --split-config Data20 \
    --subsets train val test \
    --device gpu \
    --batch-size 128 \
    --mc-samples 20
