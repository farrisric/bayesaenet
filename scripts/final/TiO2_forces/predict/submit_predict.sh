#!/bin/bash
# Submit prediction jobs for all force-trained models on TiO2
# NN on iqtc10, BNNs on iqtc13

cd /home/g15farris/bin/bayesaenet
mkdir -p bnn_aenet/logs/forces_pred
mkdir -p logs/predict

DATA_DIR="/home/g15farris/bin/bayesaenet/data/TiO/train_forces.in"

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

echo "Submitting prediction jobs..."

# NN Forces - iqtc10 (no module load cuda needed)
qsub << 'HEREDOC'
#!/bin/bash
#$ -N pred_nn_f
#$ -q iqtc10.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o logs/predict/pred_nn_forces.out
#$ -e logs/predict/pred_nn_forces.err

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

python scripts/final/TiO2_forces/predict/predict_forces.py \
    --model-type nn \
    --runs-dir bnn_aenet/logs/nn_forces/nn_forces_train \
    --output-dir bnn_aenet/logs/forces_pred/nn \
    --data-dir data/TiO/train_forces.in \
    --subsets train val test \
    --device gpu \
    --batch-size 64
HEREDOC

echo "NN Forces prediction submitted"

# LRT Forces - iqtc13
qsub << 'HEREDOC'
#!/bin/bash
#$ -N pred_lrt_f
#$ -q iqtc13.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o logs/predict/pred_lrt_forces.out
#$ -e logs/predict/pred_lrt_forces.err

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

python scripts/final/TiO2_forces/predict/predict_forces.py \
    --model-type lrt \
    --runs-dir bnn_aenet/logs/lrt_forces/lrt_forces_train \
    --output-dir bnn_aenet/logs/forces_pred/lrt \
    --data-dir data/TiO/train_forces.in \
    --subsets train val test \
    --device gpu \
    --batch-size 32 \
    --mc-samples 20
HEREDOC

echo "LRT Forces prediction submitted"

# FO Forces - iqtc13
qsub << 'HEREDOC'
#!/bin/bash
#$ -N pred_fo_f
#$ -q iqtc13.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o logs/predict/pred_fo_forces.out
#$ -e logs/predict/pred_fo_forces.err

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

python scripts/final/TiO2_forces/predict/predict_forces.py \
    --model-type fo \
    --runs-dir bnn_aenet/logs/fo_forces/fo_forces_train \
    --output-dir bnn_aenet/logs/forces_pred/fo \
    --data-dir data/TiO/train_forces.in \
    --subsets train val test \
    --device gpu \
    --batch-size 32 \
    --mc-samples 20
HEREDOC

echo "FO Forces prediction submitted"

# RAD Forces - iqtc13
qsub << 'HEREDOC'
#!/bin/bash
#$ -N pred_rad_f
#$ -q iqtc13.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o logs/predict/pred_rad_forces.out
#$ -e logs/predict/pred_rad_forces.err

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

python scripts/final/TiO2_forces/predict/predict_forces.py \
    --model-type rad \
    --runs-dir bnn_aenet/logs/rad_forces/rad_forces_train \
    --output-dir bnn_aenet/logs/forces_pred/rad \
    --data-dir data/TiO/train_forces.in \
    --subsets train val test \
    --device gpu \
    --batch-size 32 \
    --mc-samples 20
HEREDOC

echo "RAD Forces prediction submitted"
echo "All prediction jobs submitted! Check status with: qstat"
