#!/bin/bash
# Submit prediction jobs for all force-trained models on TiO2_small (20% data)
# NN on iqtc13, BNNs split across iqtc10 + iqtc13
#
# Directory convention:
#   Predictions: bnn_aenet/logs/TiO2_small/forces_pred/{model}/
#   SGE logs:    logs/predict/TiO2_small_pred_{model}.{out,err}

cd /home/g15farris/bin/bayesaenet
mkdir -p bnn_aenet/logs/TiO2_small/forces_pred
mkdir -p logs/predict

echo "=== TiO2_small Prediction Jobs ==="

# NN Forces - iqtc13
qsub << 'HEREDOC'
#!/bin/bash
#$ -N pred_nn_sm
#$ -q iqtc13.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o /home/g15farris/bin/bayesaenet/logs/predict/TiO2_small_pred_nn.out
#$ -e /home/g15farris/bin/bayesaenet/logs/predict/TiO2_small_pred_nn.err

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

python -m bnn_aenet.tasks.predict_forces \
    --model-type nn \
    --runs-dir bnn_aenet/logs/TiO2_small/nn_train \
    --output-dir bnn_aenet/logs/TiO2_small/forces_pred/nn \
    --data-dir data/TiO/train_forces.in \
    --subsets train val test \
    --device gpu \
    --batch-size 64
HEREDOC

echo "  NN prediction submitted (iqtc13)"

# LRT Forces - iqtc10
qsub << 'HEREDOC'
#!/bin/bash
#$ -N pred_lrt_sm
#$ -q iqtc10.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o /home/g15farris/bin/bayesaenet/logs/predict/TiO2_small_pred_lrt.out
#$ -e /home/g15farris/bin/bayesaenet/logs/predict/TiO2_small_pred_lrt.err

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

python -m bnn_aenet.tasks.predict_forces \
    --model-type lrt \
    --runs-dir bnn_aenet/logs/TiO2_small/lrt_train \
    --output-dir bnn_aenet/logs/TiO2_small/forces_pred/lrt \
    --data-dir data/TiO/train_forces.in \
    --subsets train val test \
    --device gpu \
    --batch-size 32 \
    --mc-samples 20
HEREDOC

echo "  LRT prediction submitted (iqtc10)"

# RAD Forces - iqtc13
qsub << 'HEREDOC'
#!/bin/bash
#$ -N pred_rad_sm
#$ -q iqtc13.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o /home/g15farris/bin/bayesaenet/logs/predict/TiO2_small_pred_rad.out
#$ -e /home/g15farris/bin/bayesaenet/logs/predict/TiO2_small_pred_rad.err

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

python -m bnn_aenet.tasks.predict_forces \
    --model-type rad \
    --runs-dir bnn_aenet/logs/TiO2_small/rad_train \
    --output-dir bnn_aenet/logs/TiO2_small/forces_pred/rad \
    --data-dir data/TiO/train_forces.in \
    --subsets train val test \
    --device gpu \
    --batch-size 32 \
    --mc-samples 20
HEREDOC

echo "  RAD prediction submitted (iqtc13)"
echo "=== All TiO2_small prediction jobs submitted ==="
