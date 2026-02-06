#!/bin/bash
# Submit training jobs for best HPS models to SGE cluster
#
# Usage:
#   ./scripts/submit_best_training.sh
#   ./scripts/submit_best_training.sh --dataset TiO_Data20

DATASET="${1:-TiO_Data100}"
EPOCHS="${2:-10000}"

cd "$(dirname "$0")/.."

# Create job scripts directory
mkdir -p scripts/best_training

# Generate and submit jobs for each model
for model in nn lrt fo rad; do
    JOB_SCRIPT="scripts/best_training/train_best_${model}.sh"
    
    # Set queue and precision based on model
    if [ "$model" == "lrt" ]; then
        QUEUE="iqtc13.q"
        PRECISION=""  # No mixed precision for LRT
    elif [ "$model" == "rad" ]; then
        QUEUE="iqtc10.q"
        PRECISION="+trainer.precision=16-mixed"
    else
        QUEUE="iqtc13.q"
        PRECISION="+trainer.precision=16-mixed"
    fi
    
    cat > "$JOB_SCRIPT" << EOF
#!/bin/bash
#\$ -N train_best_${model}
#\$ -q ${QUEUE}
#\$ -l iqtcgpu=1
#\$ -pe smp 4
#\$ -S /bin/bash
#\$ -cwd
#\$ -o /home/g15farris/bin/bayesaenet/logs/train_best_${model}.out
#\$ -e /home/g15farris/bin/bayesaenet/logs/train_best_${model}.err

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

module load cuda/12.4
conda activate bnn

cd /home/g15farris/bin/bayesaenet

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4

python scripts/train_best_model.py \\
    --model ${model} \\
    --dataset ${DATASET} \\
    --epochs ${EPOCHS}
EOF

    chmod +x "$JOB_SCRIPT"
    echo "Submitting ${model}..."
    qsub "$JOB_SCRIPT"
done

echo ""
echo "All jobs submitted. Check status with: qstat -u \$USER"
