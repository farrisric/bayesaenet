#!/bin/bash
# Post-HPS workflow: Analyze results, train best models, evaluate
#
# Usage:
#   ./scripts/run_post_hps.sh              # Analyze all HPS results
#   ./scripts/run_post_hps.sh --train      # Also train best models
#   ./scripts/run_post_hps.sh --evaluate   # Evaluate existing checkpoints

set -e

cd "$(dirname "$0")/.."

# Activate conda environment
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

echo "========================================"
echo "Step 1: Analyzing HPS Results"
echo "========================================"
python scripts/analyze_hps_results.py --all --export

if [[ "$*" == *"--train"* ]]; then
    echo ""
    echo "========================================"
    echo "Step 2: Training Best Models"
    echo "========================================"
    
    # Train each model with best hyperparameters
    for model in nn lrt fo rad; do
        echo ""
        echo "Training ${model^^}..."
        python scripts/train_best_model.py --model $model --dataset TiO_Data100 --epochs 10000
    done
fi

if [[ "$*" == *"--evaluate"* ]]; then
    echo ""
    echo "========================================"
    echo "Step 3: Evaluating Models"
    echo "========================================"
    
    # Find latest checkpoints and evaluate
    for model in nn lrt fo rad; do
        # Find the most recent checkpoint
        ckpt=$(find bnn_aenet/logs/${model}_forces -name "*.ckpt" -type f 2>/dev/null | head -1)
        
        if [ -n "$ckpt" ]; then
            echo ""
            echo "Evaluating ${model^^}: $ckpt"
            python scripts/evaluate_model.py \
                --checkpoint "$ckpt" \
                --model $model \
                --dataset TiO_Data100 \
                --output "results/${model}_test_metrics.csv"
        else
            echo "No checkpoint found for $model"
        fi
    done
fi

echo ""
echo "========================================"
echo "Done!"
echo "========================================"
