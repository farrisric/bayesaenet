import os
import numpy as np

# Get current working directory
cwd = os.path.dirname(os.path.abspath(__file__))

# Best NN Forces HPS parameters (trial #15, total_rmse=1.3276)
LR = 0.0002497627476497723
WEIGHT_DECAY = 0.0024552339549495416
FORCE_WEIGHT = 0.5081669394595789
BATCH_SIZE = 128

# SGE script template
template = """#!/bin/bash
#$ -N nn_forces_train_{i}
#$ -q iqtc13.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o /home/g15farris/bin/bayesaenet/logs/nn_forces/nn_forces_train_{i}.out
#$ -e /home/g15farris/bin/bayesaenet/logs/nn_forces/nn_forces_train_{i}.err

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
cd /home/g15farris/bin/bayesaenet

python bnn_aenet/tasks/train.py \\
    experiment=nn_forces \\
    datamodule=TiO \\
    trainer.accelerator=gpu \\
    trainer.devices=1 \\
    trainer.min_epochs=50000 \\
    trainer.max_epochs=50000 \\
    trainer.deterministic=False \\
    task_name=nn_forces_train \\
    run_name=nn_forces_train_{i} \\
    datamodule.batch_size={batch_size} \\
    model.optimizer.lr={lr} \\
    model.optimizer.weight_decay={weight_decay} \\
    model.force_weight={force_weight} \\
    callbacks.model_checkpoint.monitor=total_rmse/val \\
    callbacks.early_stopping.monitor=total_rmse/val \\
    callbacks.early_stopping.patience=500 \\
    seed={seed}
"""

# Seed numpy RNG
np.random.seed(42)

# Create log directory
os.makedirs("/home/g15farris/bin/bayesaenet/logs/nn_forces", exist_ok=True)

# Generate 10 scripts
for i in range(10):
    seed = int(np.random.randint(0, 1e6))
    script_content = template.format(
        i=i,
        seed=seed,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        force_weight=FORCE_WEIGHT,
        batch_size=BATCH_SIZE,
    )
    filename = os.path.join(cwd, f"nn_forces_train_{i}.sh")
    with open(filename, "w") as f:
        f.write(script_content)
    os.chmod(filename, 0o755)
    print(f"Written {filename}")
    os.system(f"qsub {filename}")
