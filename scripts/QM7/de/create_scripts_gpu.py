#!/usr/bin/env python
"""Create GPU-enabled training scripts for QM7 Deep Ensemble"""
import os
import numpy as np

# Get current working directory and extract method name
cwd = os.path.dirname(os.path.abspath(__file__))
method = 'de'

# GPU-enabled SLURM script template (based on Job.sh)
template = """#!/bin/bash
#$ -N {method}_train_gpu
#$ -pe smp 1
#$ -l iqtcgpu=1
#$ -q iqtc13.q
#$ -S /bin/bash
#$ -cwd
#$ -o train_{method}_gpu_{i}.out
#$ -e train_{method}_gpu_{i}.err
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
#export CUDA_VISIBLE_DEVICES=`cat $TMPDIR/.gpus`
module load cuda/12.4
conda activate bnn
export PYTHONPATH="${{PYTHONPATH}}:/home/g15farris/bin/bayesaenet/bnn_aenet"
export OMP_NUM_THREADS=1
cd /home/g15farris/bin/bayesaenet

python bnn_aenet/tasks/train.py \\
    trainer.min_epochs=100000 \\
    trainer.max_epochs=100000 \\
    trainer.accelerator=gpu \\
    trainer.devices=1 \\
    +trainer.precision=16 \\
    experiment=nn \\
    trainer.deterministic=False \\
    task_name=de_train_gpu \\
    run_name=de_gpu_{i} \\
    datamodule=QM7 \\
    datamodule.valid_split=100 \\
    datamodule.batch_size=256 \\
    seed={seed}
"""

# Seed numpy RNG
np.random.seed(54325)

# Generate multiple scripts
print("="*80)
print(f"Creating GPU training scripts for QM7 Deep Ensemble")
print("="*80)

for i in range(10):
    seed = int(np.random.randint(0, 1e6))
    run_name = f"{method}_train_gpu_{i}"
    script_content = template.format(
            i=i,
            run_name=run_name,
            seed=seed,
            method=method
    )
    filename = os.path.join(cwd, f"{run_name}.sh")
    with open(filename, "w") as f:
        f.write(script_content)
    os.chmod(filename, 0o755)
    print(f"✓ Created: {filename}")

print("\n" + "="*80)
print("To submit all jobs:")
print("="*80)
print(f"cd {cwd}")
print(f"for script in de_train_gpu_*.sh; do qsub $script; done")
print("\nOr submit individually:")
print(f"qsub {cwd}/de_train_gpu_0.sh")
print("="*80)
