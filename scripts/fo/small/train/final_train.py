import os
import numpy as np

# Get current working directory and extract method name
cwd = os.path.dirname(os.path.abspath(__file__))
method = cwd.split('/')[-3]

# SLURM script template
template = """#!/bin/bash
#$ -N {method}_train
#$ -pe smp* 1
#$ -q iqtc12.q
#$ -S /bin/bash
#$ -cwd
#$ -o train_{method}.out
#$ -e train_{method}.err
. /etc/profile
__conda_setup="$('/aplic/anaconda/2020.02/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup" 
else
    if [ -f "/aplic/anaconda/2020.02/etc/profile.d/conda.sh" ]; then
        . "/aplic/anaconda/2020.02/etc/profile.d/conda.sh"
    else
        export PATH="/aplic/anaconda/2020.02/bin:$PATH"
    fi
fi
unset __conda_setup
export CUDA_VISIBLE_DEVICES=`cat $TMPDIR/.gpus`
conda activate bnn
export PYTHONPATH="${{PYTHONPATH}}:/home/g15farris/bin/bayesaenet/bnn_aenet"
export OMP_NUM_THREADS=1
cd /home/g15farris/bin/bayesaenet
python bnn_aenet/tasks/train.py \\
    trainer.min_epochs=100000 \\
    trainer.max_epochs=100000 \\
    experiment=bnn_{method} \\
    trainer.deterministic=False \\
    task_name={method}_train \\
    run_name={run_name} \\
    datamodule=TiO \\
    datamodule.valid_split=20 \\
    datamodule.batch_size=64 \\
    model.lr=0.00032539246208848594 \\
    model.mc_samples_train=2 \\
    model.prior_scale=0.17499196760652302 \\
    model.q_scale=0.0018321771442657703 \\
    model.obs_scale=0.25960006761089327\\
    seed={seed}
"""

# Seed numpy RNG
np.random.seed(234)

# Generate multiple scripts
for i in range(5):
    seed = int(np.random.randint(0, 1e6))
    run_name = f"{method}_small_train_best_{i}"
    script_content = template.format(
        run_name=run_name,
        seed=seed,
        method=method
    )
    filename = os.path.join(cwd, f"{run_name}.sh")
    with open(filename, "w") as f:
        f.write(script_content)
    os.chmod(filename, 0o755)
    print(f"Written {filename}")
    os.system(f"qsub {filename}")