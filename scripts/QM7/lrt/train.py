import os
import numpy as np

# Get current working directory and extract method name
cwd = os.path.dirname(os.path.abspath(__file__))
method = 'lrt'

# SLURM script template
template = """#!/bin/bash
#$ -N lrt_hps
#$ -pe smp* 1
#$ -q iqtc09.q
#$ -S /bin/bash
#$ -cwd
#$ -o out
#$ -e err
#$ -m e
#$ -M farrisric@outlook.com
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
    datamodule=QM7 \\
    datamodule.valid_split=100 \\
    datamodule.batch_size=32 \\
    model.lr=0.0001834577852050578 \\
    model.mc_samples_train=2 \\
    model.prior_scale=0.12457376609465613 \\
    model.q_scale=0.0007539921163280931 \\
    model.obs_scale=0.48983719064601916 \\
    seed={seed}
"""

# Seed numpy RNG
np.random.seed(431)

# Generate multiple scripts
for i in range(10):
    seed = int(np.random.randint(0, 1e6))
    run_name = f"{method}_train_{i}"
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