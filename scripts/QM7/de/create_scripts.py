import os
import numpy as np

# Get current working directory and extract method name
cwd = os.path.dirname(os.path.abspath(__file__))
method = 'de'

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
cd /home/g15farris/bin/bayesaenet
python bnn_aenet/tasks/train.py \
    trainer.min_epochs=100000 \
    trainer.max_epochs=100000 \
    experiment=nn \
    trainer.deterministic=False \
    task_name=de_train \
    run_name=de_{i} \
    datamodule=QM7 \
    datamodule.valid_split=100 \
    datamodule.batch_size=64 \
    seed={seed}
"""

# Seed numpy RNG
np.random.seed(54325)

# Generate multiple scripts
for i in range(10):
    seed = int(np.random.randint(0, 1e6))
    run_name = f"{method}_train{i}"
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
    print(f"Written {filename}")
    os.system(f"qsub {filename}")
