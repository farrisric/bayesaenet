import os

prior_scales = [10,9,8,7,6,5,4,3,2,1, 0.1]
q_scales = [0.01, 0.001, 0.0001, 1e-05]

# Constants
base_name = "lrt_grid_pred"
task_name = "hom_lrt_train_100perc_grid"
output_dir = "grid_pred"
runs_dir= "bnn_aenet/logs"
script_template = """#!/bin/bash
#$ -N {job_name}
#$ -pe smp* 1
#$ -q iqtc12.q
#$ -S /bin/bash
#$ -cwd
#$ -o out
#$ -e err
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
python bnn_aenet/tasks/predict.py \\
        task_name={task_name}_pred_{i}_{j} \\
        prediction=TiO \\
        ckpt_path=all \\
        datamodule.valid_split=100 \\
        +method=HLRT \\
        runs_dir={runs_dir}/{task_name}_{i}_{j} \\
"""

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Generate scripts
for i, q in enumerate(q_scales):
    for j, p in enumerate(prior_scales):
        job_name = f"{base_name}_q_{q}_p{p}"
        script_path = os.path.join(output_dir, f"{job_name}.sh")
        with open(script_path, "w") as f:
            f.write(script_template.format(
                job_name=job_name,
                task_name=task_name,
                runs_dir=runs_dir,
                i=q,
                j=p,
                q_scale=q,
                prior_scale=p
            ))

print(f"✅ Generated {len(q_scales) * len(prior_scales)} job scripts in '{output_dir}'")

