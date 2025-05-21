import os
import glob

# Base values
prior_scales = [10,9,8,7,6,5,4,3,2,1, 0.1]
q_scales = [0.01, 0.001, 0.0001, 0.00001]
base_name = "lrt_grid"
task_name = "hom_lrt_train_100perc_grid"
output_dir = "new_grid"
script_template = """#!/bin/bash
#$ -N {job_name}
#$ -pe smp* 1
#$ -q iqtc12.q
#$ -S /bin/bash
#$ -cwd
#$ -o out
#$ -e err
#$ -m e
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
seed=$1
python bnn_aenet/tasks/train.py \\
    trainer.min_epochs=300000 \\
    trainer.max_epochs=300000 \\
    experiment=hom_lrt \\
    trainer.deterministic=False \\
    task_name={task_name}_{i}_{j} \\
    datamodule=TiO \\
    datamodule.valid_split=100 \\
    datamodule.batch_size=512 \\
    model.lr=0.000013265605241311912 \\
    model.mc_samples_train=2 \\
    model.prior_scale={prior_scale} \\
    model.q_scale={q_scale} \\
    model.obs_scale=0.1\\
    ckpt_path={ckpt_path}
"""

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Generate scripts
for i, q in enumerate(q_scales):
    for j, p in enumerate(prior_scales):
        job_name = f"{base_name}_q_{q:.5f}_p{p:.1f}"
        script_path = os.path.join(output_dir, f"{job_name}.sh")
        ii=round(q, 5)
        jj=round(p, 1)
        ckpt_path = glob.glob(f'/home/g15farris/bin/bayesaenet/bnn_aenet/logs/{task_name}_{ii}_{jj}/**/last.ckpt', recursive=True)[-1]
        with open(script_path, "w") as f:
            f.write(script_template.format(
                job_name=job_name,
                task_name=task_name,
                i=round(q, 5),
                j=round(p, 1),
                q_scale=q,
                prior_scale=p,
                ckpt_path=ckpt_path
            ))

print(f"✅ Generated {len(q_scales) * len(prior_scales)} job scripts in '{output_dir}'")

