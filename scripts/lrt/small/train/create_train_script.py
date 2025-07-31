import re
import os

word_dir = os.path.dirname(os.path.abspath(__file__))

template = """#!/bin/bash
#$ -N lrt_train_small
#$ -pe smp* 1
#$ -q iqtc12.q
#$ -S /bin/bash
#$ -cwd
#$ -o train_lrt.out
#$ -e train_lrt.err
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
    trainer.min_epochs=50000 \\
    trainer.max_epochs=50000 \\
    experiment=bnn_lrt \\
    trainer.deterministic=False \\
    task_name=lrt_train \\
    run_name={run_name} \\
    datamodule=TiO \\
    datamodule.valid_split=20 \\
    datamodule.batch_size={batch_size} \\
    model.lr={lr} \\
    model.mc_samples_train={mc_samples_train} \\
    model.prior_scale={prior_scale} \\
    model.q_scale={q_scale} \\
    model.obs_scale={obs_scale}
"""

# Path to the log file
log_file_path = f"{word_dir}/../hps/err"

# Regular expression to extract trial data
pattern = re.compile(
    r"Trial (\d+) finished with value: ([\d.]+) and parameters: ({.*?})"
)
matches = []
with open(log_file_path, "r") as file:
    for line in file:
        match = pattern.search(line)
        if match:
            trial = int(match.group(1))
            score = float(match.group(2))
            params = eval(match.group(3), {"__builtins__": {}})
            matches.append((score, trial, params))

# Sort by score (ascending) and keep top 3
top_matches = sorted(matches, key=lambda x: x[0])[:3]

for match in top_matches:
    score, trial, params = match
    run_name = f"training_{score:.4f}_small"
    
    script_content = template.format(
        run_name=run_name,
        lr=params["lr"],
        mc_samples_train=params["mc_samples_train"],
        prior_scale=params["prior_scale"],
        q_scale=params["q_scale"],
        obs_scale=params["obs_scale"],
        batch_size=params["batch_size"]
    )

    filename = f"{word_dir}/training_{score:.4f}.sh"
    with open(filename, "w") as f:
        f.write(script_content)
    os.chmod(filename, 0o755)
    print(f"Written {filename}")
