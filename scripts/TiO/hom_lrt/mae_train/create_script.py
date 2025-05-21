import re
import os

train_template = """#!/bin/bash
#$ -N lrt_train_100
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
    experiment=hom_lrt \\
    trainer.deterministic=False \\
    task_name=lrt_train_{score:.4f} \\
    datamodule=TiO \\
    datamodule.valid_split=100 \\
    datamodule.batch_size={batch_size} \\
    model.lr={lr} \\
    model.mc_samples_train={mc_samples_train} \\
    model.prior_scale={prior_scale} \\
    model.q_scale={q_scale} \\
    model.obs_scale={obs_scale}
"""

predict_template = """#!/bin/bash
#$ -N lrt_predict
#$ -pe smp* 8
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
conda activate bnn
export HYDRA_FULL_ERROR=1
export PYTHONPATH="${{PYTHONPATH}}:/home/g15farris/bin/bayesaenet/bnn_aenet"
export OMP_NUM_THREADS=8
cd /home/g15farris/bin/bayesaenet
python bnn_aenet/tasks/predict.py \\
    task_name=hom_lrt_pred_{score:.4f} \\
    prediction=TiO \\
    ckpt_path=all \\
    datamodule.valid_split=100 \\
    +method=HLRT \\
    runs_dir=bnn_aenet/logs/lrt_train_{score:.4f}
"""

log_file_path = "/home/g15farris/bin/bayesaenet/scripts/TiO/hom_lrt/hpsearch_mse/err"
output_dir = "/home/g15farris/bin/bayesaenet/scripts/TiO/hom_lrt/mae_train"
os.makedirs(output_dir, exist_ok=True)

pattern = re.compile(r"Trial (\d+) finished with value: ([\d.]+) and parameters: ({.*?})")

with open(log_file_path, "r") as file:
    for line in file:
        match = pattern.search(line)
        if match:
            trial = int(match.group(1))
            score = float(match.group(2))
            if score > 0.0122:
                continue
            params = eval(match.group(3), {"__builtins__": {}})

            train_script = train_template.format(
                score=score,
                lr=params["lr"],
                mc_samples_train=params["mc_samples_train"],
                prior_scale=params["prior_scale"],
                q_scale=params["q_scale"],
                obs_scale=params["obs_scale"],
                batch_size=params["batch_size"]
            )

            predict_script = predict_template.format(score=score)

            train_filename = os.path.join(output_dir, f"training_{score:.4f}.sh")
            predict_filename = os.path.join(output_dir, f"predict_{score:.4f}.sh")

            with open(train_filename, "w") as f:
                f.write(train_script)
            os.chmod(train_filename, 0o755)

            with open(predict_filename, "w") as f:
                f.write(predict_script)
            os.chmod(predict_filename, 0o755)

            print(f"Written {train_filename} and {predict_filename}")
