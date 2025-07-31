import re
import os
import glob 

cwd = os.path.dirname(os.path.abspath(__file__))
logs = '/home/g15farris/bin/bayesaenet/bnn_aenet/logs'
method = cwd.split('/')[-2]

predict_template = """#!/bin/bash
#$ -N lrt_predict
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
conda activate bnn
export HYDRA_FULL_ERROR=1
export PYTHONPATH="${{PYTHONPATH}}:/home/g15farris/bin/bayesaenet/bnn_aenet"
export OMP_NUM_THREADS=1
cd /home/g15farris/bin/bayesaenet
python bnn_aenet/tasks/predict.py \\
    task_name=lrt_pred \\
    run_name={run_name} \\
    prediction=TiO \\
    ckpt_path=all \\
    datamodule.valid_split=100 \\
    +method={method} \\
    runs_dir=bnn_aenet/logs/lrt_train/runs/{train_name} \\
"""

for train in glob.glob(f'{logs}/{method}_train/runs/*'):
    train_name = train.split('/')[-1]
    run_name = train_name.replace('training', 'pred')

    predict_script = predict_template.format(run_name=run_name,
                                             method=method.upper(),
                                             train_name=train_name)
    predict_filename = f"{cwd}/{run_name}.sh"
    with open(predict_filename, "w") as f:
        f.write(predict_script)
    os.chmod(predict_filename, 0o755)
    os.system(f'qsub {predict_filename}')
