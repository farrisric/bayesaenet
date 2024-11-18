# This script creates a job file for the pretraining of the Bayesian Neural Network

import glob


python_path = "export PYTHONPATH=\"${PYTHONPATH}:/home/g15farris/bin/bayesaenet/bnn_aenet\""
for perc in [5, 10, 20, 30, 40, 50, 60, 70, 80]:

    job_file = """#!/bin/bash
#$ -N pred_commitie_{}perc
#$ -pe smp* 1
#$ -q iqtc12.q
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
{}
export OMP_NUM_THREADS=1
cd /home/g15farris/bin/bayesaenet
python bnn_aenet/tasks/predict.py \
task_name=predict_commitie_{}perc \
prediction=PdO \
ckpt_path=all \
+method=NN \
runs_dir=bnn_aenet/logs/train_nn_commitie_{}perc
""".format(perc, python_path, perc, perc)
    
    with open(f"/home/g15farris/bin/bayesaenet/results/PdO/committie/predict/predict_commitie_{perc}.sh", "w") as f:
        f.write(job_file)