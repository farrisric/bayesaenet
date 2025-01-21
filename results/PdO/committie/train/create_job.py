# This script creates a job file for the pretraining of the Bayesian Neural Network

# Create the job file

python_path = "export PYTHONPATH=\"${PYTHONPATH}:/home/g15farris/bin/bayesaenet/bnn_aenet\""

for perc in [5, 10, 20, 30, 40, 50, 60, 70, 80]:

    job_file = """#!/bin/bash
#$ -N commitie_{}perc
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
python bnn_aenet/tasks/train.py \
task_name=train_nn_commitie_{}perc \
experiment=nn \
datamodule=PdO \
trainer.min_epochs=30000 \
trainer.max_epochs=50000 \
datamodule.test_split={} \
datamodule.valid_split=0.1 \
datamodule.batch_size=32 \
model.optimizer.lr=0.0005640905763748237 \
seed=1,2,3,4,5,6,7,8,9,10 \
--multirun""".format(perc, python_path, perc, (90-perc)/100)
    
    with open(f"train_commitie_{perc}.sh", "w") as f:
        f.write(job_file)
