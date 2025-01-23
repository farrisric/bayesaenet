#!/bin/bash
#$ -N train_lrt_20perc_TiO
#$ -pe smp* 1
#$ -q iqtc09.q
#$ -S /bin/bash
#$ -cwd
#$ -o train_lrt.out
#$ -e train_lrt.err
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
export CUDA_VISIBLE_DEVICES=`cat $TMPDIR/.gpus`
conda activate bnn
export PYTHONPATH="${PYTHONPATH}:/home/g15farris/bin/bayesaenet/bnn_aenet"
export OMP_NUM_THREADS=1
cd /home/g15farris/bin/bayesaenet
python bnn_aenet/tasks/train.py \
    trainer.min_epochs=20000 \
    trainer.min_epochs=50000 \
    experiment=bnn_lrt \
    seed=15 \
    trainer.deterministic=False \
    task_name=TiO_train_lrt_50 \
    datamodule=TiO \
    datamodule.test_split=0.4 \
    datamodule.valid_split=0.1 \
    datamodule.batch_size=256 \
    model.lr=0.00008142328235860698 \
    model.mc_samples_train=2 \
    model.prior_scale=0.020676708101278182 \
    model.q_scale=0.00016666027828979754
