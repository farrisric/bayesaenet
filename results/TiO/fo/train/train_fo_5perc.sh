#!/bin/bash
#$ -N train_fo_20perc_TiO
#$ -pe smp* 1
#$ -q iqtc12.q
#$ -S /bin/bash
#$ -cwd
#$ -o train_fo.out
#$ -e train_fo.err
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
    experiment=bnn_fo \
    seed=15 \
    trainer.deterministic=False \
    task_name=TiO_train_fo_5 \
    datamodule=TiO \
    datamodule.test_split=0.85 \
    datamodule.valid_split=0.1 \
    datamodule.batch_size=32 \
    model.lr=0.0007892292501819303 \
    model.pretrain_epochs=5 \
    model.mc_samples_train=2 \
    model.mc_samples_eval=20 \
    model.prior_loc=0 \
    model.prior_scale=0.09520703526103874 \
    model.q_scale=0.002290405685846643 \
    model.obs_scale=1.0853562544854425
