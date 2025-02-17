#!/bin/bash
#$ -N fin_free_train_lrt_80perc_TiO
#$ -pe smp* 1
#$ -q iqtc09.q
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
conda activate bayesian
export PYTHONPATH="${PYTHONPATH}:/home/g15telari/TiO/bayesaenet/bnn_aenet"
export OMP_NUM_THREADS=1
cd /home/g15telari/TiO/bayesaenet
python bnn_aenet/tasks/train.py \
    trainer.min_epochs=20000 \
    trainer.min_epochs=50000 \
    experiment=bnn_lrt \
    seed=249152917 \
    trainer.deterministic=False \
    task_name=final_TiO_train_lrt_80_sigmaoptuna \
    datamodule=TiO \
    datamodule.test_split=0.1 \
    datamodule.valid_split=0.1 \
    datamodule.batch_size=32 \
    model.lr=0.0005998764560655444 \
    model.pretrain_epochs=5 \
    model.mc_samples_train=2 \
    model.mc_samples_eval=20 \
    model.prior_loc=0 \
    model.prior_scale=0.4202843767131272 \
    model.q_scale=0.0007910538045185004 \
    model.obs_scale=0.1429396942458169
