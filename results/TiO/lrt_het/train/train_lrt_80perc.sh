#!/bin/bash
#$ -N train_lrt_80perc_TiO
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
    seed=38136957 \
    trainer.deterministic=False \
    task_name=TiO_train_lrt_80_128batch_5mc \
    datamodule=TiO \
    datamodule.test_split=0.1 \
    datamodule.valid_split=0.1 \
    datamodule.batch_size=128 \
    model.lr=0.0004892606819863511 \
    model.mc_samples_train=5 \
    model.prior_scale=0.06753350752446152 \
    model.q_scale=0.0005855051866323676
