#!/bin/bash
#$ -N lrt_grid_q_0.01000_p8.0
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
export PYTHONPATH="${PYTHONPATH}:/home/g15farris/bin/bayesaenet/bnn_aenet"
export OMP_NUM_THREADS=1
cd /home/g15farris/bin/bayesaenet
seed=$1
python bnn_aenet/tasks/train.py \
    trainer.min_epochs=300000 \
    trainer.max_epochs=300000 \
    experiment=hom_lrt \
    trainer.deterministic=False \
    task_name=hom_lrt_train_100perc_grid_0.01_8 \
    datamodule=TiO \
    datamodule.valid_split=100 \
    datamodule.batch_size=512 \
    model.lr=0.000013265605241311912 \
    model.mc_samples_train=2 \
    model.prior_scale=8 \
    model.q_scale=0.01 \
    model.obs_scale=0.1\
    ckpt_path=/home/g15farris/bin/bayesaenet/bnn_aenet/logs/hom_lrt_train_100perc_grid_0.01_8/runs/2025-05-16_14-21-46/checkpoints/last.ckpt
