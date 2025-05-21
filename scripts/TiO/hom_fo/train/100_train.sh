#!/bin/bash
#$ -N fo_train_100
#$ -pe smp* 1
#$ -q iqtc12.q
#$ -S /bin/bash
#$ -cwd
#$ -o train_lrt.out
#$ -e train_lrt.err
#$ -m e
##$ -M farrisric@outlook.com
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
    trainer.min_epochs=10000 \
    trainer.min_epochs=20000 \
    experiment=hom_lrt \
    trainer.deterministic=False \
    task_name=hom_fo_train_100perc_44 \
    datamodule=TiO \
    datamodule.valid_split=100 \
    ckpt_path=bnn_aenet/logs/hom_fo_hps_100perc/runs/2025-05-09_16-38-44/036/checkpoints/epoch_1989-step_23880.ckpt\
    datamodule.batch_size=512 \
    model.lr=0.000023959944346130677\
    model.mc_samples_train=2 \
    model.prior_scale=0.05964167723490831\
    model.q_scale=0.00010531435235600597\
    model.obs_scale=0.1327402514309527\
