#!/bin/bash
#$ -N lrt_train_100
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
    trainer.min_epochs=2000 \
    trainer.min_epochs=5000 \
    experiment=hom_lrt \
    trainer.deterministic=False \
    task_name=hom_lrt_train_100perc_${seed} \
    datamodule=TiO \
    datamodule.valid_split=100 \
    datamodule.batch_size=64 \
    model.lr=0.0007298662778869769 \
    model.mc_samples_train=1 \
    model.prior_scale=0.8243090352929017\
    model.q_scale=0.00438900888415861\
    model.obs_scale=0.19048442316907535\
    seed=${seed}
