#!/bin/bash
#$ -N tr_lrt_e044
#$ -q iqtc13.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o /home/g15farris/bin/bayesaenet/log/train/TiO2_small_lrt_elbo_t044.out
#$ -e /home/g15farris/bin/bayesaenet/log/train/TiO2_small_lrt_elbo_t044.err

. /etc/profile
__conda_setup="$('/aplic/anaconda/2020.02/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then eval "$__conda_setup"; else
  if [ -f "/aplic/anaconda/2024.10/etc/profile.d/conda.sh" ]; then . "/aplic/anaconda/2024.10/etc/profile.d/conda.sh"; else export PATH="/aplic/anaconda/2024.10/bin:$PATH"; fi
fi
unset __conda_setup
module load cuda/12.4
conda activate bnn
export OMP_NUM_THREADS=4
export TMPDIR=/tmp/g15farris
export PYTHONPATH=/home/g15farris/bin/bayesaenet:$PYTHONPATH
cd /home/g15farris/bin/bayesaenet
mkdir -p log/train

python -m bnn_aenet.tasks.train \
  experiment=bnn_lrt datamodule=TiO_Forces_Data20 trainer.accelerator=gpu trainer.devices=1 \
  trainer.max_epochs=50000 dataset=TiO2_small task_name=train run_name=lrt_elbo_top3_t044 \
  datamodule.batch_size=128 model.lr=0.0004834599071127569 model.mc_samples_train=2 \
  model.prior_scale=0.13871641785021233 model.q_scale=0.0037970830271932916 model.obs_scale=0.5 \
  model.pretrain_epochs=0 model.scale_force=0.1 \
  model.learn_noise=true \
  callbacks.model_checkpoint.monitor=elbo/val callbacks.early_stopping.monitor=elbo/val \
  callbacks.early_stopping.patience=1500 seed=365838 \
  'tags=["TiO2_small","lrt","learn_noise","top3","elbo"]'
