#!/bin/bash
#$ -N tr_lrt_e025
#$ -q iqtc13.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o /home/g15farris/bin/bayesaenet/log/train/TiO2_small_lrt_elbo_t025.out
#$ -e /home/g15farris/bin/bayesaenet/log/train/TiO2_small_lrt_elbo_t025.err

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
  trainer.max_epochs=50000 dataset=TiO2_small task_name=train run_name=lrt_elbo_top3_t025 \
  datamodule.batch_size=256 model.lr=0.00044301496601635137 model.mc_samples_train=2 \
  model.prior_scale=0.11872314099237079 model.q_scale=0.0023887406119392158 model.obs_scale=0.5 \
  model.pretrain_epochs=0 model.scale_force=0.1 \
  model.learn_noise=true \
  callbacks.model_checkpoint.monitor=elbo/val callbacks.early_stopping.monitor=elbo/val \
  callbacks.early_stopping.patience=1500 seed=259178 \
  'tags=["TiO2_small","lrt","learn_noise","top3","elbo"]'
