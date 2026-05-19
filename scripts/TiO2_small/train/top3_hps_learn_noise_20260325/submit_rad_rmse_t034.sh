#!/bin/bash
#$ -N tr_rad_r034
#$ -q iqtc13.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o /home/g15farris/bin/bayesaenet/log/train/TiO2_small_rad_rmse_t034.out
#$ -e /home/g15farris/bin/bayesaenet/log/train/TiO2_small_rad_rmse_t034.err

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
  experiment=bnn_rad datamodule=TiO_Forces_Data20 trainer.accelerator=gpu trainer.devices=1 \
  trainer.max_epochs=50000 dataset=TiO2_small task_name=train run_name=rad_rmse_top3_t034 \
  datamodule.batch_size=256 model.lr=0.0005408522500695875 model.mc_samples_train=2 \
  model.prior_scale=0.22163101409512656 model.q_scale=0.0003027076976034826 model.obs_scale=0.5 \
  model.pretrain_epochs=0 model.scale_force=0.1 \
  model.learn_noise=true \
  callbacks.model_checkpoint.monitor=total_rmse/val callbacks.early_stopping.monitor=total_rmse/val \
  callbacks.early_stopping.patience=1500 seed=54886 \
  'tags=["TiO2_small","rad","learn_noise","top3","rmse"]'
