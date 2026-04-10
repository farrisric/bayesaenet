#!/bin/bash
#$ -N timing_sweep_5model
#$ -q iqtc13.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o /home/g15farris/bin/bayesaenet/log/time/timing_sweep_5model.out
#$ -e /home/g15farris/bin/bayesaenet/log/time/timing_sweep_5model.err

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
mkdir -p log/time

echo "=========================================="
echo "BNN Training Time Benchmarking"
echo "5 Models: LRT, LRT Hetero, RAD, RAD Hetero, NN"
echo "=========================================="
echo ""
echo "Models to test:"
echo "  1. LRT with RMSE objective (learn_noise=True)"
echo "  2. LRT Hetero with RMSE objective (learn_noise=False)"
echo "  3. RAD with RMSE objective (learn_noise=True)"
echo "  4. RAD Hetero with RMSE objective (learn_noise=False)"
echo "  5. NN (standard neural network baseline)"
echo ""
echo "Each model will run for 100 epochs."
echo "=========================================="

# Model parameters (common across models)
BATCH_SIZE_LRT=1024
LR_LRT=0.001
PRIOR_SCALE_LRT=0.3
Q_SCALE_LRT=0.0002

BATCH_SIZE_RAD=128
LR_RAD=0.00034
PRIOR_SCALE_RAD=0.371
Q_SCALE_RAD=0.0001

# Test 1: LRT RMSE (learn_noise=True)
echo ""
echo "=== MODEL 1: LRT with RMSE objective ==="
START_TIME_LRT=$(date +%s)

python -m bnn_aenet.tasks.train \
  experiment=bnn_lrt datamodule=TiO_Forces_Data20 trainer.accelerator=gpu trainer.devices=1 \
  trainer.max_epochs=100 dataset=TiO2_small task_name=train run_name=time_lrt_100ep \
  datamodule.batch_size=$BATCH_SIZE_LRT model.lr=$LR_LRT model.mc_samples_train=1 \
  model.prior_scale=$PRIOR_SCALE_LRT model.q_scale=$Q_SCALE_LRT model.obs_scale=0.5 \
  model.pretrain_epochs=0 model.scale_force=0.1 \
  model.learn_noise=true \
  callbacks.model_checkpoint.monitor=total_rmse/val callbacks.early_stopping.monitor=total_rmse/val \
  callbacks.early_stopping.patience=20 seed=42

END_TIME_LRT=$(date +%s)
ELAPSED_LRT=$((END_TIME_LRT - START_TIME_LRT))
echo "LRT completed in ${ELAPSED_LRT} seconds"

# Test 2: LRT Hetero (learn_noise=False, heteroscedastic)
echo ""
echo "=== MODEL 2: LRT Hetero with RMSE objective ==="
START_TIME_LRT_HETERO=$(date +%s)

python -m bnn_aenet.tasks.train \
  experiment=bnn_lrt datamodule=TiO_Forces_Data20 trainer.accelerator=gpu trainer.devices=1 \
  trainer.max_epochs=100 dataset=TiO2_small task_name=train run_name=time_lrt_hetero_100ep \
  datamodule.batch_size=$BATCH_SIZE_LRT model.lr=$LR_LRT model.mc_samples_train=1 \
  model.prior_scale=$PRIOR_SCALE_LRT model.q_scale=$Q_SCALE_LRT model.obs_scale=0.5 \
  model.pretrain_epochs=0 model.scale_force=0.1 \
  model.learn_noise=false \
  callbacks.model_checkpoint.monitor=total_rmse/val callbacks.early_stopping.monitor=total_rmse/val \
  callbacks.early_stopping.patience=20 seed=43

END_TIME_LRT_HETERO=$(date +%s)
ELAPSED_LRT_HETERO=$((END_TIME_LRT_HETERO - START_TIME_LRT_HETERO))
echo "LRT Hetero completed in ${ELAPSED_LRT_HETERO} seconds"

# Test 3: RAD RMSE (learn_noise=True)
echo ""
echo "=== MODEL 3: RAD with RMSE objective ==="
START_TIME_RAD=$(date +%s)

python -m bnn_aenet.tasks.train \
  experiment=bnn_rad datamodule=TiO_Forces_Data20 trainer.accelerator=gpu trainer.devices=1 \
  trainer.max_epochs=100 dataset=TiO2_small task_name=train run_name=time_rad_100ep \
  datamodule.batch_size=$BATCH_SIZE_RAD model.lr=$LR_RAD model.mc_samples_train=1 \
  model.prior_scale=$PRIOR_SCALE_RAD model.q_scale=$Q_SCALE_RAD model.obs_scale=0.5 \
  model.pretrain_epochs=0 model.scale_force=0.1 \
  model.learn_noise=true \
  callbacks.model_checkpoint.monitor=total_rmse/val callbacks.early_stopping.monitor=total_rmse/val \
  callbacks.early_stopping.patience=20 seed=44

END_TIME_RAD=$(date +%s)
ELAPSED_RAD=$((END_TIME_RAD - START_TIME_RAD))
echo "RAD completed in ${ELAPSED_RAD} seconds"

# Test 4: RAD Hetero (learn_noise=False, heteroscedastic)
echo ""
echo "=== MODEL 4: RAD Hetero with RMSE objective ==="
START_TIME_RAD_HETERO=$(date +%s)

python -m bnn_aenet.tasks.train \
  experiment=bnn_rad datamodule=TiO_Forces_Data20 trainer.accelerator=gpu trainer.devices=1 \
  trainer.max_epochs=100 dataset=TiO2_small task_name=train run_name=time_rad_hetero_100ep \
  datamodule.batch_size=$BATCH_SIZE_RAD model.lr=$LR_RAD model.mc_samples_train=1 \
  model.prior_scale=$PRIOR_SCALE_RAD model.q_scale=$Q_SCALE_RAD model.obs_scale=0.5 \
  model.pretrain_epochs=0 model.scale_force=0.1 \
  model.learn_noise=false \
  callbacks.model_checkpoint.monitor=total_rmse/val callbacks.early_stopping.monitor=total_rmse/val \
  callbacks.early_stopping.patience=20 seed=45

END_TIME_RAD_HETERO=$(date +%s)
ELAPSED_RAD_HETERO=$((END_TIME_RAD_HETERO - START_TIME_RAD_HETERO))
echo "RAD Hetero completed in ${ELAPSED_RAD_HETERO} seconds"

# Test 5: NN (standard neural network baseline)
echo ""
echo "=== MODEL 5: NN (baseline) ==="
START_TIME_NN=$(date +%s)

python -m bnn_aenet.tasks.train \
  experiment=bnn_nn datamodule=TiO_Forces_Data20 trainer.accelerator=gpu trainer.devices=1 \
  trainer.max_epochs=100 dataset=TiO2_small task_name=train run_name=time_nn_100ep \
  datamodule.batch_size=512 model.lr=0.001 model.pretrain_epochs=0 model.scale_force=0.1 \
  callbacks.model_checkpoint.monitor=total_rmse/val callbacks.early_stopping.monitor=total_rmse/val \
  callbacks.early_stopping.patience=20 seed=46

END_TIME_NN=$(date +%s)
ELAPSED_NN=$((END_TIME_NN - START_TIME_NN))
echo "NN completed in ${ELAPSED_NN} seconds"

# Summary
echo ""
echo "=========================================="
echo "TIME SUMMARY (100 epochs each)"
echo "=========================================="
printf "%-35s %12s\n" "Model" "Time (seconds)"
printf "%-35s %12d\n" "LRT" $ELAPSED_LRT
printf "%-35s %12d\n" "LRT Hetero" $ELAPSED_LRT_HETERO
printf "%-35s %12d\n" "RAD" $ELAPSED_RAD
printf "%-35s %12d\n" "RAD Hetero" $ELAPSED_RAD_HETERO
printf "%-35s %12d\n" "NN (baseline)" $ELAPSED_NN

TOTAL_TIME=$((ELAPSED_LRT + ELAPSED_LRT_HETERO + ELAPSED_RAD + ELAPSED_RAD_HETERO + ELAPSED_NN))
echo ""
printf "%-35s %12d\n" "TOTAL TIME" $TOTAL_TIME
echo "=========================================="

# Save to CSV for later analysis
cat >> log/time/timing_results_5model.csv << EOF
model,epochs,time_seconds
lrt,100,$ELAPSED_LRT
lrt_hetero,100,$ELAPSED_LRT_HETERO
rad,100,$ELAPSED_RAD
rad_hetero,100,$ELAPSED_RAD_HETERO
nn_baseline,100,$ELAPSED_NN
EOF

echo ""
echo "Results saved to log/time/timing_results_5model.csv"
