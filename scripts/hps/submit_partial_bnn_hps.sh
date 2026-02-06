#!/bin/bash
# Submit HPS jobs for all Partial BNN models on TiO2 with forces
# Uses iqtc13 (3 GPUs) and iqtc10 (1 GPU) for parallel execution

cd /home/g15farris/bin/bayesaenet

# Create log directory
mkdir -p logs/hps

echo "Submitting HPS jobs for Partial BNN models..."

# Job 1: Partial LRT Last (iqtc13)
qsub << 'EOF1'
#!/bin/bash
#$ -N hps_partial_lrt_last
#$ -q iqtc13.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -cwd
#$ -o logs/hps/hps_partial_lrt_last.out
#$ -e logs/hps/hps_partial_lrt_last.err

source ~/.bashrc
conda activate bnn
export CUDA_VISIBLE_DEVICES=$SGE_GPU

cd /home/g15farris/bin/bayesaenet
python bnn_aenet/tasks/hpsearch.py hpsearch=partial_lrt_last_forces datamodule=TiO
EOF1

# Job 2: Partial LRT First+Last (iqtc13)
qsub << 'EOF2'
#!/bin/bash
#$ -N hps_partial_lrt_fl
#$ -q iqtc13.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -cwd
#$ -o logs/hps/hps_partial_lrt_first_last.out
#$ -e logs/hps/hps_partial_lrt_first_last.err

source ~/.bashrc
conda activate bnn
export CUDA_VISIBLE_DEVICES=$SGE_GPU

cd /home/g15farris/bin/bayesaenet
python bnn_aenet/tasks/hpsearch.py hpsearch=partial_lrt_first_last_forces datamodule=TiO
EOF2

# Job 3: Partial Flipout Last (iqtc13)
qsub << 'EOF3'
#!/bin/bash
#$ -N hps_partial_fo_last
#$ -q iqtc13.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -cwd
#$ -o logs/hps/hps_partial_fo_last.out
#$ -e logs/hps/hps_partial_fo_last.err

source ~/.bashrc
conda activate bnn
export CUDA_VISIBLE_DEVICES=$SGE_GPU

cd /home/g15farris/bin/bayesaenet
python bnn_aenet/tasks/hpsearch.py hpsearch=partial_fo_last_forces datamodule=TiO
EOF3

# Job 4: Partial Flipout First+Last (iqtc10)
qsub << 'EOF4'
#!/bin/bash
#$ -N hps_partial_fo_fl
#$ -q iqtc10.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -cwd
#$ -o logs/hps/hps_partial_fo_first_last.out
#$ -e logs/hps/hps_partial_fo_first_last.err

source ~/.bashrc
conda activate bnn
export CUDA_VISIBLE_DEVICES=$SGE_GPU

cd /home/g15farris/bin/bayesaenet
python bnn_aenet/tasks/hpsearch.py hpsearch=partial_fo_first_last_forces datamodule=TiO
EOF4

# Job 5: Partial Radial Last (wait for a GPU slot)
qsub << 'EOF5'
#!/bin/bash
#$ -N hps_partial_rad_last
#$ -q iqtc10.q
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -cwd
#$ -o logs/hps/hps_partial_rad_last.out
#$ -e logs/hps/hps_partial_rad_last.err

source ~/.bashrc
conda activate bnn
export CUDA_VISIBLE_DEVICES=$SGE_GPU

cd /home/g15farris/bin/bayesaenet
python bnn_aenet/tasks/hpsearch.py hpsearch=partial_rad_last_forces datamodule=TiO
EOF5

echo "All HPS jobs submitted!"
echo "Check status with: qstat"
echo "View logs in: logs/hps/"
