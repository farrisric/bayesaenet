#!/bin/bash

# Quick script to run TiO2 analysis

echo "Activating conda environment 'bnn'..."
source ~/.bashrc
conda activate bnn

echo ""
echo "Running analysis..."
cd /home/g15farris/bin/bayesaenet/bnn_aenet/notebooks/TiO2
python run_analysis.py

echo ""
echo "Results saved to: results/"
echo "To create plots, run: python create_plots.py"
