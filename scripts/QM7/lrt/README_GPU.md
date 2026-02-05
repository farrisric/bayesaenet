# LRT GPU Training for QM7

## GPU Multirun Training

Train all 10 LRT ensemble members sequentially on a single GPU using Hydra's multirun feature.

### Submit Job

```bash
cd /home/g15farris/bin/bayesaenet/scripts/QM7/lrt
qsub lrt_train_gpu_multirun.sh
```

### Monitor Training

```bash
# Watch output
tail -f lrt_train_gpu_multirun.out

# Check errors
tail -f lrt_train_gpu_multirun.err

# Check GPU usage
ssh merry04 nvidia-smi

# Monitor real-time GPU stats
ssh merry04 nvidia-smi dmon -c 10
```

### Configuration

- **Experiment**: `bnn_lrt` (Local Reparameterization Trick)
- **Epochs**: 100,000
- **Batch size**: 32 (optimized for LRT)
- **Precision**: 16-bit mixed precision
- **GPU**: Single RTX 4090
- **Seeds**: 10 different random seeds for ensemble

### Hyperparameters (from optimization)

- Learning rate: 0.0001835
- MC samples (train): 2
- Prior scale: 0.1246
- Q scale: 0.000754
- Obs scale: 0.4898

### Output Location

Results will be saved to:
```
/home/g15farris/bin/bayesaenet/bnn_aenet/logs/lrt_train_gpu/multiruns/YYYY-MM-DD_HH-MM-SS/
```

Each run (0-9) will have its own subdirectory with:
- Checkpoints: `checkpoints/*.ckpt`
- TensorBoard logs: `tensorboard/`
- Config files: `.hydra/`

### Expected Runtime

- Per model: ~2-3 hours (depends on convergence)
- Total for 10 models: ~20-30 hours

### Notes

- Training runs sequentially (one after another) on the same GPU
- More memory efficient than parallel training
- Uses the same seeds as the DE training for consistency
- Batch size is smaller (32 vs 256) due to LRT's higher memory requirements
