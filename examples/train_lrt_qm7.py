#!/usr/bin/env python3
"""Example: Train LRT BNN on QM7 dataset.

This script demonstrates how to programmatically train a Local Reparameterization
Trick (LRT) Bayesian Neural Network on the QM7 molecular dataset.

Usage:
    python examples/train_lrt_qm7.py
    
For GPU training:
    CUDA_VISIBLE_DEVICES=0 python examples/train_lrt_qm7.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT / "bnn_aenet"))

import hydra
from omegaconf import DictConfig, OmegaConf


def main():
    # Configuration overrides
    overrides = [
        "experiment=final/lrt_qm7",
        "seed=42",
        "run_name=lrt_qm7_example",
        "trainer.max_epochs=1000",  # Shorter for demo
        "callbacks.early_stopping.patience=50",
    ]
    
    # Initialize Hydra
    with hydra.initialize(config_path="../bnn_aenet/configs", version_base="1.3"):
        cfg = hydra.compose(config_name="train", overrides=overrides)
    
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Import training function
    from tasks.train import train
    
    # Run training
    metrics = train(cfg)
    
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    
    if metrics:
        print("\nFinal Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
