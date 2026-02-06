"""Script to pretrain a deterministic NN for BNN initialization.

This creates pretrained weights that can be loaded when using pretrain_epochs > 0
in BNN training.

Usage:
    python scripts/pretrain/pretrain_nn.py datamodule=TiO epochs=5
    python scripts/pretrain/pretrain_nn.py datamodule=QM7 epochs=5
"""
import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import lightning.pytorch as L
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint


# Add bnn_aenet to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "bnn_aenet" / "tasks"))

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


@hydra.main(version_base=None, config_path="../../bnn_aenet/configs", config_name="train")
def main(cfg: DictConfig):
    """Pretrain a deterministic NN and save checkpoint for BNN initialization."""
    
    # Get epochs from trainer config (use trainer.max_epochs)
    epochs = cfg.trainer.get("max_epochs", 5)
    tag = cfg.tags[0] if cfg.get("tags") else "bayesian"
    
    print(f"Pretraining NN for {epochs} epochs...")
    print(f"Tag: {tag}")
    
    # Instantiate datamodule
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    
    # Set up model config for NN
    cfg.model.net.input_size = datamodule.input_size
    cfg.model.net.hidden_size = datamodule.hidden_size
    cfg.model.net.species = datamodule.species
    cfg.model.net.active_names = datamodule.active_names
    cfg.model.net.alpha = datamodule.alpha
    cfg.model.net.e_scaling = datamodule.e_scaling
    cfg.model.net.e_shift = datamodule.e_shift
    
    # Create the network
    net = hydra.utils.instantiate(cfg.model.net)
    
    # Create NN model (deterministic)
    from bnn_aenet.models.bnn import NN
    model = NN(
        net=net,
        optimizer=hydra.utils.instantiate(cfg.model.optimizer, _partial_=True),
    )
    
    # Set up checkpoint directory
    # Format: {results_dir}/{tag}/pretrained/{epochs-1}/checkpoints/
    ckpt_dir = Path(cfg.paths.results_dir) / tag / "pretrained" / str(epochs - 1) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Checkpoint will be saved to: {ckpt_dir}/pretrained.ckpt")
    
    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="pretrained",
        save_top_k=1,
        monitor="rmse/val",
        mode="min",
    )
    
    # Trainer
    trainer = Trainer(
        max_epochs=epochs,
        accelerator=cfg.trainer.get("accelerator", "auto"),
        devices=cfg.trainer.get("devices", 1),
        callbacks=[checkpoint_callback],
        enable_progress_bar=True,
    )
    
    # Set seed for reproducibility
    seed = cfg.get("seed", 42)
    L.seed_everything(seed, workers=True)
    
    # Train
    trainer.fit(model=model, datamodule=datamodule)
    
    print(f"\nPretraining complete!")
    print(f"Best checkpoint saved to: {checkpoint_callback.best_model_path}")
    print(f"\nTo use these pretrained weights, run BNN training with:")
    print(f"  pretrain_epochs={epochs}")


if __name__ == "__main__":
    main()
