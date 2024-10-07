from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import hydra
import optuna
import lightning.pytorch as L
import rootutils
import torch
from lightning.pytorch import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf
import pyrootutils
from src.models.bnn import NN


from utils import (
    get_pylogger,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

#log = RankedLogger(__name__, rank_zero_only=True)
log = get_pylogger(__name__)

@task_wrapper
def train(cfg: DictConfig, trial: Optional[optuna.trial.Trial] = None):
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    cfg.model.net.input_size = datamodule.input_size
    cfg.model.net.hidden_size = datamodule.hidden_size
    cfg.model.net.species = datamodule.species
    cfg.model.net.active_names = datamodule.active_names
    cfg.model.net.alpha = datamodule.alpha
    if OmegaConf.is_missing(cfg.model, "dataset_size"):
        cfg.model.dataset_size = datamodule.train_size
    model: LightningModule = hydra.utils.instantiate(
        cfg.model, _convert_="partial"
        )

    log.info("Instantiating callbacks...")
    callbacks = instantiate_callbacks(cfg, trial)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
        )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        if cfg.model.get("pretrain_epochs") and not cfg.get("ckpt_path"):
            if cfg.model.pretrain_epochs > 0:
                model.net = load_pretrained_net(cfg, model)
        trainer.fit(
            model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path")
        )


    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = None
        if cfg.get("ckpt_path"):
            ckpt_path = cfg.ckpt_path
        else:
            ckpt_path = trainer.checkpoint_callback.best_model_path
            if ckpt_path == "":
                log.warning(
                    "Best ckpt not found! Using current weights for testing..."
                )
                ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict

def load_pretrained_net(cfg, model):
    ckpt_path_dir = Path(f"{cfg.paths.results_dir}/{cfg.tags[0]}/pretrained")
    if ckpt_path_dir is not None:
        ckpt_path = None
        for d in ckpt_path_dir.glob("*"):
            try:
                i = int(d.name)
            except:
                continue
            if i + 1 == cfg.model.pretrain_epochs:
                ckpt_path = Path(d, "checkpoints", "pretrained.ckpt")
                break
        if ckpt_path is not None:
            log.info(f"Restoring pretrained net from: {ckpt_path}")
            return NN.load_from_checkpoint(
                ckpt_path,
                net=model.net,
            ).net
        log.info(
            f"No pretrained net found in {ckpt_path_dir} for {cfg.model.pretrain_epochs}"
        )
    else:
        log.info(
            f"No pretrained nets found in {cfg.paths.results_dir}/{cfg.tags[0]}/pretrained"
        )
    return hydra.utils.instantiate(cfg.model.net)

@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> Optional[float]:
    train(cfg, trial=None)


if __name__ == "__main__":
    main()
