from pathlib import Path
from typing import Optional

import hydra
import pyrootutils
from omegaconf import DictConfig

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from train import train
from optuna import Study
from optuna.trial import Trial

from utils import get_pylogger
log = get_pylogger(__name__)


def objective(trial: Trial, cfg: DictConfig, output_dir: str):
    cfg.datamodule.batch_size = trial.suggest_categorical(
        "batch_size", [32, 64, 128, 256, 512]
    )
    log.info(f"{cfg.datamodule.batch_size} batch_size")
    log.info(
        f"_________________ Starting trial {trial.number:03d} __________________"
    )
    cfg.paths.output_dir = f"{output_dir}/{trial.number:03d}"
    metric_dict, _ = train(cfg, trial)
    return metric_dict[cfg.hpsearch.monitor]


def objective_bnn(trial: Trial, cfg: DictConfig, output_dir: str):
    cfg.model.pretrain_epochs = trial.suggest_categorical(
        "pretrain_epochs", [0]
    )
    log.info(f"{cfg.model.pretrain_epochs} pretrain_epochs")
    cfg.model.lr = trial.suggest_float(
        "lr", 1e-5, 1e-3, log=True
    )
    log.info(f"{cfg.model.lr}, lr")
    cfg.model.mc_samples_train = trial.suggest_categorical(
        "mc_samples_train", [1, 2]
    )
    log.info(f"{cfg.model.mc_samples_train} mc_samples_train")
    cfg.model.prior_scale = trial.suggest_float(
        "prior_scale", 0.1, 1.5, log=True
    )
    log.info(f"{cfg.model.prior_scale} prior_scale")
    cfg.model.q_scale = trial.suggest_float(
        "q_scale", 1e-4, 0.1, log=True
        )
    log.info(f"{cfg.model.q_scale} q_scale")
    cfg.model.obs_scale = trial.suggest_float(
        "obs_scale", 0.1, 2, log=True
        )
    log.info(f"{cfg.model.obs_scale} obs_scale")
    return objective(trial, cfg, output_dir)


@hydra.main(version_base=None, config_path="../configs", config_name="hpsearch")
def main(cfg: DictConfig) -> Optional[float]:
    print(cfg.trainer)
    log.info(f"Instantiating study <{cfg.hpsearch.study._target_}>")
    
    path = Path(f"{cfg.paths.results_dir}")
    log.info(f"Results will be stored in sqlite:///{path.as_posix()}/{cfg.tags[0]}/{cfg.hpsearch.study.study_name}.db")
    study: Study = hydra.utils.instantiate(
        cfg.hpsearch.study,
        storage=f"sqlite:///{path.as_posix()}/{cfg.tags[0]}/{cfg.hpsearch.study.study_name}.db",
    )
    log.info(f"Instantiating objective <{cfg.hpsearch.objective._target_}>")
    objective = hydra.utils.instantiate(cfg.hpsearch.objective, _partial_=True)

    output_dir = cfg.paths.output_dir
    log.info(f"Starting hyperparameter search ...")
    study.optimize(
        lambda trial: objective(trial, cfg, output_dir),
        n_trials=cfg.hpsearch.n_trials,
        timeout=None,
        catch=(RuntimeError,),
    )

    log.info("Number of finished trials: {}".format(len(study.trials)))


if __name__ == "__main__":
    main()
