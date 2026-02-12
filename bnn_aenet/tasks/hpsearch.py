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

from bnn_aenet.tasks.train import train
from optuna import Study
from optuna.trial import Trial

from bnn_aenet.tasks.utils import get_pylogger
log = get_pylogger(__name__)


def objective(trial: Trial, cfg: DictConfig, output_dir: str):
    cfg.datamodule.batch_size = trial.suggest_categorical(
        "batch_size", [128, 256, 512, 1024]  # Removed 32/64 for stability, added 1024 for speed
    )
    log.info(f"{cfg.datamodule.batch_size} batch_size")
    log.info(
        f"_________________ Starting trial {trial.number:03d} __________________"
    )
    cfg.paths.output_dir = f"{output_dir}/{trial.number:03d}"
    metric_dict, _ = train(cfg, trial)
    return metric_dict[cfg.hpsearch.monitor]


def objective_nn(trial: Trial, cfg: DictConfig, output_dir: str):
    """Objective function for Deep Ensemble (NN) hyperparameter search."""
    cfg.model.optimizer.lr = trial.suggest_float(
        "lr", 1e-5, 1e-2, log=True
    )
    log.info(f"{cfg.model.optimizer.lr} lr")
    cfg.model.optimizer.weight_decay = trial.suggest_float(
        "weight_decay", 1e-6, 1e-2, log=True
    )
    log.info(f"{cfg.model.optimizer.weight_decay} weight_decay")
    return objective(trial, cfg, output_dir)


def objective_nn_forces(trial: Trial, cfg: DictConfig, output_dir: str):
    """Objective function for NN_Forces (Deep Ensemble with forces) hyperparameter search.
    
    Optimizes:
    - lr: Learning rate
    
    Note: alpha is FIXED at 0.1 (from config) to avoid biasing Optuna toward lower values.
    No pretraining for NN models - they train from scratch.
    """
    cfg.model.optimizer.lr = trial.suggest_float(
        "lr", 1e-5, 1e-2, log=True
    )
    log.info(f"{cfg.model.optimizer.lr} lr")
    
    # alpha is fixed at 0.1 in config (not optimized to avoid bias)
    log.info(f"{cfg.model.alpha} alpha (fixed)")
    
    return objective(trial, cfg, output_dir)


def objective_bnn(trial: Trial, cfg: DictConfig, output_dir: str):
    cfg.model.pretrain_epochs = trial.suggest_categorical(
        "pretrain_epochs", [0, 5]
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


def objective_bnn_forces_likelihood(trial: Trial, cfg: DictConfig, output_dir: str):
    """Objective function for BNN_Forces_Likelihood hyperparameter search.
    
    Optimizes the same BNN hyperparameters as auxiliary, plus:
    - scale_force: Observation noise scale for force likelihood (analogous to obs_scale)
    """
    cfg.model.pretrain_epochs = 0
    log.info(f"{cfg.model.pretrain_epochs} pretrain_epochs (fixed at 0)")
    
    cfg.model.lr = trial.suggest_float(
        "lr", 1e-5, 1e-3, log=True
    )
    log.info(f"{cfg.model.lr} lr")
    
    cfg.model.mc_samples_train = trial.suggest_categorical(
        "mc_samples_train", [1, 2]
    )
    log.info(f"{cfg.model.mc_samples_train} mc_samples_train")
    
    cfg.model.prior_scale = trial.suggest_float(
        "prior_scale", 0.1, 0.5, log=True
    )
    log.info(f"{cfg.model.prior_scale} prior_scale")
    
    cfg.model.q_scale = trial.suggest_float(
        "q_scale", 1e-5, 0.005, log=True
    )
    log.info(f"{cfg.model.q_scale} q_scale")
    
    cfg.model.obs_scale = trial.suggest_float(
        "obs_scale", 0.1, 2.0, log=True
    )
    log.info(f"{cfg.model.obs_scale} obs_scale")
    
    # scale_force: force likelihood noise (critical for energy/force balance)
    cfg.model.scale_force = trial.suggest_float(
        "scale_force", 0.05, 2.0, log=True
    )
    log.info(f"{cfg.model.scale_force} scale_force")
    
    log.info(f"{cfg.model.net.alpha} alpha (fixed)")
    
    return objective(trial, cfg, output_dir)


@hydra.main(version_base=None, config_path="../configs", config_name="hpsearch")
def main(cfg: DictConfig) -> Optional[float]:
    print(cfg.trainer)
    log.info(f"Instantiating study <{cfg.hpsearch.study._target_}>")
    
    path = Path(f"{cfg.paths.results_dir}")
    # Store DBs in dataset-specific subdirectory: results/{dataset}/{method}.db
    # Use hpsearch.results_subdir if set (TiO2_big, TiO2_small, QM7), else tags[0]
    results_subdir = cfg.hpsearch.get("results_subdir", None)
    if results_subdir is None or results_subdir == "":
        results_subdir = cfg.tags[0]
    db_dir = path / str(results_subdir)
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / f"{cfg.hpsearch.study.study_name}.db"
    log.info(f"Results will be stored in sqlite:///{db_path.as_posix()}")
    study: Study = hydra.utils.instantiate(
        cfg.hpsearch.study,
        storage=f"sqlite:///{db_path.as_posix()}",
    )
    log.info(f"Instantiating objective <{cfg.hpsearch.objective._target_}>")
    objective = hydra.utils.instantiate(cfg.hpsearch.objective, _partial_=True)

    output_dir = cfg.paths.output_dir
    log.info(f"Starting hyperparameter search ...")
    study.optimize(
        lambda trial: objective(trial, cfg, output_dir),
        n_trials=cfg.hpsearch.n_trials,
        timeout=None,
        catch=(RuntimeError, ValueError),
    )

    log.info("Number of finished trials: {}".format(len(study.trials)))


if __name__ == "__main__":
    main()
