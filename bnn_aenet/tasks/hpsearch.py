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


def objective_bnn_forces(trial: Trial, cfg: DictConfig, output_dir: str):
    """Objective function for BNN_Forces_Aux hyperparameter search.
    
    Optimizes both standard BNN hyperparameters and force-specific ones:
    - pretrain_epochs: Number of pretraining epochs (0 or 5)
    - lr: Learning rate
    - prior_scale: Prior distribution scale (affects regularization)
    - q_scale: Initial variational parameter scale (affects uncertainty)
    - obs_scale: Observation noise scale (affects likelihood)
    - force_weight: Additional weight multiplier for force loss
    - force_lr_scale: Learning rate scale for force updates vs energy
    - scale_lr_factor: Learning rate factor for updating scale (uncertainty) params
    """
    # Standard BNN hyperparameters
    # Pretraining disabled for now - requires proper checkpoint setup first
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
        "prior_scale", 0.1, 0.5, log=True  # Narrowed range for numerical stability
    )
    log.info(f"{cfg.model.prior_scale} prior_scale")
    
    cfg.model.q_scale = trial.suggest_float(
        "q_scale", 1e-5, 0.005, log=True  # Narrowed range for numerical stability
    )
    log.info(f"{cfg.model.q_scale} q_scale")
    
    cfg.model.obs_scale = trial.suggest_float(
        "obs_scale", 0.1, 2.0, log=True
    )
    log.info(f"{cfg.model.obs_scale} obs_scale")
    
    # alpha is fixed at 0.1 in net config (not optimized to avoid bias toward lower values)
    log.info(f"{cfg.model.net.alpha} alpha (fixed)")
    
    return objective(trial, cfg, output_dir)


def objective_partial_bnn_forces(trial: Trial, cfg: DictConfig, output_dir: str):
    """Objective function for PartialBNN_Forces_Aux hyperparameter search.
    
    Similar to full BNN but with reduced hyperparameter space since partial BNNs
    have fewer Bayesian parameters and are more numerically stable.
    
    Note: bayesian_layers is fixed in the experiment config (last, first, etc.)
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
    
    # Partial BNNs can handle wider ranges since fewer params are Bayesian
    cfg.model.prior_scale = trial.suggest_float(
        "prior_scale", 0.05, 1.0, log=True
    )
    log.info(f"{cfg.model.prior_scale} prior_scale")
    
    cfg.model.q_scale = trial.suggest_float(
        "q_scale", 1e-5, 0.01, log=True
    )
    log.info(f"{cfg.model.q_scale} q_scale")
    
    cfg.model.obs_scale = trial.suggest_float(
        "obs_scale", 0.1, 2.0, log=True
    )
    log.info(f"{cfg.model.obs_scale} obs_scale")
    
    # Log which layers are Bayesian (fixed from config)
    log.info(f"{cfg.model.bayesian_layers} bayesian_layers (fixed)")
    log.info(f"{cfg.model.net.alpha} alpha (fixed)")
    
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
        catch=(RuntimeError, ValueError),
    )

    log.info("Number of finished trials: {}".format(len(study.trials)))


if __name__ == "__main__":
    main()
