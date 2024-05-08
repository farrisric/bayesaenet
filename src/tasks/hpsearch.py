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

# def objective_nn(trial: Trial, cfg: DictConfig, output_dir: str):
#     hidden_size = [[], []]
#     activations = [[], []]
#     layer_number = trial.suggest_categorical("layer_number", [1,2,3])
#     log.info(f"{layer_number} layer_number")
#     for n_layer in range(layer_number):
#         name_layer = f'layer_{n_layer}'
#         # node1 = trial.suggest_int(f'node0_{n_layer}', 5, 20)
#         # node2 = trial.suggest_int(f'node1_{n_layer}', 5, 20)
#         #activation = trial.suggest_categorical("activation", ['linear', 'sigmoid', 'tanh'])
#         activations[0].append(trial.suggest_categorical("activation", ['linear', 'sigmoid', 'tanh']))
#         activations[1].append(trial.suggest_categorical("activation", ['linear', 'sigmoid', 'tanh']))
        
#         hidden_size[0].append(trial.suggest_int(f'node0_{n_layer}', 5, 20))
#         hidden_size[1].append(trial.suggest_int(f'node1_{n_layer}', 5, 20))
        
#     cfg.model.optimizer.lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
#     log.info(f"{cfg.model.optimizer.lr} lr")
    
#     cfg.model.net.hidden_size = hidden_size
#     log.info(f"{cfg.model.net.hidden_size} hidden_size")
    
#     cfg.model.net.active_names = activations
#     log.info(f"{cfg.model.net.active_names} active_names")
#     return objective(trial, cfg, output_dir)

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
        "prior_scale", 1e-2, 0.5, log=True
    )
    log.info(f"{cfg.model.prior_scale} prior_scale")
    cfg.model.q_scale = trial.suggest_float(
        "q_scale", 1e-4, 1e-2, log=True
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