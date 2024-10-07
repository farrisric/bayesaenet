from typing import Optional

import yaml
import hydra
import pyrootutils
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch import LightningDataModule, LightningModule, Trainer

import pandas as pd
from pathlib import Path


from utils import get_pylogger
import sys

from src.utils.miscellaneous import ResultSaver

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

log = get_pylogger(__name__)

def load_model_hyperparams(model, hp_path):
    with open(hp_path, 'r') as file:
        hp_override = yaml.safe_load(file)
    for x in hp_override:
        if x[:5] == 'model':
            key, value = x[6:].split('=')
            model[key] = float(value)
    return model

def predict(cfg: DictConfig):
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)

    ckpt_paths = []
    hp_paths = []
    if cfg.ckpt_path == "all":
        path = Path(f"{cfg.runs_dir}")
        print(path)
        for ckpt_path in path.glob("**/epoch*.ckpt"):
            ckpt_paths.append(ckpt_path)
        for hp_path in path.glob("**/overrides.yaml"):
            hp_paths.append(hp_path)            
    else:
        ckpt_paths.append(Path(cfg.ckpt_path))
    
    
    for ckpt_path, hp_path in zip(ckpt_paths, hp_paths):
        method, run = ckpt_path.as_posix().split("/")[-4:-2]
        run = f"{int(run):03d}"
        model = cfg[method]
        model = load_model_hyperparams(model, hp_path)
        print(model['lr'])
        log.info(f"Instantiating model <{model._target_}>")
        model.net.input_size = datamodule.input_size
        model.net.hidden_size = datamodule.hidden_size
        model.net.species = datamodule.species
        model.net.active_names = datamodule.active_names
        model.net.alpha = datamodule.alpha
        if OmegaConf.is_missing(model, "dataset_size"):
            model.dataset_size = datamodule.train_size
        model: LightningModule = hydra.utils.instantiate(
            model, _convert_="partial"
        )
        for subset in cfg.subsets:
            
            filename = f"{method}_{run}_{subset}.parquet"
            #datamodule.set_predict_dataset(subset)
            predictions = trainer.predict(
                model=model, datamodule=datamodule, ckpt_path=ckpt_path
            )
            predictions = pd.DataFrame.from_records(predictions)
            predictions = predictions.explode(
                predictions.columns.tolist()
            ).reset_index(drop=True)
            log.info(f"Saving predicions: {cfg.paths.output_dir}")
            results = ResultSaver(f"{cfg.paths.output_dir}", f"{filename}")
            results.save(predictions)


@hydra.main(version_base=None, config_path="../configs", config_name="predict")
def main(cfg: DictConfig) -> Optional[float]:
    predict(cfg)


if __name__ == "__main__":
    main()