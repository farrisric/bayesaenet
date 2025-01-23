import sys
sys.path.append('../')

import os
import pyro
from bnn_aenet.models.bnn import BNN, NN
from bnn_aenet.models.nets.network_h2 import NetAtom
import torch
import lightning.pytorch as L
import torch.utils.data as data
import numpy as np
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from optuna import Study
import optuna
from optuna.trial import Trial

from bnn_aenet.datamodule.aenet_datamodule import AenetDataModule

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
import uncertainty_toolbox as uct
import glob

from torch.optim import Adam

datamodele = AenetDataModule(
    data_dir='data/TiO/train.in',
    batch_size=100,
    test_split=0.1,
    valid_split=0.1
    )

net = NetAtom(
    datamodele.input_size, 
    datamodele.hidden_size, 
    datamodele.species, 
    datamodele.active_names, 
    datamodele.alpha,
    energy_scaling=1,
    energy_shift=1,
    device='cpu'
)

model_kwargs = {'net': net,
        'lr': 0.00025616626859234823,
        'pretrain_epochs': 5,
        'mc_samples_train': 1,
        'mc_samples_eval': 20, 
        'dataset_size': datamodele.train_size, 
        'fit_context': 'lrt', 
        'prior_loc': 0, 
        'prior_scale': 0.3726682199695302, 
        'guide': 'normal', 
        'q_scale': 0.00127000766093029489207278289472795143,
        'obs_scale' :  0.8115512648735741}


early_stopping = EarlyStopping(monitor='elbo/val', min_delta = 0., # minimum change in the monitored quantity to qualify as an improvement
  patience= 3, # number of checks with no improvement after which training will be stopped
  verbose= False, # verbosity mode
  mode= "min", # "max" means higher metric value is better, can be also "min"
  strict= True, # whether to crash the training if monitor is not found in the validation metrics
  check_finite= True,) # when set True, stops training when the monitor becomes NaN or infinite
#   'stopping_threshold'= null, # stop training immediately once the monitored quantity reaches this threshold
#   'divergence_threshold'= null, # stop training as soon as the monitored quantity becomes worse than this threshold
#   'check_on_train_epoch_end'= null,)

early_stopping = EarlyStopping(
    monitor='elbo/val',
    patience=100,
    mode='min' )

model = BNN(**model_kwargs)
trainer = L.Trainer(max_epochs = 2000)
trainer.fit(model=model, datamodule=datamodele)