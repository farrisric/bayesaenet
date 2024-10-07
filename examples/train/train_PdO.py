import sys
sys.path.append('../')

from ase.db import connect
import os
import pyro
from bnn_aenet.models.bnn import BNN, NN
from bnn_aenet.models.nets.network import NetAtom
import torch
import pytorch_lightning as L
import torch.utils.data as data
import numpy as np
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

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

from lightning.pytorch import seed_everything

seed_everything(143, workers=True)

datamodele = AenetDataModule(data_dir='../data/PdO/train.in',batch_size=64,test_split=0.3,valid_split=0.3)

net = NetAtom(datamodele.input_size, datamodele.hidden_size, datamodele.species, datamodele.active_names, datamodele.alpha,'cpu')

model_kwargs = {'net': net,
        'dataset_size': datamodele.train_size, 
        'fit_context': 'lrt', 
        'prior_loc': 0, 
        'guide': 'normal', 
        'mc_samples_eval': 20,
        'lr': 0.0009549627167707569, 
        'mc_samples_train': 2, 
        'obs_scale': 0.47612142594698453,
        'pretrain_epochs': 0,
        'prior_scale': 0.026705525180274015,
        'q_scale': 0.0005171823159504436
        }



early_stopping = EarlyStopping(monitor='elbo/val', min_delta = 0., # minimum change in the monitored quantity to qualify as an improvement
  patience= 3, # number of checks with no improvement after which training will be stopped
  verbose= False, # verbosity mode
  mode= "min", # "max" means higher metric value is better, can be also "min"
  strict= True, # whether to crash the training if monitor is not found in the validation metrics
  check_finite= True,)

ckpt_path = 'a/home/riccardo/bin/repos/aenet-bnn/train/lightning_logs/version_6/checkpoints/epoch=63-step=13056.ckpt'

model = BNN(**model_kwargs)
trainer = L.Trainer(max_epochs = 1000, deterministic=True)
if os.path.isfile(ckpt_path):
    trainer.fit(model=model, datamodule=datamodele, ckpt_path=ckpt_path)
    trainer.test(model=model, datamodule=datamodele, ckpt_path=ckpt_path)
else:
    trainer.fit(model=model, datamodule=datamodele)
    trainer.test(model=model, datamodule=datamodele, ckpt_path=trainer.checkpoint_callback.best_model_path)


names = ['train', 'valid', 'test']
datasets = [datamodele.grouped_train_data, datamodele.grouped_valid_data, datamodele.grouped_test_data]
for name, dataset in zip(names, datasets):
    y_pred, y_std, y_true = np.array([]), np.array([]), np.array([])
    for batch in dataset:
        x = batch[10], batch[12]
        y_true_i = batch[11]
        n_atoms = batch[14]
        y_pred_i, y_std_i = model.bnn.predict(x[0], x[1], num_predictions=20)


        y_pred = np.concatenate((y_pred, y_pred_i))
        y_std = np.concatenate((y_std, y_std_i)) 
        y_true = np.concatenate((y_true, y_true_i))
    
    #y_pred, y_std, y_true = y_pred[:10], y_std[:10], y_true[:10]
    print(f'Size {name} dataset {len(y_pred)}')


    plt.rcParams.update({
        "figure.dpi" : 200,
        "font.family": "sans-serif",
        "font.sans-serif": "Arial",
        "font.size" : 12,
        'mathtext.default': 'regular'
    })

    regr = LinearRegression()

    x_model = abs(y_pred-y_true).reshape(-1,1)
    y_model = y_std.reshape(-1,1)

    regr.fit(x_model, y_model)
    y_model_pred = regr.predict(x_model)
    x_modelito = np.linspace(0, max(x_model), 100).reshape(-1,1)


    fig, ax = plt.subplots(1,2, figsize=(6,3), sharey=True)

    ax[0].scatter(x_model, y_model, alpha=0.3)
    ax[0].plot(x_modelito, regr.predict(x_modelito), '--', color='tab:orange', label='f(x)=x')


    print('R2 score: {:.3f}'.format(r2_score(y_model, y_model_pred)))
    ax[1].scatter(x_model, y_model, alpha=0.3, label='Predictions')

    ax[1].plot(x_modelito, regr.predict(x_modelito), '--', color='tab:orange', label='f(x)=x')

    ax[1].set_xlim(-0.1,1)
    #ax[0].text(0.1,1.7, 'R2 score: {:.3f}'.format(r2_score(y_model, y_model_pred)))
    ax[1].set_xlabel('MAE')
    ax[0].set_xlabel('MAE')
    ax[0].set_ylabel('std. dev.')
    ax[1].legend()
    fig.savefig(f'correlation_{name}.png')

    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    ax = ax.flat
    uct.plot_intervals(y_pred, y_std, y_true, ax=ax[0])
    uct.plot_intervals_ordered(y_pred, y_std, y_true, ax=ax[1])
    uct.plot_calibration(y_pred, y_std, y_true, ax=ax[2])
    uct.plot_adversarial_group_calibration(y_pred, y_std, y_true, ax=ax[3])
    uct.plot_sharpness(y_std, ax=ax[4])
    uct.plot_residuals_vs_stds(y_pred, y_std- model_kwargs['obs_scale'], y_true, ax=ax[5])
    fig.savefig(f'uncertanty_analysis_{name}.png')
