import sys
sys.path.append('/home/g15farris/bin/forks/bayesaenet/')

import os
import pyro
from bnn_aenet.models.bnn import BNN, NN
from bnn_aenet.models.nets.network_old import NetAtom
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
import glob

datamodele = AenetDataModule(data_dir='/home/g15farris/bin/forks/bayesaenet/data/PdO/train.in',batch_size=100,test_split=0.8,valid_split=0.1)

net = NetAtom(datamodele.input_size, datamodele.hidden_size, datamodele.species, datamodele.active_names, datamodele.alpha,'cpu')

model_kwargs = {'net': net,
        'mc_samples_eval': 20, 
        'dataset_size': datamodele.train_size, 
        'lr': 0.0009410248956021665,
        'pretrain_epochs': 0,
        'mc_samples_train': 2,
        'mc_samples_eval': 20,
        'fit_context': 'lrt',
        'guide': 'normal',
        'prior_loc': 0,
        'prior_scale': 0.06343664261871752,
        'q_scale': 0.00020116582693680395,
        'obs_scale': 0.3555713872117341,
        }

early_stopping = EarlyStopping(monitor='elbo/val', min_delta = 0., # minimum change in the monitored quantity to qualify as an improvement
  patience= 3, # number of checks with no improvement after which training will be stopped
  verbose= False, # verbosity mode
  mode= "min", # "max" means higher metric value is better, can be also "min"
  strict= True, # whether to crash the training if monitor is not found in the validation metrics
  check_finite= True,)

ckpt_path = glob.glob('/home/g15farris/bin/forks/bayesaenet/src/logs/train_lrt/runs/2024-05-16_11-01-39/checkpoints/epoch_530-step_7965.ckpt')[0]

model = BNN(**model_kwargs)
trainer = L.Trainer(deterministic=True)

trainer.test(model=model, datamodule=datamodele, ckpt_path=ckpt_path)
print(trainer.callback_metrics)
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
    
    y_std = y_std - model_kwargs['obs_scale']
    #y_pred, y_std, y_true = y_pred[:10], y_std[:10], y_true[:10]
    print(f'Size {name} dataset {len(y_pred)}')


    plt.rcParams.update({
        "figure.dpi" : 200,
        #"font.family": "sans-serif",
        #"font.sans-serif": "Arial",
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
    ax[0].set_title('{} R2 score: {:.3f}'.format(name, r2_score(y_model, y_model_pred)))
    ax[1].set_xlabel('MAE')
    ax[0].set_xlabel('MAE')
    ax[0].set_ylabel('std. dev.')
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(f'predict/lrt_correlation_{name}_5perc.png')
    

    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    ax = ax.flat
    uct.plot_intervals(y_pred, y_std, y_true, ax=ax[0])
    uct.plot_intervals_ordered(y_pred, y_std, y_true, ax=ax[1])
    uct.plot_calibration(y_pred, y_std, y_true, ax=ax[2])
    uct.plot_adversarial_group_calibration(y_pred, y_std, y_true, ax=ax[3])
    uct.plot_sharpness(y_std, ax=ax[4])
    uct.plot_residuals_vs_stds(y_pred, y_std, y_true, ax=ax[5])
    fig.savefig(f'predict/lrt_uncertanty_analysis_{name}_5perc.png')
