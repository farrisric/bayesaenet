import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO

import torch
from torch import nn
import torch.nn.functional as F
import tyxe
from tyxe import guides, priors, likelihoods, VariationalBNN
from tyxe.guides import AutoNormal, AutoRadial
import pytorch_lightning as L
from functools import partial
import copy
import contextlib
import numpy as np
class BNN(L.LightningModule):
    def __init__(
            self,
            net: torch.nn.Module,
            lr: float,
            pretrain_epochs: bool,
            mc_samples_train: int,
            mc_samples_eval: int,
            dataset_size: int,
            fit_context: str,
            prior_loc: float,
            prior_scale: float,
            guide: str,
            q_scale: float,
            obs_scale: float
    ):
        super().__init__()
        pyro.clear_param_store()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net  

    def define_bnn(self):
        if not self.hparams.pretrain_epochs == 0:
            self.net.apply(weights_init)

        prior_kwargs = {}  # {'hide_all': True}
        prior = tyxe.priors.IIDPrior(
            dist.Normal(
            torch.tensor(
                self.hparams.prior_loc,
                dtype=torch.float32,
                device=self.device,
            ),
            torch.tensor(
                self.hparams.prior_scale,
                dtype=torch.float32,
                device=self.device,
            ),
        ),
        **prior_kwargs,
        )

        if self.hparams.fit_context == "lrt":
            self.fit_ctxt = tyxe.poutine.local_reparameterization
        elif self.hparams.fit_context == "flipout":
            self.fit_ctxt = tyxe.poutine.flipout
        else:
            self.fit_ctxt = contextlib.nullcontext

        guide_kwargs = {"init_scale": self.hparams.q_scale}
        if self.hparams.guide == "normal":
            guide_base = tyxe.guides.AutoNormal
        elif self.hparams.guide == "radial":
            guide_base = AutoRadial
            self.fit_ctxt = contextlib.nullcontext
        else:
            raise RuntimeError("Guide unknown. Choose from 'normal', 'radial'.")

        if self.hparams.pretrain_epochs > 0:
            guide_kwargs[
                "init_loc_fn"
            ] = tyxe.guides.PretrainedInitializer.from_net(self.net)
        guide = partial(guide_base, **guide_kwargs)

        likelihood = tyxe.likelihoods.HomoskedasticGaussian(
            self.hparams.dataset_size,
            scale=self.hparams.obs_scale,
        )

        self.bnn = VariationalBNN(
            copy.deepcopy(self.net.to(self.device)),
            prior,
            likelihood,
            guide,
        )
         
    def on_fit_start(self):
        self.define_bnn()
        param_store_to(self.device)
        self.configure_optimizers()

        self.optimizer = pyro.optim.ClippedAdam({'lr': self.hparams.lr, 'betas': [0.95, 0.999], 'clip_norm': 15})
        self.loss = (
            TraceMeanField_ELBO(self.hparams.mc_samples_train)
            if self.hparams.guide != "radial"
            else Trace_ELBO(self.hparams.mc_samples_train)
        )

        self.svi = self.svi = SVI(
            pyro.poutine.scale(self.bnn.model,scale=1.0/self.hparams.dataset_size,),
            pyro.poutine.scale(self.bnn.guide,scale=1.0/self.hparams.dataset_size,),
            self.optimizer,
            self.loss,)

    def training_step(self, batch, batch_idx):
        x = batch[10], batch[12]
        y = batch[11]
        self.bnn_no_obs = pyro.poutine.block(self.bnn, hide=["obs"])
        self.svi_no_obs = SVI(
            self.bnn_no_obs, self.bnn.guide, self.optimizer, self.loss
        )
        self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.increment_ready()

        with self.fit_ctxt():
            elbo = self.svi.step(x,y)
            loc, scale = self.bnn.predict(x[0], x[1],num_predictions=self.hparams.mc_samples_train)
            kl = self.svi_no_obs.evaluate_loss(x[0], x[1])

        self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.increment_ready()
        
        #mse = rmse(loc, y, batch[14])
        mse = F.mse_loss(loc, y)
        self.log("mse/train", mse, on_step=False, on_epoch=True, batch_size=len(y) )
        self.log("elbo/train", elbo, on_step=False, on_epoch=True, batch_size=len(y))
        self.log("kl/train", kl, on_step=False, on_epoch=True, batch_size=len(y))
        self.log("likelihood/train", elbo - kl, on_step=False, on_epoch=True, batch_size=len(y))

    def validation_step(self, batch, batch_idx):
        x = batch[10], batch[12]
        y = batch[11]

        self.bnn_no_obs = pyro.poutine.block(self.bnn, hide=["obs"])
        self.svi_no_obs = SVI(
            self.bnn_no_obs, self.bnn.guide, self.optimizer, self.loss
        )
        elbo = self.svi.evaluate_loss(x, y.squeeze())
        # Aggregate = False if num_prediction = 1, else nans in sd
        loc, scale = self.bnn.predict(x[0], x[1], num_predictions=self.hparams.mc_samples_eval)
        kl = self.svi_no_obs.evaluate_loss(x[0], x[1])

        #mse = rmse(loc, y, batch[14])
        mse = F.mse_loss(loc, y)
        self.log("elbo/val", elbo, batch_size=len(y))
        self.log("mse/val", mse, batch_size=len(y))
        self.log("kl/val", kl, batch_size=len(y))
        self.log("likelihood/val", elbo - kl, batch_size=len(y))

    def on_test_start(self) -> None:
        self.define_bnn()
        param_store_to(self.device)

    def test_step(self, batch, batch_idx):
        x = batch[10], batch[12]
        y = batch[11]
        loc, scale = self.bnn.predict(x[0], x[1], num_predictions=self.hparams.mc_samples_eval)

        nll = F.gaussian_nll_loss(loc.squeeze(), y.squeeze(), torch.square(scale))
        #mse = rmse(loc, y, batch[14])
        mse = F.mse_loss(loc, y)
        self.log("nll/test", nll, batch_size=len(y))
        self.log("mse/test", mse, batch_size=len(y))
        return nll
    
    def on_predict_start(self) -> None:
        self.define_bnn()
        param_store_to(self.device)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch[10], batch[12]
        y = batch[11]
        pred = dict()
        loc, scale = self.bnn.predict(
            x[0], x[1],
            num_predictions=self.hparams.mc_samples_eval
        )
        pred["preds"] = loc.cpu().numpy()
        pred["stds"] = scale.cpu().numpy()
        return pred

    def configure_optimizers(self):
        pass

    def on_save_checkpoint(self, checkpoint):
        """Saving Pyro's param_store for the bnn's parameters"""
        checkpoint["param_store"] = pyro.get_param_store().get_state()

    def on_load_checkpoint(self, checkpoint):
        pyro.get_param_store().set_state(checkpoint["param_store"])
        if not hasattr(self, "bnn"):
            checkpoint["state_dict"] = remove_dict_entry_startswith(
                checkpoint["state_dict"], "bnn"
            )

def param_store_to(device: str):
    ps = pyro.get_param_store().get_state()
    for k in ps["params"].keys():
        ps["params"][k] = ps["params"][k].to(device)
    pyro.get_param_store().set_state(ps)


def remove_dict_entry_startswith(dictionary, string):
    """Used to remove entries with 'bnn' in checkpoint state dict"""
    n = len(string)
    for key in dictionary:
        if string == key[:n]:
            dict2 = dictionary.copy()
            dict2.pop(key)
            dictionary = dict2
    return dictionary

def weights_init(m):
    """Initializes weights of a nn.Module : xavier for conv
    and kaiming for linear

    """
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_normal_(m.weight)
    elif isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)

class NN(L.LightningModule):
    """
    Class used by BNNs to pretrain their weights. This class is instantiated,
    trained for X epochs and then it stores its weights in the log directory.
    VIBnnWrapper then loads the weights and starts the Bayesian training
    """

    def __init__(self, net: torch.nn.Module, optimizer: torch.optim.Optimizer):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        self.net.apply(weights_init)

    def forward(self, x):
        return self.net(x)

    def step(self, batch):
        grp_descrp, grp_energy, logic_reduce, grp_N_atom = batch[10], batch[11], batch[12], batch[14]
        for x in grp_descrp:
            x = x.float()
        for x in logic_reduce:
            x = x.float()
        grp_energy = batch[11].float()
        y_hat = self.net.forward(grp_descrp, logic_reduce)
        return rmse(y_hat, grp_energy, grp_N_atom)

    def training_step(self, batch, batch_idx):
        mse = self.step(batch)
        self.log("mse/train", mse, on_step=False, on_epoch=True, batch_size=len(batch[11]))
        return mse

    def validation_step(self, batch, batch_idx):
        mse = self.step(batch)
        self.log("mse/val", mse, on_step=False, on_epoch=True, batch_size=len(batch[11]))

    def test_step(self, batch, batch_idx):
        mse = self.step(batch)
        self.log("mse/test", mse, on_step=False, on_epoch=True, batch_size=len(batch[11]))

    def configure_optimizers(self):
        return self.hparams.optimizer(params=self.parameters())
    

def rmse(list_E_ann, grp_energy, grp_N_atom):
    differences = (list_E_ann - grp_energy)
    l2 = torch.sum( differences**2/grp_N_atom**2 )
    return l2