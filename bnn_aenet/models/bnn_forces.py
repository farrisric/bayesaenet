"""
BNN with joint energy+force likelihood (ELBO-integrated).
Canonical BNN model; always uses forces.
"""

from typing import Any, Dict, List
import warnings

import numpy as np
import pyro
import pyro.poutine as poutine
import torch
import torch.nn.functional as F
import tyxe
from tyxe import priors, guides
from tyxe.bnn import VariationalBNN
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO
from functools import partial
import copy
import contextlib

from .bnn import BNN
from .utils import get_rmse_atom, weights_init
from .likelihoods import make_energy_force_model
from ..results.metrics import sharpness, rms_calibration_error
from ..datamodule.aenet.batch_constants import BatchIdx

from bnn_aenet.tasks.utils import get_pylogger
from bnn_aenet.models.utils import param_store_to

log = get_pylogger(__name__)


class BNN_Forces(BNN):
    """
    BNN with joint energy+force likelihood in the ELBO.
    
    Uses a custom Pyro model that:
    - Samples weights from the guide
    - Computes E_pred and F_pred with the same weights
    - Adds log p(E_obs|E_pred) + log p(F_obs|F_pred) to the ELBO
    
    This is theoretically more rigorous than the auxiliary loss approach.
    """

    def __init__(
        self,
        net,
        lr,
        pretrain_epochs,
        mc_samples_train,
        mc_samples_eval,
        dataset_size,
        fit_context,
        prior_loc,
        prior_scale,
        guide,
        q_scale,
        obs_scale,
        scale_force: float = 0.1,
        grad_clip_val: float = 1.0,
        name: str = "BNN_Forces",
    ):
        super().__init__(
            net, lr, pretrain_epochs, mc_samples_train, mc_samples_eval,
            dataset_size, fit_context, prior_loc, prior_scale, guide,
            q_scale, obs_scale, name=name,
        )
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.scale_force = scale_force

    def define_bnn(self):
        if self.hparams.pretrain_epochs > 0:
            self.net.apply(weights_init)

        prior = tyxe.priors.IIDPrior(
            pyro.distributions.Normal(
                torch.tensor(self.hparams.prior_loc, dtype=torch.float32, device=self.device),
                torch.tensor(self.hparams.prior_scale, dtype=torch.float32, device=self.device),
            ),
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
            from .guides.radial import AutoRadial
            guide_base = AutoRadial
            self.fit_ctxt = contextlib.nullcontext
        else:
            raise RuntimeError("Guide unknown. Choose from 'normal', 'radial'.")

        if self.hparams.pretrain_epochs > 0:
            guide_kwargs["init_loc_fn"] = tyxe.guides.PretrainedInitializer.from_net(self.net)
        guide_builder = partial(guide_base, **guide_kwargs)

        net_copy = copy.deepcopy(self.net.to(self.device))
        pyro.nn.module.to_pyro_module_(net_copy)
        prior.apply_(net_copy)
        self.net_guide = guide_builder(net_copy)

        self.bnn_net = net_copy
        self.net = net_copy  # for likelihood's forward_F
        self.bnn_prior = prior

        scale_energy = self.hparams.obs_scale
        scale_force = getattr(self, "scale_force", 0.1)
        self._model_fn = make_energy_force_model(
            self,
            scale_energy=scale_energy,
            scale_force=scale_force,
            dataset_size=self.hparams.dataset_size,
        )

    def model(self, batch):
        """Custom model for joint energy+force likelihood."""
        return self._model_fn(batch)

    def guide(self, batch):
        """Guide: net_guide needs (descrp, logic_reduce), we extract from batch."""
        E_descrp = batch[BatchIdx.E_DESCRP]
        E_logic_reduce = batch[BatchIdx.E_LOGIC_REDUCE]
        return self.net_guide(E_descrp, E_logic_reduce) or {}

    def guided_forward(self, E_descrp, E_logic_reduce, guide_tr=None):
        """Run net with guide samples (same weights for energy and force)."""
        if guide_tr is None:
            guide_tr = poutine.trace(self.net_guide).get_trace(E_descrp, E_logic_reduce)
        return poutine.replay(self.bnn_net, trace=guide_tr)(E_descrp, E_logic_reduce)

    def on_fit_start(self) -> None:
        self.define_bnn()
        param_store_to(self.device)
        self.configure_optimizers()

        if self.hparams.fit_context == "lrt" and hasattr(self.trainer, "precision"):
            if "16" in str(self.trainer.precision):
                warnings.warn("LRT incompatible with mixed precision.", UserWarning)

        clip_norm = getattr(self.hparams, "grad_clip_val", 10.0)
        self.optimizer = pyro.optim.ClippedAdam({
            "lr": self.hparams.lr,
            "betas": [0.95, 0.999],
            "clip_norm": clip_norm,
        })

        self.loss = (
            TraceMeanField_ELBO(self.hparams.mc_samples_train)
            if self.hparams.guide != "radial"
            else Trace_ELBO(self.hparams.mc_samples_train)
        )

        scale = 1.0 / self.hparams.dataset_size
        self.svi = SVI(
            pyro.poutine.scale(self.model, scale=scale),
            pyro.poutine.scale(self.guide, scale=scale),
            self.optimizer,
            self.loss,
        )

    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> None:
        x = batch[BatchIdx.E_DESCRP], batch[BatchIdx.E_LOGIC_REDUCE]
        y = batch[BatchIdx.E_ENERGY]
        n_atoms = batch[BatchIdx.E_N_ATOM]

        self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.increment_ready()

        with self.fit_ctxt():
            elbo = self.svi.step(batch)

            with torch.no_grad():
                for name, val in pyro.get_param_store().items():
                    if torch.isnan(val).any():
                        val.data = torch.nan_to_num(val.data, nan=0.0)

            loc = self.guided_forward(x[0], x[1])
            if loc.dim() == 1:
                loc = loc.unsqueeze(-1)

        self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.increment_ready()

        rmse = get_rmse_atom(loc.squeeze(-1), y, n_atoms)  # normalized units
        force_rmse = self._compute_force_rmse(batch)
        alpha = self.net.alpha.item() if hasattr(self.net, "alpha") else 0.1
        total_rmse = (1 - alpha) * rmse + alpha * force_rmse

        self.log("rmse/train", rmse, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(y))
        self.log("force_rmse/train", force_rmse, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(y))
        self.log("total_rmse/train", total_rmse, on_step=False, on_epoch=True, batch_size=len(y))
        self.log("elbo/train", elbo, on_step=False, on_epoch=True, batch_size=len(y))

    def _compute_force_rmse(self, batch):
        F_descrp = batch[BatchIdx.F_DESCRP]
        F_forces = batch[BatchIdx.F_FORCES]
        if F_descrp is None or F_forces is None:
            return torch.tensor(0.0, device=self.device)
        if isinstance(F_descrp, list) and len(F_descrp) == 0:
            return torch.tensor(0.0, device=self.device)

        net = self.bnn_net
        E_descrp = batch[BatchIdx.E_DESCRP]
        E_logic_reduce = batch[BatchIdx.E_LOGIC_REDUCE]
        F_logic_reduce = batch[BatchIdx.F_LOGIC_REDUCE]
        F_sfderiv_i = batch[BatchIdx.F_SFDERIV_I]
        F_sfderiv_j = batch[BatchIdx.F_SFDERIV_J]
        F_indices = batch[BatchIdx.F_INDICES]
        F_indices_i = batch[BatchIdx.F_INDICES_I]
        max_nnb = F_sfderiv_j[0].shape[1] if len(F_sfderiv_j) > 0 else 0

        # forward_F needs grad for autograd; use guide samples via replay
        guide_tr = poutine.trace(self.net_guide).get_trace(E_descrp, E_logic_reduce)
        with torch.enable_grad():
            F_descrp_grad = [d.clone().detach().float().requires_grad_(True) for d in F_descrp]
            F_sfderiv_i_f = [s.float() for s in F_sfderiv_i]
            F_sfderiv_j_f = [s.float() for s in F_sfderiv_j]
            F_logic_reduce_f = [l.float() for l in F_logic_reduce]
            with poutine.replay(trace=guide_tr):
                _, F_pred = net.forward_F(
                    F_descrp_grad,
                    F_sfderiv_i_f,
                    F_sfderiv_j_f,
                    F_indices,
                    F_indices_i,
                    F_logic_reduce_f,
                    net.input_size,
                    max_nnb,
                )
        diff = F_pred.detach() - F_forces.float()
        return torch.sqrt(torch.mean(diff ** 2))  # normalized; meV/Å in prediction

    def validation_step(self, batch: List[torch.Tensor], batch_idx: int) -> None:
        x = batch[BatchIdx.E_DESCRP], batch[BatchIdx.E_LOGIC_REDUCE]
        y = batch[BatchIdx.E_ENERGY]
        n_atoms = batch[BatchIdx.E_N_ATOM]

        elbo = self.svi.evaluate_loss(batch)

        with torch.no_grad():
            preds = []
            for _ in range(self.hparams.mc_samples_eval):
                p = self.guided_forward(x[0], x[1])
                preds.append(p.unsqueeze(0) if p.dim() == 1 else p)
            loc = torch.cat(preds, 0).mean(0)
            if loc.dim() == 1:
                loc = loc.unsqueeze(-1)

        rmse = get_rmse_atom(loc.squeeze(-1), y, n_atoms)  # normalized units
        force_rmse = self._compute_force_rmse(batch)
        alpha = getattr(self.net, "alpha", 0.1)
        if hasattr(alpha, "item"):
            alpha = alpha.item()
        total_rmse = (1 - alpha) * rmse + alpha * force_rmse

        self.log("rmse/val", rmse, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(y))
        self.log("force_rmse/val", force_rmse, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(y))
        self.log("total_rmse/val", total_rmse, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(y))
        self.log("elbo/val", elbo, on_step=False, on_epoch=True, batch_size=len(y))

    def on_predict_start(self) -> None:
        """Initialize BNN for prediction."""
        self.define_bnn()
        param_store_to(self.device)

    def predict_step(
        self,
        batch: List[torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Dict[str, np.ndarray]:
        """Predict energies and forces with uncertainty."""
        x = batch[BatchIdx.E_DESCRP], batch[BatchIdx.E_LOGIC_REDUCE]
        y = batch[BatchIdx.E_ENERGY]
        n_atoms = batch[BatchIdx.E_N_ATOM]

        pred = {}
        energy_samples = []
        for _ in range(self.hparams.mc_samples_eval):
            p = self.guided_forward(x[0], x[1])
            energy_samples.append(p.detach().cpu().numpy())
        energy_preds = np.stack(energy_samples).mean(axis=0)
        energy_stds = np.stack(energy_samples).std(axis=0)

        F_group_descrp = batch[BatchIdx.F_DESCRP]
        F_group_forces = batch[BatchIdx.F_FORCES]
        has_force_data = (
            F_group_descrp is not None
            and F_group_forces is not None
            and not (isinstance(F_group_descrp, list) and len(F_group_descrp) == 0)
        )

        if has_force_data:
            F_logic_reduce = batch[BatchIdx.F_LOGIC_REDUCE]
            F_sfderiv_i = batch[BatchIdx.F_SFDERIV_I]
            F_sfderiv_j = batch[BatchIdx.F_SFDERIV_J]
            F_indices = batch[BatchIdx.F_INDICES]
            F_indices_i = batch[BatchIdx.F_INDICES_I]
            max_nnb = F_sfderiv_j[0].shape[1] if len(F_sfderiv_j) > 0 and F_sfderiv_j[0].shape[0] > 0 else 0
            net_params = dict(self.net.named_parameters())

            force_samples = []
            for _ in range(self.hparams.mc_samples_eval):
                guide_tr = poutine.trace(self.net_guide).get_trace(x[0], x[1])
                for site_name, site in guide_tr.nodes.items():
                    if site.get("type") == "sample" and not site.get("is_observed", False):
                        if site_name.startswith("net."):
                            param_name = site_name[len("net.") :]
                            if param_name in net_params:
                                net_params[param_name].data.copy_(site["value"].data)
                with torch.inference_mode(False), torch.enable_grad():
                    F_descrp_f = [d.clone().detach().float().requires_grad_(True) for d in F_group_descrp]
                    F_sfderiv_i_f = [s.clone().detach().float() for s in F_sfderiv_i]
                    F_sfderiv_j_f = [s.clone().detach().float() for s in F_sfderiv_j]
                    F_logic_reduce_f = [l.clone().detach().float() for l in F_logic_reduce]
                    _, F_pred = self.net.forward_F(
                        F_descrp_f, F_sfderiv_i_f, F_sfderiv_j_f,
                        F_indices, F_indices_i, F_logic_reduce_f,
                        self.net.input_size, max_nnb,
                    )
                force_samples.append(F_pred.detach().cpu().numpy())

            force_preds = np.stack(force_samples).mean(axis=0)
            force_stds = np.stack(force_samples).std(axis=0)
            true_forces_flat = F_group_forces.cpu().numpy().flatten()
            pred_forces_flat = force_preds.flatten()
            std_forces_flat = force_stds.flatten()
            scale = float(self.net.e_scaling) if hasattr(self.net.e_scaling, "item") else float(self.net.e_scaling)
            force_rmse = np.sqrt(np.mean((true_forces_flat - pred_forces_flat) ** 2)) / scale * 1000
            force_mae = np.mean(np.abs(true_forces_flat - pred_forces_flat)) / scale * 1000
        else:
            true_forces_flat = None
            pred_forces_flat = None
            std_forces_flat = None
            force_rmse = np.nan
            force_mae = np.nan

        pred["true"] = y.cpu().numpy()
        pred["preds"] = energy_preds
        pred["stds"] = energy_stds
        pred["n_atoms"] = n_atoms.cpu().numpy()
        pred["true_forces"] = true_forces_flat
        pred["pred_forces"] = pred_forces_flat
        pred["std_forces"] = std_forces_flat
        pred["force_rmse"] = force_rmse
        pred["force_mae"] = force_mae
        return pred


# Backward compatibility for checkpoints
BNN_Forces_Likelihood = BNN_Forces
