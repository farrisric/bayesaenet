"""
BNN with joint energy+force likelihood (ELBO-integrated).
Canonical BNN model; always uses forces.
"""

from typing import Any, Dict, List, Union
import warnings

import numpy as np
import torch.nn as nn
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
from ..utils.metrics import sharpness, rms_calibration_error
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
        learn_noise: bool = False,
        name: str = "BNN_Forces",
    ):
        super().__init__(
            net, lr, pretrain_epochs, mc_samples_train, mc_samples_eval,
            dataset_size, fit_context, prior_loc, prior_scale, guide,
            q_scale, obs_scale, name=name,
        )
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.scale_force = scale_force
        self.learn_noise = learn_noise
        # Will be populated during training if learn_noise=True
        self.learned_scale_energy = None
        self.learned_scale_force = None

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
        learn_noise = getattr(self, "learn_noise", False)
        self._model_fn = make_energy_force_model(
            self,
            scale_energy=scale_energy,
            scale_force=scale_force,
            dataset_size=self.hparams.dataset_size,
            learn_noise=learn_noise,
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

        # Log learned noise scales if learn_noise is enabled
        if getattr(self, "learn_noise", False) and self.learned_scale_force is not None:
            self.log("obs_scale_force/train", self.learned_scale_force, on_step=False, on_epoch=True, batch_size=len(y))
        if getattr(self, "learn_noise", False) and self.learned_scale_energy is not None:
            self.log("obs_scale_energy/train", self.learned_scale_energy, on_step=False, on_epoch=True, batch_size=len(y))

    # ------------------------------------------------------------------
    # Force helpers using poutine.replay
    # ------------------------------------------------------------------
    # After ``to_pyro_module_``, the net's parameters are ``PyroSample``
    # sites.  ``poutine.replay`` is the ONLY correct way to inject guide
    # weights -- direct tensor copy via ``named_parameters()`` does NOT
    # work because the PyroModule exposes an *empty* parameter dict once
    # all weights have been converted to ``PyroSample`` by the prior.
    # ------------------------------------------------------------------

    def _get_guide_trace(self, E_descrp, E_logic_reduce, use_mean=False):
        """Get a guide trace, optionally replacing samples with means.

        Args:
            E_descrp: Energy descriptors (needed to run the guide).
            E_logic_reduce: Logic reduce tensors.
            use_mean: If True, overwrite each sample site value with the
                posterior mean (``site["fn"].mean``), giving a deterministic,
                low-variance force prediction.

        Returns:
            A Pyro trace usable with ``poutine.replay``.
        """
        guide_tr = poutine.trace(self.net_guide).get_trace(
            E_descrp, E_logic_reduce,
        )
        if use_mean:
            for site_name, site in guide_tr.nodes.items():
                if (
                    site.get("type") == "sample"
                    and not site.get("is_observed", False)
                ):
                    site["value"] = site["fn"].mean
        return guide_tr

    def _replay_forward_F(self, batch, guide_tr):
        """Compute forces by replaying guide weights through ``forward_F``.

        ``poutine.replay`` intercepts ``pyro.sample`` calls issued by the
        PyroModule ``__getattr__`` and returns the trace values, so the
        correct (guide-sampled or posterior-mean) weights are used in the
        autograd-based force computation.

        Args:
            batch: Full data batch.
            guide_tr: A Pyro trace (from ``_get_guide_trace``).

        Returns:
            Predicted force tensor.
        """
        net = self.bnn_net
        F_descrp = batch[BatchIdx.F_DESCRP]
        F_logic_reduce = batch[BatchIdx.F_LOGIC_REDUCE]
        F_sfderiv_i = batch[BatchIdx.F_SFDERIV_I]
        F_sfderiv_j = batch[BatchIdx.F_SFDERIV_J]
        F_indices = batch[BatchIdx.F_INDICES]
        F_indices_i = batch[BatchIdx.F_INDICES_I]
        max_nnb = (
            F_sfderiv_j[0].shape[1]
            if len(F_sfderiv_j) > 0 and F_sfderiv_j[0].shape[0] > 0
            else 0
        )

        with torch.inference_mode(False):
            F_descrp_grad = [
                d.clone().detach().float().requires_grad_(True)
                for d in F_descrp
            ]
            F_sfderiv_i_f = [s.clone().float() for s in F_sfderiv_i]
            F_sfderiv_j_f = [s.clone().float() for s in F_sfderiv_j]
            F_logic_reduce_f = [l.clone().float() for l in F_logic_reduce]
            F_indices_c = F_indices.clone() if isinstance(F_indices, torch.Tensor) else [idx.clone() for idx in F_indices]
            F_indices_i_c = F_indices_i.clone() if isinstance(F_indices_i, torch.Tensor) else F_indices_i
            with poutine.replay(trace=guide_tr):
                _, F_pred = net.forward_F(
                    F_descrp_grad,
                    F_sfderiv_i_f,
                    F_sfderiv_j_f,
                    F_indices_c,
                    F_indices_i_c,
                    F_logic_reduce_f,
                    net.input_size,
                    max_nnb,
                )
        return F_pred

    def _compute_force_rmse(self, batch):
        """Compute force RMSE using a single guide sample (via replay)."""
        F_descrp = batch[BatchIdx.F_DESCRP]
        F_forces = batch[BatchIdx.F_FORCES]
        if F_descrp is None or F_forces is None:
            return torch.tensor(0.0, device=self.device)
        if isinstance(F_descrp, list) and len(F_descrp) == 0:
            return torch.tensor(0.0, device=self.device)

        E_descrp = batch[BatchIdx.E_DESCRP]
        E_logic_reduce = batch[BatchIdx.E_LOGIC_REDUCE]

        guide_tr = self._get_guide_trace(E_descrp, E_logic_reduce, use_mean=False)
        F_pred = self._replay_forward_F(batch, guide_tr)

        diff = F_pred.detach() - F_forces.float()
        return torch.sqrt(torch.mean(diff ** 2))  # normalized units

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

    def on_test_start(self) -> None:
        """Initialize BNN for testing (skip if already defined from training)."""
        if not hasattr(self, "bnn_net") or self.bnn_net is None:
            self.define_bnn()
            param_store_to(self.device)

    def test_step(self, batch: List[torch.Tensor], batch_idx: int) -> None:
        """Test step with energy, force, and total RMSE metrics."""
        x = batch[BatchIdx.E_DESCRP], batch[BatchIdx.E_LOGIC_REDUCE]
        y = batch[BatchIdx.E_ENERGY]
        n_atoms = batch[BatchIdx.E_N_ATOM]

        with torch.no_grad():
            preds = []
            for _ in range(self.hparams.mc_samples_eval):
                p = self.guided_forward(x[0], x[1])
                preds.append(p.unsqueeze(0) if p.dim() == 1 else p)
            loc = torch.cat(preds, 0).mean(0)
            if loc.dim() == 1:
                loc = loc.unsqueeze(-1)

        rmse = get_rmse_atom(loc.squeeze(-1), y, n_atoms)  # normalized units
        # Force RMSE needs autograd; exit inference_mode (set by Lightning test)
        with torch.inference_mode(False):
            force_rmse = self._compute_force_rmse(batch)
        alpha = getattr(self.net, "alpha", 0.1)
        if hasattr(alpha, "item"):
            alpha = alpha.item()
        total_rmse = (1 - alpha) * rmse + alpha * force_rmse

        self.log("rmse/test", rmse, on_step=False, on_epoch=True, batch_size=len(y))
        self.log("force_rmse/test", force_rmse, on_step=False, on_epoch=True, batch_size=len(y))
        self.log("total_rmse/test", total_rmse, on_step=False, on_epoch=True, batch_size=len(y))

    def on_predict_start(self) -> None:
        """Initialize BNN for prediction (skip if already defined)."""
        if not hasattr(self, "bnn_net") or self.bnn_net is None:
            self.define_bnn()
            param_store_to(self.device)
        # Retrieve learned noise scales from param store (if trained with learn_noise=True)
        ps = pyro.get_param_store()
        if "obs_scale_force" in ps:
            self.learned_scale_force = ps["obs_scale_force"].item()
        if "obs_scale_energy" in ps:
            self.learned_scale_energy = ps["obs_scale_energy"].item()

    def predict_step(
        self,
        batch: List[torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Dict[str, np.ndarray]:
        """Predict energies and forces with uncertainty.

        Forces are predicted at the **posterior mean** weights (guide loc),
        which gives much better force quality than MC-averaging because the
        BNN posterior is too wide for individual samples to produce
        consistent forces.  Uncertainty (std) is still estimated from MC
        samples.
        """
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
            # --- Posterior-mean force prediction (via poutine.replay) ---
            # The posterior mean gives much better forces than MC averaging
            # because the wide BNN posterior causes individual samples to
            # produce highly variable forces that degrade when averaged.
            # NOTE: guide trace must be created inside inference_mode(False)
            # so sampled weight tensors are not inference tensors (which
            # cannot be used by autograd in forward_F).
            with torch.inference_mode(False):
                mean_tr = self._get_guide_trace(x[0], x[1], use_mean=True)
                F_pred_mean = self._replay_forward_F(batch, mean_tr)
            force_preds = F_pred_mean.detach().cpu().numpy()

            # MC samples for force uncertainty (std) estimation
            force_samples = []
            for _ in range(self.hparams.mc_samples_eval):
                with torch.inference_mode(False):
                    sample_tr = self._get_guide_trace(x[0], x[1], use_mean=False)
                    F_pred_s = self._replay_forward_F(batch, sample_tr)
                force_samples.append(F_pred_s.detach().cpu().numpy())
            epistemic_std = np.stack(force_samples).std(axis=0)

            # Total predictive uncertainty = epistemic (MC weight std) + aleatoric (obs noise)
            # σ_total = sqrt(σ_epistemic² + σ_aleatoric²)
            # Use learned scale_force if available (from learn_noise=True), else fixed.
            if getattr(self, "learned_scale_force", None) is not None:
                aleatoric_std = self.learned_scale_force
            else:
                aleatoric_std = getattr(self, "scale_force", self.hparams.scale_force)
            force_stds = np.sqrt(epistemic_std ** 2 + aleatoric_std ** 2)

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


class PartialBNN_Forces(BNN_Forces):
    """
    Partial BNN with joint energy+force likelihood.

    Same as BNN_Forces but only specified layers are Bayesian; others are
    deterministic (scale frozen to near-zero). Uses bayesian_layers as in
    PartialBNN.

    Args:
        bayesian_layers: Which layers should be Bayesian.
            - "all": All layers (default BNN_Forces behavior)
            - "last": Only the last linear layer of each species subnet
            - "first": Only the first linear layer of each species subnet
            - "first_last": First and last layers
            - List[int]: Specific layer indices, e.g., [0, 2]
            - Dict: Per-species, e.g., {"Ti": "last", "O": "first"}
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
        learn_noise: bool = False,
        bayesian_layers: Union[str, List[int], Dict] = "all",
        name: str = "PartialBNN_Forces",
    ):
        self._bayesian_layers_config = bayesian_layers
        super().__init__(
            net, lr, pretrain_epochs, mc_samples_train, mc_samples_eval,
            dataset_size, fit_context, prior_loc, prior_scale, guide,
            q_scale, obs_scale, scale_force=scale_force,
            grad_clip_val=grad_clip_val, learn_noise=learn_noise, name=name,
        )
        self.save_hyperparameters(logger=False, ignore=["net"])

    def _get_linear_layer_names(self) -> List[str]:
        """Get names of all linear layers in the network."""
        linear_layers = []
        for name, module in self.net.named_modules():
            if isinstance(module, nn.Linear):
                linear_layers.append(name)
        return linear_layers

    def _get_bayesian_layer_names(self) -> List[str]:
        """Determine which layers should be Bayesian based on configuration."""
        all_linear_layers = self._get_linear_layer_names()
        config = self._bayesian_layers_config

        if config == "all":
            return all_linear_layers

        species_layers = {}
        for layer_name in all_linear_layers:
            if "functions." in layer_name:
                parts = layer_name.split(".")
                species_idx = int(parts[1])
                if species_idx not in species_layers:
                    species_layers[species_idx] = []
                species_layers[species_idx].append(layer_name)

        bayesian_layers = []

        if config == "last":
            for species_idx, layers in species_layers.items():
                if layers:
                    bayesian_layers.append(layers[-1])
        elif config == "first":
            for species_idx, layers in species_layers.items():
                if layers:
                    bayesian_layers.append(layers[0])
        elif config == "first_last":
            for species_idx, layers in species_layers.items():
                if layers:
                    bayesian_layers.append(layers[0])
                    if len(layers) > 1:
                        bayesian_layers.append(layers[-1])
        elif isinstance(config, list):
            for species_idx, layers in species_layers.items():
                for idx in config:
                    if 0 <= idx < len(layers):
                        bayesian_layers.append(layers[idx])
        elif isinstance(config, dict):
            for species_idx, layers in species_layers.items():
                species_name = (
                    self.net.species[species_idx]
                    if hasattr(self.net, "species")
                    else str(species_idx)
                )
                species_config = config.get(species_name, config.get(species_idx, "all"))
                if species_config == "all":
                    bayesian_layers.extend(layers)
                elif species_config == "last" and layers:
                    bayesian_layers.append(layers[-1])
                elif species_config == "first" and layers:
                    bayesian_layers.append(layers[0])
                elif species_config == "first_last" and layers:
                    bayesian_layers.append(layers[0])
                    if len(layers) > 1:
                        bayesian_layers.append(layers[-1])
                elif isinstance(species_config, list):
                    for idx in species_config:
                        if 0 <= idx < len(layers):
                            bayesian_layers.append(layers[idx])

        return bayesian_layers

    def define_bnn(self):
        """Override to apply selective Bayesian treatment."""
        super().define_bnn()

        bayesian_layer_names = self._get_bayesian_layer_names()
        all_layers = self._get_linear_layer_names()
        deterministic_layers = [l for l in all_layers if l not in bayesian_layer_names]

        if self.trainer and self.trainer.is_global_zero:
            log.info(
                "[PartialBNN_Forces] Bayesian layers: %s",
                bayesian_layer_names,
            )
            log.info(
                "[PartialBNN_Forces] Deterministic layers: %s",
                deterministic_layers,
            )

        param_store = pyro.get_param_store()
        frozen_count = 0
        for param_name in list(param_store.keys()):
            if ".scale" in param_name:
                is_bayesian = any(
                    layer in param_name for layer in bayesian_layer_names
                )
                if not is_bayesian:
                    param = param_store[param_name]
                    param.data.fill_(1e-8)
                    param.requires_grad = False
                    frozen_count += 1

        if self.trainer and self.trainer.is_global_zero:
            log.info("  Frozen %d scale parameters for deterministic layers", frozen_count)


# Backward compatibility for checkpoints
BNN_Forces_Likelihood = BNN_Forces
