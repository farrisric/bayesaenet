"""
Base Bayesian Neural Network models for atomic energy prediction.

Classes:
    - BNN: Full Bayesian Neural Network with variational inference
    - PartialBNN: Selective Bayesian treatment of layers

Note on Mixed Precision:
    - Flipout and Radial methods work with mixed precision (16-mixed)
    - LRT (Local Reparameterization Trick) should NOT use mixed precision
      as it causes NaN values in variational parameters. Use full precision
      (precision=32-true or no precision override) for LRT models.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO

import torch
from torch import nn
import torch.nn.functional as F
import tyxe
from tyxe import guides, priors, likelihoods, VariationalBNN
from tyxe.guides import AutoNormal
from .guides.radial import AutoRadial
import lightning.pytorch as L
from functools import partial
import copy
import contextlib
import numpy as np

from .utils import param_store_to, remove_dict_entry_startswith, weights_init, get_rmse_atom
from ..results.metrics import sharpness, rms_calibration_error
from ..datamodule.aenet.batch_constants import BatchIdx


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
            obs_scale: float,
            name: str = "BNN",  # Model name for logging organization
    ):
        super().__init__()
        pyro.clear_param_store()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        self.name = name  

    def define_bnn(self):
        if self.hparams.pretrain_epochs > 0:
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
         
    def on_fit_start(self) -> None:
        """Initialize BNN components at the start of training.
        
        Sets up the variational BNN, optimizer, loss function, and SVI objects.
        Caches svi_no_obs to avoid recreating it every step.
        
        Warning:
            For LRT (Local Reparameterization Trick), do NOT use mixed precision
            (trainer.precision='16-mixed') as it causes NaN in variational parameters.
        """
        # Warn about LRT and mixed precision
        if self.hparams.fit_context == "lrt":
            if hasattr(self.trainer, 'precision') and '16' in str(self.trainer.precision):
                warnings.warn(
                    "LRT (Local Reparameterization Trick) is incompatible with mixed precision. "
                    "This may cause NaN values in variational parameters. "
                    "Consider using precision='32-true' or removing precision override.",
                    UserWarning
                )
        
        self.define_bnn()
        param_store_to(self.device)
        self.configure_optimizers()

        # Use grad_clip_val if available (BNN_Forces_Aux), otherwise default to 10.0
        clip_norm = getattr(self.hparams, 'grad_clip_val', 10.0)
        self.optimizer = pyro.optim.ClippedAdam({
            'lr': self.hparams.lr, 
            'betas': [0.95, 0.999], 
            'clip_norm': clip_norm
        })
        self.loss = (
            TraceMeanField_ELBO(self.hparams.mc_samples_train)
            if self.hparams.guide != "radial"
            else Trace_ELBO(self.hparams.mc_samples_train)
        )

        self.svi = SVI(
            pyro.poutine.scale(self.bnn.model, scale=1.0/self.hparams.dataset_size),
            pyro.poutine.scale(self.bnn.guide, scale=1.0/self.hparams.dataset_size),
            self.optimizer,
            self.loss,
        )
        
        # Cache svi_no_obs to avoid recreation every step (performance optimization)
        self.bnn_no_obs = pyro.poutine.block(self.bnn, hide=["obs"])
        self.svi_no_obs = SVI(
            self.bnn_no_obs, self.bnn.guide, self.optimizer, self.loss
        )

    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> None:
        """Execute one training step.
        
        Args:
            batch: List of tensors containing energy and optionally force data.
                   Energy data at indices BatchIdx.E_* (10-14).
            batch_idx: Index of the current batch.
        """
        # Extract energy data using BatchIdx constants
        x = batch[BatchIdx.E_DESCRP], batch[BatchIdx.E_LOGIC_REDUCE]
        y = batch[BatchIdx.E_ENERGY]
        n_atoms = batch[BatchIdx.E_N_ATOM]
        
        # Note: svi_no_obs is cached in on_fit_start() for performance
        self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.increment_ready()

        with self.fit_ctxt():
            elbo = self.svi.step(x, y)
            
            # NaN protection: check guide parameters after SVI step
            with torch.no_grad():
                for name, val in pyro.get_param_store().items():
                    if torch.isnan(val).any():
                        warnings.warn(f"NaN detected in Pyro param '{name}' after SVI step, clamping.")
                        val.data = torch.nan_to_num(val.data, nan=0.0)
            
            loc, scale = self.bnn.predict(x[0], x[1], num_predictions=self.hparams.mc_samples_train)
            kl = self.svi_no_obs.evaluate_loss(x[0], x[1])

        self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.increment_ready()
        
        rmse = get_rmse_atom(loc, y, n_atoms)  # normalized units
        mse = F.mse_loss(loc, y)
        
        # NLL for training monitoring
        nll = F.gaussian_nll_loss(loc.squeeze(), y.squeeze(), torch.square(scale))
        
        # Calibration metrics (only compute if scale is valid)
        try:
            if scale.min() > 0:
                rmsce = rms_calibration_error(loc.squeeze(), scale.squeeze(), y.squeeze())
                sharp = sharpness(scale.squeeze())
                self.log("rmsce/train", rmsce, on_step=False, on_epoch=True, batch_size=len(y))
                self.log("sharp/train", sharp, on_step=False, on_epoch=True, batch_size=len(y))
        except (ValueError, RuntimeError):
            pass  # Skip calibration metrics if computation fails
        
        self.log("mse/train", mse, on_step=False, on_epoch=True, batch_size=len(y))
        self.log("rmse/train", rmse, on_step=False, prog_bar=True, on_epoch=True, batch_size=len(y))
        self.log("nll/train", nll, on_step=False, on_epoch=True, batch_size=len(y))
        self.log("elbo/train", elbo, on_step=False, on_epoch=True, batch_size=len(y))
        self.log("kl/train", kl, on_step=False, on_epoch=True, batch_size=len(y))
        self.log("likelihood/train", elbo - kl, on_step=False, on_epoch=True, batch_size=len(y))

    def validation_step(self, batch: List[torch.Tensor], batch_idx: int) -> None:
        """Execute one validation step.
        
        Args:
            batch: List of tensors containing energy data at BatchIdx.E_* indices.
            batch_idx: Index of the current batch.
        """
        # Extract energy data using BatchIdx constants
        x = batch[BatchIdx.E_DESCRP], batch[BatchIdx.E_LOGIC_REDUCE]
        y = batch[BatchIdx.E_ENERGY]
        n_atoms = batch[BatchIdx.E_N_ATOM]

        # Note: svi_no_obs is cached in on_fit_start() for performance
        elbo = self.svi.evaluate_loss(x, y.squeeze())
        # Aggregate = False if num_prediction = 1, else nans in sd
        loc, scale = self.bnn.predict(x[0], x[1], num_predictions=self.hparams.mc_samples_eval)
        kl = self.svi_no_obs.evaluate_loss(x[0], x[1])

        mse = F.mse_loss(loc, y)
        rmse = get_rmse_atom(loc, y, n_atoms)  # normalized units
        
        # NLL (Negative Log-Likelihood) for proper Bayesian evaluation
        nll = F.gaussian_nll_loss(loc.squeeze(), y.squeeze(), torch.square(scale))
        
        # Calibration metrics (only compute if scale is valid)
        try:
            if scale.min() > 0:
                rmsce = rms_calibration_error(loc.squeeze(), scale.squeeze(), y.squeeze())
                sharp = sharpness(scale.squeeze())
                self.log("rmsce/val", rmsce, on_step=False, on_epoch=True, batch_size=len(y))
                self.log("sharp/val", sharp, on_step=False, on_epoch=True, batch_size=len(y))
        except (ValueError, RuntimeError):
            pass  # Skip calibration metrics if computation fails
        
        self.log("rmse/val", rmse, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(y))
        self.log("nll/val", nll, on_step=False, on_epoch=True, batch_size=len(y))
        self.log("elbo/val", elbo, on_step=False, on_epoch=True, batch_size=len(y))
        self.log("mse/val", mse, on_step=False, on_epoch=True, batch_size=len(y))
        self.log("kl/val", kl, on_step=False, on_epoch=True, batch_size=len(y))
        self.log("likelihood/val", elbo - kl, on_step=False, on_epoch=True, batch_size=len(y))

    def on_test_start(self) -> None:
        """Initialize BNN for testing."""
        self.define_bnn()
        param_store_to(self.device)

    def test_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Execute one test step.
        
        Args:
            batch: List of tensors containing energy data.
            batch_idx: Index of the current batch.
            
        Returns:
            NLL loss value.
        """
        x = batch[BatchIdx.E_DESCRP], batch[BatchIdx.E_LOGIC_REDUCE]
        y = batch[BatchIdx.E_ENERGY]
        n_atoms = batch[BatchIdx.E_N_ATOM]
        
        loc, scale = self.bnn.predict(x[0], x[1], num_predictions=self.hparams.mc_samples_eval)

        nll = F.gaussian_nll_loss(loc.squeeze(), y.squeeze(), torch.square(scale))

        mse = F.mse_loss(loc, y)
        rmse = get_rmse_atom(loc, y, n_atoms)  # normalized units
        
        # Calibration metrics
        try:
            if scale.min() > 0:
                rmsce = rms_calibration_error(loc.squeeze(), scale.squeeze(), y.squeeze())
                sharp = sharpness(scale.squeeze())
                self.log("rmsce/test", rmsce, batch_size=len(y))
                self.log("sharp/test", sharp, batch_size=len(y))
        except Exception:
            pass
        
        self.log("nll/test", nll, batch_size=len(y))
        self.log("mse/test", mse, batch_size=len(y))
        self.log("rmse/test", rmse, batch_size=len(y))
        return nll
    
    def on_predict_start(self) -> None:
        """Initialize BNN for prediction."""
        self.define_bnn()
        param_store_to(self.device)
        # Create bnn_no_obs (needed by BNN_Forces_Aux.predict_step for force sampling)
        self.bnn_no_obs = pyro.poutine.block(self.bnn, hide=["obs"])

    def predict_step(
        self, 
        batch: List[torch.Tensor], 
        batch_idx: int, 
        dataloader_idx: int = 0
    ) -> Dict[str, np.ndarray]:
        """Execute one prediction step with uncertainty estimation.
        
        Args:
            batch: List of tensors with energy data.
            batch_idx: Batch index.
            dataloader_idx: Dataloader index.
            
        Returns:
            Dictionary with 'true', 'preds', 'stds', and 'n_atoms' arrays.
        """
        x = batch[BatchIdx.E_DESCRP], batch[BatchIdx.E_LOGIC_REDUCE]
        y = batch[BatchIdx.E_ENERGY]
        n_atoms = batch[BatchIdx.E_N_ATOM]
        
        pred = {}
        
        output = self.bnn.predict(
            x[0], x[1],
            num_predictions=self.hparams.mc_samples_eval,
            aggregate=False
        )
        preds = output.mean(axis=0)
        stds = output.std(axis=0)        
        
        pred["true"] = y.cpu().numpy()
        pred["preds"] = preds.cpu().numpy()
        pred["stds"] = stds.cpu().numpy()
        pred["n_atoms"] = n_atoms.cpu().numpy()
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


class PartialBNN(BNN):
    """
    Partially Bayesian Neural Network - selective Bayesian treatment of layers.
    
    Allows specifying which layers should have weight uncertainty (Bayesian)
    and which should remain deterministic. This is useful for:
    - Last-layer Bayesian: Only uncertainty in final layer (fast, often effective)
    - First-layer Bayesian: Uncertainty in input processing
    - Custom selection: Any combination of layers
    
    The non-Bayesian layers have their variational scale (uncertainty) frozen
    to a very small value, making them effectively deterministic.
    
    Args:
        bayesian_layers: Specification of which layers should be Bayesian.
            - "all": All layers are Bayesian (default BNN behavior)
            - "last": Only the last linear layer of each species subnet
            - "first": Only the first linear layer of each species subnet
            - "first_last": First and last layers
            - List[int]: Specific layer indices (0-indexed), e.g., [0, 2]
            - Dict: Per-species specification, e.g., {"Ti": [0, 2], "O": "last"}
        
    Example:
        ```python
        # Last-layer Bayesian (recommended for speed + good UQ)
        model = PartialBNN(net=net, bayesian_layers="last", ...)
        
        # First and last layers Bayesian
        model = PartialBNN(net=net, bayesian_layers="first_last", ...)
        
        # Custom: only layers 0 and 2
        model = PartialBNN(net=net, bayesian_layers=[0, 2], ...)
        ```
    """
    
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
            obs_scale: float,
            bayesian_layers: Union[str, List[int], Dict] = "all",
            name: str = "PartialBNN",
    ):
        # Store bayesian_layers before calling parent __init__
        self._bayesian_layers_config = bayesian_layers
        super().__init__(
            net=net,
            lr=lr,
            pretrain_epochs=pretrain_epochs,
            mc_samples_train=mc_samples_train,
            mc_samples_eval=mc_samples_eval,
            dataset_size=dataset_size,
            fit_context=fit_context,
            prior_loc=prior_loc,
            prior_scale=prior_scale,
            guide=guide,
            q_scale=q_scale,
            obs_scale=obs_scale,
            name=name,
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
        
        # Group layers by species (e.g., "functions.0.Linear_Sp1_F1" -> species 0)
        species_layers = {}
        for layer_name in all_linear_layers:
            # Extract species index from layer name
            if "functions." in layer_name:
                parts = layer_name.split(".")
                species_idx = int(parts[1])  # functions.{idx}.Linear_...
                if species_idx not in species_layers:
                    species_layers[species_idx] = []
                species_layers[species_idx].append(layer_name)
        
        bayesian_layers = []
        
        if config == "last":
            # Last layer of each species subnet
            for species_idx, layers in species_layers.items():
                if layers:
                    bayesian_layers.append(layers[-1])
        
        elif config == "first":
            # First layer of each species subnet
            for species_idx, layers in species_layers.items():
                if layers:
                    bayesian_layers.append(layers[0])
        
        elif config == "first_last":
            # First and last layers of each species subnet
            for species_idx, layers in species_layers.items():
                if layers:
                    bayesian_layers.append(layers[0])
                    if len(layers) > 1:
                        bayesian_layers.append(layers[-1])
        
        elif isinstance(config, list):
            # Specific layer indices for all species
            for species_idx, layers in species_layers.items():
                for idx in config:
                    if 0 <= idx < len(layers):
                        bayesian_layers.append(layers[idx])
        
        elif isinstance(config, dict):
            # Per-species configuration
            for species_idx, layers in species_layers.items():
                species_name = self.net.species[species_idx] if hasattr(self.net, 'species') else str(species_idx)
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
        # First, call parent to create the full BNN
        super().define_bnn()
        
        # Then freeze non-Bayesian layers by setting their scale to near-zero
        bayesian_layer_names = self._get_bayesian_layer_names()
        
        # Log which layers are Bayesian
        all_layers = self._get_linear_layer_names()
        deterministic_layers = [l for l in all_layers if l not in bayesian_layer_names]
        
        if self.trainer and self.trainer.is_global_zero:
            print(f"\n[PartialBNN] Layer configuration:")
            print(f"  Bayesian layers ({len(bayesian_layer_names)}): {bayesian_layer_names}")
            print(f"  Deterministic layers ({len(deterministic_layers)}): {deterministic_layers}")
        
        # Freeze scale parameters for non-Bayesian layers
        # This makes them effectively deterministic (point estimates)
        param_store = pyro.get_param_store()
        frozen_count = 0
        
        for param_name in list(param_store.keys()):
            # Scale parameters control uncertainty - freeze them for deterministic layers
            if '.scale' in param_name:
                # Check if this parameter belongs to a non-Bayesian layer
                is_bayesian = False
                for bayesian_layer in bayesian_layer_names:
                    if bayesian_layer in param_name:
                        is_bayesian = True
                        break
                
                if not is_bayesian:
                    # Freeze by setting requires_grad=False and value to very small
                    param = param_store[param_name]
                    param.data.fill_(1e-8)  # Very small scale = nearly deterministic
                    param.requires_grad = False
                    frozen_count += 1
        
        if self.trainer and self.trainer.is_global_zero:
            print(f"  Frozen {frozen_count} scale parameters for deterministic layers\n")
    
    def get_bayesian_param_count(self) -> Dict[str, int]:
        """Get count of Bayesian vs deterministic parameters."""
        bayesian_layers = self._get_bayesian_layer_names()
        bayesian_params = 0
        total_params = 0
        
        for name, param in self.net.named_parameters():
            param_count = param.numel()
            total_params += param_count
            
            # Check if this parameter is in a Bayesian layer
            for layer_name in bayesian_layers:
                if layer_name in name:
                    bayesian_params += param_count
                    break
        
        return {
            "bayesian_params": bayesian_params,
            "deterministic_params": total_params - bayesian_params,
            "total_params": total_params,
            "bayesian_fraction": bayesian_params / total_params if total_params > 0 else 0,
        }
