"""
BNN-AENET Model Implementations.

This module contains the core model classes for Bayesian Neural Networks
and standard Neural Networks for atomic energy and force prediction.

Models:
    - BNN: Base Bayesian Neural Network with variational inference
    - BNN_Forces_Aux: BNN with auxiliary force training
    - NN: Deterministic Neural Network (also used for BNN pretraining)
    - NN_Forces: NN with force training for Deep Ensemble

Note on Mixed Precision:
    - NN, Flipout, and Radial methods work with mixed precision (16-mixed)
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

# Import calibration metrics from results module
from ..results.metrics import sharpness, rms_calibration_error

# Import batch index constants for readable code
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
        
        rmse = get_rmse_atom(loc, y, n_atoms)
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
        rmse = get_rmse_atom(loc, y, n_atoms)
        
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
        rmse = get_rmse_atom(loc, y, n_atoms)
        
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


class NN(L.LightningModule):
    """
    Class used by BNNs to pretrain their weights. This class is instantiated,
    trained for X epochs and then it stores its weights in the log directory.
    VIBnnWrapper then loads the weights and starts the Bayesian training
    """

    def __init__(self,
                 net: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 name: str = "NN"):  # Model name for logging organization
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        self.name = name
        self.net.apply(weights_init)

    def forward(self, grp_descrp, logic_reduce):
        return self.net.forward(grp_descrp, logic_reduce)

    def step(self, batch: List[torch.Tensor]) -> torch.Tensor:
        """Compute RMSE for a batch.
        
        Args:
            batch: List of tensors with energy data at BatchIdx.E_* indices.
            
        Returns:
            RMSE per atom.
        """
        grp_descrp = batch[BatchIdx.E_DESCRP]
        grp_energy = batch[BatchIdx.E_ENERGY]
        logic_reduce = batch[BatchIdx.E_LOGIC_REDUCE]
        grp_N_atom = batch[BatchIdx.E_N_ATOM]
        
        list_E_ann = self.forward(grp_descrp, logic_reduce)   
        return get_rmse_atom(list_E_ann, grp_energy, grp_N_atom)

    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Execute one training step."""
        mse = self.step(batch)
        self.log("rmse/train", 
                 mse, 
                 on_step=False, 
                 on_epoch=True, 
                 prog_bar=True,
                 batch_size=len(batch[BatchIdx.E_ENERGY]))
        return mse

    def validation_step(self, batch: List[torch.Tensor], batch_idx: int) -> None:
        """Execute one validation step."""
        mse = self.step(batch)
        self.log("rmse/val",
                 mse,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 batch_size=len(batch[BatchIdx.E_ENERGY]))

    def test_step(self, batch: List[torch.Tensor], batch_idx: int) -> None:
        """Execute one test step."""
        mse = self.step(batch)
        self.log("rmse/test", mse, on_step=False, on_epoch=True, 
                 batch_size=len(batch[BatchIdx.E_ENERGY]))

    def predict_step(
        self, 
        batch: List[torch.Tensor], 
        batch_idx: int, 
        dataloader_idx: int = 0
    ) -> Dict[str, np.ndarray]:
        """Execute one prediction step.
        
        Args:
            batch: List of tensors with energy data.
            batch_idx: Batch index.
            dataloader_idx: Dataloader index for multi-dataloader setups.
            
        Returns:
            Dictionary with 'true', 'preds', and 'n_atoms' arrays.
        """
        grp_descrp = batch[BatchIdx.E_DESCRP]
        grp_energy = batch[BatchIdx.E_ENERGY]
        logic_reduce = batch[BatchIdx.E_LOGIC_REDUCE]
        grp_N_atom = batch[BatchIdx.E_N_ATOM]
        
        pred = {}
        
        true = grp_energy / self.net.e_scaling + self.net.e_shift * grp_N_atom
        list_E_ann = self.net.forward(grp_descrp, logic_reduce)
        preds = list_E_ann / self.net.e_scaling + self.net.e_shift * grp_N_atom

        pred["true"] = true.cpu().numpy()
        pred["preds"] = preds.cpu().numpy()
        pred["n_atoms"] = grp_N_atom.cpu().numpy()
        return pred
    
    def configure_optimizers(self):
        return self.hparams.optimizer(params=self.parameters())


class NN_Forces(NN):
    """
    Neural Network with force training for Deep Ensemble.
    
    Extends NN to include force training via auxiliary loss.
    Uses weighted combination of energy RMSE and force RMSE.
    """
    
    def __init__(self,
                 net: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 alpha: float = 0.1,
                 name: str = "NN_Forces"):  # Model name for logging organization
        super().__init__(net=net, optimizer=optimizer, name=name)
        self.alpha = alpha  # Weight for force loss: (1-alpha)*E_loss + alpha*F_loss
    
    def compute_force_loss(self, batch: List[torch.Tensor]) -> torch.Tensor:
        """Compute force RMSE from batch using forward_F.
        
        Args:
            batch: List of tensors with force data at BatchIdx.F_* indices.
            
        Returns:
            Force RMSE in mHa/Bohr, or 0.0 if no force data available.
        """
        # Get force data from batch using BatchIdx constants
        F_group_descrp = batch[BatchIdx.F_DESCRP]
        F_group_forces = batch[BatchIdx.F_FORCES]
        
        # Check if force data is available
        if F_group_descrp is None or F_group_forces is None:
            return torch.tensor(0.0, device=self.device)
        if isinstance(F_group_descrp, list) and len(F_group_descrp) == 0:
            return torch.tensor(0.0, device=self.device)
        
        # Extract remaining force-related data
        F_logic_reduce = batch[BatchIdx.F_LOGIC_REDUCE]
        F_sfderiv_i = batch[BatchIdx.F_SFDERIV_I]
        F_sfderiv_j = batch[BatchIdx.F_SFDERIV_J]
        F_indices = batch[BatchIdx.F_INDICES]
        F_indices_i = batch[BatchIdx.F_INDICES_I]
        
        # Compute max_nnb from sfderiv_j shape
        max_nnb = F_sfderiv_j[0].shape[1] if len(F_sfderiv_j) > 0 and F_sfderiv_j[0].shape[0] > 0 else 0
        
        net = self.net
        
        with torch.enable_grad():
            # Clone descriptors for gradient tracking, convert to float32
            F_descrp_grad = [d.clone().detach().float().requires_grad_(True) for d in F_group_descrp]
            F_sfderiv_i_f = [s.float() for s in F_sfderiv_i]
            F_sfderiv_j_f = [s.float() for s in F_sfderiv_j]
            F_logic_reduce_f = [l.float() for l in F_logic_reduce]
            
            # Compute forces via autodiff through forward_F
            E_pred, F_pred = net.forward_F(
                F_descrp_grad, F_sfderiv_i_f, F_sfderiv_j_f,
                F_indices, F_indices_i, F_logic_reduce_f,
                net.input_size,
                max_nnb
            )
        
        # Compute RMSE for forces (in mHa/Bohr)
        force_diff = F_pred - F_group_forces.float()
        force_rmse = torch.sqrt(torch.mean(force_diff ** 2)) * 1000
        
        return force_rmse
    
    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        # Energy loss
        energy_rmse = self.step(batch)
        
        # Force loss
        force_rmse = self.compute_force_loss(batch)
        
        # Combined loss with alpha weighting
        alpha = self.alpha
        total_loss = (1 - alpha) * energy_rmse + alpha * force_rmse
        
        # Logging
        batch_size = len(batch[BatchIdx.E_ENERGY])
        self.log("rmse/train", energy_rmse, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("force_rmse/train", force_rmse, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("total_loss/train", total_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("alpha", alpha, on_step=False, on_epoch=True)
        
        return total_loss
    
    def validation_step(self, batch: List[torch.Tensor], batch_idx: int) -> None:
        """Validation step with energy and force metrics."""
        energy_rmse = self.step(batch)
        force_rmse = self.compute_force_loss(batch)
        
        # Combined metric for HPS optimization: weighted sum of energy and force RMSE
        total_rmse = (1 - self.alpha) * energy_rmse + self.alpha * force_rmse
        
        batch_size = len(batch[BatchIdx.E_ENERGY])
        self.log("rmse/val", energy_rmse, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("force_rmse/val", force_rmse, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("total_rmse/val", total_rmse, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
    
    def test_step(self, batch: List[torch.Tensor], batch_idx: int) -> None:
        """Test step with energy and force metrics."""
        energy_rmse = self.step(batch)
        force_rmse = self.compute_force_loss(batch)
        
        # Combined metric
        total_rmse = (1 - self.alpha) * energy_rmse + self.alpha * force_rmse
        
        batch_size = len(batch[BatchIdx.E_ENERGY])
        self.log("rmse/test", energy_rmse, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("force_rmse/test", force_rmse, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("total_rmse/test", total_rmse, on_step=False, on_epoch=True, batch_size=batch_size)


class BNN_Forces_Aux(BNN):
    """
    BNN with auxiliary force loss for energy+force training.
    
    Inherits from BNN and adds force training via auxiliary loss term.
    Energy training uses standard TyXe ELBO (unchanged), forces added as separate loss.
    This approach is simpler than custom likelihood but less rigorous Bayesian-wise.
    
    Key features:
    - Energy ELBO computed via TyXe (standard variational inference)
    - Force loss computed separately and added as auxiliary term
    - Gradients from both losses flow through shared network parameters
    - Fully backward compatible with energy-only BNN training
    """
    
    def __init__(self, net, lr, pretrain_epochs, mc_samples_train, mc_samples_eval,
                 dataset_size, fit_context, prior_loc, prior_scale, guide, q_scale, 
                 obs_scale, force_lr_scale: float = 0.1,
                 scale_lr_factor: float = 0.5, grad_clip_val: float = 1.0,
                 name: str = "BNN_Forces"):
        """
        BNN with auxiliary force loss.
        
        Args:
            force_lr_scale: Learning rate scale for force updates (default 0.1 = 10% of main lr)
            scale_lr_factor: Learning rate factor for scale (uncertainty) param updates (default 0.5)
            grad_clip_val: Max gradient norm for force gradient clipping (default 1.0)
            name: Model name for logging organization
        
        Note: The actual force loss weight is: alpha * force_rmse
        where alpha comes from self.net.alpha (fixed at 0.1)
        """
        super().__init__(net, lr, pretrain_epochs, mc_samples_train, mc_samples_eval,
                        dataset_size, fit_context, prior_loc, prior_scale, guide, 
                        q_scale, obs_scale, name=name)
        self.save_hyperparameters(logger=False, ignore=["net"])
    
    def compute_force_loss(self, batch):
        """
        Compute force RMSE loss using forward_F through the BNN's sampled network.
        
        Args:
            batch: Data batch containing force information at indices [0-9]
        
        Returns:
            force_rmse: Root mean squared error of force predictions, or zero if no force data
        """
        # Extract force-related data from batch using BatchIdx constants
        F_group_descrp = batch[BatchIdx.F_DESCRP]
        F_group_forces = batch[BatchIdx.F_FORCES]
        
        # Check if force data is available (datamodule sets these to None if no force data)
        if F_group_descrp is None or F_group_forces is None:
            # Return zero loss if no force data available
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Check if force data is not empty list
        if isinstance(F_group_descrp, list) and len(F_group_descrp) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Extract remaining force-related data
        F_group_energy = batch[BatchIdx.F_ENERGY]  
        F_logic_reduce = batch[BatchIdx.F_LOGIC_REDUCE]
        F_group_N_atom = batch[BatchIdx.F_N_ATOM]
        F_sfderiv_i = batch[BatchIdx.F_SFDERIV_I]
        F_sfderiv_j = batch[BatchIdx.F_SFDERIV_J]
        F_indices = batch[BatchIdx.F_INDICES]
        F_indices_i = batch[BatchIdx.F_INDICES_I]
        
        # Get the underlying network from BNN
        net = self.net
        
        # Sync self.net parameters with Pyro guide's loc params (for validation/prediction)
        # This ensures force predictions use the current learned weights
        with torch.no_grad():
            for name, param in net.named_parameters():
                pyro_name = f'net_guide.net.{name}.loc'
                if pyro_name in pyro.get_param_store().keys():
                    loc_param = pyro.get_param_store()._params[pyro_name]
                    param.data.copy_(loc_param.data)
        
        # Compute max_nnb from sfderiv_j shape
        max_nnb = F_sfderiv_j[0].shape[1] if len(F_sfderiv_j) > 0 and F_sfderiv_j[0].shape[0] > 0 else 0
        
        # Ensure gradient computation is enabled for force calculation
        with torch.enable_grad():
            # Clone descriptors to ensure proper gradient tracking
            # Also convert to float32 to match network dtype
            F_descrp_grad = [d.clone().detach().float().requires_grad_(True) for d in F_group_descrp]
            
            # Convert other tensors to float32 for consistency
            F_sfderiv_i_f = [s.float() for s in F_sfderiv_i]
            F_sfderiv_j_f = [s.float() for s in F_sfderiv_j]
            F_logic_reduce_f = [l.float() for l in F_logic_reduce]
            
            # Compute forces via autodiff through forward_F
            E_pred, F_pred = net.forward_F(
                F_descrp_grad, F_sfderiv_i_f, F_sfderiv_j_f,
                F_indices, F_indices_i, F_logic_reduce_f,
                net.input_size,
                max_nnb
            )
        
        # Compute RMSE for forces (in mHa/Bohr for consistency)
        # Convert target forces to float32 for consistency
        force_diff = F_pred - F_group_forces.float()
        force_rmse = torch.sqrt(torch.mean(force_diff ** 2)) * 1000  # Convert to mHa/Bohr
        
        return force_rmse
    
    def compute_force_loss_and_update(self, batch):
        """
        Compute force RMSE loss and update TyXe guide parameters directly.
        
        TyXe's SVI.step() handles its own backward pass internally, so calling
        backward() on a separate loss doesn't affect the variational parameters.
        This method properly updates the guide's parameters (both loc AND scale) by:
        1. Setting self.net's parameters to the guide's loc values
        2. Computing force loss through self.net
        3. Backpropagating to get gradients
        4. Applying gradients to both loc (mean) and scale (uncertainty) params
        
        The loss uses alpha from self.net.alpha to balance energy/force contributions,
        following the original aenet formula: loss = (1-alpha)*E_loss + alpha*F_loss
        
        Args:
            batch: Data batch containing force information at BatchIdx.F_* indices
        
        Returns:
            force_rmse: Root mean squared error of force predictions (for logging)
        """
        # Extract force-related data from batch using BatchIdx constants
        F_group_descrp = batch[BatchIdx.F_DESCRP]
        F_group_forces = batch[BatchIdx.F_FORCES]
        
        # Check if force data is available
        if F_group_descrp is None or F_group_forces is None:
            return torch.tensor(0.0, device=self.device)
        
        if isinstance(F_group_descrp, list) and len(F_group_descrp) == 0:
            return torch.tensor(0.0, device=self.device)
        
        # Extract remaining force-related data
        F_logic_reduce = batch[BatchIdx.F_LOGIC_REDUCE]
        F_sfderiv_i = batch[BatchIdx.F_SFDERIV_I]
        F_sfderiv_j = batch[BatchIdx.F_SFDERIV_J]
        F_indices = batch[BatchIdx.F_INDICES]
        F_indices_i = batch[BatchIdx.F_INDICES_I]
        
        # Compute max_nnb from sfderiv_j shape
        max_nnb = F_sfderiv_j[0].shape[1] if len(F_sfderiv_j) > 0 and F_sfderiv_j[0].shape[0] > 0 else 0
        
        # Convert force data to float32
        F_descrp_f = [d.float().requires_grad_(True) for d in F_group_descrp]
        F_sfderiv_i_f = [s.float() for s in F_sfderiv_i]
        F_sfderiv_j_f = [s.float() for s in F_sfderiv_j]
        F_logic_reduce_f = [l.float() for l in F_logic_reduce]
        F_target = F_group_forces.float()
        
        # Build mapping from self.net param names to Pyro param store names
        # Pyro stores guide params as: net_guide.net.{net_param_name}.loc and .scale
        param_mapping_loc = {}
        param_mapping_scale = {}
        
        for name, param in self.net.named_parameters():
            loc_name = f'net_guide.net.{name}.loc'
            scale_name = f'net_guide.net.{name}.scale'
            if loc_name in pyro.get_param_store().keys():
                param_mapping_loc[name] = loc_name
            if scale_name in pyro.get_param_store().keys():
                param_mapping_scale[name] = scale_name
        
        if len(param_mapping_loc) == 0:
            # No Pyro params found - SVI might not be initialized yet
            return self.compute_force_loss(batch)
        
        # Get alpha from network (balances energy vs force loss)
        # alpha=0.5 means 50% energy, 50% force
        alpha = self.net.alpha.item() if hasattr(self.net, 'alpha') else 0.5
        
        # Set self.net's parameters to the guide's loc values
        with torch.enable_grad():
            # Save original params to restore later
            original_params = {name: p.data.clone() for name, p in self.net.named_parameters()}
            
            # Sync net params with Pyro loc params
            with torch.no_grad():
                for name, param in self.net.named_parameters():
                    if name in param_mapping_loc:
                        pyro_p = pyro.get_param_store()._params[param_mapping_loc[name]]
                        param.data.copy_(pyro_p.data)
            
            # Compute forces using self.net
            E_pred, F_pred = self.net.forward_F(
                F_descrp_f, F_sfderiv_i_f, F_sfderiv_j_f,
                F_indices, F_indices_i, F_logic_reduce_f,
                self.net.input_size,
                max_nnb
            )
            
            # Compute RMSE for forces (in mHa/Bohr)
            force_diff = F_pred - F_target
            force_rmse = torch.sqrt(torch.mean(force_diff ** 2)) * 1000
            
            # Apply alpha weighting: the force contribution should be scaled by alpha
            # (energy contribution is already handled by ELBO with implicit 1-alpha weight)
            weighted_loss = alpha * force_rmse
            
            if weighted_loss.requires_grad:
                # Zero existing gradients
                for p in self.net.parameters():
                    if p.grad is not None:
                        p.grad.zero_()
                
                # Backprop to get gradients on self.net params
                weighted_loss.backward(retain_graph=False)
                
                # Clip gradients to prevent NaN in variational parameters
                grad_clip_val = getattr(self.hparams, 'grad_clip_val', 1.0)
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), grad_clip_val)
                
                # Apply gradients to Pyro param store directly
                force_lr_scale = getattr(self.hparams, 'force_lr_scale', 0.1)
                force_lr = self.hparams.lr * force_lr_scale
                
                # Scale learning rate for scale params (uncertainty)
                # Use smaller updates to avoid destabilizing the uncertainty estimates
                scale_lr_factor = getattr(self.hparams, 'scale_lr_factor', 0.5)
                
                with torch.no_grad():
                    for name, param in self.net.named_parameters():
                        if param.grad is not None:
                            grad = param.grad
                            
                            # Update loc (mean) parameters
                            if name in param_mapping_loc:
                                pyro_p = pyro.get_param_store()._params[param_mapping_loc[name]]
                                pyro_p -= force_lr * grad
                            
                            # Update scale (uncertainty) parameters
                            # The gradient for scale is approximated as grad * current_scale
                            # This increases uncertainty where gradients are large (high error)
                            if name in param_mapping_scale:
                                scale_p = pyro.get_param_store()._params[param_mapping_scale[name]]
                                # Scale params use softplus transform, so we update the unconstrained
                                # Increase scale where error gradients are large
                                scale_grad = torch.abs(grad) * torch.exp(scale_p)
                                scale_p += force_lr * scale_lr_factor * scale_grad.mean()
                
                # NaN protection after force gradient update
                for pname, val in pyro.get_param_store().items():
                    if torch.isnan(val).any():
                        warnings.warn(f"NaN detected in Pyro param '{pname}' after force update, clamping.")
                        val.data = torch.nan_to_num(val.data, nan=0.0)
            
            # Restore original params
            for name, param in self.net.named_parameters():
                if name in original_params:
                    param.data.copy_(original_params[name])
        
        return force_rmse.detach()
    
    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> None:
        """Training step with energy ELBO and auxiliary force loss."""
        # Extract energy data using BatchIdx constants
        x = batch[BatchIdx.E_DESCRP], batch[BatchIdx.E_LOGIC_REDUCE]
        y = batch[BatchIdx.E_ENERGY]
        n_atoms = batch[BatchIdx.E_N_ATOM]
        
        # Note: svi_no_obs is cached in on_fit_start() for performance
        self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.increment_ready()
        
        with self.fit_ctxt():
            # 1. Energy ELBO step (unchanged from parent BNN)
            elbo = self.svi.step(x, y)
            
            # NaN protection: check guide parameters after SVI step
            with torch.no_grad():
                for name, val in pyro.get_param_store().items():
                    if torch.isnan(val).any():
                        warnings.warn(f"NaN detected in Pyro param '{name}' after SVI step, clamping.")
                        val.data = torch.nan_to_num(val.data, nan=0.0)
            
            loc, scale = self.bnn.predict(x[0], x[1], num_predictions=self.hparams.mc_samples_train)
            kl = self.svi_no_obs.evaluate_loss(x[0], x[1])
            
            # 2. Compute auxiliary force loss and update guide parameters
            force_loss = self.compute_force_loss_and_update(batch)
        
        self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.increment_ready()
        
        # Compute energy metrics
        rmse = get_rmse_atom(loc, y, n_atoms)
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
            pass
        
        # Get alpha for logging
        alpha = self.net.alpha.item() if hasattr(self.net, 'alpha') else 0.5
        
        # Log all metrics
        self.log("mse/train", mse, on_step=False, on_epoch=True, batch_size=len(y))
        self.log("rmse/train", rmse, on_step=False, prog_bar=True, on_epoch=True, batch_size=len(y))
        self.log("nll/train", nll, on_step=False, on_epoch=True, batch_size=len(y))
        self.log("force_rmse/train", force_loss, on_step=False, on_epoch=True, batch_size=len(y), prog_bar=True)
        self.log("alpha", alpha, on_step=False, on_epoch=True, batch_size=len(y))
        self.log("elbo/train", elbo, on_step=False, on_epoch=True, batch_size=len(y))
        self.log("kl/train", kl, on_step=False, on_epoch=True, batch_size=len(y))
        self.log("likelihood/train", elbo - kl, on_step=False, on_epoch=True, batch_size=len(y))
    
    def validation_step(self, batch: List[torch.Tensor], batch_idx: int) -> None:
        """Validation step with energy ELBO and force metrics."""
        # Extract energy data using BatchIdx constants
        x = batch[BatchIdx.E_DESCRP], batch[BatchIdx.E_LOGIC_REDUCE]
        y = batch[BatchIdx.E_ENERGY]
        n_atoms = batch[BatchIdx.E_N_ATOM]
        
        # Note: svi_no_obs is cached in on_fit_start() for performance
        # Energy ELBO
        elbo = self.svi.evaluate_loss(x, y.squeeze())
        loc, scale = self.bnn.predict(x[0], x[1], num_predictions=self.hparams.mc_samples_eval)
        kl = self.svi_no_obs.evaluate_loss(x[0], x[1])
        
        # Force loss
        force_loss = self.compute_force_loss(batch)
        
        # Energy metrics
        mse = F.mse_loss(loc, y)
        rmse = get_rmse_atom(loc, y, n_atoms)
        
        # NLL for validation monitoring
        nll = F.gaussian_nll_loss(loc.squeeze(), y.squeeze(), torch.square(scale))
        
        # Calibration metrics (only compute if scale is valid)
        try:
            if scale.min() > 0:
                rmsce = rms_calibration_error(loc.squeeze(), scale.squeeze(), y.squeeze())
                sharp = sharpness(scale.squeeze())
                self.log("rmsce/val", rmsce, on_step=False, on_epoch=True, batch_size=len(y))
                self.log("sharp/val", sharp, on_step=False, on_epoch=True, batch_size=len(y))
        except (ValueError, RuntimeError):
            pass
        
        # Combined metric for HPS optimization: weighted sum of energy and force RMSE
        alpha = getattr(self.net, 'alpha', 0.5)  # Get alpha from network config
        total_rmse = (1 - alpha) * rmse + alpha * force_loss
        
        # Log all metrics
        self.log("rmse/val", rmse, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(y))
        self.log("nll/val", nll, on_step=False, on_epoch=True, batch_size=len(y))
        self.log("force_rmse/val", force_loss, on_step=False, on_epoch=True, batch_size=len(y), prog_bar=True)
        self.log("total_rmse/val", total_rmse, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(y))
        self.log("elbo/val", elbo, on_step=False, on_epoch=True, batch_size=len(y))
        self.log("mse/val", mse, on_step=False, on_epoch=True, batch_size=len(y))
        self.log("kl/val", kl, on_step=False, on_epoch=True, batch_size=len(y))
        self.log("likelihood/val", elbo - kl, on_step=False, on_epoch=True, batch_size=len(y))
    
    def test_step(self, batch: List[torch.Tensor], batch_idx: int) -> None:
        """Test step with combined energy+force metrics."""
        # Extract energy data using BatchIdx constants
        x = batch[BatchIdx.E_DESCRP], batch[BatchIdx.E_LOGIC_REDUCE]
        y = batch[BatchIdx.E_ENERGY]
        n_atoms = batch[BatchIdx.E_N_ATOM]
        
        loc, scale = self.bnn.predict(x[0], x[1], num_predictions=self.hparams.mc_samples_eval)
        
        # Energy metrics
        rmse = get_rmse_atom(loc, y, n_atoms)
        nll = F.gaussian_nll_loss(loc.squeeze(), y.squeeze(), torch.square(scale))
        
        # Force metrics
        force_loss = self.compute_force_loss(batch)
        
        # Combined metric
        alpha = getattr(self.net, 'alpha', 0.5)
        total_rmse = (1 - alpha) * rmse + alpha * force_loss
        
        # Calibration metrics
        try:
            if scale.min() > 0:
                rmsce = rms_calibration_error(loc.squeeze(), scale.squeeze(), y.squeeze())
                sharp = sharpness(scale.squeeze())
                self.log("rmsce/test", rmsce, on_step=False, on_epoch=True, batch_size=len(y))
                self.log("sharp/test", sharp, on_step=False, on_epoch=True, batch_size=len(y))
        except (ValueError, RuntimeError):
            pass
        
        self.log("rmse/test", rmse, on_step=False, on_epoch=True, batch_size=len(y))
        self.log("nll/test", nll, on_step=False, on_epoch=True, batch_size=len(y))
        self.log("force_rmse/test", force_loss, on_step=False, on_epoch=True, batch_size=len(y))
        self.log("total_rmse/test", total_rmse, on_step=False, on_epoch=True, batch_size=len(y))
    
    def predict_step(
        self, 
        batch: List[torch.Tensor], 
        batch_idx: int, 
        dataloader_idx: int = 0
    ) -> Dict[str, np.ndarray]:
        """Predict both energies and forces with uncertainty.
        
        Args:
            batch: List of tensors with energy and force data.
            batch_idx: Batch index.
            dataloader_idx: Dataloader index.
            
        Returns:
            Dictionary with energy predictions, force predictions, and uncertainties.
        """
        # Extract energy data using BatchIdx constants
        x = batch[BatchIdx.E_DESCRP], batch[BatchIdx.E_LOGIC_REDUCE]
        y = batch[BatchIdx.E_ENERGY]
        n_atoms = batch[BatchIdx.E_N_ATOM]
        
        pred = {}
        
        # Energy predictions (same as parent BNN)
        output = self.bnn.predict(
            x[0], x[1],
            num_predictions=self.hparams.mc_samples_eval,
            aggregate=False
        )
        energy_preds = output.mean(axis=0)
        energy_stds = output.std(axis=0)
        
        # Check if force data is available using BatchIdx constants
        F_group_descrp = batch[BatchIdx.F_DESCRP]
        F_group_forces = batch[BatchIdx.F_FORCES]
        has_force_data = (F_group_descrp is not None and 
                          F_group_forces is not None and
                          not (isinstance(F_group_descrp, list) and len(F_group_descrp) == 0))
        
        if has_force_data:
            # Force predictions (compute for each MC sample)
            force_samples = []
            for _ in range(self.hparams.mc_samples_eval):
                with torch.no_grad():
                    # Sample network from guide
                    guide_trace = pyro.poutine.trace(self.bnn.guide).get_trace(x[0], x[1])
                    model_trace = pyro.poutine.trace(pyro.poutine.replay(self.bnn_no_obs, guide_trace)).get_trace(x[0], x[1])
                    
                    # Compute forces with this sampled network using BatchIdx constants
                    F_logic_reduce = batch[BatchIdx.F_LOGIC_REDUCE]
                    F_sfderiv_i = batch[BatchIdx.F_SFDERIV_I]
                    F_sfderiv_j = batch[BatchIdx.F_SFDERIV_J]
                    F_indices = batch[BatchIdx.F_INDICES]
                    F_indices_i = batch[BatchIdx.F_INDICES_I]
                    max_nnb = F_sfderiv_j[0].shape[1] if len(F_sfderiv_j) > 0 and F_sfderiv_j[0].shape[0] > 0 else 0
                    
                    # Convert to float32 for consistency (same as compute_force_loss)
                    F_descrp_f = [d.clone().detach().float().requires_grad_(True) for d in F_group_descrp]
                    F_sfderiv_i_f = [s.float() for s in F_sfderiv_i]
                    F_sfderiv_j_f = [s.float() for s in F_sfderiv_j]
                    F_logic_reduce_f = [l.float() for l in F_logic_reduce]
                    
                    with torch.enable_grad():
                        _, F_pred = self.bnn.net.forward_F(
                            F_descrp_f, F_sfderiv_i_f, F_sfderiv_j_f,
                            F_indices, F_indices_i, F_logic_reduce_f,
                            self.bnn.net.input_size, max_nnb
                        )
                    force_samples.append(F_pred.detach().cpu().numpy())
            
            force_samples = np.array(force_samples)
            force_preds = force_samples.mean(axis=0)
            force_stds = force_samples.std(axis=0)
            
            # Flatten forces for storage (Nx3 -> N*3)
            true_forces_flat = F_group_forces.cpu().numpy().flatten()
            pred_forces_flat = force_preds.flatten()
            std_forces_flat = force_stds.flatten()
            
            # Compute per-atom force errors
            force_errors = np.abs(true_forces_flat - pred_forces_flat)
            force_rmse = np.sqrt(np.mean((true_forces_flat - pred_forces_flat)**2))
            force_mae = np.mean(force_errors)
        else:
            # No force data - set to None/NaN
            true_forces_flat = None
            pred_forces_flat = None
            std_forces_flat = None
            force_rmse = np.nan
            force_mae = np.nan
        
        # Store all predictions
        pred["true"] = y.cpu().numpy()
        pred["preds"] = energy_preds.cpu().numpy()
        pred["stds"] = energy_stds.cpu().numpy()
        pred["n_atoms"] = n_atoms.cpu().numpy()
        pred["true_forces"] = true_forces_flat
        pred["pred_forces"] = pred_forces_flat
        pred["std_forces"] = std_forces_flat
        pred["force_rmse"] = force_rmse
        pred["force_mae"] = force_mae
        
        return pred


class PartialBNN_Forces_Aux(BNN_Forces_Aux, PartialBNN):
    """
    Partially Bayesian Neural Network with auxiliary force loss.
    
    Combines selective Bayesian treatment of layers (PartialBNN) with
    force training (BNN_Forces_Aux). Use this for:
    - Fast UQ with forces using "last" layer Bayesian
    - Targeted uncertainty modeling with force-aware training
    
    Args:
        bayesian_layers: Same as PartialBNN - "all", "last", "first", "first_last",
                        List[int], or Dict for per-species configuration.
        force_weight: Multiplier for force loss (default 1.0)
        force_lr_scale: Learning rate scale for force updates (default 0.1)
        scale_lr_factor: Learning rate factor for scale updates (default 0.5)
    
    Example:
        ```python
        # Last-layer Bayesian with force training
        model = PartialBNN_Forces_Aux(
            net=net,
            bayesian_layers="last",
            force_weight=1.0,
            ...
        )
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
            bayesian_layers: Union[str, List[int], Dict] = "last",
            force_lr_scale: float = 0.1,
            scale_lr_factor: float = 0.5,
            grad_clip_val: float = 1.0,
            name: str = "PartialBNN_Forces",
    ):
        # Call BNN_Forces_Aux init (which calls PartialBNN.__init__ due to MRO)
        BNN_Forces_Aux.__init__(
            self,
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
            force_lr_scale=force_lr_scale,
            scale_lr_factor=scale_lr_factor,
            grad_clip_val=grad_clip_val,
            name=name,
        )
        
        # Set bayesian_layers AFTER parent init (MRO causes PartialBNN.__init__ 
        # to be called which sets default "all", so we override it here)
        self._bayesian_layers_config = bayesian_layers
        self.save_hyperparameters(logger=False, ignore=["net"])
    
    def define_bnn(self):
        """Override to use PartialBNN's selective Bayesian treatment."""
        # Call PartialBNN's define_bnn which applies selective freezing
        PartialBNN.define_bnn(self)


def get_rmse_atom(list_E_ann, grp_energy, grp_N_atom):
    """
    Compute RMSE per atom in meV.
    
    Matches original aenet_pytorch formula:
    RMSE = sqrt(mean((E_pred - E_true)^2 / N_atom^2)) * 1000
    
    This computes the RMSE of per-atom energy errors.
    """
    # Per-atom energy error for each structure
    per_atom_err = (list_E_ann - grp_energy) / grp_N_atom
    # RMSE of per-atom errors, converted to meV
    return torch.sqrt(torch.mean(per_atom_err**2)) * 1000
