"""
BNN models with auxiliary force training for energy+force prediction.

Classes:
    - BNN_Forces_Aux: Full BNN with auxiliary force loss
    - PartialBNN_Forces_Aux: Partial BNN with auxiliary force loss
"""

from typing import Any, Dict, List, Union
import warnings

import pyro
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import lightning.pytorch as L

from .bnn import BNN, PartialBNN
from .utils import get_rmse_atom
from ..results.metrics import sharpness, rms_calibration_error
from ..datamodule.aenet.batch_constants import BatchIdx


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
        scale = float(net.e_scaling) if hasattr(net.e_scaling, "item") else float(net.e_scaling)
        force_rmse = torch.sqrt(torch.mean(force_diff ** 2)) / scale * 1000  # meV/Å
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
            
            # Compute RMSE for forces (meV/Å)
            force_diff = F_pred - F_target
            scale = float(self.net.e_scaling) if hasattr(self.net.e_scaling, "item") else float(self.net.e_scaling)
            force_rmse = torch.sqrt(torch.mean(force_diff ** 2)) / scale * 1000
            
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
        rmse = get_rmse_atom(loc, y, n_atoms, self.net.e_scaling)
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
        rmse = get_rmse_atom(loc, y, n_atoms, self.net.e_scaling)
        
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
        rmse = get_rmse_atom(loc, y, n_atoms, self.net.e_scaling)
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
            # Extract force tensors ONCE outside the MC loop
            F_logic_reduce = batch[BatchIdx.F_LOGIC_REDUCE]
            F_sfderiv_i = batch[BatchIdx.F_SFDERIV_I]
            F_sfderiv_j = batch[BatchIdx.F_SFDERIV_J]
            F_indices = batch[BatchIdx.F_INDICES]
            F_indices_i = batch[BatchIdx.F_INDICES_I]
            max_nnb = F_sfderiv_j[0].shape[1] if len(F_sfderiv_j) > 0 and F_sfderiv_j[0].shape[0] > 0 else 0
            
            # Build mapping: guide trace site name -> self.net param name
            # Guide trace sites are "net.{param_name}" -> strip "net." to get param_name
            # We use self.net (regular nn.Module) not self.bnn.net (PyroModule)
            # because forward_F needs autograd through regular parameters
            net_params = dict(self.net.named_parameters())
            
            for _ in range(self.hparams.mc_samples_eval):
                with torch.no_grad():
                    # Sample network weights from the guide
                    guide_trace = pyro.poutine.trace(self.bnn.guide).get_trace(x[0], x[1])
                
                    # Copy sampled weights into self.net so forward_F uses them
                    # Guide trace sites: "net.{param_name}" -> sampled value
                    for site_name, site in guide_trace.nodes.items():
                        if site.get("type") == "sample" and not site.get("is_observed", False):
                            # Strip "net." prefix to get param name matching self.net
                            if site_name.startswith("net."):
                                param_name = site_name[len("net."):]
                                if param_name in net_params:
                                    net_params[param_name].data.copy_(site["value"].data)
                
                # Force computation needs gradients (autograd.grad inside forward_F)
                # Must exit both inference_mode and no_grad
                with torch.inference_mode(False):
                    with torch.enable_grad():
                        # Clone/detach all tensors to escape inference mode
                        F_descrp_f = [d.clone().detach().float().requires_grad_(True) for d in F_group_descrp]
                        F_sfderiv_i_f = [s.clone().detach().float() for s in F_sfderiv_i]
                        F_sfderiv_j_f = [s.clone().detach().float() for s in F_sfderiv_j]
                        F_logic_reduce_f = [l.clone().detach().float() for l in F_logic_reduce]
                        F_indices_c = F_indices.clone().detach()
                        F_indices_i_c = F_indices_i.clone().detach()
                        
                        # Use self.net (regular network) with sampled weights
                        _, F_pred = self.net.forward_F(
                            F_descrp_f, F_sfderiv_i_f, F_sfderiv_j_f,
                            F_indices_c, F_indices_i_c, F_logic_reduce_f,
                            self.net.input_size, max_nnb
                        )
                force_samples.append(F_pred.detach().cpu().numpy())
            
            force_samples = np.array(force_samples)
            force_preds = force_samples.mean(axis=0)
            force_stds = force_samples.std(axis=0)
            
            # Flatten forces for storage (Nx3 -> N*3)
            true_forces_flat = F_group_forces.cpu().numpy().flatten()
            pred_forces_flat = force_preds.flatten()
            std_forces_flat = force_stds.flatten()
            
            # Compute per-atom force errors (meV/Å)
            scale = float(self.net.e_scaling) if hasattr(self.net.e_scaling, "item") else float(self.net.e_scaling)
            force_errors = np.abs(true_forces_flat - pred_forces_flat) / scale * 1000
            force_rmse = np.sqrt(np.mean((true_forces_flat - pred_forces_flat) ** 2)) / scale * 1000
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
