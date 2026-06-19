"""
BNN with heteroscedastic (input-dependent) observation noise.

Subclasses ``BNN_Forces`` -- only overrides the methods that differ:
- ``__init__``: creates the ``NoiseNet``
- ``define_bnn``: uses the heteroscedastic likelihood
- ``training_step``: logs mean noise scales
- ``predict_step``: uses per-atom aleatoric noise
- ``on_predict_start``: no special param-store retrieval needed
"""

from typing import Any, Dict, List

import numpy as np
import pyro
import torch
import torch.nn.functional as Fn

from bnn_aenet.models.utils import param_store_to

from ..datamodule.aenet.batch_constants import BatchIdx
from .bnn_forces import BNN_Forces
from .likelihoods_hetero import make_hetero_energy_force_model
from .nets.noise_net import NoiseNet
from .utils import get_rmse_atom


class BNN_Forces_Hetero(BNN_Forces):
    """BNN with heteroscedastic observation noise.

    Adds a small deterministic ``NoiseNet`` that predicts per-atom
    observation noise for both energy and force likelihoods.  The noise
    network is optimized jointly with the BNN guide via SVI.

    Extra hyperparameters (beyond ``BNN_Forces``):
        noise_hidden_size: Hidden layer width of the noise MLP per species.
        noise_min: Minimum noise floor (prevents collapse to zero).
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
        noise_hidden_size: int = 15,
        noise_min: float = 0.01,
        name: str = "BNN_Forces_Hetero",
    ):
        # learn_noise is superseded by heteroscedastic noise; pass False
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
            scale_force=scale_force,
            grad_clip_val=grad_clip_val,
            learn_noise=False,
            name=name,
        )
        self.save_hyperparameters(logger=False, ignore=["net"])

        # Create the noise network (deterministic -- NOT converted to Pyro)
        self.noise_net = NoiseNet(
            input_size=net.input_size,
            species=net.species,
            hidden_size=noise_hidden_size,
            min_noise=noise_min,
        )

    # ------------------------------------------------------------------
    # Override define_bnn: use heteroscedastic likelihood
    # ------------------------------------------------------------------

    def define_bnn(self):
        """Set up BNN, then replace the model function with heteroscedastic version."""
        # Call parent to set up bnn_net, net_guide, fit_ctxt, etc.
        super().define_bnn()

        # Move noise_net to the same device as the BNN
        self.noise_net = self.noise_net.to(self.device)

        # Replace model function with heteroscedastic version
        # (obs_scale / scale_force are NOT used -- NoiseNet predicts all noise)
        self._model_fn = make_hetero_energy_force_model(
            self,
            noise_net=self.noise_net,
            dataset_size=self.hparams.dataset_size,
        )

    # ------------------------------------------------------------------
    # Override training_step: log mean noise scales
    # ------------------------------------------------------------------

    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> None:
        """Training step with heteroscedastic noise logging."""
        # Run the parent training step (SVI, RMSE, etc.)
        super().training_step(batch, batch_idx)

        # Log mean noise scales from the noise network
        with torch.no_grad():
            E_descrp = batch[BatchIdx.E_DESCRP]
            E_logic_reduce = batch[BatchIdx.E_LOGIC_REDUCE]
            y = batch[BatchIdx.E_ENERGY]

            sigma_E = self.noise_net.forward_energy(E_descrp, E_logic_reduce)
            self.log(
                "noise_energy_mean/train",
                sigma_E.mean(),
                on_step=False,
                on_epoch=True,
                batch_size=len(y),
            )

            F_descrp = batch[BatchIdx.F_DESCRP]
            F_indices_i = batch[BatchIdx.F_INDICES_I]
            has_force = F_descrp is not None and not (
                isinstance(F_descrp, list) and len(F_descrp) == 0
            )
            if has_force:
                sigma_F = self.noise_net.forward_forces(F_descrp, F_indices_i)
                self.log(
                    "noise_force_mean/train",
                    sigma_F.mean(),
                    on_step=False,
                    on_epoch=True,
                    batch_size=len(y),
                )
                self.log(
                    "noise_force_std/train",
                    sigma_F.std(),
                    on_step=False,
                    on_epoch=True,
                    batch_size=len(y),
                )

    # ------------------------------------------------------------------
    # Override on_predict_start: no param-store noise to retrieve
    # ------------------------------------------------------------------

    def on_predict_start(self) -> None:
        """Initialize BNN for prediction (skip if already defined).

        Noise network params are saved in the Lightning checkpoint as
        regular PyTorch parameters, so no special param-store handling
        is needed (unlike learn_noise scalar params).
        """
        if not hasattr(self, "bnn_net") or self.bnn_net is None:
            self.define_bnn()
            param_store_to(self.device)

    # ------------------------------------------------------------------
    # Override predict_step: per-atom aleatoric noise
    # ------------------------------------------------------------------

    def predict_step(
        self,
        batch: List[torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Dict[str, np.ndarray]:
        """Predict energies and forces with heteroscedastic uncertainty."""
        x = batch[BatchIdx.E_DESCRP], batch[BatchIdx.E_LOGIC_REDUCE]
        y = batch[BatchIdx.E_ENERGY]
        n_atoms = batch[BatchIdx.E_N_ATOM]

        # --- Energy predictions (MC sampling) ---
        pred = {}
        energy_samples = []
        for _ in range(self.hparams.mc_samples_eval):
            p = self.guided_forward(x[0], x[1])
            energy_samples.append(p.detach().cpu().numpy())
        energy_preds = np.stack(energy_samples).mean(axis=0)
        energy_stds_epistemic = np.stack(energy_samples).std(axis=0)

        # Per-structure aleatoric energy noise
        with torch.no_grad():
            sigma_E = self.noise_net.forward_energy(x[0], x[1])
        energy_stds_aleatoric = sigma_E.cpu().numpy()
        energy_stds = np.sqrt(energy_stds_epistemic**2 + energy_stds_aleatoric**2)

        # --- Force predictions ---
        F_group_descrp = batch[BatchIdx.F_DESCRP]
        F_group_forces = batch[BatchIdx.F_FORCES]
        F_indices_i = batch[BatchIdx.F_INDICES_I]
        has_force_data = (
            F_group_descrp is not None
            and F_group_forces is not None
            and not (isinstance(F_group_descrp, list) and len(F_group_descrp) == 0)
        )

        if has_force_data:
            # Posterior-mean force prediction
            with torch.inference_mode(False):
                mean_tr = self._get_guide_trace(x[0], x[1], use_mean=True)
                F_pred_mean = self._replay_forward_F(batch, mean_tr)
            force_preds = F_pred_mean.detach().cpu().numpy()

            # MC samples for epistemic force uncertainty
            force_samples = []
            for _ in range(self.hparams.mc_samples_eval):
                with torch.inference_mode(False):
                    sample_tr = self._get_guide_trace(x[0], x[1], use_mean=False)
                    F_pred_s = self._replay_forward_F(batch, sample_tr)
                force_samples.append(F_pred_s.detach().cpu().numpy())
            epistemic_std = np.stack(force_samples).std(axis=0)

            # Per-atom aleatoric force noise from NoiseNet
            with torch.no_grad():
                sigma_F = self.noise_net.forward_forces(
                    F_group_descrp,
                    F_indices_i,
                )  # (n_force_atoms, 3)
            aleatoric_std = sigma_F.cpu().numpy()

            # Total: sqrt(epistemic² + aleatoric²)
            force_stds = np.sqrt(epistemic_std**2 + aleatoric_std**2)

            true_forces_flat = F_group_forces.cpu().numpy().flatten()
            pred_forces_flat = force_preds.flatten()
            std_forces_flat = force_stds.flatten()
            scale = (
                float(self.net.e_scaling)
                if hasattr(self.net.e_scaling, "item")
                else float(self.net.e_scaling)
            )
            force_rmse = (
                np.sqrt(np.mean((true_forces_flat - pred_forces_flat) ** 2)) / scale * 1000
            )
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
