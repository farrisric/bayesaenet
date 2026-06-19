"""
Deterministic Neural Network models for atomic energy and force prediction.

Classes:
    - NNBase: Base deterministic NN (used for BNN pretraining)
    - NN: NN with force training (canonical; always uses forces)
"""

from typing import Any, Dict, List

import lightning.pytorch as L
import numpy as np
import torch

from ..datamodule.aenet.batch_constants import BatchIdx
from .utils import get_rmse_atom, weights_init


class NNBase(L.LightningModule):
    """
    Class used by BNNs to pretrain their weights. This class is instantiated,
    trained for X epochs and then it stores its weights in the log directory.
    VIBnnWrapper then loads the weights and starts the Bayesian training
    """

    def __init__(
        self, net: torch.nn.Module, optimizer: torch.optim.Optimizer, name: str = "NN"
    ):  # Model name for logging organization
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
        return get_rmse_atom(list_E_ann, grp_energy, grp_N_atom)  # normalized units

    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Execute one training step."""
        mse = self.step(batch)
        self.log(
            "rmse/train",
            mse,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch[BatchIdx.E_ENERGY]),
        )
        return mse

    def validation_step(self, batch: List[torch.Tensor], batch_idx: int) -> None:
        """Execute one validation step."""
        mse = self.step(batch)
        self.log(
            "rmse/val",
            mse,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch[BatchIdx.E_ENERGY]),
        )

    def test_step(self, batch: List[torch.Tensor], batch_idx: int) -> None:
        """Execute one test step."""
        mse = self.step(batch)
        self.log(
            "rmse/test",
            mse,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch[BatchIdx.E_ENERGY]),
        )

    def predict_step(
        self, batch: List[torch.Tensor], batch_idx: int, dataloader_idx: int = 0
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
        # Keep normalized units (match BNN) for analysis compatibility
        true = grp_energy
        list_E_ann = self.net.forward(grp_descrp, logic_reduce)
        preds = list_E_ann

        pred["true"] = true.cpu().numpy()
        pred["preds"] = preds.cpu().numpy()
        pred["n_atoms"] = grp_N_atom.cpu().numpy()
        return pred

    def configure_optimizers(self):
        return self.hparams.optimizer(params=self.parameters())


class NN(NNBase):
    """
    Neural Network with force training (canonical; always uses forces).

    Used for Deep Ensemble baseline. Weighted combination of energy RMSE and force RMSE.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        alpha: float = 0.1,
        name: str = "NN",
    ):
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
        max_nnb = (
            F_sfderiv_j[0].shape[1] if len(F_sfderiv_j) > 0 and F_sfderiv_j[0].shape[0] > 0 else 0
        )

        net = self.net

        with torch.enable_grad():
            # Clone descriptors for gradient tracking, convert to float32
            F_descrp_grad = [
                d.clone().detach().float().requires_grad_(True) for d in F_group_descrp
            ]
            F_sfderiv_i_f = [s.float() for s in F_sfderiv_i]
            F_sfderiv_j_f = [s.float() for s in F_sfderiv_j]
            F_logic_reduce_f = [l.float() for l in F_logic_reduce]

            # Compute forces via autodiff through forward_F
            E_pred, F_pred = net.forward_F(
                F_descrp_grad,
                F_sfderiv_i_f,
                F_sfderiv_j_f,
                F_indices,
                F_indices_i,
                F_logic_reduce_f,
                net.input_size,
                max_nnb,
            )

        # Compute RMSE for forces (normalized units; meV/Å only in prediction)
        force_diff = F_pred - F_group_forces.float()
        return torch.sqrt(torch.mean(force_diff**2))

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
        self.log(
            "rmse/train",
            energy_rmse,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "force_rmse/train",
            force_rmse,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "total_rmse/train", total_loss, on_step=False, on_epoch=True, batch_size=batch_size
        )
        self.log("alpha", alpha, on_step=False, on_epoch=True)

        return total_loss

    def validation_step(self, batch: List[torch.Tensor], batch_idx: int) -> None:
        """Validation step with energy and force metrics."""
        energy_rmse = self.step(batch)
        force_rmse = self.compute_force_loss(batch)

        # Combined metric for HPS optimization: weighted sum of energy and force RMSE
        total_rmse = (1 - self.alpha) * energy_rmse + self.alpha * force_rmse

        batch_size = len(batch[BatchIdx.E_ENERGY])
        self.log(
            "rmse/val",
            energy_rmse,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "force_rmse/val",
            force_rmse,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "total_rmse/val",
            total_rmse,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

    def test_step(self, batch: List[torch.Tensor], batch_idx: int) -> None:
        """Test step with energy and force metrics."""
        energy_rmse = self.step(batch)
        force_rmse = self.compute_force_loss(batch)

        # Combined metric
        total_rmse = (1 - self.alpha) * energy_rmse + self.alpha * force_rmse

        batch_size = len(batch[BatchIdx.E_ENERGY])
        self.log("rmse/test", energy_rmse, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(
            "force_rmse/test", force_rmse, on_step=False, on_epoch=True, batch_size=batch_size
        )
        self.log(
            "total_rmse/test", total_rmse, on_step=False, on_epoch=True, batch_size=batch_size
        )

    def predict_step(
        self, batch: List[torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Dict[str, np.ndarray]:
        """Predict energies and forces (deterministic, stds=0).

        Returns dict matching BNN_Forces.predict_step() format for
        compatibility with Deep Ensemble creation.
        """
        # Energy predictions (deterministic); keep normalized units (match BNN) for analysis
        grp_descrp = batch[BatchIdx.E_DESCRP]
        grp_energy = batch[BatchIdx.E_ENERGY]
        logic_reduce = batch[BatchIdx.E_LOGIC_REDUCE]
        grp_N_atom = batch[BatchIdx.E_N_ATOM]

        true = grp_energy
        list_E_ann = self.net.forward(grp_descrp, logic_reduce)
        preds = list_E_ann

        pred = {}
        pred["true"] = true.cpu().numpy()
        pred["preds"] = preds.detach().cpu().numpy()
        pred["stds"] = np.zeros_like(pred["preds"])  # Deterministic model
        pred["n_atoms"] = grp_N_atom.cpu().numpy()

        # Force predictions
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
            max_nnb = (
                F_sfderiv_j[0].shape[1]
                if len(F_sfderiv_j) > 0 and F_sfderiv_j[0].shape[0] > 0
                else 0
            )

            # inference_mode(False) needed for PyTorch 2.x (enable_grad cant override inference_mode)
            with torch.inference_mode(False):
                F_descrp_grad = [
                    d.clone().detach().float().requires_grad_(True) for d in F_group_descrp
                ]
                F_sfderiv_i_f = [s.float() for s in F_sfderiv_i]
                F_sfderiv_j_f = [s.float() for s in F_sfderiv_j]
                F_logic_reduce_f = [l.float() for l in F_logic_reduce]

                _, F_pred = self.net.forward_F(
                    F_descrp_grad,
                    F_sfderiv_i_f,
                    F_sfderiv_j_f,
                    F_indices,
                    F_indices_i,
                    F_logic_reduce_f,
                    self.net.input_size,
                    max_nnb,
                )

            true_forces_flat = F_group_forces.cpu().numpy().flatten()
            pred_forces_flat = F_pred.detach().cpu().numpy().flatten()
            std_forces_flat = np.zeros_like(pred_forces_flat)  # Deterministic

            scale = (
                float(self.net.e_scaling)
                if hasattr(self.net.e_scaling, "item")
                else float(self.net.e_scaling)
            )
            force_errors = np.abs(true_forces_flat - pred_forces_flat) / scale * 1000
            force_rmse = (
                np.sqrt(np.mean((true_forces_flat - pred_forces_flat) ** 2)) / scale * 1000
            )
            force_mae = np.mean(force_errors)
        else:
            true_forces_flat = None
            pred_forces_flat = None
            std_forces_flat = None
            force_rmse = np.nan
            force_mae = np.nan

        pred["true_forces"] = true_forces_flat
        pred["pred_forces"] = pred_forces_flat
        pred["std_forces"] = std_forces_flat
        pred["force_rmse"] = force_rmse
        pred["force_mae"] = force_mae

        return pred


# Backward compatibility for checkpoints
NN_Forces = NN
