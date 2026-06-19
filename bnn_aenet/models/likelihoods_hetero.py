"""
Heteroscedastic likelihood for BNN energy+force training.

Extends the base likelihood with input-dependent observation noise
predicted by a ``NoiseNet``.  Per-atom noise scales replace the global
scalar used in the homoscedastic version.
"""

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import torch

from ..datamodule.aenet.batch_constants import BatchIdx


def make_hetero_energy_force_model(bnn, noise_net, dataset_size):
    """Create a Pyro model with heteroscedastic energy+force likelihood.

    The ``noise_net`` is registered via ``pyro.module`` so its parameters
    are optimized jointly with the BNN guide by SVI.

    All observation noise is predicted per-sample by ``noise_net``; there
    are no global ``obs_scale`` / ``scale_force`` scalars.

    Args:
        bnn: The BNN_Forces_Hetero model instance.
        noise_net: A ``NoiseNet`` instance (deterministic, not Bayesian).
        dataset_size: Training set size for minibatch scaling.
    """

    def model(batch):
        # Register noise_net so Pyro optimizes its parameters
        pyro.module("noise_net", noise_net)

        # ---------- Energy data ----------
        E_descrp = batch[BatchIdx.E_DESCRP]
        E_logic_reduce = batch[BatchIdx.E_LOGIC_REDUCE]
        E_obs = batch[BatchIdx.E_ENERGY]

        # ---------- Force data (may be None) ----------
        F_descrp = batch[BatchIdx.F_DESCRP]
        F_forces = batch[BatchIdx.F_FORCES]
        F_logic_reduce = batch[BatchIdx.F_LOGIC_REDUCE]
        F_sfderiv_i = batch[BatchIdx.F_SFDERIV_I]
        F_sfderiv_j = batch[BatchIdx.F_SFDERIV_J]
        F_indices = batch[BatchIdx.F_INDICES]
        F_indices_i = batch[BatchIdx.F_INDICES_I]

        has_force = (
            F_descrp is not None
            and F_forces is not None
            and not (isinstance(F_descrp, list) and len(F_descrp) == 0)
        )

        # ---------- Per-structure energy noise ----------
        obs_scale_energy = noise_net.forward_energy(E_descrp, E_logic_reduce)

        # ---------- BNN energy prediction ----------
        energy_trace = poutine.trace(bnn.bnn_net).get_trace(
            E_descrp,
            E_logic_reduce,
        )
        weight_sites = energy_trace.stochastic_nodes
        with poutine.block(hide=weight_sites), poutine.replay(trace=energy_trace):
            E_pred = bnn.bnn_net(E_descrp, E_logic_reduce)

        # Energy likelihood: N(E_pred, sigma_E(x))
        with pyro.plate("energy_plate", len(E_obs)):
            pyro.sample(
                "energy_obs",
                dist.Normal(E_pred.squeeze(-1), obs_scale_energy).to_event(0),
                obs=E_obs,
            )

        # ---------- Force prediction + heteroscedastic likelihood ----------
        if has_force:
            net = bnn.net
            max_nnb = (
                F_sfderiv_j[0].shape[1]
                if len(F_sfderiv_j) > 0 and F_sfderiv_j[0].shape[0] > 0
                else 0
            )

            F_descrp_grad = [d.clone().detach().float().requires_grad_(True) for d in F_descrp]
            F_sfderiv_i_f = [s.float() for s in F_sfderiv_i]
            F_sfderiv_j_f = [s.float() for s in F_sfderiv_j]
            F_logic_reduce_f = [l.float() for l in F_logic_reduce]

            with (
                poutine.block(hide=weight_sites),
                poutine.replay(trace=energy_trace),
                torch.enable_grad(),
            ):
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

            # Per-atom -> per-component noise
            # F_descrp (not _grad) for noise -- no gradients needed
            obs_scale_force = noise_net.forward_forces(
                F_descrp,
                F_indices_i,
            )  # (n_force_atoms, 3)

            F_obs_flat = F_forces.float().flatten()
            F_pred_flat = F_pred.flatten()
            obs_scale_force_flat = obs_scale_force.flatten()

            with pyro.plate("force_plate", len(F_obs_flat)):
                pyro.sample(
                    "force_obs",
                    dist.Normal(F_pred_flat, obs_scale_force_flat).to_event(0),
                    obs=F_obs_flat,
                )

        return E_pred

    return model
