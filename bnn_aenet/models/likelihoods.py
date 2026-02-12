"""
Custom likelihoods for BNN energy+force training.

Provides a custom Pyro model for joint energy+force Gaussian likelihood,
integrating both into the ELBO for proper Bayesian training.
"""

import torch
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine

from ..datamodule.aenet.batch_constants import BatchIdx


def make_energy_force_model(bnn, scale_energy, scale_force, dataset_size):
    """
    Create a Pyro model function for joint energy+force likelihood.
    
    Returns a callable model(batch) that:
    - Uses guided_forward for energy (samples weights)
    - Uses forward_F for forces (same weights)
    - Adds pyro.sample for both energy and force observations
    
    Log p(E, F | theta) = log p(E | theta) + log p(F | theta)
    with Gaussian observation noise.
    """
    def model(batch):
        # Energy data
        E_descrp = batch[BatchIdx.E_DESCRP]
        E_logic_reduce = batch[BatchIdx.E_LOGIC_REDUCE]
        E_obs = batch[BatchIdx.E_ENERGY]
        
        # Force data (may be None)
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
        
        # E_pred from BNN - trace to get weight samples; replay for E_pred and F_pred (avoid duplicate sites)
        energy_trace = poutine.trace(bnn.bnn_net).get_trace(E_descrp, E_logic_reduce)
        weight_sites = energy_trace.stochastic_nodes
        with poutine.block(hide=weight_sites), poutine.replay(trace=energy_trace):
            E_pred = bnn.bnn_net(E_descrp, E_logic_reduce)

        # Energy likelihood: p(E_obs | E_pred)
        # Outer scale (1/dataset_size) handles minibatch scaling
        with pyro.plate("energy_plate", len(E_obs)):
            pyro.sample(
                "energy_obs",
                dist.Normal(E_pred.squeeze(-1), scale_energy).to_event(0),
                obs=E_obs,
            )

        # Force likelihood: p(F_obs | F_pred) — same weights as energy (replay trace)
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

            with poutine.block(hide=weight_sites), poutine.replay(trace=energy_trace), torch.enable_grad():
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
            
            F_obs_flat = F_forces.float().flatten()
            F_pred_flat = F_pred.flatten()

            # Force likelihood - plate for proper batching
            with pyro.plate("force_plate", len(F_obs_flat)):
                pyro.sample(
                    "force_obs",
                    dist.Normal(F_pred_flat, scale_force).to_event(0),
                    obs=F_obs_flat,
                )
        
        return E_pred
    return model
