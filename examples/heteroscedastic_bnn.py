"""
Heteroscedastic Bayesian Neural Network for Atomic Energy Networks (BNN-AENET)
==============================================================================

This example explains how the heteroscedastic BNN works for joint
energy and force prediction with input-dependent uncertainty
quantification.

Background
----------
Standard (homoscedastic) BNNs assume a fixed observation noise scale
for all data points:

    y ~ Normal(f(x), sigma)          # sigma is a scalar, same for all x

This is unrealistic -- some atomic configurations are inherently harder
to predict than others.  The heteroscedastic BNN replaces the fixed
scalar with an input-dependent noise network:

    y ~ Normal(f(x), sigma(x))       # sigma(x) varies per input

This gives two independent sources of uncertainty:

    1. **Epistemic** (model uncertainty): from the posterior distribution
       over network weights.  Reducible with more training data.

    2. **Aleatoric** (data/noise uncertainty): from the NoiseNet.
       Reflects irreducible noise or model misspecification for each input.

The total predictive uncertainty combines both:

    sigma_total = sqrt(sigma_epistemic^2 + sigma_aleatoric^2)


Architecture
------------
The model has three components:

    ┌─────────────────────────────────────────────────────────┐
    │                 Atomic Descriptors (G_i)                │
    │              (symmetry functions per atom)               │
    └────────────────┬────────────────────┬───────────────────┘
                     │                    │
            ┌────────▼────────┐  ┌────────▼────────┐
            │   Main BNN      │  │   NoiseNet       │
            │   (Bayesian)    │  │   (Deterministic) │
            │                 │  │                   │
            │  Per-species    │  │  Per-species      │
            │  MLPs with      │  │  small MLPs:      │
            │  weight         │  │  Linear → Tanh    │
            │  distributions  │  │  → Linear         │
            │                 │  │  → softplus+floor │
            └────────┬────────┘  └────────┬──────────┘
                     │                    │
                E_pred, F_pred     sigma_E(x), sigma_F(x)
                     │                    │
            ┌────────▼────────────────────▼──────────┐
            │         Likelihood (Pyro)              │
            │                                        │
            │  E_obs ~ Normal(E_pred, sigma_E(x))    │
            │  F_obs ~ Normal(F_pred, sigma_F(x))    │
            └────────────────────────────────────────┘

Key design choices:

    - The **main network** is Bayesian: its weights have prior
      distributions and are inferred via variational inference
      (SVI with a guide, either Local Reparameterization Trick
      or Radial approximation).

    - The **NoiseNet** is deterministic (standard nn.Module): aleatoric
      noise is a property of the data, not the model.  Its parameters
      are registered with Pyro via ``pyro.module("noise_net", ...)``
      and optimized jointly with the BNN guide by SVI.

    - The **NoiseNet** reuses the same atomic descriptors as the main
      network, so no extra descriptor computation is needed.


NoiseNet Details
----------------
One small MLP per atomic species (e.g. one for Ti, one for O):

    raw = Linear(descriptor_dim, hidden_size) -> Tanh -> Linear(hidden_size, 1)
    sigma_atom = softplus(raw) + min_noise

where:
    - ``hidden_size`` matches the main network (e.g. 15, from train.in)
    - ``min_noise`` is a floor (e.g. 0.01) that prevents sigma from
      collapsing to zero, which would make the log-likelihood diverge.

For **energy** noise, per-atom variances are summed over all atoms in
a structure (variances add for independent noise), then the square root
gives the per-structure standard deviation:

    sigma_E(structure) = sqrt( sum_i sigma_atom_i^2 )

For **force** noise, each atom's sigma is expanded to all 3 Cartesian
components (x, y, z) of that atom's force vector:

    sigma_F_atom = [sigma, sigma, sigma]   # same noise for x, y, z


Training
--------
Training uses Stochastic Variational Inference (SVI) with the
Trace_ELBO objective, which maximizes the Evidence Lower Bound:

    ELBO = E_q[log p(data | weights, sigma(x))] - KL(q(weights) || p(weights))

The first term is the heteroscedastic log-likelihood (data fit), and
the second is the KL divergence between the variational posterior q
and the prior p (regularization).

Both the BNN weights (via the guide) and the NoiseNet parameters are
updated in each SVI step.


Example: Training a Heteroscedastic BNN
----------------------------------------

Command-line training with Hydra configs:

.. code-block:: bash

    # LRT guide (Local Reparameterization Trick)
    python -m bnn_aenet.tasks.train \\
        experiment=bnn_lrt_hetero \\
        datamodule=TiO_Forces_Data20 \\
        dataset=TiO2_small \\
        trainer.accelerator=gpu \\
        trainer.max_epochs=50000 \\
        datamodule.batch_size=512 \\
        model.lr=3.67e-4 \\
        model.mc_samples_train=1 \\
        model.prior_scale=0.156 \\
        model.q_scale=3.18e-5 \\
        model.noise_hidden_size=15 \\
        model.noise_min=0.01 \\
        seed=12345

    # Radial guide
    python -m bnn_aenet.tasks.train \\
        experiment=bnn_rad_hetero \\
        datamodule=TiO_Forces_Data20 \\
        dataset=TiO2_small \\
        trainer.accelerator=gpu \\
        trainer.max_epochs=50000 \\
        +trainer.precision=16-mixed \\
        datamodule.batch_size=512 \\
        model.lr=3.67e-4 \\
        model.mc_samples_train=1 \\
        model.prior_scale=0.156 \\
        model.q_scale=3.18e-5 \\
        model.noise_hidden_size=15 \\
        model.noise_min=0.01 \\
        seed=12345


Key hyperparameters
^^^^^^^^^^^^^^^^^^^

+----------------------+--------------------------------------------------+
| Parameter            | Description                                      |
+======================+==================================================+
| ``lr``               | Learning rate for SVI optimizer                  |
+----------------------+--------------------------------------------------+
| ``mc_samples_train`` | MC samples per SVI step (1 or 2)                 |
+----------------------+--------------------------------------------------+
| ``prior_scale``      | Std of the Gaussian weight prior                 |
+----------------------+--------------------------------------------------+
| ``q_scale``          | Initial std of the variational posterior          |
+----------------------+--------------------------------------------------+
| ``noise_hidden_size``| Hidden width of NoiseNet MLPs (match train.in)   |
+----------------------+--------------------------------------------------+
| ``noise_min``        | Minimum noise floor for sigma(x)                 |
+----------------------+--------------------------------------------------+

Note: ``obs_scale`` and ``scale_force`` are NOT used -- all observation
noise is predicted by the NoiseNet.


Example: Prediction with Uncertainty
-------------------------------------

.. code-block:: bash

    python -m bnn_aenet.tasks.predict_forces \\
        --model-type lrt_hetero \\
        --runs-dir bnn_aenet/logs/TiO2_small/train/runs/lrt_hetero \\
        --output-dir bnn_aenet/logs/TiO2_small/pred/lrt_hetero \\
        --data-dir data/TiO/train_forces.in \\
        --use-run-config

The prediction step produces per-sample uncertainties:

    1. **Epistemic** (weight uncertainty): run MC forward passes by
       sampling from the variational posterior, compute std across samples.
    2. **Aleatoric** (noise uncertainty): single forward pass through
       NoiseNet to get sigma_E(x) and sigma_F(x).
    3. **Total**: sqrt(epistemic^2 + aleatoric^2).

Outputs are saved as:
    - ``energy_{split}.csv``: columns ``true, preds, stds, n_atoms``
    - ``forces_{split}.npz``: arrays ``true_forces, pred_forces, std_forces``


Example: Programmatic Usage
----------------------------
"""

import sys
sys.path.insert(0, "..")  # Add project root to path

import torch
import numpy as np


def create_noise_net_example():
    """Demonstrate NoiseNet creation and forward pass."""
    from bnn_aenet.models.nets.noise_net import NoiseNet

    # Two species (Ti, O) with descriptor dimensions 40 and 30
    noise_net = NoiseNet(
        input_size=[40, 30],
        species=["Ti", "O"],
        hidden_size=15,    # match main network architecture from train.in
        min_noise=0.01,    # minimum noise floor
    )
    print(f"NoiseNet parameters: {sum(p.numel() for p in noise_net.parameters())}")
    print(f"  Ti MLP: Linear(40, 15) -> Tanh -> Linear(15, 1)")
    print(f"  O  MLP: Linear(30, 15) -> Tanh -> Linear(15, 1)")

    # --- Energy noise ---
    # Suppose we have 3 structures: 2 Ti atoms + 3 O atoms in various configs
    E_descrp_Ti = torch.randn(5, 40)   # 5 Ti atoms total across structures
    E_descrp_O = torch.randn(8, 30)    # 8 O atoms total across structures

    # logic_reduce maps atoms -> structures (binary matrix)
    # Shape: (n_structures, n_atoms_of_species)
    E_logic_Ti = torch.tensor([
        [1., 1., 0., 0., 0.],   # Structure 0 has 2 Ti atoms
        [0., 0., 1., 1., 0.],   # Structure 1 has 2 Ti atoms
        [0., 0., 0., 0., 1.],   # Structure 2 has 1 Ti atom
    ])
    E_logic_O = torch.tensor([
        [1., 1., 1., 0., 0., 0., 0., 0.],   # Structure 0 has 3 O atoms
        [0., 0., 0., 1., 1., 1., 0., 0.],   # Structure 1 has 3 O atoms
        [0., 0., 0., 0., 0., 0., 1., 1.],   # Structure 2 has 2 O atoms
    ])

    sigma_E = noise_net.forward_energy(
        [E_descrp_Ti, E_descrp_O],
        [E_logic_Ti, E_logic_O],
    )
    print(f"\nEnergy noise per structure: {sigma_E.detach().numpy()}")
    print(f"  Shape: {sigma_E.shape}  (one sigma per structure)")

    # --- Force noise ---
    F_descrp_Ti = torch.randn(4, 40)   # 4 Ti atoms with force data
    F_descrp_O = torch.randn(6, 30)    # 6 O atoms with force data

    # grp_indices_F_i: reorder from per-species concat to final atom ordering
    grp_indices = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7, 8, 9])

    sigma_F = noise_net.forward_forces(
        [F_descrp_Ti, F_descrp_O],
        grp_indices,
    )
    print(f"\nForce noise per atom (x,y,z): shape {sigma_F.shape}")
    print(f"  First 3 atoms:\n{sigma_F[:3].detach().numpy()}")
    print(f"  Note: same sigma for x, y, z of each atom")


def uncertainty_decomposition_example():
    """Show how total uncertainty is decomposed into epistemic + aleatoric."""
    # Simulated predictions from 20 MC samples (epistemic)
    n_atoms = 50
    mc_samples = 20

    # Each MC sample gives slightly different force predictions
    # due to weight sampling from the variational posterior
    force_samples = np.random.randn(mc_samples, n_atoms, 3) * 0.1  # small spread
    force_mean = force_samples.mean(axis=0)
    epistemic_std = force_samples.std(axis=0)  # per-component

    # Aleatoric noise from NoiseNet (varies per atom)
    # Some atoms have low noise (well-represented in training)
    # Some atoms have high noise (unusual configurations)
    aleatoric_std = np.abs(np.random.randn(n_atoms, 3)) * 0.05
    aleatoric_std[0:5] *= 5  # first 5 atoms are "hard" -> higher noise

    # Total uncertainty: quadrature sum
    total_std = np.sqrt(epistemic_std ** 2 + aleatoric_std ** 2)

    print("Uncertainty decomposition (first 5 'hard' atoms, x-component):")
    print(f"  Epistemic:  {epistemic_std[:5, 0]}")
    print(f"  Aleatoric:  {aleatoric_std[:5, 0]}")
    print(f"  Total:      {total_std[:5, 0]}")
    print()
    print("Uncertainty decomposition (atoms 5-10 'easy' atoms, x-component):")
    print(f"  Epistemic:  {epistemic_std[5:10, 0]}")
    print(f"  Aleatoric:  {aleatoric_std[5:10, 0]}")
    print(f"  Total:      {total_std[5:10, 0]}")
    print()
    print("Key insight: 'hard' atoms have larger aleatoric noise,")
    print("while epistemic uncertainty is similar across atoms.")
    print("A homoscedastic model would assign the SAME noise to all atoms.")


def compare_homo_vs_hetero():
    """Illustrate the difference between homoscedastic and heteroscedastic noise.

    Homoscedastic (standard BNN / Deep Ensemble):
        All predictions share a single global noise scale.
        -> Uncertainty is flat (same for easy and hard configurations).
        -> Poor error-uncertainty correlation.

    Heteroscedastic (this model):
        Each atom gets its own noise from NoiseNet.
        -> Uncertainty varies with input (high where errors are expected).
        -> Strong error-uncertainty correlation.
        -> Better calibrated prediction intervals.
    """
    n_points = 100

    # True errors: some points are easy (small error), some hard (large)
    true_errors = np.concatenate([
        np.random.randn(50) * 0.01,     # easy configurations
        np.random.randn(50) * 0.10,     # hard configurations
    ])

    # Homoscedastic: same sigma for all points
    sigma_homo = np.full(n_points, 0.05)  # fixed global noise

    # Heteroscedastic: NoiseNet learns to assign higher noise to hard configs
    sigma_hetero = np.concatenate([
        np.full(50, 0.015),   # low noise for easy configs
        np.full(50, 0.09),    # high noise for hard configs
    ])

    # Error-uncertainty correlation
    corr_homo = np.corrcoef(np.abs(true_errors), sigma_homo)[0, 1]
    corr_hetero = np.corrcoef(np.abs(true_errors), sigma_hetero)[0, 1]

    print(f"Error-UQ correlation (homoscedastic):   {corr_homo:.3f}")
    print(f"Error-UQ correlation (heteroscedastic): {corr_hetero:.3f}")
    print()
    print("The heteroscedastic model assigns uncertainty proportional")
    print("to the expected error, enabling reliable confidence intervals.")


if __name__ == "__main__":
    print("=" * 70)
    print("Example 1: NoiseNet creation and forward pass")
    print("=" * 70)
    create_noise_net_example()

    print()
    print("=" * 70)
    print("Example 2: Uncertainty decomposition (epistemic + aleatoric)")
    print("=" * 70)
    uncertainty_decomposition_example()

    print()
    print("=" * 70)
    print("Example 3: Homoscedastic vs Heteroscedastic comparison")
    print("=" * 70)
    compare_homo_vs_hetero()
