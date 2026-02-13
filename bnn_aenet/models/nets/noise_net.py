"""
Heteroscedastic noise network for per-atom observation noise prediction.

Predicts input-dependent (heteroscedastic) observation noise for both
energy and force likelihoods.  Each atomic species has a small MLP that
maps descriptors to a positive noise scale.

The noise network is kept **deterministic** (not converted to a
PyroModule) because aleatoric noise is a property of the data, not the
model.  Its parameters are registered with Pyro via ``pyro.module()``
so they are optimized jointly by SVI.
"""

import torch
import torch.nn as nn
import torch.nn.functional as Fn


class NoiseNet(nn.Module):
    """Per-atom noise prediction network.

    One small MLP per atomic species:
        Linear(input_dim, hidden) -> Tanh -> Linear(hidden, 1)
    Output goes through ``softplus + min_noise`` to ensure a positive,
    bounded-below noise scale.

    Args:
        input_size: List of descriptor dimensions per species.
        species: List of species names (e.g. ``["Ti", "O"]``).
        hidden_size: Hidden layer width (same for all species).
        min_noise: Minimum noise floor to prevent collapse.
    """

    def __init__(
        self,
        input_size,
        species,
        hidden_size: int = 15,
        min_noise: float = 0.01,
    ):
        super().__init__()
        self.species = species
        self.min_noise = min_noise

        heads = []
        for i, sp in enumerate(species):
            heads.append(nn.Sequential(
                nn.Linear(input_size[i], hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1),
            ))
        self.heads = nn.ModuleList(heads)

    # ------------------------------------------------------------------
    # Energy noise
    # ------------------------------------------------------------------

    def forward_energy(self, E_descrp, E_logic_reduce):
        """Compute per-structure energy noise from energy descriptors.

        Per-atom noise variances are summed across atoms belonging to
        the same structure (variances add for independent noise), then
        the square root gives the per-structure standard deviation.

        Args:
            E_descrp: List of descriptor tensors per species.
            E_logic_reduce: List of logic-reduce tensors per species
                (maps atoms -> structures).

        Returns:
            Tensor of shape ``(n_structures,)`` -- per-structure sigma_E.
        """
        device = E_descrp[0].device
        n_structures = E_logic_reduce[0].shape[0]
        noise_var = torch.zeros(n_structures, device=device)

        for iesp in range(len(self.species)):
            # Per-atom noise scale -> variance
            raw = self.heads[iesp](E_descrp[iesp].float())  # (n_atoms, 1)
            sigma_atom = Fn.softplus(raw) + self.min_noise   # (n_atoms, 1)
            var_atom = sigma_atom ** 2                        # (n_atoms, 1)

            # Sum variances per structure using logic_reduce
            # logic_reduce[iesp]: (n_structures, n_atoms_of_species)
            # var_atom: (n_atoms_of_species, 1)
            noise_var = noise_var + torch.einsum(
                "ij,ki->k", var_atom, E_logic_reduce[iesp]
            )

        return torch.sqrt(noise_var + 1e-8)  # per-structure sigma

    # ------------------------------------------------------------------
    # Force noise
    # ------------------------------------------------------------------

    def forward_forces(self, F_descrp, grp_indices_F_i):
        """Compute per-force-component noise from force descriptors.

        Each atom gets a single noise scale; it is expanded to all 3
        Cartesian components (x, y, z) of that atom's force.

        Args:
            F_descrp: List of descriptor tensors per species (force batch).
            grp_indices_F_i: Index tensor that maps from the per-species
                concatenation to the final force-atom ordering (same
                indexing as ``forward_F`` uses for ``aux_F_i``).

        Returns:
            Tensor of shape ``(n_force_atoms, 3)`` -- per-component sigma_F.
        """
        noise_per_species = []
        for iesp in range(len(self.species)):
            raw = self.heads[iesp](F_descrp[iesp].float())  # (n_atoms, 1)
            sigma = Fn.softplus(raw) + self.min_noise         # (n_atoms, 1)
            noise_per_species.append(sigma)

        # Concatenate across species (same order as aux_F_i in forward_F)
        noise_all = torch.cat(noise_per_species, dim=0)  # (total_atoms, 1)

        # Reorder to match F_ann atom ordering
        noise_per_atom = torch.index_select(
            noise_all.squeeze(-1), 0, grp_indices_F_i
        )  # (n_force_atoms,)

        # Expand to 3 Cartesian components
        return noise_per_atom.unsqueeze(1).expand(-1, 3)  # (n_force_atoms, 3)
