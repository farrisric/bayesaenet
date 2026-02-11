"""
Utility functions for BNN-AENET models.

Provides helper functions for parameter management, weight initialization,
and per-atom RMSE computation.
"""

import pyro
import torch
from torch import nn


def param_store_to(device: str):
    """Move all Pyro parameter store tensors to the specified device."""
    ps = pyro.get_param_store().get_state()
    for k in ps["params"].keys():
        ps["params"][k] = ps["params"][k].to(device)
    pyro.get_param_store().set_state(ps)


def remove_dict_entry_startswith(dictionary, string):
    """Remove entries whose keys start with the given string.

    Used to remove 'bnn'-prefixed entries from checkpoint state dicts.
    """
    n = len(string)
    for key in dictionary:
        if string == key[:n]:
            dict2 = dictionary.copy()
            dict2.pop(key)
            dictionary = dict2
    return dictionary


def weights_init(m):
    """Initialize weights of a nn.Module: xavier for conv, kaiming for linear."""
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_normal_(m.weight)
    elif isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)


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
