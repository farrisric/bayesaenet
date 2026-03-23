"""Analysis and plotting utilities for BNN-AENET.

This module provides functions for:
- Loading and analyzing predictions
- Computing uncertainty metrics
- Creating publication-quality plots
"""

from .metrics import (
    compute_energy_metrics,
    compute_force_metrics,
    compute_uncertainty_metrics,
)

try:
    from .plots import (
        plot_parity,
        plot_residuals,
        plot_uncertainty_calibration,
        plot_training_curves,
    )
except ModuleNotFoundError:
    # Plot helpers are optional in this repo snapshot.
    pass

__all__ = [
    "compute_energy_metrics",
    "compute_force_metrics",
    "compute_uncertainty_metrics",
]

try:
    __all__.extend(
        [
            "plot_parity",
            "plot_residuals",
            "plot_uncertainty_calibration",
            "plot_training_curves",
        ]
    )
except NameError:
    pass
