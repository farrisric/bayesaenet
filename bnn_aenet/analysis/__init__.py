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
from .plots import (
    plot_parity,
    plot_residuals,
    plot_uncertainty_calibration,
    plot_training_curves,
)

__all__ = [
    "compute_energy_metrics",
    "compute_force_metrics",
    "compute_uncertainty_metrics",
    "plot_parity",
    "plot_residuals",
    "plot_uncertainty_calibration",
    "plot_training_curves",
]
