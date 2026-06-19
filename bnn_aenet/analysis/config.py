"""Dataset configurations for analysis."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class DatasetConfig:
    """Configuration for a dataset analysis.

    Attributes:
        name: Dataset name (e.g., 'TiO2', 'QM7')
        logs_dir: Path to logs directory containing predictions
        indices_dir: Path to directory containing train/val/test indices
        e_scaling: Energy scaling factor for denormalization
        e_shift: Energy shift factor for denormalization
        methods: List of methods to analyze (e.g., ['lrt', 'fo', 'rad', 'de'])
        data_sizes: Mapping of size labels to data directories
    """

    name: str
    logs_dir: Path
    indices_dir: Path
    e_scaling: float
    e_shift: float
    methods: List[str]
    data_sizes: Dict[str, str]

    def get_indices(self, size: str, split: str) -> np.ndarray:
        """Load indices for a specific data size and split.

        Args:
            size: Data size key ('high' or 'low')
            split: Data split ('train', 'valid', or 'test')

        Returns:
            Array of indices
        """
        size_dir = self.data_sizes[size]
        idx_file = self.indices_dir / size_dir / f"{split}_set_idxes.txt"

        if not idx_file.exists():
            raise FileNotFoundError(f"Indices file not found: {idx_file}")

        return np.genfromtxt(idx_file).astype(int)

    def get_pred_dir(self, method: str) -> Path:
        """Get prediction directory for a method.

        Args:
            method: Method name (e.g., 'lrt', 'fo', 'rad', 'de')

        Returns:
            Path to prediction directory
        """
        if method == "de":
            return self.logs_dir / "DE"
        else:
            return self.logs_dir / f"{method}_pred" / "runs"


# Repository root - always resolve from this file's actual location
# __file__ is in: repo_root/bnn_aenet/analysis/config.py
# So parent.parent.parent gives us repo_root
_config_path = Path(__file__).resolve()
REPO_ROOT = _config_path.parent.parent.parent

# TiO2 Configuration
TIO2_CONFIG = DatasetConfig(
    name="TiO2",
    logs_dir=REPO_ROOT / "bnn_aenet" / "logs" / "TiO2_big",
    indices_dir=REPO_ROOT / "bnn_aenet" / "data" / "TiO2" / "indices",
    e_scaling=0.06565926932648217,
    e_shift=6.6588702845000975,
    methods=["lrt", "fo", "rad", "de"],
    data_sizes={"high": "Data100", "low": "Data20"},
)

# QM7 Configuration (for future use)
QM7_CONFIG = DatasetConfig(
    name="QM7",
    logs_dir=REPO_ROOT / "bnn_aenet" / "logs",
    indices_dir=REPO_ROOT / "bnn_aenet" / "data" / "QM7" / "indices",
    e_scaling=0.9754923797786934,
    e_shift=-4.652443333333333,
    methods=["lrt", "nn"],
    data_sizes={"high": "Data100"},  # Only 100% data for QM7
)
