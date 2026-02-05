"""Load predictions from log directories."""
import pandas as pd
import glob
from pathlib import Path
from typing import Tuple, Optional, List
import numpy as np


class PredictionLoader:
    """Load and process predictions from log directories.
    
    This class handles loading predictions for different methods:
    - BNN methods (lrt, fo, rad): Single parquet file per run
    - DE (deep ensemble): Multiple parquet files that need to be aggregated
    
    Attributes:
        config: DatasetConfig instance with paths and parameters
    """
    
    def __init__(self, config):
        """Initialize loader with dataset configuration.
        
        Args:
            config: DatasetConfig instance
        """
        self.config = config
    
    def load_method_predictions(
        self, 
        method: str, 
        size: str, 
        run: int, 
        split: str = 'test'
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
        """Load predictions for a specific method, data size, and run.
        
        Args:
            method: Method name ('lrt', 'fo', 'rad', 'de')
            size: Data size ('high' or 'low')
            run: Run number (0-4 typically)
            split: Data split ('train', 'valid', or 'test')
            
        Returns:
            Tuple of (y_true, y_pred, y_std, n_atoms)
            - y_true: True energy values (eV/atom)
            - y_pred: Predicted energy values (eV/atom)
            - y_std: Prediction uncertainty (eV/atom), None for deterministic
            - n_atoms: Number of atoms per structure
        """
        # Get file pattern based on method
        if method == 'de':
            files = self._find_de_files(size, run)
            return self._process_de_predictions(files, size, split)
        else:
            file_path = self._find_bnn_file(method, size, run)
            return self._process_bnn_predictions(file_path, size, split)
    
    def _find_bnn_file(self, method: str, size: str, run: int) -> Path:
        """Find BNN prediction file.
        
        Args:
            method: Method name ('lrt', 'fo', 'rad')
            size: Data size ('high' or 'low')
            run: Run number
            
        Returns:
            Path to parquet file
        """
        size_suffix = '_small' if size == 'low' else ''
        pred_dir = self.config.get_pred_dir(method)
        
        # Try standard naming convention
        file_path = pred_dir / f"{method}{size_suffix}_train_best_{run}" / f"{method.upper()}_0_val.parquet"
        
        if not file_path.exists():
            raise FileNotFoundError(
                f"Prediction file not found: {file_path}\n"
                f"Available runs: {list(pred_dir.glob('*'))}"
            )
        
        return file_path
    
    def _find_de_files(self, size: str, run: int) -> List[Path]:
        """Find DE (deep ensemble) prediction files.
        
        Args:
            size: Data size ('high' or 'low')
            run: Run number
            
        Returns:
            List of paths to parquet files (one per ensemble member)
        """
        size_num = '20' if size == 'low' else '100'
        de_dir = self.config.get_pred_dir('de')
        
        # Find the specific run directory
        run_dir = de_dir / f"de_pred_{size_num}_{run}" / "runs"
        
        if not run_dir.exists():
            raise FileNotFoundError(
                f"DE run directory not found: {run_dir}"
            )
        
        # Find timestamped subdirectories (e.g., 2025-03-11_10-24-21)
        timestamp_dirs = sorted([d for d in run_dir.iterdir() if d.is_dir()])
        
        if not timestamp_dirs:
            raise FileNotFoundError(
                f"No timestamp directories found in: {run_dir}"
            )
        
        # Use the most recent (last) timestamp directory
        latest_dir = timestamp_dirs[-1]
        
        # Find all parquet files in this directory
        files = list(latest_dir.glob("*.parquet"))
        
        if not files:
            raise FileNotFoundError(
                f"No DE predictions found in: {latest_dir}"
            )
        
        return sorted(files)
    
    def _process_bnn_predictions(
        self, 
        file_path: Path, 
        size: str, 
        split: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Process BNN predictions from a single parquet file.
        
        Args:
            file_path: Path to parquet file
            size: Data size ('high' or 'low')
            split: Data split name
            
        Returns:
            Tuple of (y_true, y_pred, y_std, n_atoms)
        """
        # Load predictions
        rs = pd.read_csv(file_path)
        
        # Get indices for this split
        indices = self.config.get_indices(size, split)
        
        # Extract data
        n_atoms = rs['n_atoms'].to_numpy()[indices]
        true_raw = rs['true'].to_numpy()[indices]
        pred_raw = rs['preds'].to_numpy()[indices]
        std_raw = rs['stds'].to_numpy()[indices] if 'stds' in rs.columns else None
        
        # Denormalize energies
        y_true = self._denormalize_energy(true_raw, n_atoms)
        y_pred = self._denormalize_energy(pred_raw, n_atoms)
        y_std = self._denormalize_energy(std_raw, n_atoms) if std_raw is not None else None
        
        return y_true, y_pred, y_std, n_atoms
    
    def _process_de_predictions(
        self, 
        file_paths: List[Path], 
        size: str, 
        split: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Process DE predictions from multiple ensemble members.
        
        Args:
            file_paths: List of paths to parquet files
            size: Data size ('high' or 'low')
            split: Data split name
            
        Returns:
            Tuple of (y_true, y_pred, y_std, n_atoms)
            - y_pred is the ensemble mean
            - y_std is the ensemble standard deviation
        """
        # Load first file to check size
        rs_first = pd.read_csv(file_paths[0])
        n_samples = len(rs_first)
        
        # Get indices for this split
        indices = self.config.get_indices(size, split)
        
        # Check if data is already filtered (indices out of bounds)
        # This happens for DE low data which was predicted with valid_split=100
        if indices.max() >= n_samples:
            # Data is already filtered to the test set, use all rows
            print(f"  Note: DE {size} data appears pre-filtered ({n_samples} rows), using all data")
            indices = np.arange(n_samples)
        
        y_preds = []
        
        # Load all ensemble members
        for parquet in file_paths:
            rs = pd.read_csv(parquet)
            n_atoms_full = rs['n_atoms'].to_numpy()
            pred_raw_full = rs['preds'].to_numpy()
            
            # Apply indices
            n_atoms = n_atoms_full[indices]
            pred_raw = pred_raw_full[indices]
            
            y_pred = self._denormalize_energy(pred_raw, n_atoms)
            y_preds.append(y_pred)
        
        # Compute ensemble statistics
        y_pred_all = np.array(y_preds).mean(axis=0)
        y_std_all = np.std(y_preds, axis=0)
        
        # Get true values from first file
        n_atoms_full = rs_first['n_atoms'].to_numpy()
        true_raw_full = rs_first['true'].to_numpy()
        
        # Apply indices
        n_atoms = n_atoms_full[indices]
        true_raw = true_raw_full[indices]
        
        y_true_all = self._denormalize_energy(true_raw, n_atoms)
        
        return y_true_all, y_pred_all, y_std_all, n_atoms
    
    def _denormalize_energy(self, energy_raw: np.ndarray, n_atoms: np.ndarray) -> np.ndarray:
        """Denormalize energy values to eV/atom.
        
        Args:
            energy_raw: Raw energy values (scaled)
            n_atoms: Number of atoms per structure
            
        Returns:
            Denormalized energy per atom (eV/atom)
        """
        return (energy_raw / self.config.e_scaling + n_atoms * self.config.e_shift) / n_atoms
    
    def get_available_runs(self, method: str, size: str) -> List[int]:
        """Get list of available run numbers for a method and size.
        
        Args:
            method: Method name
            size: Data size ('high' or 'low')
            
        Returns:
            List of available run numbers
        """
        if method == 'de':
            size_num = '20' if size == 'low' else '100'
            de_dir = self.config.get_pred_dir('de')
            pattern = f"de_pred_{size_num}_*"
            
            if not de_dir.exists():
                return []
            
            runs = []
            for p in de_dir.glob(pattern):
                try:
                    # Extract run number from de_pred_100_1 -> 1
                    run_num = int(p.name.split('_')[-1])
                    runs.append(run_num)
                except (ValueError, IndexError):
                    continue
        else:
            size_suffix = '_small' if size == 'low' else ''
            pred_dir = self.config.get_pred_dir(method)
            
            if not pred_dir.exists():
                return []
            
            pattern = f"{method}{size_suffix}_train_best_*"
            runs = []
            for p in pred_dir.glob(pattern):
                try:
                    run_num = int(p.name.split('_')[-1])
                    runs.append(run_num)
                except (ValueError, IndexError):
                    continue
        
        return sorted(runs)
