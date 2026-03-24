"""Metrics computation for energy and force predictions.

Provides functions to compute:
- Energy metrics: MAE, RMSE, R², MaxErr
- Force metrics: MAE, RMSE, magnitude error, angular error
- Uncertainty metrics: NLL, calibration error, sharpness
"""

import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional


def compute_energy_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: Optional[np.ndarray] = None,
    n_atoms: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute energy prediction metrics.

    When n_atoms is provided and has one entry per structure, deterministic error
    metrics are computed from per-atom errors to match TensorBoard-style reporting
    ((E_pred - E_true) / n_atoms). If n_atoms is missing or mismatched in length,
    deterministic metrics fall back to total-energy errors.
    
    Args:
        y_true: Ground truth energies (total per structure)
        y_pred: Predicted energies (total per structure)
        y_std: Predicted standard deviations (optional)
        n_atoms: Number of atoms per structure (optional; when set, uses per-atom errors)
    
    Returns:
        Dictionary with metrics
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    errors = y_true - y_pred
    metric_errors = errors

    if n_atoms is not None:
        n_atoms = np.asarray(n_atoms).flatten()
        if len(n_atoms) == len(errors):
            metric_errors = errors / np.clip(n_atoms, 1, None)
        else:
            n_atoms = None

    mae = np.mean(np.abs(metric_errors))
    rmse = np.sqrt(np.mean(metric_errors**2))
    max_err = np.max(np.abs(metric_errors))
    
    # R-squared
    ss_res = np.sum(metric_errors**2)
    ss_tot = np.sum((metric_errors - np.mean(metric_errors))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    metrics = {
        "mae": mae,
        "rmse": rmse,
        "max_err": max_err,
        "r2": r2,
        "mean_error": np.mean(metric_errors),
        "std_error": np.std(metric_errors),
    }
    
    # Uncertainty metrics if std provided
    if y_std is not None:
        y_std = np.asarray(y_std).flatten()
        y_std = np.clip(y_std, 1e-6, None)  # Avoid division by zero
        
        # Negative log-likelihood (Gaussian)
        nll = 0.5 * np.mean(np.log(2 * np.pi * y_std**2) + (errors / y_std)**2)
        metrics["nll"] = nll
        
        # Mean predicted uncertainty
        metrics["mean_std"] = np.mean(y_std)
        
        # Calibration: fraction of true values within predicted intervals
        for confidence in [0.68, 0.95, 0.99]:
            z = stats.norm.ppf((1 + confidence) / 2)
            within = np.mean(np.abs(errors) < z * y_std)
            metrics[f"calib_{int(confidence*100)}"] = within
    
    return metrics


def compute_force_metrics(
    f_true: np.ndarray,
    f_pred: np.ndarray,
    f_std: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute force prediction metrics.
    
    Args:
        f_true: Ground truth forces (N_atoms, 3) or flattened
        f_pred: Predicted forces
        f_std: Predicted standard deviations (optional)
    
    Returns:
        Dictionary with metrics
    """
    f_true = np.asarray(f_true)
    f_pred = np.asarray(f_pred)
    
    # Flatten if needed
    if f_true.ndim > 1:
        f_true = f_true.flatten()
        f_pred = f_pred.flatten()
    
    # Component-wise metrics
    errors = f_true - f_pred
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    max_err = np.max(np.abs(errors))
    
    # R-squared
    ss_res = np.sum(errors**2)
    ss_tot = np.sum((f_true - np.mean(f_true))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    metrics = {
        "mae": mae,
        "rmse": rmse,
        "max_err": max_err,
        "r2": r2,
        "mean_error": np.mean(errors),
        "std_error": np.std(errors),
    }
    
    # If we can compute magnitude errors (3D vectors)
    if f_true.shape[0] % 3 == 0:
        n_atoms = f_true.shape[0] // 3
        f_true_3d = f_true.reshape(n_atoms, 3)
        f_pred_3d = f_pred.reshape(n_atoms, 3)
        
        # Magnitude errors
        mag_true = np.linalg.norm(f_true_3d, axis=1)
        mag_pred = np.linalg.norm(f_pred_3d, axis=1)
        mag_errors = mag_true - mag_pred
        
        metrics["mag_mae"] = np.mean(np.abs(mag_errors))
        metrics["mag_rmse"] = np.sqrt(np.mean(mag_errors**2))
        
        # Angular errors (for non-zero vectors)
        mask = (mag_true > 1e-6) & (mag_pred > 1e-6)
        if np.sum(mask) > 0:
            dot_products = np.sum(f_true_3d[mask] * f_pred_3d[mask], axis=1)
            cos_angles = dot_products / (mag_true[mask] * mag_pred[mask])
            cos_angles = np.clip(cos_angles, -1, 1)
            angles = np.arccos(cos_angles) * 180 / np.pi  # degrees
            metrics["mean_angle_error"] = np.mean(angles)
            metrics["max_angle_error"] = np.max(angles)
    
    # Uncertainty metrics
    if f_std is not None:
        f_std = np.asarray(f_std).flatten()
        f_std = np.clip(f_std, 1e-6, None)
        
        nll = 0.5 * np.mean(np.log(2 * np.pi * f_std**2) + (errors / f_std)**2)
        metrics["nll"] = nll
        metrics["mean_std"] = np.mean(f_std)
    
    return metrics


def compute_uncertainty_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
) -> Dict[str, float]:
    """Compute uncertainty quantification metrics.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        y_std: Predicted standard deviations
    
    Returns:
        Dictionary with UQ metrics
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    y_std = np.asarray(y_std).flatten()
    y_std = np.clip(y_std, 1e-8, None)
    
    errors = y_true - y_pred
    abs_errors = np.abs(errors)
    
    metrics = {}
    
    # Negative Log-Likelihood
    nll = 0.5 * np.mean(np.log(2 * np.pi * y_std**2) + (errors / y_std)**2)
    metrics["nll"] = nll
    
    # Calibration error (Expected Calibration Error)
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        confidence = (bin_edges[i] + bin_edges[i + 1]) / 2
        z = stats.norm.ppf((1 + confidence) / 2)
        
        # Expected fraction within confidence interval
        expected_frac = confidence
        
        # Actual fraction
        actual_frac = np.mean(abs_errors < z * y_std)
        
        ece += np.abs(expected_frac - actual_frac) / n_bins
    
    metrics["ece"] = ece
    
    # Sharpness (average predicted variance)
    metrics["sharpness"] = np.mean(y_std**2)
    metrics["mean_std"] = np.mean(y_std)
    
    # Correlation between error and uncertainty
    if np.std(y_std) > 0:
        corr, _ = stats.pearsonr(abs_errors, y_std)
        metrics["error_std_corr"] = corr
    else:
        metrics["error_std_corr"] = 0.0
    
    # PICP (Prediction Interval Coverage Probability) at 95%
    z_95 = stats.norm.ppf(0.975)
    picp_95 = np.mean(abs_errors < z_95 * y_std)
    metrics["picp_95"] = picp_95
    
    # MPIW (Mean Prediction Interval Width) at 95%
    metrics["mpiw_95"] = np.mean(2 * z_95 * y_std)
    
    return metrics


def compute_calibration_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    n_bins: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute calibration curve data.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values  
        y_std: Predicted standard deviations
        n_bins: Number of confidence levels
    
    Returns:
        expected_freq: Expected frequencies (confidence levels)
        observed_freq: Observed frequencies
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    y_std = np.asarray(y_std).flatten()
    y_std = np.clip(y_std, 1e-8, None)
    
    errors = y_true - y_pred
    abs_errors = np.abs(errors)
    
    expected_freq = np.linspace(0.05, 0.95, n_bins)
    observed_freq = np.zeros(n_bins)
    
    for i, conf in enumerate(expected_freq):
        z = stats.norm.ppf((1 + conf) / 2)
        observed_freq[i] = np.mean(abs_errors < z * y_std)
    
    return expected_freq, observed_freq


def compute_combined_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: Optional[np.ndarray] = None,
    f_true: Optional[np.ndarray] = None,
    f_pred: Optional[np.ndarray] = None,
    f_std: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute combined energy and force metrics.
    
    Wrapper that calls compute_energy_metrics and compute_force_metrics,
    returning a merged dict with 'force_' prefix for force metrics.
    
    Args:
        y_true: Ground truth energies
        y_pred: Predicted energies
        y_std: Predicted energy standard deviations (optional)
        f_true: Ground truth forces (optional)
        f_pred: Predicted forces (optional)
        f_std: Predicted force standard deviations (optional)
    
    Returns:
        Dictionary with all metrics (energy keys as-is, force keys prefixed with 'force_')
    """
    # Energy metrics
    metrics = compute_energy_metrics(y_true, y_pred, y_std)
    
    # Energy uncertainty metrics
    if y_std is not None:
        uq_metrics = compute_uncertainty_metrics(y_true, y_pred, y_std)
        for k, v in uq_metrics.items():
            if k not in metrics:
                metrics[k] = v
    
    # Force metrics
    if f_true is not None and f_pred is not None:
        force_metrics = compute_force_metrics(f_true, f_pred, f_std)
        for k, v in force_metrics.items():
            metrics[f"force_{k}"] = v
        
        # Force uncertainty metrics
        if f_std is not None:
            force_uq = compute_uncertainty_metrics(f_true, f_pred, f_std)
            for k, v in force_uq.items():
                metrics[f"force_{k}"] = v
    
    return metrics
