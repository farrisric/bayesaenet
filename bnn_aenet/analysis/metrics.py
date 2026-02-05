"""Compute performance and uncertainty quantification metrics."""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Optional
import warnings

# Try to import uncertainty_toolbox, but make it optional
try:
    import uncertainty_toolbox as uct
    HAS_UCT = True
except ImportError:
    HAS_UCT = False
    warnings.warn(
        "uncertainty_toolbox not installed. "
        "Some advanced calibration metrics will not be available. "
        "Install with: pip install uncertainty-toolbox"
    )


def compute_overlap_metric(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray) -> float:
    """Compute overlap between high-uncertainty and high-error quartiles.
    
    This is a custom metric that measures how well uncertainty estimates
    align with prediction errors. Higher values indicate better calibration.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        y_std: Prediction uncertainties (standard deviations)
        
    Returns:
        Percentage of high-uncertainty points falling in top error quartile
    """
    errors = np.abs(y_true - y_pred)
    
    # Compute quartile thresholds
    q3_error = np.percentile(errors, 75)
    q3_uncertainty = np.percentile(y_std, 75)
    
    # Boolean masks
    high_error = errors > q3_error
    high_uncertainty = y_std > q3_uncertainty
    high_both = high_error & high_uncertainty
    
    n_overlap = np.sum(high_both)
    n_high_uncertainty = np.sum(high_uncertainty)
    
    if n_high_uncertainty == 0:
        return 0.0
    
    return 100 * n_overlap / n_high_uncertainty


def compute_nll(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray) -> float:
    """Compute negative log-likelihood assuming Gaussian distribution.
    
    Args:
        y_true: True values
        y_pred: Predicted mean values
        y_std: Predicted standard deviations
        
    Returns:
        Negative log-likelihood
    """
    # Ensure std is positive
    y_std = np.maximum(y_std, 1e-8)
    
    nll = F.gaussian_nll_loss(
        torch.tensor(y_pred, dtype=torch.float32),
        torch.tensor(y_true, dtype=torch.float32),
        torch.square(torch.tensor(y_std, dtype=torch.float32))
    ).item()
    
    return nll


def compute_performance_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute standard performance metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metric names and values
    """
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'maxerr': np.max(np.abs(y_true - y_pred)),
        'r2score': r2_score(y_true, y_pred),
    }
    
    return metrics


def compute_uq_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray) -> Dict[str, float]:
    """Compute uncertainty quantification metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted mean values
        y_std: Predicted standard deviations
        
    Returns:
        Dictionary of UQ metric names and values
    """
    metrics = {
        'sharp': np.mean(y_std),  # Average uncertainty (sharpness)
        'overlap': compute_overlap_metric(y_true, y_pred, y_std),
        'nll': compute_nll(y_true, y_pred, y_std),
    }
    
    # Add uncertainty_toolbox metrics if available
    if HAS_UCT:
        try:
            uct_metrics = uct.metrics.get_all_metrics(y_pred, y_std, y_true)
            
            # Add selected UCT metrics
            for key in ['ece', 'rmsce', 'ma', 'rms_cal', 'miscal_area']:
                if key in uct_metrics:
                    metrics[key] = uct_metrics[key]
        except Exception as e:
            warnings.warn(f"Failed to compute uncertainty_toolbox metrics: {e}")
    
    return metrics


def compute_all_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    y_std: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Compute all performance and UQ metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        y_std: Prediction uncertainties (optional, for UQ metrics)
        
    Returns:
        Dictionary of all metric names and values
    """
    # Performance metrics (always computed)
    metrics = compute_performance_metrics(y_true, y_pred)
    
    # UQ metrics (only if uncertainties provided)
    if y_std is not None:
        uq_metrics = compute_uq_metrics(y_true, y_pred, y_std)
        metrics.update(uq_metrics)
    
    return metrics
