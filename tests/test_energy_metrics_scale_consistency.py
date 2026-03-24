"""Regression tests for energy metric scale consistency."""

import numpy as np

from bnn_aenet.analysis.metrics import compute_energy_metrics


def _expected_r2(values: np.ndarray) -> float:
    ss_res = np.sum(values**2)
    ss_tot = np.sum((values - np.mean(values)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


def test_energy_metrics_use_per_atom_scale_for_all_deterministic_metrics() -> None:
    """When n_atoms is provided, deterministic metrics should all be per-atom."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([8.0, 18.0, 24.0])
    n_atoms = np.array([2.0, 2.0, 3.0])

    metrics = compute_energy_metrics(y_true, y_pred, n_atoms=n_atoms)
    per_atom_errors = (y_true - y_pred) / n_atoms

    assert np.isclose(metrics["mae"], np.mean(np.abs(per_atom_errors)))
    assert np.isclose(metrics["rmse"], np.sqrt(np.mean(per_atom_errors**2)))
    assert np.isclose(metrics["max_err"], np.max(np.abs(per_atom_errors)))
    assert np.isclose(metrics["mean_error"], np.mean(per_atom_errors))
    assert np.isclose(metrics["std_error"], np.std(per_atom_errors))
    assert np.isclose(metrics["r2"], _expected_r2(per_atom_errors))


def test_energy_metrics_without_n_atoms_match_total_energy_behavior() -> None:
    """Without n_atoms, metrics should use total-energy errors."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([8.0, 18.0, 24.0])
    errors = y_true - y_pred

    metrics = compute_energy_metrics(y_true, y_pred)

    assert np.isclose(metrics["mae"], np.mean(np.abs(errors)))
    assert np.isclose(metrics["rmse"], np.sqrt(np.mean(errors**2)))
    assert np.isclose(metrics["max_err"], np.max(np.abs(errors)))
    assert np.isclose(metrics["mean_error"], np.mean(errors))
    assert np.isclose(metrics["std_error"], np.std(errors))
    assert np.isclose(metrics["r2"], _expected_r2(errors))


def test_energy_metrics_mismatched_n_atoms_falls_back_to_total_energy() -> None:
    """Mismatched n_atoms length should keep total-energy metric behavior."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([8.0, 18.0, 24.0])
    n_atoms = np.array([2.0, 2.0])  # mismatched length
    errors = y_true - y_pred

    metrics = compute_energy_metrics(y_true, y_pred, n_atoms=n_atoms)

    assert np.isclose(metrics["mae"], np.mean(np.abs(errors)))
    assert np.isclose(metrics["rmse"], np.sqrt(np.mean(errors**2)))
    assert np.isclose(metrics["max_err"], np.max(np.abs(errors)))
    assert np.isclose(metrics["mean_error"], np.mean(errors))
    assert np.isclose(metrics["std_error"], np.std(errors))
    assert np.isclose(metrics["r2"], _expected_r2(errors))
