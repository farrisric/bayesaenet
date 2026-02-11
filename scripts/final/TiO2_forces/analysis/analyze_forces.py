"""Comprehensive analysis of TiO2 force-trained models.

Creates Deep Ensembles from NN predictions, selects best BNN models,
computes metrics, and generates comparison plots.

Usage:
    python scripts/final/TiO2_forces/analysis/analyze_forces.py \
        --pred-dir bnn_aenet/logs/forces_pred \
        --output-dir plots/TiO2_forces
"""

import argparse
import os
import sys
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for cluster
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from bnn_aenet.analysis.metrics import (
    compute_energy_metrics,
    compute_force_metrics,
    compute_uncertainty_metrics,
    compute_calibration_curve,
)

# Plotting style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox_inches": "tight",
})

# Method display names and colors
METHOD_NAMES = {
    "DE": "Deep Ensemble",
    "DE_sub": "DE (5-model)",
    "nn": "NN (individual)",
    "lrt": "LRT",
    "fo": "Flipout",
    "rad": "Radial",
}

METHOD_COLORS = {
    "DE": "#2196F3",
    "DE_sub": "#90CAF9",
    "nn": "#4CAF50",
    "lrt": "#FF9800",
    "fo": "#9C27B0",
    "rad": "#F44336",
}


# ============================================================================
# Data Loading
# ============================================================================

def load_run_predictions(pred_dir: Path, model_type: str, subset: str = "val"):
    """Load all predictions for a model type and subset.

    Returns list of dicts, one per run, each with:
        - energy_df: DataFrame with true, preds, stds, n_atoms
        - forces: dict with true_forces, pred_forces, std_forces arrays (or None)
        - run_name: str
    """
    model_dir = pred_dir / model_type
    if not model_dir.exists():
        print(f"WARNING: {model_dir} does not exist")
        return []

    runs = []
    # Find all energy CSV files for this subset
    energy_files = sorted(model_dir.glob(f"*_{subset}_energy.csv"))

    for energy_file in energy_files:
        run_name = energy_file.name.replace(f"_{subset}_energy.csv", "")
        energy_df = pd.read_csv(energy_file)

        # Try to load corresponding force data
        force_file = model_dir / f"{run_name}_{subset}_forces.npz"
        forces = None
        if force_file.exists():
            fdata = np.load(force_file)
            forces = {
                "true_forces": fdata["true_forces"],
                "pred_forces": fdata["pred_forces"],
                "std_forces": fdata["std_forces"],
            }

        runs.append({
            "energy_df": energy_df,
            "forces": forces,
            "run_name": run_name,
        })

    return runs


# ============================================================================
# Deep Ensemble Creation
# ============================================================================

def create_deep_ensemble(runs: list) -> dict:
    """Create a Deep Ensemble from multiple NN predictions.

    Combines energy and force predictions from all runs.
    For energy: mu = mean(mu_m), sigma = sqrt(mean(mu_m^2 + sigma_m^2) - mu^2)
    For forces: same formula applied component-wise.

    Args:
        runs: List of run dicts from load_run_predictions.

    Returns:
        Dict with ensemble energy_df and forces.
    """
    n_models = len(runs)
    if n_models == 0:
        return None

    # Energy ensemble
    energy_preds = np.stack([r["energy_df"]["preds"].values for r in runs])
    energy_stds = np.stack([r["energy_df"]["stds"].values for r in runs])

    mu = energy_preds.mean(axis=0)
    sigma = np.sqrt((energy_preds**2 + energy_stds**2).mean(axis=0) - mu**2)

    energy_df = pd.DataFrame({
        "true": runs[0]["energy_df"]["true"].values,
        "preds": mu,
        "stds": sigma,
        "n_atoms": runs[0]["energy_df"]["n_atoms"].values,
    })

    # Force ensemble
    forces = None
    if all(r["forces"] is not None for r in runs):
        force_preds = np.stack([r["forces"]["pred_forces"] for r in runs])
        force_stds = np.stack([r["forces"]["std_forces"] for r in runs])

        f_mu = force_preds.mean(axis=0)
        f_sigma = np.sqrt((force_preds**2 + force_stds**2).mean(axis=0) - f_mu**2)

        forces = {
            "true_forces": runs[0]["forces"]["true_forces"],
            "pred_forces": f_mu,
            "std_forces": f_sigma,
        }

    return {
        "energy_df": energy_df,
        "forces": forces,
        "run_name": f"DE_{n_models}models",
        "n_models": n_models,
    }


def create_sub_ensembles(runs: list, n_per_ensemble: int = 5, max_ensembles: int = 20) -> list:
    """Create sub-ensembles from combinations of NN models.

    Args:
        runs: List of all NN run dicts.
        n_per_ensemble: Number of models per sub-ensemble.
        max_ensembles: Maximum number of sub-ensembles to create.

    Returns:
        List of ensemble dicts.
    """
    import random
    random.seed(42)

    n_models = len(runs)
    if n_models < n_per_ensemble:
        return []

    all_combos = list(combinations(range(n_models), n_per_ensemble))
    if len(all_combos) > max_ensembles:
        selected = random.sample(all_combos, max_ensembles)
    else:
        selected = all_combos

    ensembles = []
    for i, combo in enumerate(selected):
        sub_runs = [runs[j] for j in combo]
        ens = create_deep_ensemble(sub_runs)
        if ens is not None:
            ens["run_name"] = f"DE_sub_{i:02d}"
            ensembles.append(ens)

    return ensembles


# ============================================================================
# BNN Model Selection
# ============================================================================

def compute_run_metrics(run: dict, alpha: float = 0.1) -> dict:
    """Compute metrics for a single run."""
    df = run["energy_df"]
    y_true = df["true"].values
    y_pred = df["preds"].values
    y_std = df["stds"].values if "stds" in df.columns else None

    e_metrics = compute_energy_metrics(y_true, y_pred, y_std)

    metrics = {
        "run_name": run["run_name"],
        "energy_rmse": e_metrics["rmse"],
        "energy_mae": e_metrics["mae"],
        "energy_r2": e_metrics["r2"],
        "energy_maxerr": e_metrics["max_err"],
    }

    # Uncertainty metrics for energy
    if y_std is not None and np.any(y_std > 0):
        uq = compute_uncertainty_metrics(y_true, y_pred, y_std)
        metrics["energy_nll"] = uq["nll"]
        metrics["energy_ece"] = uq["ece"]
        metrics["energy_picp_95"] = uq["picp_95"]
        metrics["energy_sharpness"] = uq["sharpness"]
        metrics["energy_mean_std"] = uq["mean_std"]
        metrics["energy_error_std_corr"] = uq["error_std_corr"]

    if run["forces"] is not None:
        f_true = run["forces"]["true_forces"]
        f_pred = run["forces"]["pred_forces"]
        f_std = run["forces"]["std_forces"]

        f_metrics = compute_force_metrics(f_true, f_pred, f_std)
        metrics["force_rmse"] = f_metrics["rmse"]
        metrics["force_mae"] = f_metrics["mae"]
        metrics["force_r2"] = f_metrics["r2"]
        metrics["force_maxerr"] = f_metrics["max_err"]
        if "mag_mae" in f_metrics:
            metrics["force_mag_mae"] = f_metrics["mag_mae"]
            metrics["force_mag_rmse"] = f_metrics["mag_rmse"]
        if "mean_angle_error" in f_metrics:
            metrics["force_angle_error"] = f_metrics["mean_angle_error"]

        # Force uncertainty
        if f_std is not None and np.any(f_std > 0):
            f_uq = compute_uncertainty_metrics(f_true, f_pred, f_std)
            metrics["force_nll"] = f_uq["nll"]
            metrics["force_ece"] = f_uq["ece"]
            metrics["force_picp_95"] = f_uq["picp_95"]
            metrics["force_sharpness"] = f_uq["sharpness"]
            metrics["force_mean_std"] = f_uq["mean_std"]

        # Combined total RMSE
        metrics["total_rmse"] = (1 - alpha) * metrics["energy_rmse"] + alpha * metrics.get("force_rmse", 0)
    else:
        metrics["total_rmse"] = metrics["energy_rmse"]

    return metrics


def select_best_bnn(runs: list, alpha: float = 0.1) -> dict:
    """Select best BNN models by different criteria.

    Returns dict with keys: best_overall, best_energy, best_forces, all_metrics
    """
    all_metrics = [compute_run_metrics(r, alpha) for r in runs]
    df = pd.DataFrame(all_metrics)

    result = {"all_metrics": df}

    # Best overall (lowest total_rmse)
    best_idx = df["total_rmse"].idxmin()
    result["best_overall"] = {
        "run": runs[best_idx],
        "metrics": all_metrics[best_idx],
        "index": best_idx,
    }

    # Best energy
    best_e_idx = df["energy_rmse"].idxmin()
    result["best_energy"] = {
        "run": runs[best_e_idx],
        "metrics": all_metrics[best_e_idx],
        "index": best_e_idx,
    }

    # Best forces
    if "force_rmse" in df.columns:
        best_f_idx = df["force_rmse"].idxmin()
        result["best_forces"] = {
            "run": runs[best_f_idx],
            "metrics": all_metrics[best_f_idx],
            "index": best_f_idx,
        }

    return result


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_energy_parity_comparison(method_data: dict, output_dir: Path, subset: str = "val"):
    """Plot energy parity plots for all methods side by side."""
    methods = list(method_data.keys())
    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5))
    if n_methods == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        data = method_data[method]
        y_true = data["energy_df"]["true"].values
        y_pred = data["energy_df"]["preds"].values
        y_std = data["energy_df"]["stds"].values

        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        mae = np.mean(np.abs(y_true - y_pred))

        all_vals = np.concatenate([y_true, y_pred])
        vmin, vmax = all_vals.min(), all_vals.max()
        margin = 0.05 * (vmax - vmin)
        lim = [vmin - margin, vmax + margin]

        ax.plot(lim, lim, "k--", lw=1, alpha=0.5)

        if np.any(y_std > 0):
            sc = ax.scatter(y_true, y_pred, c=y_std, cmap="viridis", alpha=0.7, s=20, edgecolors="none")
            plt.colorbar(sc, ax=ax, label="Uncertainty")
        else:
            ax.scatter(y_true, y_pred, alpha=0.6, s=20, c=METHOD_COLORS.get(method, "steelblue"), edgecolors="none")

        name = METHOD_NAMES.get(method, method)
        ax.set_title(f"{name}\nRMSE: {rmse:.4f}, MAE: {mae:.4f}")
        ax.set_xlabel("True Energy")
        ax.set_ylabel("Predicted Energy")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")

    fig.suptitle(f"Energy Parity ({subset})", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / f"energy_parity_{subset}.png")
    plt.close(fig)


def plot_force_parity_comparison(method_data: dict, output_dir: Path, subset: str = "val"):
    """Plot force parity plots for all methods side by side."""
    methods = [m for m in method_data if method_data[m].get("forces") is not None]
    n_methods = len(methods)
    if n_methods == 0:
        return

    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5))
    if n_methods == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        forces = method_data[method]["forces"]
        f_true = forces["true_forces"]
        f_pred = forces["pred_forces"]
        f_std = forces["std_forces"]

        rmse = np.sqrt(np.mean((f_true - f_pred)**2))
        mae = np.mean(np.abs(f_true - f_pred))

        # Subsample for plotting
        n_pts = len(f_true)
        if n_pts > 10000:
            idx = np.random.choice(n_pts, 10000, replace=False)
        else:
            idx = np.arange(n_pts)

        all_vals = np.concatenate([f_true, f_pred])
        vmin, vmax = all_vals.min(), all_vals.max()
        margin = 0.1 * (vmax - vmin)
        lim = [vmin - margin, vmax + margin]

        ax.plot(lim, lim, "k--", lw=1, alpha=0.5)

        if np.any(f_std > 0):
            sc = ax.scatter(f_true[idx], f_pred[idx], c=f_std[idx], cmap="plasma", alpha=0.4, s=10, edgecolors="none")
            plt.colorbar(sc, ax=ax, label="Uncertainty")
        else:
            ax.scatter(f_true[idx], f_pred[idx], alpha=0.3, s=10, c=METHOD_COLORS.get(method, "coral"), edgecolors="none")

        name = METHOD_NAMES.get(method, method)
        ax.set_title(f"{name}\nRMSE: {rmse:.4f}, MAE: {mae:.4f}")
        ax.set_xlabel("True Force Component")
        ax.set_ylabel("Predicted Force Component")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")

    fig.suptitle(f"Force Parity ({subset})", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / f"force_parity_{subset}.png")
    plt.close(fig)


def plot_force_components_comparison(method_data: dict, output_dir: Path, subset: str = "val"):
    """Plot force x, y, z component parity for each method."""
    methods = [m for m in method_data if method_data[m].get("forces") is not None]

    for method in methods:
        forces = method_data[method]["forces"]
        f_true = forces["true_forces"].flatten()
        f_pred = forces["pred_forces"].flatten()

        if len(f_true) % 3 != 0:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        components = ["x", "y", "z"]
        colors = ["#e74c3c", "#2ecc71", "#3498db"]

        for i, (comp, color) in enumerate(zip(components, colors)):
            ct = f_true[i::3]
            cp = f_pred[i::3]
            errors = cp - ct
            rmse = np.sqrt(np.mean(errors**2))

            all_vals = np.concatenate([ct, cp])
            vmin, vmax = all_vals.min(), all_vals.max()
            margin = 0.1 * (vmax - vmin)
            lim = [vmin - margin, vmax + margin]

            axes[i].plot(lim, lim, "k--", lw=1, alpha=0.5)

            # Subsample
            n = len(ct)
            idx = np.random.choice(n, min(n, 5000), replace=False) if n > 5000 else np.arange(n)
            axes[i].scatter(ct[idx], cp[idx], alpha=0.4, s=10, c=color, edgecolors="none")
            axes[i].set_title(f"F_{comp}: RMSE={rmse:.4f}")
            axes[i].set_xlabel(f"True F_{comp}")
            axes[i].set_ylabel(f"Pred F_{comp}")
            axes[i].set_xlim(lim)
            axes[i].set_ylim(lim)
            axes[i].set_aspect("equal")

        name = METHOD_NAMES.get(method, method)
        fig.suptitle(f"{name} - Force Components ({subset})", fontsize=13, y=1.02)
        fig.tight_layout()
        fig.savefig(output_dir / f"force_components_{method}_{subset}.png")
        plt.close(fig)


def plot_calibration_comparison(method_data: dict, output_dir: Path, subset: str = "val"):
    """Plot uncertainty calibration curves for energy and forces."""
    # Energy calibration
    methods_with_uq = [m for m in method_data
                       if np.any(method_data[m]["energy_df"]["stds"].values > 0)]

    if len(methods_with_uq) > 0:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")

        for method in methods_with_uq:
            data = method_data[method]
            y_true = data["energy_df"]["true"].values
            y_pred = data["energy_df"]["preds"].values
            y_std = data["energy_df"]["stds"].values

            expected, observed = compute_calibration_curve(y_true, y_pred, y_std)
            name = METHOD_NAMES.get(method, method)
            color = METHOD_COLORS.get(method, None)
            ax.plot(expected, observed, "o-", markersize=3, label=name, color=color)

        ax.set_xlabel("Expected Coverage")
        ax.set_ylabel("Observed Coverage")
        ax.set_title(f"Energy Uncertainty Calibration ({subset})")
        ax.legend(fontsize=9)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(output_dir / f"energy_calibration_{subset}.png")
        plt.close(fig)

    # Force calibration
    methods_with_f_uq = [m for m in method_data
                         if method_data[m].get("forces") is not None
                         and np.any(method_data[m]["forces"]["std_forces"] > 0)]

    if len(methods_with_f_uq) > 0:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")

        for method in methods_with_f_uq:
            forces = method_data[method]["forces"]
            f_true = forces["true_forces"]
            f_pred = forces["pred_forces"]
            f_std = forces["std_forces"]

            expected, observed = compute_calibration_curve(f_true, f_pred, f_std)
            name = METHOD_NAMES.get(method, method)
            color = METHOD_COLORS.get(method, None)
            ax.plot(expected, observed, "o-", markersize=3, label=name, color=color)

        ax.set_xlabel("Expected Coverage")
        ax.set_ylabel("Observed Coverage")
        ax.set_title(f"Force Uncertainty Calibration ({subset})")
        ax.legend(fontsize=9)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(output_dir / f"force_calibration_{subset}.png")
        plt.close(fig)


def plot_error_vs_uncertainty(method_data: dict, output_dir: Path, subset: str = "val"):
    """Plot error vs predicted uncertainty scatter plots."""
    methods_with_uq = [m for m in method_data
                       if np.any(method_data[m]["energy_df"]["stds"].values > 0)]

    if len(methods_with_uq) == 0:
        return

    n = len(methods_with_uq)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, method in zip(axes, methods_with_uq):
        data = method_data[method]
        y_true = data["energy_df"]["true"].values
        y_pred = data["energy_df"]["preds"].values
        y_std = data["energy_df"]["stds"].values

        errors = np.abs(y_true - y_pred)
        corr, _ = stats.pearsonr(errors, y_std) if np.std(y_std) > 0 else (0, 1)

        ax.scatter(y_std, errors, alpha=0.5, s=15, c=METHOD_COLORS.get(method, "steelblue"), edgecolors="none")

        # Add 1:1 line
        max_val = max(y_std.max(), errors.max())
        ax.plot([0, max_val], [0, max_val], "k--", alpha=0.5)

        name = METHOD_NAMES.get(method, method)
        ax.set_title(f"{name}\nCorr: {corr:.3f}")
        ax.set_xlabel("Predicted Uncertainty")
        ax.set_ylabel("|Error|")

    fig.suptitle(f"Energy: Error vs Uncertainty ({subset})", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / f"energy_error_vs_uq_{subset}.png")
    plt.close(fig)

    # Force error vs uncertainty
    methods_with_f_uq = [m for m in method_data
                         if method_data[m].get("forces") is not None
                         and np.any(method_data[m]["forces"]["std_forces"] > 0)]

    if len(methods_with_f_uq) == 0:
        return

    n = len(methods_with_f_uq)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, method in zip(axes, methods_with_f_uq):
        forces = method_data[method]["forces"]
        f_true = forces["true_forces"]
        f_pred = forces["pred_forces"]
        f_std = forces["std_forces"]

        errors = np.abs(f_true - f_pred)
        # Subsample
        n_pts = len(errors)
        idx = np.random.choice(n_pts, min(n_pts, 5000), replace=False) if n_pts > 5000 else np.arange(n_pts)

        corr, _ = stats.pearsonr(errors, f_std) if np.std(f_std) > 0 else (0, 1)
        ax.scatter(f_std[idx], errors[idx], alpha=0.3, s=10, c=METHOD_COLORS.get(method, "coral"), edgecolors="none")

        max_val = max(f_std[idx].max(), errors[idx].max())
        ax.plot([0, max_val], [0, max_val], "k--", alpha=0.5)

        name = METHOD_NAMES.get(method, method)
        ax.set_title(f"{name}\nCorr: {corr:.3f}")
        ax.set_xlabel("Predicted Uncertainty")
        ax.set_ylabel("|Error|")

    fig.suptitle(f"Forces: Error vs Uncertainty ({subset})", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / f"force_error_vs_uq_{subset}.png")
    plt.close(fig)


def plot_method_comparison_bars(summary_df: pd.DataFrame, output_dir: Path, subset: str = "val"):
    """Plot bar charts comparing methods on key metrics."""
    metrics_to_plot = [
        ("energy_rmse", "Energy RMSE", True),
        ("force_rmse", "Force RMSE", True),
        ("total_rmse", "Total RMSE", True),
        ("energy_nll", "Energy NLL", True),
        ("energy_ece", "Energy ECE", True),
        ("force_nll", "Force NLL", True),
    ]

    available = [(m, label, lower) for m, label, lower in metrics_to_plot if m in summary_df.columns]

    n_plots = len(available)
    if n_plots == 0:
        return

    ncols = min(3, n_plots)
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    for i, (metric, label, lower_better) in enumerate(available):
        row, col = divmod(i, ncols)
        ax = axes[row, col]

        methods = summary_df["method"].values
        values = summary_df[metric].values
        colors = [METHOD_COLORS.get(m, "#888888") for m in methods]

        bars = ax.bar(range(len(methods)), values, color=colors, alpha=0.8, edgecolor="white")

        # Highlight best
        if lower_better:
            best_idx = np.nanargmin(values)
        else:
            best_idx = np.nanargmax(values)
        bars[best_idx].set_edgecolor("black")
        bars[best_idx].set_linewidth(2)

        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([METHOD_NAMES.get(m, m) for m in methods], rotation=45, ha="right", fontsize=9)
        ax.set_title(label)
        ax.set_ylabel(label)

    # Hide unused axes
    for i in range(n_plots, nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle(f"Method Comparison ({subset})", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / f"method_comparison_{subset}.png")
    plt.close(fig)


# ============================================================================
# Main Analysis
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze TiO2 force model predictions")
    parser.add_argument("--pred-dir", type=str, default="bnn_aenet/logs/forces_pred",
                       help="Directory with prediction outputs")
    parser.add_argument("--output-dir", type=str, default="plots/TiO2_forces",
                       help="Output directory for plots and tables")
    parser.add_argument("--subsets", type=str, nargs="+", default=["train", "val", "test"],
                       help="Data subsets to analyze")
    parser.add_argument("--alpha", type=float, default=0.1,
                       help="Alpha for total_rmse = (1-alpha)*E_rmse + alpha*F_rmse")
    parser.add_argument("--n-sub-ensemble", type=int, default=5,
                       help="Number of models per sub-ensemble")
    parser.add_argument("--max-sub-ensembles", type=int, default=20,
                       help="Maximum number of sub-ensembles")

    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TiO2 Force Model Analysis")
    print("=" * 80)
    print(f"Predictions: {pred_dir}")
    print(f"Output: {output_dir}")
    print()

    for subset in args.subsets:
        print(f"\n{'='*80}")
        print(f"ANALYZING SUBSET: {subset}")
        print(f"{'='*80}\n")

        # Load all predictions
        print("Loading predictions...")
        nn_runs = load_run_predictions(pred_dir, "nn", subset)
        lrt_runs = load_run_predictions(pred_dir, "lrt", subset)
        fo_runs = load_run_predictions(pred_dir, "fo", subset)
        rad_runs = load_run_predictions(pred_dir, "rad", subset)

        print(f"  NN: {len(nn_runs)} runs")
        print(f"  LRT: {len(lrt_runs)} runs")
        print(f"  FO: {len(fo_runs)} runs")
        print(f"  RAD: {len(rad_runs)} runs")

        if len(nn_runs) == 0 and len(lrt_runs) == 0:
            print("  No predictions found, skipping this subset.")
            continue

        # ============================================
        # Create Deep Ensemble
        # ============================================
        print("\nCreating Deep Ensembles...")
        de_full = None
        de_subs = []
        if len(nn_runs) >= 2:
            de_full = create_deep_ensemble(nn_runs)
            print(f"  Full DE: {de_full['n_models']} models")

            de_subs = create_sub_ensembles(nn_runs, args.n_sub_ensemble, args.max_sub_ensembles)
            print(f"  Sub-ensembles: {len(de_subs)} ({args.n_sub_ensemble}-model ensembles)")
        else:
            print("  Not enough NN runs for DE")

        # ============================================
        # Select Best BNN Models
        # ============================================
        print("\nSelecting best BNN models...")
        bnn_selections = {}
        for name, runs in [("lrt", lrt_runs), ("fo", fo_runs), ("rad", rad_runs)]:
            if len(runs) == 0:
                continue
            sel = select_best_bnn(runs, args.alpha)
            bnn_selections[name] = sel

            print(f"\n  {name.upper()} ({len(runs)} runs):")
            print(f"    Best overall: {sel['best_overall']['metrics']['run_name']}")
            print(f"      E_RMSE={sel['best_overall']['metrics']['energy_rmse']:.4f}, "
                  f"F_RMSE={sel['best_overall']['metrics'].get('force_rmse', 'N/A')}, "
                  f"Total={sel['best_overall']['metrics']['total_rmse']:.4f}")

            print(f"    Best energy:  {sel['best_energy']['metrics']['run_name']}")
            print(f"      E_RMSE={sel['best_energy']['metrics']['energy_rmse']:.4f}")

            if "best_forces" in sel:
                print(f"    Best forces:  {sel['best_forces']['metrics']['run_name']}")
                print(f"      F_RMSE={sel['best_forces']['metrics']['force_rmse']:.4f}")

            # Save per-run metrics
            sel["all_metrics"].to_csv(
                output_dir / f"{name}_all_runs_{subset}.csv", index=False
            )

        # ============================================
        # Build method_data for comparison plots
        # (Use best_overall for each BNN)
        # ============================================
        method_data = {}

        if de_full is not None:
            method_data["DE"] = de_full

        for name in ["lrt", "fo", "rad"]:
            if name in bnn_selections:
                method_data[name] = bnn_selections[name]["best_overall"]["run"]

        # ============================================
        # Compute Summary Metrics Table
        # ============================================
        print("\nComputing summary metrics...")
        summary_rows = []

        for method, data in method_data.items():
            m = compute_run_metrics(data, args.alpha)
            m["method"] = method
            summary_rows.append(m)

        # Add sub-ensemble stats if available
        if len(de_subs) > 0:
            sub_metrics = [compute_run_metrics(s, args.alpha) for s in de_subs]
            sub_df = pd.DataFrame(sub_metrics)
            mean_row = {"method": "DE_sub"}
            for col in sub_df.select_dtypes(include=[np.number]).columns:
                mean_row[col] = sub_df[col].mean()
                mean_row[f"{col}_std"] = sub_df[col].std()
            summary_rows.append(mean_row)

        # Add individual NN stats
        if len(nn_runs) > 0:
            nn_metrics = [compute_run_metrics(r, args.alpha) for r in nn_runs]
            nn_df = pd.DataFrame(nn_metrics)
            nn_row = {"method": "nn"}
            for col in nn_df.select_dtypes(include=[np.number]).columns:
                nn_row[col] = nn_df[col].mean()
                nn_row[f"{col}_std"] = nn_df[col].std()
            summary_rows.append(nn_row)

        summary_df = pd.DataFrame(summary_rows)

        # Print summary
        print("\n" + "=" * 100)
        print(f"SUMMARY TABLE ({subset})")
        print("=" * 100)

        key_cols = ["method", "energy_rmse", "energy_mae", "force_rmse", "force_mae",
                    "total_rmse", "energy_nll", "force_nll", "energy_ece"]
        avail_cols = [c for c in key_cols if c in summary_df.columns]
        print(summary_df[avail_cols].to_string(index=False, float_format="%.4f"))

        # Save
        summary_df.to_csv(output_dir / f"summary_{subset}.csv", index=False)
        print(f"\nSaved: {output_dir / f'summary_{subset}.csv'}")

        # LaTeX table
        latex_df = summary_df[avail_cols].copy()
        latex_str = latex_df.to_latex(index=False, float_format="%.4f",
                                      caption=f"TiO2 Force Model Comparison ({subset})")
        with open(output_dir / f"summary_{subset}.tex", "w") as f:
            f.write(latex_str)
        print(f"Saved: {output_dir / f'summary_{subset}.tex'}")

        # ============================================
        # Generate Plots
        # ============================================
        print("\nGenerating plots...")

        plot_energy_parity_comparison(method_data, output_dir, subset)
        print(f"  Saved energy_parity_{subset}.png")

        plot_force_parity_comparison(method_data, output_dir, subset)
        print(f"  Saved force_parity_{subset}.png")

        plot_force_components_comparison(method_data, output_dir, subset)
        print(f"  Saved force_components_*_{subset}.png")

        plot_calibration_comparison(method_data, output_dir, subset)
        print(f"  Saved calibration_{subset}.png")

        plot_error_vs_uncertainty(method_data, output_dir, subset)
        print(f"  Saved error_vs_uq_{subset}.png")

        plot_method_comparison_bars(summary_df, output_dir, subset)
        print(f"  Saved method_comparison_{subset}.png")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print(f"All outputs saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
