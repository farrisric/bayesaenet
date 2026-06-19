"""Publication-ready analysis of force-trained models (test set only).

Produces a small set of condensed, high-quality figures suitable for
journal publication, plus a LaTeX summary table.

Usage:
    python -m bnn_aenet.tasks.analyze_publication \
        --pred-dir bnn_aenet/logs/TiO2_big/pred \
        --output-dir plots/TiO2_big/publication \
        --train-dir bnn_aenet/logs/TiO2_big/train \
        --time-dirs bnn_aenet/logs/nn_train bnn_aenet/logs/lrt_train \
                    bnn_aenet/logs/rad_train
"""

import argparse
import importlib.util
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

try:
    from bnn_aenet.utils.metrics import (
        compute_calibration_curve,
        compute_energy_metrics,
        compute_force_metrics,
        compute_uncertainty_metrics,
    )
except (ModuleNotFoundError, ImportError):
    # Fallback when utils.metrics is unavailable or does not expose needed symbols.
    # Load analysis/metrics.py directly to avoid package __init__ side effects.
    _metrics_path = Path(__file__).resolve().parents[1] / "analysis" / "metrics.py"
    _metrics_spec = importlib.util.spec_from_file_location(
        "analysis_metrics_fallback", _metrics_path
    )
    _metrics_mod = importlib.util.module_from_spec(_metrics_spec)
    _metrics_spec.loader.exec_module(_metrics_mod)
    compute_calibration_curve = _metrics_mod.compute_calibration_curve
    compute_energy_metrics = _metrics_mod.compute_energy_metrics
    compute_force_metrics = _metrics_mod.compute_force_metrics
    compute_uncertainty_metrics = _metrics_mod.compute_uncertainty_metrics

# Reuse data-loading helpers from analysis modules.
# Prefer package module when available; fallback to script module in this repo.
try:
    from bnn_aenet.tasks.analyze import (
        compute_run_metrics,
        create_deep_ensemble,
        create_sub_ensembles,
        load_run_predictions,
        load_tensorboard_scalars,
        select_best_bnn,
    )
except ModuleNotFoundError:
    _analysis_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "TiO2_big"
        / "analysis"
        / "analyze_forces.py"
    )
    _spec = importlib.util.spec_from_file_location("analyze_forces_fallback", _analysis_path)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    create_deep_ensemble = _mod.create_deep_ensemble
    create_sub_ensembles = _mod.create_sub_ensembles
    compute_run_metrics = _mod.compute_run_metrics
    load_run_predictions = _mod.load_run_predictions
    load_tensorboard_scalars = _mod.load_tensorboard_scalars
    select_best_bnn = _mod.select_best_bnn

# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------
SINGLE_COL = 3.5  # inches  (single-column figure)
DOUBLE_COL = 7.0  # inches  (double-column / full-width figure)
DPI = 300

plt.rcParams.update(
    {
        # Fonts
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "cm",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        # Figure
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        # Axes
        "axes.linewidth": 0.6,
        "axes.grid": False,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        # Lines
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
    }
)

# Method display names and colors (consistent across all figures)
METHOD_ORDER = ["DE", "lrt", "rad", "lrt_hetero", "rad_hetero"]
METHOD_NAMES = {
    "DE": "Deep Ens.",
    "DE_sub": "DE (5-model)",
    "lrt": "LRT",
    "rad": "Radial",
    "lrt_hetero": "LRT Het.",
    "rad_hetero": "Radial Het.",
}
METHOD_COLORS = {
    "DE": "#2196F3",
    "DE_sub": "#90CAF9",
    "lrt": "#FF9800",
    "rad": "#F44336",
    "lrt_hetero": "#7E57C2",
    "rad_hetero": "#009688",
}
METHOD_MARKERS = {
    "DE": "o",
    "lrt": "^",
    "rad": "v",
    "lrt_hetero": "D",
    "rad_hetero": "P",
}

# For training time chart (maps to exec_time.log directories)
TIME_METHOD_MAP = {
    "nn_train": "DE",
    "lrt_train": "lrt",
    "rad_train": "rad",
}

# Dataset-level defaults when scaling metadata is not passed explicitly.
DATASET_SCALING_DEFAULTS = {
    "tio2": (0.06565926932648217, 6.6588702845000975),
    "qm7": (0.9754923797786934, -4.652443333333333),
}


def _save(fig, path, close=True):
    """Save figure as both PNG and PDF."""
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    if close:
        plt.close(fig)


def _get_default_scaling_from_path(pred_dir: Path):
    """Infer (e_scaling, e_shift) from prediction path name."""
    p = str(pred_dir).lower()
    if "tio2" in p:
        return DATASET_SCALING_DEFAULTS["tio2"]
    if "qm7" in p:
        return DATASET_SCALING_DEFAULTS["qm7"]
    return None, None


def _denormalize_run_inplace(run: dict, e_scaling: float, e_shift: float = 0.0):
    """De-normalize run predictions in-place before metrics/plots.

    Energy (per-atom) follows:
      y = (y_norm / e_scaling + n_atoms * e_shift) / n_atoms

    Forces use:
      f = f_norm / e_scaling
    """
    if run.get("_denormalized", False):
        return
    if e_scaling is None or e_scaling == 0:
        return

    df = run.get("energy_df")
    if df is not None and {"true", "preds", "n_atoms"}.issubset(df.columns):
        n_atoms = df["n_atoms"].to_numpy().astype(float)
        y_true = (df["true"].to_numpy() / e_scaling + n_atoms * e_shift) / n_atoms
        y_pred = (df["preds"].to_numpy() / e_scaling + n_atoms * e_shift) / n_atoms
        df["true"] = y_true
        df["preds"] = y_pred
        if "stds" in df.columns:
            # Shift does not affect uncertainty; divide by both scale and n_atoms.
            df["stds"] = df["stds"].to_numpy() / (e_scaling * n_atoms)

    if run.get("forces") is not None:
        run["forces"]["true_forces"] = run["forces"]["true_forces"] / e_scaling
        run["forces"]["pred_forces"] = run["forces"]["pred_forces"] / e_scaling
        run["forces"]["std_forces"] = run["forces"]["std_forces"] / e_scaling

    run["e_scaling"] = e_scaling
    run["e_shift"] = e_shift
    run["_denormalized"] = True


def _load_run_e_scaling_from_npz(pred_dir: Path, model_type: str, run_name: str, subset: str):
    """Load e_scaling stored by prediction task in force npz, if available."""
    npz_path = pred_dir / model_type / f"{run_name}_{subset}_forces.npz"
    if not npz_path.exists():
        return None
    try:
        with np.load(npz_path) as data:
            if "e_scaling" in data:
                val = np.asarray(data["e_scaling"]).reshape(-1)
                if val.size > 0:
                    return float(val[0])
    except Exception:
        return None
    return None


# ============================================================================
# Figure 1: Energy Parity
# ============================================================================


def plot_energy_parity(method_data: dict, output_dir: Path):
    """Energy parity plots -- one panel per method, shared axes."""
    methods = [m for m in METHOD_ORDER if m in method_data]
    n = len(methods)
    nrows = 2 if n > 1 else 1
    ncols = int(np.ceil(n / nrows))
    fig, axes = plt.subplots(nrows, ncols, figsize=(DOUBLE_COL, DOUBLE_COL * 0.65), squeeze=False)
    axes = axes.flatten()

    # Determine global axis range
    all_true, all_pred = [], []
    for m in methods:
        df = method_data[m]["energy_df"]
        all_true.extend(df["true"].values)
        all_pred.extend(df["preds"].values)
    lo = min(min(all_true), min(all_pred))
    hi = max(max(all_true), max(all_pred))
    margin = (hi - lo) * 0.03
    lo, hi = lo - margin, hi + margin

    for i, method in enumerate(methods):
        ax = axes[i]
        df = method_data[method]["energy_df"]
        y_true = df["true"].values
        y_pred = df["preds"].values

        color = METHOD_COLORS[method]
        ax.scatter(y_true, y_pred, s=6, alpha=0.5, color=color, edgecolors="none", rasterized=True)
        ax.plot([lo, hi], [lo, hi], "k-", linewidth=0.6, alpha=0.6)

        # Metrics annotation
        em = compute_energy_metrics(y_true, y_pred)
        rmse_str = f"RMSE = {em['rmse']:.3f}"
        r2_str = f"$R^2$ = {em['r2']:.6f}"
        ax.text(
            0.05,
            0.95,
            f"{rmse_str}\n{r2_str}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.85),
        )

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_title(METHOD_NAMES.get(method, method))
        if i % ncols == 0:
            ax.set_ylabel("Predicted energy (meV/atom)")
        ax.set_xlabel("True energy (meV/atom)")

    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout(h_pad=1.0, w_pad=0.8)
    _save(fig, output_dir / "energy_parity.png")


# ============================================================================
# Figure 2: Force Parity
# ============================================================================


def plot_force_parity(method_data: dict, output_dir: Path, max_points: int = 8000):
    """Force parity plots -- one panel per method, density-colored."""
    methods = [
        m for m in METHOD_ORDER if m in method_data and method_data[m].get("forces") is not None
    ]
    n = len(methods)
    if n == 0:
        return

    nrows = 2 if n > 1 else 1
    ncols = int(np.ceil(n / nrows))
    fig, axes = plt.subplots(nrows, ncols, figsize=(DOUBLE_COL, DOUBLE_COL * 0.65), squeeze=False)
    axes = axes.flatten()

    # Global axis range
    all_f = []
    for m in methods:
        f = method_data[m]["forces"]
        all_f.extend(f["true_forces"].flatten()[:max_points])
        all_f.extend(f["pred_forces"].flatten()[:max_points])
    lo = min(all_f)
    hi = max(all_f)
    margin = (hi - lo) * 0.03
    lo, hi = lo - margin, hi + margin

    for i, method in enumerate(methods):
        ax = axes[i]
        f = method_data[method]["forces"]
        ft = f["true_forces"].flatten()
        fp = f["pred_forces"].flatten()

        # Subsample
        if len(ft) > max_points:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(ft), max_points, replace=False)
            ft, fp = ft[idx], fp[idx]

        # 2D histogram for density coloring
        from matplotlib.colors import LogNorm

        h = ax.hist2d(
            ft,
            fp,
            bins=100,
            range=[[lo, hi], [lo, hi]],
            cmap="viridis",
            norm=LogNorm(),
            rasterized=True,
        )
        ax.plot([lo, hi], [lo, hi], "w-", linewidth=0.6, alpha=0.8)

        # Metrics
        fm = compute_force_metrics(ft, fp)
        rmse_str = f"RMSE = {fm['rmse']:.3f}"
        r2_str = f"$R^2$ = {fm['r2']:.4f}"
        ax.text(
            0.05,
            0.95,
            f"{rmse_str}\n{r2_str}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=7,
            color="white",
            bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="0.3", alpha=0.6),
        )

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_title(METHOD_NAMES.get(method, method))
        if i % ncols == 0:
            ax.set_ylabel(r"Predicted force (eV/$\mathrm{\AA}$)")
        ax.set_xlabel(r"True force (eV/$\mathrm{\AA}$)")

    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout(h_pad=1.0, w_pad=0.8)
    _save(fig, output_dir / "force_parity.png")


# ============================================================================
# Figure 3: Calibration
# ============================================================================


def plot_calibration(method_data: dict, output_dir: Path):
    """Combined calibration plot: energy (left) + forces (right), all methods
    overlaid on the same axes."""
    methods_e = [
        m
        for m in METHOD_ORDER
        if m in method_data and method_data[m]["energy_df"]["stds"].values.std() > 0
    ]
    methods_f = [
        m
        for m in METHOD_ORDER
        if m in method_data
        and method_data[m].get("forces") is not None
        and method_data[m]["forces"]["std_forces"].std() > 0
    ]

    if not methods_e and not methods_f:
        return

    n_panels = (1 if methods_e else 0) + (1 if methods_f else 0)
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(SINGLE_COL * 1.05, SINGLE_COL * n_panels),
        squeeze=False,
    )
    axes = axes[:, 0]
    ax_idx = 0

    if methods_e:
        ax = axes[ax_idx]
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.6, alpha=0.5, label="Ideal")
        for method in methods_e:
            df = method_data[method]["energy_df"]
            exp_f, obs_f = compute_calibration_curve(
                df["true"].values, df["preds"].values, df["stds"].values
            )
            ax.plot(
                exp_f,
                obs_f,
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS.get(method, "o"),
                markersize=3,
                label=METHOD_NAMES.get(method, method),
            )
        ax.set_xlabel("Expected confidence")
        ax.set_ylabel("Observed coverage")
        ax.set_title("Energy calibration")
        ax.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="0.8", framealpha=0.9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax_idx += 1

    if methods_f:
        ax = axes[ax_idx]
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.6, alpha=0.5, label="Ideal")
        for method in methods_f:
            f = method_data[method]["forces"]
            exp_f, obs_f = compute_calibration_curve(
                f["true_forces"], f["pred_forces"], f["std_forces"]
            )
            ax.plot(
                exp_f,
                obs_f,
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS.get(method, "o"),
                markersize=3,
                label=METHOD_NAMES.get(method, method),
            )
        ax.set_xlabel("Expected confidence")
        ax.set_ylabel("Observed coverage")
        ax.set_title("Force calibration")
        ax.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="0.8", framealpha=0.9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")

    fig.tight_layout(h_pad=1.0)
    _save(fig, output_dir / "calibration.png")


# ============================================================================
# Figure 4: Binned Error vs Uncertainty
# ============================================================================


def _binned_error_uq(y_true, y_pred, y_std, n_bins=10):
    """Compute binned mean |error| vs uncertainty."""
    errors = np.abs(y_true.flatten() - y_pred.flatten())
    stds = y_std.flatten()

    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(stds, quantiles)
    bx, by, be = [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (stds >= lo) & (stds < hi) if i < n_bins - 1 else (stds >= lo) & (stds <= hi)
        if mask.sum() > 0:
            bx.append(stds[mask].mean())
            by.append(errors[mask].mean())
            be.append(errors[mask].std() / np.sqrt(mask.sum()))
    return np.array(bx), np.array(by), np.array(be)


def _compute_rmsce(y_true, y_pred, y_std):
    """Compute RMS calibration error from expected vs observed coverage."""
    exp_f, obs_f = compute_calibration_curve(y_true, y_pred, y_std)
    if len(exp_f) == 0:
        return np.nan
    return float(np.sqrt(np.mean((obs_f - exp_f) ** 2)))


def _corr_text(x, y):
    """Return formatted Spearman/Pearson correlation text."""
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return r"$\rho$=n/a" + "\n" + r"$r$=n/a"

    if np.std(x) > 0 and np.std(y) > 0:
        rho, _ = stats.spearmanr(x, y)
        r = np.corrcoef(x, y)[0, 1]
        rho_txt = "n/a" if not np.isfinite(rho) else f"{rho:.2f}"
        r_txt = "n/a" if not np.isfinite(r) else f"{r:.2f}"
    else:
        rho_txt, r_txt = "n/a", "n/a"

    return rf"$\rho$={rho_txt}" + "\n" + rf"$r$={r_txt}"


def _corr_text_energy_normalized(run: dict):
    """Correlation text in normalized space for energy."""
    df = run["energy_df"]
    e_scaling = run.get("e_scaling")
    e_shift = run.get("e_shift", 0.0)
    if e_scaling is None or e_scaling == 0:
        x = np.abs(df["stds"].values)
        y = np.abs(df["true"].values - df["preds"].values)
        return _corr_text(x, y)

    n_atoms = df["n_atoms"].to_numpy().astype(float)
    y_true_n = (df["true"].to_numpy() - e_shift) * n_atoms * e_scaling
    y_pred_n = (df["preds"].to_numpy() - e_shift) * n_atoms * e_scaling
    y_std_n = df["stds"].to_numpy() * n_atoms * e_scaling
    return _corr_text(np.abs(y_std_n), np.abs(y_true_n - y_pred_n))


def _corr_text_force_normalized(run: dict):
    """Correlation text in normalized space for forces."""
    f = run["forces"]
    e_scaling = run.get("e_scaling")
    if e_scaling is None or e_scaling == 0:
        x = np.abs(f["std_forces"]).flatten()
        y = np.abs(f["true_forces"].flatten() - f["pred_forces"].flatten())
        return _corr_text(x, y)

    x = np.abs(f["std_forces"] * e_scaling).flatten()
    y = np.abs((f["true_forces"] - f["pred_forces"]) * e_scaling).flatten()
    return _corr_text(x, y)


def plot_error_vs_uncertainty(
    method_data: dict,
    output_dir: Path,
    corr_space: str = "denorm",
    output_name: str = "error_vs_uncertainty.png",
):
    """Error vs uncertainty with one panel per model (force_parity style).

    Uses density-colored 2D histograms (LogNorm).
    Layout is 5 columns (one per method in METHOD_ORDER) and up to 2 rows:
    - row 1: energy |error| vs std
    - row 2: force  |error| vs std (if forces are available)
    """
    methods_e = [
        m
        for m in METHOD_ORDER
        if m in method_data and method_data[m]["energy_df"]["stds"].values.std() > 0
    ]
    methods_f = [
        m
        for m in METHOD_ORDER
        if m in method_data
        and method_data[m].get("forces") is not None
        and method_data[m]["forces"]["std_forces"].std() > 0
    ]

    if not methods_e and not methods_f:
        return

    methods = [m for m in METHOD_ORDER if m in method_data]
    ncols = len(methods)
    nrows = (1 if methods_e else 0) + (1 if methods_f else 0)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(DOUBLE_COL * 1.8, 2.6 * nrows),
        squeeze=False,
    )

    from matplotlib.colors import LogNorm

    row_idx = 0

    if methods_e:
        for col_idx, method in enumerate(methods):
            ax = axes[row_idx, col_idx]
            if method not in methods_e:
                ax.axis("off")
                continue
            df = method_data[method]["energy_df"]
            if corr_space == "normalized":
                e_scaling = method_data[method].get("e_scaling")
                e_shift = method_data[method].get("e_shift", 0.0)
                if e_scaling is not None and e_scaling != 0:
                    n_atoms = df["n_atoms"].to_numpy().astype(float)
                    y_true_n = (df["true"].to_numpy() - e_shift) * n_atoms * e_scaling
                    y_pred_n = (df["preds"].to_numpy() - e_shift) * n_atoms * e_scaling
                    y_std_n = df["stds"].to_numpy() * n_atoms * e_scaling
                    x = np.abs(y_std_n)
                    y = np.abs(y_true_n - y_pred_n)
                else:
                    x = np.abs(df["stds"].values)
                    y = np.abs(df["true"].values - df["preds"].values)
            else:
                x = np.abs(df["stds"].values)
                y = np.abs(df["true"].values - df["preds"].values)

            # Keep positive bins for LogNorm and stable limits.
            x = np.clip(x, 1e-12, None)
            y = np.clip(y, 1e-12, None)
            x_hi = np.percentile(x, 99.5)
            y_hi = np.percentile(y, 99.5)

            ax.hist2d(
                x,
                y,
                bins=90,
                range=[[0, x_hi], [0, y_hi]],
                cmap="viridis",
                norm=LogNorm(),
                rasterized=True,
            )

            lim = max(x_hi, y_hi)
            ax.plot([0, lim], [0, lim], "w--", linewidth=0.7, alpha=0.8)
            corr_label = _corr_text(x, y)
            ax.text(
                0.04,
                0.96,
                corr_label,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=7,
                color="white",
                bbox=dict(boxstyle="round,pad=0.25", fc="black", ec="0.3", alpha=0.65),
            )

            ax.set_title(METHOD_NAMES.get(method, method))
            if corr_space == "normalized":
                ax.set_xlabel("Predicted uncertainty (normalized)")
            else:
                ax.set_xlabel("Predicted uncertainty (eV/atom)")
            if col_idx == 0:
                if corr_space == "normalized":
                    ax.set_ylabel("Absolute error (normalized)")
                else:
                    ax.set_ylabel("Absolute error (eV/atom)")
        row_idx += 1

    if methods_f:
        for col_idx, method in enumerate(methods):
            ax = axes[row_idx, col_idx]
            if method not in methods_f:
                ax.axis("off")
                continue
            f = method_data[method]["forces"]
            if corr_space == "normalized":
                e_scaling = method_data[method].get("e_scaling")
                if e_scaling is not None and e_scaling != 0:
                    x = np.abs(f["std_forces"] * e_scaling).flatten()
                    y = np.abs((f["true_forces"] - f["pred_forces"]) * e_scaling).flatten()
                else:
                    x = np.abs(f["std_forces"]).flatten()
                    y = np.abs(f["true_forces"].flatten() - f["pred_forces"].flatten())
            else:
                x = np.abs(f["std_forces"]).flatten()
                y = np.abs(f["true_forces"].flatten() - f["pred_forces"].flatten())

            x = np.clip(x, 1e-12, None)
            y = np.clip(y, 1e-12, None)
            x_hi = np.percentile(x, 99.5)
            y_hi = np.percentile(y, 99.5)

            ax.hist2d(
                x,
                y,
                bins=90,
                range=[[0, x_hi], [0, y_hi]],
                cmap="viridis",
                norm=LogNorm(),
                rasterized=True,
            )

            lim = max(x_hi, y_hi)
            ax.plot([0, lim], [0, lim], "w--", linewidth=0.7, alpha=0.8)
            corr_label = _corr_text(x, y)
            ax.text(
                0.04,
                0.96,
                corr_label,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=7,
                color="white",
                bbox=dict(boxstyle="round,pad=0.25", fc="black", ec="0.3", alpha=0.65),
            )

            ax.set_title(METHOD_NAMES.get(method, method))
            if corr_space == "normalized":
                ax.set_xlabel("Predicted uncertainty (normalized)")
            else:
                ax.set_xlabel(r"Predicted uncertainty (eV/$\mathrm{\AA}$)")
            if col_idx == 0:
                if corr_space == "normalized":
                    ax.set_ylabel("Absolute error (normalized)")
                else:
                    ax.set_ylabel(r"Absolute error (eV/$\mathrm{\AA}$)")

    fig.tight_layout(h_pad=1.0, w_pad=0.8)
    _save(fig, output_dir / output_name)


# ============================================================================
# Figure 5: Summary Bar Chart
# ============================================================================


def plot_metric_bars(summary_df: pd.DataFrame, output_dir: Path):
    """Grouped bar chart for key metrics across methods."""
    metric_defs = [
        ("energy_rmse", "Energy RMSE\n(meV/atom)"),
        ("force_rmse", r"Force RMSE" + "\n" + r"(meV/$\mathrm{\AA}$)"),
        ("energy_nll", "Energy NLL"),
        ("energy_sharpness", "Energy sharpness"),
        ("energy_rmsce", "Energy RMSCE"),
        ("force_ece", "Force ECE"),
        ("force_sharpness", "Force sharpness"),
        ("force_rmsce", "Force RMSCE"),
    ]
    # Filter to metrics that actually exist
    metric_defs = [(k, lab) for k, lab in metric_defs if k in summary_df.columns]
    n_metrics = len(metric_defs)
    if n_metrics == 0:
        return

    ncols = 2
    nrows = (n_metrics + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(DOUBLE_COL, 1.8 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    ordered_methods = [m for m in METHOD_ORDER if m in summary_df["method"].values]
    x = np.arange(len(ordered_methods))
    width = 0.6

    for i, (metric_key, ylabel) in enumerate(metric_defs):
        ax = axes_flat[i]
        vals, errs, colors = [], [], []
        for m in ordered_methods:
            row = summary_df[summary_df["method"] == m].iloc[0]
            v = row.get(metric_key, np.nan)
            std_key = f"{metric_key}_std"
            e = row.get(std_key, 0) if std_key in row.index else 0
            vals.append(v)
            errs.append(e if pd.notna(e) else 0)
            colors.append(METHOD_COLORS.get(m, "#888888"))

        bars = ax.bar(
            x,
            vals,
            width,
            yerr=errs,
            capsize=3,
            color=colors,
            edgecolor="0.3",
            linewidth=0.5,
            error_kw=dict(linewidth=0.8),
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [METHOD_NAMES.get(m, m) for m in ordered_methods], rotation=25, ha="right", fontsize=7
        )
        ax.set_ylabel(ylabel)
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    # Hide unused axes
    for j in range(n_metrics, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout(h_pad=1.5, w_pad=1.2)
    _save(fig, output_dir / "metric_bars.png")


# ============================================================================
# Figure 6: Training Time
# ============================================================================


def _load_exec_times(time_dirs: list) -> dict:
    """Load execution times from exec_time.log files.

    Returns: {method_name: [time_seconds, ...]}
    """
    result = {}
    for d in time_dirs:
        d = Path(d)
        method_dir_name = d.name  # e.g. nn_forces
        method = TIME_METHOD_MAP.get(method_dir_name, method_dir_name)

        times = []
        # Walk subdirectories looking for exec_time.log
        for log in sorted(d.rglob("exec_time.log")):
            text = log.read_text().strip()
            # Parse: 'name' execution time: 1234.56 (s)
            match = re.search(r"execution time:\s*([\d.]+)", text)
            if match:
                times.append(float(match.group(1)))
        if times:
            result[method] = times
    return result


def plot_training_time(time_dirs: list, output_dir: Path):
    """Bar chart of training wall-clock time per method."""
    times = _load_exec_times(time_dirs)
    if not times:
        print("  No training time data found, skipping.")
        return

    ordered = [m for m in ["DE", "lrt", "rad"] if m in times]
    if not ordered:
        return

    means = [np.mean(times[m]) / 60 for m in ordered]  # minutes
    stds = [np.std(times[m]) / 60 for m in ordered]
    n_runs = [len(times[m]) for m in ordered]
    total_hrs = [np.sum(times[m]) / 3600 for m in ordered]
    colors = [METHOD_COLORS.get(m, "#888") for m in ordered]

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.5))
    x = np.arange(len(ordered))
    bars = ax.bar(
        x,
        means,
        0.6,
        yerr=stds,
        capsize=3,
        color=colors,
        edgecolor="0.3",
        linewidth=0.5,
        error_kw=dict(linewidth=0.8),
    )

    # Annotate total GPU-hours on each bar
    for i, (bar, hrs, nr) in enumerate(zip(bars, total_hrs, n_runs)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + stds[i] + 1,
            f"{hrs:.1f} h\n({nr} runs)",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_NAMES.get(m, m) for m in ordered], fontsize=8)
    ax.set_ylabel("Training time per run (min)")
    ax.set_title("Training Wall-Clock Time")
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    fig.tight_layout()
    _save(fig, output_dir / "training_time.png")

    return times  # Return for table use


# ============================================================================
# Figure 7: Training Curves (condensed)
# ============================================================================


def plot_training_curves(train_dir: Path, output_dir: Path):
    """Condensed training curves: energy RMSE + force RMSE (validation),
    all methods on the same axes."""
    model_types = ["lrt", "rad", "lrt_hetero", "rad_hetero"]
    available = []
    for mt in model_types:
        mdir = train_dir / mt
        if mdir.exists() and any(mdir.iterdir()):
            available.append(mt)

    if not available:
        print("  No training logs found, skipping training curves.")
        return

    tags = ["rmse/val", "force_rmse/val"]

    # Load all data
    all_data = {}
    for mt in available:
        mdir = train_dir / mt
        runs = sorted([d for d in mdir.iterdir() if d.is_dir()])
        model_data = {}
        for run_dir in runs:
            rd = load_tensorboard_scalars(run_dir, tags)
            if rd:
                model_data[run_dir.name] = rd
        if model_data:
            all_data[mt] = model_data

    if not all_data:
        return

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.8))

    for tag_idx, (tag, ylabel) in enumerate(
        [
            ("rmse/val", "Energy RMSE (meV/atom)"),
            ("force_rmse/val", r"Force RMSE (meV/$\mathrm{\AA}$)"),
        ]
    ):
        ax = axes[tag_idx]
        for mt in available:
            if mt not in all_data:
                continue
            model_data = all_data[mt]
            color = METHOD_COLORS.get(mt, "#888")
            name = METHOD_NAMES.get(mt, mt)

            all_vals = []
            for run_name, rd in model_data.items():
                if tag in rd:
                    all_vals.append(rd[tag]["values"])

            if all_vals:
                min_len = min(len(v) for v in all_vals)
                truncated = np.array([v[:min_len] for v in all_vals])
                mean_vals = truncated.mean(axis=0)
                std_vals = truncated.std(axis=0)
                steps = list(model_data.values())[0][tag]["steps"][:min_len]
                ax.plot(steps, mean_vals, color=color, linewidth=1.2, label=name)
                ax.fill_between(
                    steps, mean_vals - std_vals, mean_vals + std_vals, color=color, alpha=0.12
                )

        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(
            loc="upper right",
            frameon=True,
            fancybox=False,
            edgecolor="0.8",
            framealpha=0.9,
            fontsize=7,
        )
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())

    fig.tight_layout(w_pad=1.5)
    _save(fig, output_dir / "training_curves.png")


# ============================================================================
# LaTeX Table
# ============================================================================


def generate_latex_table(summary_df: pd.DataFrame, time_data: dict, output_dir: Path):
    """Publication-quality LaTeX table with bold best values."""
    # Columns to include (with display names and formatting)
    col_defs = [
        ("energy_rmse", "E RMSE", ".3f", "min"),
        ("energy_mae", "E MAE", ".3f", "min"),
        ("force_rmse", "F RMSE", ".3f", "min"),
        ("force_mae", "F MAE", ".3f", "min"),
        ("total_rmse", "Total RMSE", ".3f", "min"),
        ("energy_nll", "E NLL", ".3f", "min"),
        ("energy_ece", "E ECE", ".3f", "min"),
        ("force_ece", "F ECE", ".3f", "min"),
    ]
    # Filter to columns actually present
    col_defs = [(k, lab, fmt, opt) for k, lab, fmt, opt in col_defs if k in summary_df.columns]

    ordered_methods = [m for m in METHOD_ORDER if m in summary_df["method"].values]

    # Build table data
    lines = []
    lines.append(r"\begin{table*}")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Test set performance of TiO$_2$ force-trained models. "
        r"Energy in meV/atom, forces in meV/\AA{}. "
        r"Best values per column in \textbf{bold}.}"
    )
    lines.append(r"\label{tab:tio2_results}")

    # Header
    col_str = "l" + "r" * len(col_defs) + "r"  # +1 for time column
    lines.append(r"\begin{tabular}{" + col_str + "}")
    lines.append(r"\toprule")

    header = "Method"
    for _, lab, _, _ in col_defs:
        header += f" & {lab}"
    header += r" & Time (min) \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Find best (minimum) for each column
    best_vals = {}
    for key, _, _, opt in col_defs:
        vals = []
        for m in ordered_methods:
            row = summary_df[summary_df["method"] == m].iloc[0]
            v = row.get(key, np.nan)
            if pd.notna(v):
                vals.append((v, m))
        if vals:
            if opt == "min":
                best_vals[key] = min(vals, key=lambda x: x[0])[1]
            else:
                best_vals[key] = max(vals, key=lambda x: x[0])[1]

    # Rows
    for method in ordered_methods:
        row = summary_df[summary_df["method"] == method].iloc[0]
        name = METHOD_NAMES.get(method, method)
        row_str = name

        for key, _, fmt, _ in col_defs:
            v = row.get(key, np.nan)
            std_key = f"{key}_std"
            s = row.get(std_key, np.nan) if std_key in row.index else np.nan

            if pd.isna(v):
                cell = " & --"
            else:
                val_str = f"{v:{fmt}}"
                if pd.notna(s) and s > 0:
                    val_str += f" $\\pm$ {s:{fmt}}"
                if best_vals.get(key) == method:
                    val_str = r"\textbf{" + val_str + "}"
                cell = f" & {val_str}"
            row_str += cell

        # Training time
        if method in time_data:
            t_mean = np.mean(time_data[method]) / 60
            t_std = np.std(time_data[method]) / 60
            time_str = f"{t_mean:.1f} $\\pm$ {t_std:.1f}"
        elif method == "DE":
            # DE uses the same NN runs that form the ensemble.
            if "DE" in time_data:
                t_mean = np.mean(time_data["DE"]) / 60
                t_std = np.std(time_data["DE"]) / 60
                time_str = f"{t_mean:.1f} $\\pm$ {t_std:.1f}"
            else:
                time_str = "--"
        else:
            time_str = "--"
        row_str += f" & {time_str}"
        row_str += r" \\"
        lines.append(row_str)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    tex_str = "\n".join(lines)
    tex_path = output_dir / "summary_table.tex"
    tex_path.write_text(tex_str)
    print(f"  Saved: {tex_path}")

    # Also save as CSV for convenience
    csv_path = output_dir / "summary_table.csv"
    cols_to_save = ["method"] + [k for k, _, _, _ in col_defs]
    std_cols = [f"{k}_std" for k, _, _, _ in col_defs if f"{k}_std" in summary_df.columns]
    all_cols = [c for c in cols_to_save + std_cols if c in summary_df.columns]
    summary_df[summary_df["method"].isin(ordered_methods)][all_cols].to_csv(
        csv_path, index=False, float_format="%.6f"
    )
    print(f"  Saved: {csv_path}")

    return tex_str


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Publication-ready analysis (test set only)")
    parser.add_argument(
        "--pred-dir",
        type=str,
        default="bnn_aenet/logs/TiO2_big/pred/runs",
        help="Directory with prediction outputs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots/TiO2_big/publication",
        help="Output directory for publication figures",
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default="bnn_aenet/logs/TiO2_big/train",
        help="Directory with training logs (for curves)",
    )
    parser.add_argument(
        "--time-dirs",
        type=str,
        nargs="+",
        default=[
            "bnn_aenet/logs/nn_train",
            "bnn_aenet/logs/lrt_train",
            "bnn_aenet/logs/rad_train",
        ],
        help="Directories containing exec_time.log files",
    )
    parser.add_argument(
        "--e-scaling",
        type=float,
        default=None,
        help="Energy scaling factor (eV/atom) for force RMSE to meV/Å. "
        "If not set, loaded from npz when available.",
    )
    parser.add_argument(
        "--e-shift", type=float, default=None, help="Energy shift used for denormalizing energies."
    )
    parser.add_argument("--alpha", type=float, default=0.1, help="Alpha for total_rmse")
    parser.add_argument(
        "--subsets",
        type=str,
        nargs="+",
        default=["train", "val", "test"],
        help="Prediction subsets to analyze",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["DE", "lrt", "rad", "lrt_hetero", "rad_hetero"],
        help="Methods to include: DE lrt rad lrt_hetero rad_hetero",
    )
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    output_dir = Path(args.output_dir)
    train_dir = Path(args.train_dir)
    default_e_scaling, default_e_shift = _get_default_scaling_from_path(pred_dir)
    e_scaling_used = args.e_scaling if args.e_scaling is not None else default_e_scaling
    e_shift_used = (
        args.e_shift
        if args.e_shift is not None
        else (default_e_shift if default_e_shift is not None else 0.0)
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Model Comparison Analysis")
    print("=" * 70)
    print(f"  Predictions:  {pred_dir}")
    print(f"  Training:     {train_dir}")
    print(f"  Output:       {output_dir}")
    print(f"  Subsets:      {args.subsets}")
    if e_scaling_used is not None:
        print(f"  Denorm energy: y=(y/e_scaling + n_atoms*e_shift)/n_atoms")
        print(f"    e_scaling={e_scaling_used:.16g}, e_shift={e_shift_used:.16g}")
    else:
        print("  Denorm energy: disabled (missing e_scaling)")
    print()
    for subset in args.subsets:
        subset_out = output_dir / subset
        subset_out.mkdir(parents=True, exist_ok=True)
        print("-" * 70)
        print(f"Subset: {subset}")

        # ---- Load predictions ----
        print("Loading predictions...")
        _models = set(args.models)
        nn_runs = load_run_predictions(pred_dir, "nn", subset) if "DE" in _models else []
        lrt_runs = load_run_predictions(pred_dir, "lrt", subset) if "lrt" in _models else []
        rad_runs = load_run_predictions(pred_dir, "rad", subset) if "rad" in _models else []
        lrt_het_runs = (
            load_run_predictions(pred_dir, "lrt_hetero", subset) if "lrt_hetero" in _models else []
        )
        rad_het_runs = (
            load_run_predictions(pred_dir, "rad_hetero", subset) if "rad_hetero" in _models else []
        )

        for model_type, runs in (
            ("nn", nn_runs),
            ("lrt", lrt_runs),
            ("rad", rad_runs),
            ("lrt_hetero", lrt_het_runs),
            ("rad_hetero", rad_het_runs),
        ):
            for r in runs:
                run_e_scaling = e_scaling_used
                if run_e_scaling is None:
                    run_e_scaling = _load_run_e_scaling_from_npz(
                        pred_dir, model_type, r.get("run_name", ""), subset
                    )
                _denormalize_run_inplace(r, run_e_scaling, e_shift_used)

        print(
            f"  NN: {len(nn_runs)}  LRT: {len(lrt_runs)}  RAD: {len(rad_runs)}  "
            f"LRT_HET: {len(lrt_het_runs)}  RAD_HET: {len(rad_het_runs)}"
        )

        # ---- Build method_data ----
        method_data = {}

        # Deep Ensemble from NN runs
        if len(nn_runs) > 0:
            de = create_deep_ensemble(nn_runs)
            if de is not None:
                method_data["DE"] = de

        # Best BNN run per model type
        for name, runs in [
            ("lrt", lrt_runs),
            ("rad", rad_runs),
            ("lrt_hetero", lrt_het_runs),
            ("rad_hetero", rad_het_runs),
        ]:
            if len(runs) == 0:
                continue
            sel = select_best_bnn(runs, args.alpha)
            method_data[name] = sel["best_overall"]["run"]
            best_m = sel["best_overall"]["metrics"]
            print(
                f"  {name.upper()} best: {best_m['run_name']} "
                f"(total_rmse={best_m['total_rmse']:.4f})"
            )

        if not method_data:
            print(f"  No runs found for subset={subset}, skipping.")
            continue

        print(f"\nMethods: {list(method_data.keys())}")

        # ---- Compute summary metrics ----
        print("\nComputing metrics...")
        summary_rows = []
        for method, data in method_data.items():
            m = compute_run_metrics(data, args.alpha)
            # Add RMS calibration error explicitly for plotting/reporting.
            df = data["energy_df"]
            y_std = df["stds"].values if "stds" in df.columns else None
            if y_std is not None and np.any(y_std > 0):
                m["energy_rmsce"] = _compute_rmsce(df["true"].values, df["preds"].values, y_std)

            if data.get("forces") is not None:
                f = data["forces"]
                f_std = f.get("std_forces")
                if f_std is not None and np.any(f_std > 0):
                    m["force_rmsce"] = _compute_rmsce(f["true_forces"], f["pred_forces"], f_std)
            m["method"] = method
            summary_rows.append(m)

        # Add per-run std for NN and DE sub-ensembles
        if len(nn_runs) > 0:
            nn_all = [compute_run_metrics(r, args.alpha) for r in nn_runs]
            nn_df = pd.DataFrame(nn_all)
            nn_row = [r for r in summary_rows if r.get("method") == "nn"]
            if nn_row:
                for col in nn_df.select_dtypes(include=[np.number]).columns:
                    nn_row[0][f"{col}_std"] = nn_df[col].std()

            de_subs = create_sub_ensembles(nn_runs, n_per_ensemble=5, max_ensembles=20)
            if de_subs:
                sub_metrics = [compute_run_metrics(s, args.alpha) for s in de_subs]
                sub_df = pd.DataFrame(sub_metrics)
                de_row = [r for r in summary_rows if r.get("method") == "DE"]
                if de_row:
                    for col in sub_df.select_dtypes(include=[np.number]).columns:
                        de_row[0][f"{col}_std"] = sub_df[col].std()

        # BNN stds across runs
        for name, runs in [
            ("lrt", lrt_runs),
            ("rad", rad_runs),
            ("lrt_hetero", lrt_het_runs),
            ("rad_hetero", rad_het_runs),
        ]:
            if len(runs) == 0:
                continue
            bnn_all = [compute_run_metrics(r, args.alpha) for r in runs]
            bnn_df = pd.DataFrame(bnn_all)
            bnn_row = [r for r in summary_rows if r.get("method") == name]
            if bnn_row:
                for col in bnn_df.select_dtypes(include=[np.number]).columns:
                    bnn_row[0][f"{col}_std"] = bnn_df[col].std()

        summary_df = pd.DataFrame(summary_rows)
        key_cols = [
            "method",
            "energy_rmse",
            "energy_mae",
            "force_rmse",
            "force_mae",
            "total_rmse",
            "energy_nll",
            "energy_ece",
        ]
        avail = [c for c in key_cols if c in summary_df.columns]
        print("\n" + summary_df[avail].to_string(index=False, float_format="%.4f"))

        # ---- Generate figures ----
        print("\nGenerating figures...")
        print("  1/7  Energy parity")
        plot_energy_parity(method_data, subset_out)
        print("  2/7  Force parity")
        plot_force_parity(method_data, subset_out)
        print("  3/7  Calibration")
        plot_calibration(method_data, subset_out)
        print("  4/7  Error vs uncertainty")
        plot_error_vs_uncertainty(method_data, subset_out)
        print("       + normalized-correlation variant")
        plot_error_vs_uncertainty(
            method_data,
            subset_out,
            corr_space="normalized",
            output_name="error_vs_uncertainty_normcorr.png",
        )
        print("  5/7  Metric bars")
        plot_metric_bars(summary_df, subset_out)
        print("  6/7  Training time")
        time_data = plot_training_time(args.time_dirs, subset_out) or {}
        print("  7/7  Training curves")
        if train_dir.exists():
            plot_training_curves(train_dir, subset_out)
        else:
            print(f"       Training dir not found: {train_dir}")

        # ---- LaTeX table ----
        print("\nGenerating LaTeX table...")
        tex = generate_latex_table(summary_df, time_data, subset_out)
        print("\n" + tex)

        print(
            f"\nDone subset={subset}: {len(list(subset_out.glob('*.png')))} PNG + "
            f"{len(list(subset_out.glob('*.pdf')))} PDF files in {subset_out}/"
        )

    print("\n" + "=" * 70)
    print("Done all subsets.")
    print("=" * 70)


if __name__ == "__main__":
    main()
