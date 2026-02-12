"""Publication-ready analysis of force-trained models (test set only).

Produces a small set of condensed, high-quality figures suitable for
journal publication, plus a LaTeX summary table.

Usage:
    python -m bnn_aenet.tasks.analyze_publication \
        --pred-dir bnn_aenet/logs/TiO2_big/pred \
        --output-dir plots/TiO2_big/publication \
        --train-dir bnn_aenet/logs/TiO2_big/train \
        --time-dirs bnn_aenet/logs/nn_forces bnn_aenet/logs/lrt_forces \
                    bnn_aenet/logs/fo_forces bnn_aenet/logs/rad_forces
"""

import argparse
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

from bnn_aenet.analysis.metrics import (
    compute_calibration_curve,
    compute_energy_metrics,
    compute_force_metrics,
    compute_uncertainty_metrics,
)

# Reuse data-loading helpers from the main analysis module
from bnn_aenet.tasks.analyze import (
    create_deep_ensemble,
    create_sub_ensembles,
    compute_run_metrics,
    load_run_predictions,
    load_tensorboard_scalars,
    select_best_bnn,
)

# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------
SINGLE_COL = 3.5   # inches  (single-column figure)
DOUBLE_COL = 7.0   # inches  (double-column / full-width figure)
DPI = 300

plt.rcParams.update({
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
})

# Method display names and colors (consistent across all figures)
METHOD_ORDER = ["DE", "nn", "lrt", "fo", "rad"]
METHOD_NAMES = {
    "DE": "Deep Ens.",
    "DE_sub": "DE (5-model)",
    "nn": "NN (mean)",
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
METHOD_MARKERS = {
    "DE": "o",
    "nn": "s",
    "lrt": "^",
    "fo": "D",
    "rad": "v",
}

# For training time chart (maps to exec_time.log directories)
TIME_METHOD_MAP = {
    "nn_forces": "nn",
    "lrt_forces": "lrt",
    "fo_forces": "fo",
    "rad_forces": "rad",
}


def _save(fig, path, close=True):
    """Save figure as both PNG and PDF."""
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    if close:
        plt.close(fig)


# ============================================================================
# Figure 1: Energy Parity
# ============================================================================

def plot_energy_parity(method_data: dict, output_dir: Path):
    """Energy parity plots -- one panel per method, shared axes."""
    methods = [m for m in METHOD_ORDER if m in method_data]
    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(DOUBLE_COL, DOUBLE_COL / n + 0.4),
                             squeeze=False)
    axes = axes[0]

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

    for ax, method in zip(axes, methods):
        df = method_data[method]["energy_df"]
        y_true = df["true"].values
        y_pred = df["preds"].values

        color = METHOD_COLORS[method]
        ax.scatter(y_true, y_pred, s=6, alpha=0.5, color=color,
                   edgecolors="none", rasterized=True)
        ax.plot([lo, hi], [lo, hi], "k-", linewidth=0.6, alpha=0.6)

        # Metrics annotation
        em = compute_energy_metrics(y_true, y_pred)
        rmse_str = f"RMSE = {em['rmse']:.3f}"
        r2_str = f"$R^2$ = {em['r2']:.6f}"
        ax.text(0.05, 0.95, f"{rmse_str}\n{r2_str}",
                transform=ax.transAxes, va="top", ha="left", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8",
                          alpha=0.85))

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_title(METHOD_NAMES.get(method, method))
        if ax == axes[0]:
            ax.set_ylabel("Predicted energy (meV/atom)")
        ax.set_xlabel("True energy (meV/atom)")

    fig.tight_layout(w_pad=0.8)
    _save(fig, output_dir / "energy_parity.png")


# ============================================================================
# Figure 2: Force Parity
# ============================================================================

def plot_force_parity(method_data: dict, output_dir: Path,
                      max_points: int = 8000):
    """Force parity plots -- one panel per method, density-colored."""
    methods = [m for m in METHOD_ORDER if m in method_data
               and method_data[m].get("forces") is not None]
    n = len(methods)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(DOUBLE_COL, DOUBLE_COL / n + 0.4),
                             squeeze=False)
    axes = axes[0]

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

    for ax, method in zip(axes, methods):
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
        h = ax.hist2d(ft, fp, bins=100, range=[[lo, hi], [lo, hi]],
                       cmap="viridis", norm=LogNorm(), rasterized=True)
        ax.plot([lo, hi], [lo, hi], "w-", linewidth=0.6, alpha=0.8)

        # Metrics
        fm = compute_force_metrics(ft, fp)
        rmse_str = f"RMSE = {fm['rmse']:.3f}"
        r2_str = f"$R^2$ = {fm['r2']:.4f}"
        ax.text(0.05, 0.95, f"{rmse_str}\n{r2_str}",
                transform=ax.transAxes, va="top", ha="left", fontsize=7,
                color="white",
                bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="0.3",
                          alpha=0.6))

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_title(METHOD_NAMES.get(method, method))
        if ax == axes[0]:
            ax.set_ylabel(r"Predicted force (eV/$\mathrm{\AA}$)")
        ax.set_xlabel(r"True force (eV/$\mathrm{\AA}$)")

    fig.tight_layout(w_pad=0.8)
    _save(fig, output_dir / "force_parity.png")


# ============================================================================
# Figure 3: Calibration
# ============================================================================

def plot_calibration(method_data: dict, output_dir: Path):
    """Combined calibration plot: energy (left) + forces (right), all methods
    overlaid on the same axes."""
    methods_e = [m for m in METHOD_ORDER if m in method_data
                 and method_data[m]["energy_df"]["stds"].values.std() > 0]
    methods_f = [m for m in METHOD_ORDER if m in method_data
                 and method_data[m].get("forces") is not None
                 and method_data[m]["forces"]["std_forces"].std() > 0]

    if not methods_e and not methods_f:
        return

    n_panels = (1 if methods_e else 0) + (1 if methods_f else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(SINGLE_COL * n_panels, SINGLE_COL),
                             squeeze=False)
    axes = axes[0]
    ax_idx = 0

    if methods_e:
        ax = axes[ax_idx]
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.6, alpha=0.5, label="Ideal")
        for method in methods_e:
            df = method_data[method]["energy_df"]
            exp_f, obs_f = compute_calibration_curve(
                df["true"].values, df["preds"].values, df["stds"].values
            )
            ax.plot(exp_f, obs_f, color=METHOD_COLORS[method],
                    marker=METHOD_MARKERS.get(method, "o"), markersize=3,
                    label=METHOD_NAMES.get(method, method))
        ax.set_xlabel("Expected confidence")
        ax.set_ylabel("Observed coverage")
        ax.set_title("Energy calibration")
        ax.legend(loc="lower right", frameon=True, fancybox=False,
                  edgecolor="0.8", framealpha=0.9)
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
            ax.plot(exp_f, obs_f, color=METHOD_COLORS[method],
                    marker=METHOD_MARKERS.get(method, "o"), markersize=3,
                    label=METHOD_NAMES.get(method, method))
        ax.set_xlabel("Expected confidence")
        ax.set_ylabel("Observed coverage")
        ax.set_title("Force calibration")
        ax.legend(loc="lower right", frameon=True, fancybox=False,
                  edgecolor="0.8", framealpha=0.9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")

    fig.tight_layout(w_pad=1.0)
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


def plot_error_vs_uncertainty(method_data: dict, output_dir: Path):
    """Binned error vs uncertainty -- energy (left) + forces (right),
    all methods overlaid."""
    methods_e = [m for m in METHOD_ORDER if m in method_data
                 and method_data[m]["energy_df"]["stds"].values.std() > 0]
    methods_f = [m for m in METHOD_ORDER if m in method_data
                 and method_data[m].get("forces") is not None
                 and method_data[m]["forces"]["std_forces"].std() > 0]

    if not methods_e and not methods_f:
        return

    n_panels = (1 if methods_e else 0) + (1 if methods_f else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(SINGLE_COL * n_panels, SINGLE_COL),
                             squeeze=False)
    axes = axes[0]
    ax_idx = 0

    # Energy
    if methods_e:
        ax = axes[ax_idx]
        for method in methods_e:
            df = method_data[method]["energy_df"]
            bx, by, be = _binned_error_uq(
                df["true"].values, df["preds"].values, df["stds"].values
            )
            spearman_r, _ = stats.spearmanr(
                np.abs(df["true"].values - df["preds"].values),
                df["stds"].values,
            )
            label = f"{METHOD_NAMES.get(method, method)} ($\\rho$={spearman_r:.2f})"
            ax.errorbar(bx, by, yerr=be, fmt="-",
                        marker=METHOD_MARKERS.get(method, "o"),
                        color=METHOD_COLORS[method], capsize=2,
                        markersize=4, linewidth=1.0, label=label)
        # y=x reference
        all_vals = ax.get_xlim()[1]
        ax.plot([0, all_vals], [0, all_vals], "k--", linewidth=0.5, alpha=0.4)
        ax.set_xlabel("Predicted uncertainty (meV/atom)")
        ax.set_ylabel("Mean |error| (meV/atom)")
        ax.set_title("Energy")
        ax.legend(loc="upper left", frameon=True, fancybox=False,
                  edgecolor="0.8", framealpha=0.9, fontsize=7)
        ax_idx += 1

    # Forces
    if methods_f:
        ax = axes[ax_idx]
        for method in methods_f:
            f = method_data[method]["forces"]
            bx, by, be = _binned_error_uq(
                f["true_forces"], f["pred_forces"], f["std_forces"]
            )
            spearman_r, _ = stats.spearmanr(
                np.abs(f["true_forces"].flatten() - f["pred_forces"].flatten()),
                f["std_forces"].flatten(),
            )
            label = f"{METHOD_NAMES.get(method, method)} ($\\rho$={spearman_r:.2f})"
            ax.errorbar(bx, by, yerr=be, fmt="-",
                        marker=METHOD_MARKERS.get(method, "o"),
                        color=METHOD_COLORS[method], capsize=2,
                        markersize=4, linewidth=1.0, label=label)
        all_vals = ax.get_xlim()[1]
        ax.plot([0, all_vals], [0, all_vals], "k--", linewidth=0.5, alpha=0.4)
        ax.set_xlabel(r"Predicted uncertainty (eV/$\mathrm{\AA}$)")
        ax.set_ylabel(r"Mean |error| (eV/$\mathrm{\AA}$)")
        ax.set_title("Forces")
        ax.legend(loc="upper left", frameon=True, fancybox=False,
                  edgecolor="0.8", framealpha=0.9, fontsize=7)

    fig.tight_layout(w_pad=1.0)
    _save(fig, output_dir / "error_vs_uncertainty.png")


# ============================================================================
# Figure 5: Summary Bar Chart
# ============================================================================

def plot_metric_bars(summary_df: pd.DataFrame, output_dir: Path):
    """Grouped bar chart for key metrics across methods."""
    metric_defs = [
        ("energy_rmse", "Energy RMSE\n(meV/atom)"),
        ("force_rmse", r"Force RMSE" + "\n" + r"(meV/$\mathrm{\AA}$)"),
        ("energy_nll", "Energy NLL"),
        ("force_ece", "Force ECE"),
    ]
    # Filter to metrics that actually exist
    metric_defs = [(k, lab) for k, lab in metric_defs if k in summary_df.columns]
    n_metrics = len(metric_defs)
    if n_metrics == 0:
        return

    ncols = 2
    nrows = (n_metrics + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(DOUBLE_COL, 1.8 * nrows),
                             squeeze=False)
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

        bars = ax.bar(x, vals, width, yerr=errs, capsize=3,
                      color=colors, edgecolor="0.3", linewidth=0.5,
                      error_kw=dict(linewidth=0.8))
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_NAMES.get(m, m) for m in ordered_methods],
                           rotation=25, ha="right", fontsize=7)
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

    ordered = [m for m in ["nn", "lrt", "fo", "rad"] if m in times]
    if not ordered:
        return

    means = [np.mean(times[m]) / 60 for m in ordered]  # minutes
    stds = [np.std(times[m]) / 60 for m in ordered]
    n_runs = [len(times[m]) for m in ordered]
    total_hrs = [np.sum(times[m]) / 3600 for m in ordered]
    colors = [METHOD_COLORS.get(m, "#888") for m in ordered]

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.5))
    x = np.arange(len(ordered))
    bars = ax.bar(x, means, 0.6, yerr=stds, capsize=3,
                  color=colors, edgecolor="0.3", linewidth=0.5,
                  error_kw=dict(linewidth=0.8))

    # Annotate total GPU-hours on each bar
    for i, (bar, hrs, nr) in enumerate(zip(bars, total_hrs, n_runs)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + stds[i] + 1,
                f"{hrs:.1f} h\n({nr} runs)",
                ha="center", va="bottom", fontsize=6.5)

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
    model_types = ["nn", "lrt", "fo", "rad"]
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

    for tag_idx, (tag, ylabel) in enumerate([
        ("rmse/val", "Energy RMSE (meV/atom)"),
        ("force_rmse/val", r"Force RMSE (meV/$\mathrm{\AA}$)"),
    ]):
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
                ax.fill_between(steps, mean_vals - std_vals, mean_vals + std_vals,
                                color=color, alpha=0.12)

        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right", frameon=True, fancybox=False,
                  edgecolor="0.8", framealpha=0.9, fontsize=7)
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())

    fig.tight_layout(w_pad=1.5)
    _save(fig, output_dir / "training_curves.png")


# ============================================================================
# LaTeX Table
# ============================================================================

def generate_latex_table(summary_df: pd.DataFrame, time_data: dict,
                         output_dir: Path):
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
    col_defs = [(k, lab, fmt, opt) for k, lab, fmt, opt in col_defs
                if k in summary_df.columns]

    ordered_methods = [m for m in METHOD_ORDER if m in summary_df["method"].values]

    # Build table data
    lines = []
    lines.append(r"\begin{table*}")
    lines.append(r"\centering")
    lines.append(r"\caption{Test set performance of TiO$_2$ force-trained models. "
                 r"Energy in meV/atom, forces in meV/\AA{}. "
                 r"Best values per column in \textbf{bold}.}")
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
            # DE is just the NN time * 10
            if "nn" in time_data:
                t_total = np.sum(time_data["nn"]) / 60
                time_str = f"{t_total:.0f} (total)"
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
    parser = argparse.ArgumentParser(
        description="Publication-ready analysis (test set only)"
    )
    parser.add_argument("--pred-dir", type=str,
                        default="bnn_aenet/logs/TiO2_big/pred",
                        help="Directory with prediction outputs")
    parser.add_argument("--output-dir", type=str,
                        default="plots/TiO2_big/publication",
                        help="Output directory for publication figures")
    parser.add_argument("--train-dir", type=str,
                        default="bnn_aenet/logs/TiO2_big/train",
                        help="Directory with training logs (for curves)")
    parser.add_argument("--time-dirs", type=str, nargs="+",
                        default=[
                            "bnn_aenet/logs/nn_forces",
                            "bnn_aenet/logs/lrt_forces",
                            "bnn_aenet/logs/fo_forces",
                            "bnn_aenet/logs/rad_forces",
                        ],
                        help="Directories containing exec_time.log files")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Alpha for total_rmse")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    output_dir = Path(args.output_dir)
    train_dir = Path(args.train_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Publication Analysis -- Test Set")
    print("=" * 70)
    print(f"  Predictions:  {pred_dir}")
    print(f"  Training:     {train_dir}")
    print(f"  Output:       {output_dir}")
    print()

    subset = "test"

    # ---- Load predictions ----
    print("Loading predictions...")
    nn_runs = load_run_predictions(pred_dir, "nn", subset)
    lrt_runs = load_run_predictions(pred_dir, "lrt", subset)
    fo_runs = load_run_predictions(pred_dir, "fo", subset)
    rad_runs = load_run_predictions(pred_dir, "rad", subset)
    print(f"  NN: {len(nn_runs)}  LRT: {len(lrt_runs)}  "
          f"FO: {len(fo_runs)}  RAD: {len(rad_runs)}")

    # ---- Build method_data ----
    method_data = {}

    # Deep Ensemble (all NN runs)
    if len(nn_runs) > 0:
        de = create_deep_ensemble(nn_runs)
        if de is not None:
            method_data["DE"] = de

        # NN mean
        nn_mean = {
            "energy_df": pd.DataFrame({
                "true": nn_runs[0]["energy_df"]["true"].values,
                "preds": np.mean([r["energy_df"]["preds"].values for r in nn_runs], axis=0),
                "stds": np.std([r["energy_df"]["preds"].values for r in nn_runs], axis=0),
                "n_atoms": nn_runs[0]["energy_df"]["n_atoms"].values,
            }),
            "forces": None,
            "run_name": "nn_mean",
        }
        if all(r["forces"] is not None for r in nn_runs):
            nn_mean["forces"] = {
                "true_forces": nn_runs[0]["forces"]["true_forces"],
                "pred_forces": np.mean([r["forces"]["pred_forces"] for r in nn_runs], axis=0),
                "std_forces": np.std([r["forces"]["pred_forces"] for r in nn_runs], axis=0),
            }
        method_data["nn"] = nn_mean

    # Best BNN per type
    for name, runs in [("lrt", lrt_runs), ("fo", fo_runs), ("rad", rad_runs)]:
        if len(runs) == 0:
            continue
        sel = select_best_bnn(runs, args.alpha)
        method_data[name] = sel["best_overall"]["run"]
        best_m = sel["best_overall"]["metrics"]
        print(f"  {name.upper()} best: {best_m['run_name']} "
              f"(total_rmse={best_m['total_rmse']:.4f})")

    print(f"\nMethods: {list(method_data.keys())}")

    # ---- Compute summary metrics ----
    print("\nComputing metrics...")
    summary_rows = []
    for method, data in method_data.items():
        m = compute_run_metrics(data, args.alpha)
        m["method"] = method
        summary_rows.append(m)

    # Add per-run std for nn and DE sub-ensembles
    if len(nn_runs) > 0:
        nn_all = [compute_run_metrics(r, args.alpha) for r in nn_runs]
        nn_df = pd.DataFrame(nn_all)
        # Update nn row with stds
        nn_row = [r for r in summary_rows if r.get("method") == "nn"]
        if nn_row:
            for col in nn_df.select_dtypes(include=[np.number]).columns:
                nn_row[0][f"{col}_std"] = nn_df[col].std()

        # DE sub-ensembles for std
        de_subs = create_sub_ensembles(nn_runs, n_per_ensemble=5, max_ensembles=20)
        if de_subs:
            sub_metrics = [compute_run_metrics(s, args.alpha) for s in de_subs]
            sub_df = pd.DataFrame(sub_metrics)
            de_row = [r for r in summary_rows if r.get("method") == "DE"]
            if de_row:
                for col in sub_df.select_dtypes(include=[np.number]).columns:
                    de_row[0][f"{col}_std"] = sub_df[col].std()

    # BNN stds (across runs)
    for name, runs in [("lrt", lrt_runs), ("fo", fo_runs), ("rad", rad_runs)]:
        if len(runs) == 0:
            continue
        bnn_all = [compute_run_metrics(r, args.alpha) for r in runs]
        bnn_df = pd.DataFrame(bnn_all)
        bnn_row = [r for r in summary_rows if r.get("method") == name]
        if bnn_row:
            for col in bnn_df.select_dtypes(include=[np.number]).columns:
                bnn_row[0][f"{col}_std"] = bnn_df[col].std()

    summary_df = pd.DataFrame(summary_rows)

    # Print summary
    key_cols = ["method", "energy_rmse", "energy_mae", "force_rmse",
                "force_mae", "total_rmse", "energy_nll", "energy_ece"]
    avail = [c for c in key_cols if c in summary_df.columns]
    print("\n" + summary_df[avail].to_string(index=False, float_format="%.4f"))

    # ---- Generate figures ----
    print("\nGenerating figures...")

    print("  1/7  Energy parity")
    plot_energy_parity(method_data, output_dir)

    print("  2/7  Force parity")
    plot_force_parity(method_data, output_dir)

    print("  3/7  Calibration")
    plot_calibration(method_data, output_dir)

    print("  4/7  Error vs uncertainty")
    plot_error_vs_uncertainty(method_data, output_dir)

    print("  5/7  Metric bars")
    plot_metric_bars(summary_df, output_dir)

    print("  6/7  Training time")
    time_data = plot_training_time(args.time_dirs, output_dir)
    if time_data is None:
        time_data = {}

    print("  7/7  Training curves")
    if train_dir.exists():
        plot_training_curves(train_dir, output_dir)
    else:
        print(f"       Training dir not found: {train_dir}")

    # ---- LaTeX table ----
    print("\nGenerating LaTeX table...")
    tex = generate_latex_table(summary_df, time_data, output_dir)
    print("\n" + tex)

    print("\n" + "=" * 70)
    print(f"Done! {len(list(output_dir.glob('*.png')))} PNG + "
          f"{len(list(output_dir.glob('*.pdf')))} PDF files in {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
