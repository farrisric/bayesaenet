from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentId:
    """Lightweight identifier for an experiment/run."""

    dataset: str  # e.g. "tio2_small", "tio2_big", "qm7"
    task: str  # e.g. "hps", "train", "pred"
    model: str  # e.g. "lrt", "rad", "nn"
    run_id: str  # already formatted identifier, e.g. "001" or "20260219_120001"


def _repo_root() -> Path:
    """Return the repository root based on this file location."""

    # This file lives in: repo_root/bnn_aenet/utils/paths.py
    return Path(__file__).resolve().parents[2]


def get_results_root() -> Path:
    """Root directory for Optuna / HPS result databases."""

    return _repo_root() / "bnn_aenet" / "results"


def get_results_db_path(dataset: str, model: str) -> Path:
    """Return the canonical path for an Optuna SQLite DB.

    Layout: bnn_aenet/results/<dataset>/<model>.db
    Example: bnn_aenet/results/tio2_small/lrt.db
    """

    return get_results_root() / dataset / f"{model}.db"


def get_logs_root() -> Path:
    """Root directory for all logs (hps, train, prediction, etc.)."""

    return _repo_root() / "bnn_aenet" / "logs"


def get_run_log_dir(exp: ExperimentId) -> Path:
    """Return the directory for a single run's logs.

    Layout: bnn_aenet/logs/<dataset>/<task>/<model>/run_<run_id>
    Example: bnn_aenet/logs/tio2_small/hps/lrt/run_001
    """

    return (
        get_logs_root()
        / exp.dataset
        / exp.task
        / exp.model
        / f"run_{exp.run_id}"
    )

