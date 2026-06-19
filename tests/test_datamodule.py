"""Tests for data loading and preprocessing."""

from pathlib import Path

import pytest


class TestDataConfigs:
    """Tests for datamodule configurations."""

    def test_datamodule_configs_exist(self, project_root):
        """Test that datamodule configs exist."""
        dm_dir = project_root / "bnn_aenet" / "configs" / "datamodule"

        assert (dm_dir / "datamodule.yaml").exists()
        assert (dm_dir / "QM7.yaml").exists()
        assert (dm_dir / "TiO.yaml").exists()

    def test_data_directories_exist(self, project_root):
        """Test that data directories exist."""
        data_dir = project_root / "data"

        assert data_dir.exists()
        # Check for at least one dataset
        datasets = list(data_dir.iterdir())
        assert len(datasets) > 0, "No datasets found in data directory"


class TestDataModuleImport:
    """Tests for datamodule imports."""

    def test_aenet_datamodule_import(self):
        """Test AenetDataModule can be imported."""
        from bnn_aenet.datamodule.aenet_datamodule import AenetDataModule

        assert AenetDataModule is not None

    def test_prepare_batches_import(self):
        """Test prepare_batches can be imported."""
        from bnn_aenet.datamodule.aenet import prepare_batches

        assert prepare_batches is not None


class TestTrainInParsing:
    """Tests for train.in file parsing."""

    def test_input_parameters_import(self):
        """Test InputParameters class can be imported."""
        from bnn_aenet.datamodule.aenet.data_classes import InputParameters

        assert InputParameters is not None

    def test_read_train_in_import(self):
        """Test read_train_in function can be imported."""
        from bnn_aenet.datamodule.aenet.read_input import read_train_in

        assert read_train_in is not None
