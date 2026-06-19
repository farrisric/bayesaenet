"""Tests for reproducibility features."""

import numpy as np
import pytest
import torch


class TestSeedReproducibility:
    """Tests for seed-based reproducibility."""

    def test_torch_seed_consistency(self):
        """Test that PyTorch operations are reproducible with seed."""
        torch.manual_seed(42)
        x1 = torch.randn(10, 10)

        torch.manual_seed(42)
        x2 = torch.randn(10, 10)

        assert torch.allclose(x1, x2)

    def test_numpy_seed_consistency(self):
        """Test that NumPy operations are reproducible with seed."""
        np.random.seed(42)
        x1 = np.random.randn(10, 10)

        np.random.seed(42)
        x2 = np.random.randn(10, 10)

        assert np.allclose(x1, x2)

    def test_lightning_seed_everything(self):
        """Test Lightning's seed_everything function."""
        import lightning.pytorch as L

        L.seed_everything(42, workers=True)
        x1_torch = torch.randn(5, 5)
        x1_np = np.random.randn(5, 5)

        L.seed_everything(42, workers=True)
        x2_torch = torch.randn(5, 5)
        x2_np = np.random.randn(5, 5)

        assert torch.allclose(x1_torch, x2_torch)
        assert np.allclose(x1_np, x2_np)


class TestConfigReproducibility:
    """Tests for configuration-based reproducibility."""

    def test_default_seed_in_configs(self, project_root):
        """Test that final configs have deterministic seeds."""
        from omegaconf import OmegaConf

        final_dir = project_root / "bnn_aenet" / "configs" / "experiment" / "final"

        for config_file in final_dir.glob("*.yaml"):
            cfg = OmegaConf.load(config_file)
            assert "seed" in cfg, f"No seed in {config_file.name}"
            assert cfg.seed is not None, f"Seed is None in {config_file.name}"
            assert isinstance(cfg.seed, int), f"Seed is not int in {config_file.name}"
