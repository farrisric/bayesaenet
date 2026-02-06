"""Pytest configuration and shared fixtures."""

import pytest
import torch
import numpy as np
from pathlib import Path


@pytest.fixture
def device():
    """Get available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def project_root():
    """Get project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def sample_batch():
    """Create a sample batch for testing."""
    batch_size = 8
    input_dim = 50
    
    return {
        "x": torch.randn(batch_size, input_dim),
        "y": torch.randn(batch_size, 1),
        "logic": torch.ones(batch_size, 1),
    }


@pytest.fixture
def seed():
    """Set random seed for reproducibility."""
    seed_val = 42
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)
    return seed_val
