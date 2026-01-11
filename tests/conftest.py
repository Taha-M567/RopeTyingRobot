"""Pytest configuration and shared fixtures."""

import numpy as np
import pytest


@pytest.fixture
def sample_image():
    """Fixture providing a sample test image."""
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_mask():
    """Fixture providing a sample binary mask."""
    return np.zeros((100, 100), dtype=np.uint8)
