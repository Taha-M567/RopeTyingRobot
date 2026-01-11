"""Tests for rope segmentation module."""

import numpy as np
import pytest

from src.perception.rope_segmentation import RopeMask, segment_rope


def test_segment_rope_valid_image():
    """Test segmentation with valid input image."""
    # Create a dummy image
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    result = segment_rope(image)

    assert isinstance(result, RopeMask)
    assert result.mask.shape == (100, 100)
    assert 0.0 <= result.confidence <= 1.0


def test_segment_rope_invalid_image():
    """Test segmentation raises error for invalid input."""
    with pytest.raises(ValueError):
        segment_rope(None)

    with pytest.raises(ValueError):
        segment_rope(np.array([]))


def test_rope_mask_dataclass():
    """Test RopeMask dataclass structure."""
    mask = np.zeros((10, 10), dtype=np.uint8)
    rope_mask = RopeMask(
        mask=mask,
        confidence=0.8,
        image_shape=(10, 10),
    )

    assert rope_mask.confidence == 0.8
    assert rope_mask.image_shape == (10, 10)
