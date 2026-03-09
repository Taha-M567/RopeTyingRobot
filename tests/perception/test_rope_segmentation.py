"""Tests for rope segmentation module."""

import cv2
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


def test_segment_rope_saturation_method():
    """Test saturation-based segmentation with a colored line on dark bg."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    # Draw a bright saturated green line (BGR)
    cv2.line(image, (10, 50), (90, 50), (0, 255, 0), 4)

    config = {
        "method": "saturation",
        "saturation_range": {"min_saturation": 50, "min_value": 50},
        "morph_operations": {
            "opening_kernel_size": 1,
            "closing_kernel_size": 3,
        },
        "contour_filter": {
            "min_area": 10,
            "min_aspect_ratio": 1.0,
        },
        "cleanup": {"min_area": 0, "keep_largest": False},
    }

    result = segment_rope(image, config=config)
    assert isinstance(result, RopeMask)
    assert result.mask.shape == (100, 100)
    # The green line should produce some rope pixels
    assert np.count_nonzero(result.mask) > 0


def test_segment_rope_saturation_ignores_dark_bg():
    """Saturation method should not segment a dark/gray background."""
    # Uniform dark gray image — no saturated pixels
    image = np.full((100, 100, 3), 60, dtype=np.uint8)

    config = {
        "method": "saturation",
        "saturation_range": {"min_saturation": 60, "min_value": 50},
        "contour_filter": {"min_area": 10, "min_aspect_ratio": 1.0},
        "cleanup": {"min_area": 0, "keep_largest": False},
    }

    result = segment_rope(image, config=config)
    assert np.count_nonzero(result.mask) == 0


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
