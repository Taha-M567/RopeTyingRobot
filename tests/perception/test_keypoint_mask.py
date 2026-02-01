"""Tests for keypoint mask generation."""

import numpy as np

from src.perception.keypoint_detection import Keypoint
from src.perception.keypoint_mask import (
    BACKGROUND_CLASS,
    CROSSING_CLASS,
    ENDPOINT_CLASS,
    ROPE_CLASS,
    create_keypoint_class_mask,
)


def test_create_keypoint_class_mask_basic():
    """Test class mask generation on a simple rope."""
    rope_mask = np.zeros((50, 50), dtype=np.uint8)
    rope_mask[25, 5:45] = 255

    keypoints = [
        Keypoint(position=(10.0, 25.0), keypoint_type="endpoint", confidence=1.0),
        Keypoint(position=(40.0, 25.0), keypoint_type="endpoint", confidence=1.0),
        Keypoint(position=(25.0, 25.0), keypoint_type="crossing", confidence=1.0),
    ]

    config = {"endpoint_radius": 2, "crossing_radius": 2}
    class_mask = create_keypoint_class_mask(rope_mask, keypoints, config=config)

    assert class_mask.shape == rope_mask.shape
    assert class_mask.dtype == np.uint8

    # Rope pixels should be labeled as rope or overridden by keypoints
    assert class_mask[25, 20] in [ROPE_CLASS, CROSSING_CLASS, ENDPOINT_CLASS]
    # Endpoint location should be endpoint class
    assert class_mask[25, 10] == ENDPOINT_CLASS
    # Crossing location should be crossing class
    assert class_mask[25, 25] == CROSSING_CLASS
    # Background should remain background
    assert class_mask[0, 0] == BACKGROUND_CLASS
