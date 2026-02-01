"""Keypoint mask generation for rope endpoints and crossings.

This module creates a class-labeled mask restricted to rope pixels.
"""

from typing import Optional

import cv2
import numpy as np

from src.perception.keypoint_detection import Keypoint

BINARY_THRESHOLD = 127

BACKGROUND_CLASS = 0
ROPE_CLASS = 1
ENDPOINT_CLASS = 2
CROSSING_CLASS = 3

DEFAULT_ENDPOINT_RADIUS = 6
DEFAULT_CROSSING_RADIUS = 8


def _ensure_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Ensure the mask is uint8 binary with values 0 or 255."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    unique_vals = np.unique(mask)
    if not all(v in [0, 255] for v in unique_vals):
        mask = ((mask > BINARY_THRESHOLD) * 255).astype(np.uint8)
    return mask


def _apply_keypoint_class(
    class_mask: np.ndarray,
    rope_binary: np.ndarray,
    keypoint: Keypoint,
    radius: int,
    class_id: int,
) -> None:
    """Draw a keypoint class onto the mask, restricted to rope pixels."""
    overlay = np.zeros_like(rope_binary, dtype=np.uint8)
    x = int(round(keypoint.position[0]))
    y = int(round(keypoint.position[1]))
    cv2.circle(overlay, (x, y), int(radius), 255, -1)
    overlay = cv2.bitwise_and(overlay, rope_binary)
    class_mask[overlay > 0] = class_id


def create_keypoint_class_mask(
    rope_mask: np.ndarray,
    keypoints: list[Keypoint],
    config: Optional[dict] = None,
) -> np.ndarray:
    """Create a class-labeled mask for rope, endpoints, and crossings.

    Args:
        rope_mask: Binary rope mask (uint8, 0/255)
        keypoints: Detected keypoints
        config: Optional config with:
            - endpoint_radius: int, pixel radius for endpoint regions
            - crossing_radius: int, pixel radius for crossing regions

    Returns:
        Class mask (uint8) with:
            0 = background, 1 = rope, 2 = endpoint, 3 = crossing
    """
    if rope_mask is None or not isinstance(rope_mask, np.ndarray):
        return np.zeros((0, 0), dtype=np.uint8)

    if rope_mask.size == 0:
        return np.zeros_like(rope_mask, dtype=np.uint8)

    if config is None:
        config = {}

    rope_binary = _ensure_binary_mask(rope_mask)
    class_mask = np.zeros_like(rope_binary, dtype=np.uint8)
    class_mask[rope_binary > 0] = ROPE_CLASS

    endpoint_radius = int(config.get("endpoint_radius", DEFAULT_ENDPOINT_RADIUS))
    crossing_radius = int(config.get("crossing_radius", DEFAULT_CROSSING_RADIUS))

    for kp in keypoints:
        if kp.keypoint_type != "endpoint":
            continue
        _apply_keypoint_class(
            class_mask=class_mask,
            rope_binary=rope_binary,
            keypoint=kp,
            radius=endpoint_radius,
            class_id=ENDPOINT_CLASS,
        )

    for kp in keypoints:
        if kp.keypoint_type != "crossing":
            continue
        _apply_keypoint_class(
            class_mask=class_mask,
            rope_binary=rope_binary,
            keypoint=kp,
            radius=crossing_radius,
            class_id=CROSSING_CLASS,
        )

    return class_mask
