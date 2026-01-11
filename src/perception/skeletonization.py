"""Rope skeletonization for path extraction.

This module extracts the centerline/skeleton of the rope.
"""

from typing import Tuple

import cv2
import numpy as np


def skeletonize_rope(mask: np.ndarray) -> np.ndarray:
    """Extract skeleton from rope mask.

    Args:
        mask: Binary mask of the rope

    Returns:
        Binary mask of the skeleton
    """
    # TODO: Implement skeletonization using cv2 or scikit-image
    skeleton = np.zeros_like(mask)
    return skeleton


def extract_path(skeleton: np.ndarray) -> np.ndarray:
    """Extract ordered path from skeleton.

    Args:
        skeleton: Binary skeleton mask

    Returns:
        Array of (x, y) coordinates representing the rope path
    """
    # TODO: Implement path extraction
    return np.array([])
