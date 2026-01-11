"""Geometric utilities for coordinate transformations and calculations.

This module provides helper functions for geometric operations.
"""

from typing import Tuple

import numpy as np


def transform_point(
    point: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Transform a point by rotation and translation.

    Args:
        point: Point to transform (x, y, z)
        rotation: 3x3 rotation matrix
        translation: Translation vector (x, y, z)

    Returns:
        Transformed point
    """
    return rotation @ point + translation


def compute_distance(
    point1: np.ndarray,
    point2: np.ndarray,
) -> float:
    """Compute Euclidean distance between two points.

    Args:
        point1: First point
        point2: Second point

    Returns:
        Euclidean distance
    """
    return np.linalg.norm(point1 - point2)


def angle_between_vectors(
    vec1: np.ndarray,
    vec2: np.ndarray,
) -> float:
    """Compute angle between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Angle in radians
    """
    cos_angle = np.dot(vec1, vec2) / (
        np.linalg.norm(vec1) * np.linalg.norm(vec2)
    )
    return np.arccos(np.clip(cos_angle, -1.0, 1.0))
