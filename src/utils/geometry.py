"""Geometric utilities for coordinate transformations and calculations.

This module provides helper functions for geometric operations.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class CameraPose:
    """Camera pose in the world (or robot-base) frame.

    Attributes:
        position: ``(3,)`` translation of the camera optical center.
        rotation: ``(3, 3)`` rotation matrix whose columns are the camera
            +X, +Y, +Z axes expressed in the world frame.  The convention
            assumed for back-projection is the OpenCV camera frame:
            +X right, +Y down, +Z forward (along the optical axis).
    """

    position: np.ndarray
    rotation: np.ndarray


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


def pixel_to_table_world(
    uv: np.ndarray,
    camera_matrix: np.ndarray,
    camera_pose: CameraPose,
    z_table: float,
) -> np.ndarray:
    """Back-project pixel coordinates to a horizontal table plane.

    Assumes a pinhole camera (no distortion: pre-undistort the image
    first if needed).  The image ray is computed in the camera frame
    using the OpenCV convention (+Z forward), rotated into the world
    frame, then intersected with the plane ``z = z_table``.

    Args:
        uv: Pixel coordinates, either shape ``(2,)`` for a single point
            or ``(N, 2)`` for a batch.  Ordered as ``(u, v)``.
        camera_matrix: ``(3, 3)`` intrinsic matrix containing fx, fy,
            cx, cy in the standard OpenCV layout.
        camera_pose: Pose of the camera in the same frame as
            ``z_table``.  ``rotation`` follows OpenCV convention.
        z_table: World-frame Z value of the table surface.

    Returns:
        World-frame XYZ coordinates of the ray–plane intersection,
        shape ``(3,)`` for a single input or ``(N, 3)`` for a batch.
        Rays that point away from the plane produce ``nan`` entries
        (intersection is behind the camera).
    """
    single = uv.ndim == 1
    uv_2d = np.atleast_2d(uv).astype(np.float64)

    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])

    rays_cam = np.stack(
        [
            (uv_2d[:, 0] - cx) / fx,
            (uv_2d[:, 1] - cy) / fy,
            np.ones(uv_2d.shape[0], dtype=np.float64),
        ],
        axis=1,
    )

    rays_world = rays_cam @ camera_pose.rotation.T

    cam_z = float(camera_pose.position[2])
    dz = rays_world[:, 2]
    # Avoid divide-by-zero; rays parallel to the plane never intersect.
    safe_dz = np.where(np.abs(dz) < 1e-9, np.nan, dz)
    t = (z_table - cam_z) / safe_dz
    # Rays going the wrong way (t<=0) produce nan.
    t = np.where(t > 0, t, np.nan)

    world = camera_pose.position[None, :] + t[:, None] * rays_world

    return world[0] if single else world
