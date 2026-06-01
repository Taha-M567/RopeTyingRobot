"""Tests for geometry utilities."""

import numpy as np
import pytest

from src.utils.geometry import (
    CameraPose,
    angle_between_vectors,
    compute_distance,
    pixel_to_table_world,
    transform_point,
)


def test_compute_distance():
    """Test distance computation."""
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1.0, 0.0, 0.0])

    distance = compute_distance(p1, p2)
    assert distance == pytest.approx(1.0)


def test_transform_point():
    """Test point transformation."""
    point = np.array([1.0, 0.0, 0.0])
    rotation = np.eye(3)
    translation = np.array([1.0, 1.0, 1.0])

    transformed = transform_point(point, rotation, translation)
    expected = np.array([2.0, 1.0, 1.0])

    np.testing.assert_array_almost_equal(transformed, expected)


def test_angle_between_vectors():
    """Test angle computation between vectors."""
    vec1 = np.array([1.0, 0.0])
    vec2 = np.array([0.0, 1.0])

    angle = angle_between_vectors(vec1, vec2)
    assert angle == pytest.approx(np.pi / 2)


def _top_down_intrinsics(
    width: int = 640,
    height: int = 480,
    fx: float = 733.0,
) -> np.ndarray:
    return np.array(
        [
            [fx, 0.0, width / 2.0],
            [0.0, fx, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def test_pixel_to_table_world_principal_point():
    """A pixel at the principal point of a top-down camera maps to the
    camera's XY directly above it on the table plane."""
    K = _top_down_intrinsics()
    # Top-down camera: rotated 180° about X so +Z_cam points into −Z_world.
    R = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )
    pose = CameraPose(
        position=np.array([0.25, 0.0, 0.50]),
        rotation=R,
    )
    uv = np.array([320.0, 240.0])  # principal point

    world = pixel_to_table_world(uv, K, pose, z_table=0.0)

    np.testing.assert_allclose(world, np.array([0.25, 0.0, 0.0]), atol=1e-9)


def test_pixel_to_table_world_offset_pixel():
    """An off-center pixel projects out along a known direction.

    Camera at (0.25, 0.0, 0.50) looking straight down, table at z=0.
    A pixel +100 px right of center should land at +100/fx * 0.5 m to
    the right in world X (because +X_cam = +X_world for this rotation).
    """
    K = _top_down_intrinsics(fx=500.0)
    R = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )
    pose = CameraPose(
        position=np.array([0.25, 0.0, 0.50]),
        rotation=R,
    )

    uv = np.array([320.0 + 100.0, 240.0])
    world = pixel_to_table_world(uv, K, pose, z_table=0.0)

    expected_dx = (100.0 / 500.0) * 0.50  # 0.10 m
    np.testing.assert_allclose(
        world, np.array([0.25 + expected_dx, 0.0, 0.0]), atol=1e-9,
    )


def test_pixel_to_table_world_batch():
    """Batch input returns ``(N, 3)`` with matching values."""
    K = _top_down_intrinsics()
    R = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )
    pose = CameraPose(
        position=np.array([0.25, 0.0, 0.50]),
        rotation=R,
    )
    uv = np.array([[320.0, 240.0], [320.0, 240.0]])

    world = pixel_to_table_world(uv, K, pose, z_table=0.0)

    assert world.shape == (2, 3)
    np.testing.assert_allclose(world[0], world[1])


def test_pixel_to_table_world_ray_away_from_plane_is_nan():
    """A camera pointed up at the sky cannot hit a table below it."""
    K = _top_down_intrinsics()
    # Identity rotation: camera +Z = world +Z, so ray exits up away from z=0.
    pose = CameraPose(
        position=np.array([0.0, 0.0, 0.50]),
        rotation=np.eye(3),
    )

    world = pixel_to_table_world(
        np.array([320.0, 240.0]), K, pose, z_table=0.0,
    )

    assert np.all(np.isnan(world))
