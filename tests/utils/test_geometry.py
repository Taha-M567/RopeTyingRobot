"""Tests for geometry utilities."""

import numpy as np
import pytest

from src.utils.geometry import (
    angle_between_vectors,
    compute_distance,
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
