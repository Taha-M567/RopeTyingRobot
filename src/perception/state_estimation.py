"""Rope state estimation (crossings, knots, endpoints).

This module estimates the current state of the rope from perception data.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from src.perception.keypoint_detection import Keypoint


@dataclass
class RopeState:
    """Current state of the rope.

    Attributes:
        endpoints: List of endpoint positions
        crossings: List of crossing positions
        knots: List of knot positions
        path: Ordered path of rope centerline
    """

    endpoints: List[Tuple[float, float]]
    crossings: List[Tuple[float, float]]
    knots: List[Tuple[float, float]]
    path: np.ndarray


def estimate_rope_state(
    keypoints: List[Keypoint],
    path: np.ndarray,
) -> RopeState:
    """Estimate rope state from keypoints and path.

    Args:
        keypoints: List of detected keypoints
        path: Rope centerline path

    Returns:
        RopeState object with current rope configuration
    """
    endpoints = [
        kp.position
        for kp in keypoints
        if kp.keypoint_type == "endpoint"
    ]
    crossings = [
        kp.position
        for kp in keypoints
        if kp.keypoint_type == "crossing"
    ]
    knots = [
        kp.position
        for kp in keypoints
        if kp.keypoint_type == "knot"
    ]

    return RopeState(
        endpoints=endpoints,
        crossings=crossings,
        knots=knots,
        path=path,
    )
