"""Rope state estimation (crossings, knots, endpoints).

This module estimates the current state of the rope from perception data.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np

from src.perception.keypoint_detection import Keypoint
from src.perception.skeletonization import PathDict


@dataclass
class RopeState:
    """Current state of the rope.

    Attributes:
        endpoints: List of endpoint positions
        crossings: List of crossing positions
        knots: List of knot positions
        path: Ordered path of rope centerline (main_path from skeletonization)
        path_graph: Optional graph representation from skeletonization
    """

    endpoints: List[Tuple[float, float]]
    crossings: List[Tuple[float, float]]
    knots: List[Tuple[float, float]]
    path: np.ndarray
    path_graph: Optional[PathDict] = None


def estimate_rope_state(
    keypoints: List[Keypoint],
    path: Union[np.ndarray, PathDict],
) -> RopeState:
    """Estimate rope state from keypoints and path.

    Args:
        keypoints: List of detected keypoints
        path: Rope centerline path. Can be:
            - np.ndarray: Legacy format, (N, 2) array of (x, y) coordinates
            - PathDict: New format with keys "main_path", "endpoints",
              "junctions", "edges"

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

    path_graph: Optional[PathDict] = None
    # Extract path from new dictionary format or use legacy array
    if isinstance(path, dict):
        # New format: extract main_path if available
        path_graph = path
        if path.get("main_path") is not None:
            path_array = path["main_path"]
        elif len(path.get("edges", [])) > 0:
            # Use longest edge if no main_path
            edges = path["edges"]
            path_array = max(edges, key=len)
        else:
            # Empty path
            path_array = np.array([], dtype=np.float32).reshape(0, 2)
    else:
        # Legacy format: use array directly
        path_array = path

    return RopeState(
        endpoints=endpoints,
        crossings=crossings,
        knots=knots,
        path=path_array,
        path_graph=path_graph,
    )
