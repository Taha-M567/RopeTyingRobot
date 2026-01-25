"""Keypoint detection for rope endpoints and crossings.

This module extracts keypoints from segmented rope masks.
"""

from dataclasses import dataclass
import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.perception.skeletonization import skeletonize_rope

logger = logging.getLogger(__name__)

BINARY_THRESHOLD = 127
BINARY_MAX = 255
MAX_CONTOUR_SAMPLE_POINTS = 2000
CROSSING_SIZE_SCALE = 10.0
MIN_ENDPOINTS_FOR_PAIR = 2


@dataclass
class Keypoint:
    """Represents a keypoint on the rope.

    Attributes:
        position: (x, y) coordinates in image space
        keypoint_type: Type of keypoint ('endpoint', 'crossing', 'knot')
        confidence: Detection confidence (0.0 to 1.0)
    """

    position: Tuple[float, float]
    keypoint_type: str
    confidence: float


def _ensure_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Ensure the mask is uint8 binary with values 0 or 255."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    unique_vals = np.unique(mask)
    if not all(v in [0, BINARY_MAX] for v in unique_vals):
        mask = ((mask > BINARY_THRESHOLD) * BINARY_MAX).astype(np.uint8)

    return mask


def _count_neighbors(skeleton: np.ndarray) -> np.ndarray:
    """Count 8-connected neighbors for each skeleton pixel."""
    skeleton_binary = (skeleton > BINARY_THRESHOLD).astype(np.uint8)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(
        skeleton_binary,
        -1,
        kernel,
        borderType=cv2.BORDER_CONSTANT,
    )
    return neighbor_count


def _find_largest_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    """Find the largest contour in the binary mask."""
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _sample_contour_points(contour: np.ndarray) -> np.ndarray:
    """Sample contour points to limit computation."""
    points = contour.reshape(-1, 2)
    if len(points) <= MAX_CONTOUR_SAMPLE_POINTS:
        return points
    step = max(1, len(points) // MAX_CONTOUR_SAMPLE_POINTS)
    return points[::step]


def _farthest_point_pair(
    points: np.ndarray,
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Find the farthest pair of points in a set."""
    if len(points) < MIN_ENDPOINTS_FOR_PAIR:
        return None

    max_distance = -1.0
    farthest_pair = None

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dx = float(points[i, 0] - points[j, 0])
            dy = float(points[i, 1] - points[j, 1])
            distance = dx * dx + dy * dy
            if distance > max_distance:
                max_distance = distance
                farthest_pair = (
                    (float(points[i, 0]), float(points[i, 1])),
                    (float(points[j, 0]), float(points[j, 1])),
                )

    return farthest_pair


def _confidence_from_distance(
    distance: float,
    image_shape: Tuple[int, int],
) -> float:
    """Compute confidence based on endpoint separation and image size."""
    height, width = image_shape
    diag = float(np.hypot(height, width))
    if diag <= 0.0:
        return 0.0
    return min(1.0, max(0.0, distance / diag))


def _detect_endpoints_from_contour(
    mask: np.ndarray,
    min_confidence: float,
) -> List[Keypoint]:
    """Detect endpoints using contour analysis."""
    contour = _find_largest_contour(mask)
    if contour is None:
        return []

    points = _sample_contour_points(contour)
    pair = _farthest_point_pair(points)
    if pair is None:
        return []

    (x1, y1), (x2, y2) = pair
    distance = float(np.hypot(x1 - x2, y1 - y2))
    confidence = _confidence_from_distance(distance, mask.shape[:2])

    if confidence < min_confidence:
        return []

    return [
        Keypoint(position=(x1, y1), keypoint_type="endpoint", confidence=confidence),
        Keypoint(position=(x2, y2), keypoint_type="endpoint", confidence=confidence),
    ]


def _detect_endpoints_from_skeleton(
    skeleton: np.ndarray,
    min_confidence: float,
) -> List[Keypoint]:
    """Detect endpoints from skeleton pixels with one neighbor."""
    if skeleton.size == 0 or np.count_nonzero(skeleton) == 0:
        return []

    neighbor_count = _count_neighbors(skeleton)
    endpoints_mask = (neighbor_count == 1) & (skeleton > BINARY_THRESHOLD)
    coords = np.column_stack(np.where(endpoints_mask))

    keypoints = []
    for row, col in coords:
        confidence = 1.0
        if confidence < min_confidence:
            continue
        keypoints.append(
            Keypoint(
                position=(float(col), float(row)),
                keypoint_type="endpoint",
                confidence=confidence,
            )
        )

    return keypoints


def _detect_crossings_from_skeleton(
    skeleton: np.ndarray,
    min_confidence: float,
) -> List[Keypoint]:
    """Detect crossings from skeleton pixels with 3+ neighbors."""
    if skeleton.size == 0 or np.count_nonzero(skeleton) == 0:
        return []

    neighbor_count = _count_neighbors(skeleton)
    junction_mask = (neighbor_count >= 3) & (skeleton > BINARY_THRESHOLD)

    if np.count_nonzero(junction_mask) == 0:
        return []

    junction_uint8 = (junction_mask.astype(np.uint8) * BINARY_MAX)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        junction_uint8,
        connectivity=8,
    )

    keypoints = []
    for label_idx in range(1, num_labels):
        area = float(stats[label_idx, cv2.CC_STAT_AREA])
        x, y = centroids[label_idx]
        confidence = area / (area + CROSSING_SIZE_SCALE)
        if confidence < min_confidence:
            continue
        keypoints.append(
            Keypoint(
                position=(float(x), float(y)),
                keypoint_type="crossing",
                confidence=confidence,
            )
        )

    return keypoints


def detect_keypoints(
    mask: np.ndarray,
    config: dict,
) -> List[Keypoint]:
    """Detect keypoints from a rope segmentation mask.

    Args:
        mask: Binary mask of the rope
        config: Configuration dictionary with detection parameters

    Returns:
        List of Keypoint objects
    """
    if mask is None or not isinstance(mask, np.ndarray):
        logger.warning("Invalid mask input for keypoint detection")
        return []

    if mask.size == 0:
        logger.warning("Empty mask input for keypoint detection")
        return []

    if config is None:
        config = {}

    mask = _ensure_binary_mask(mask)

    endpoint_config = config.get("endpoint_detection", {})
    crossing_config = config.get("crossing_detection", {})

    endpoint_method = endpoint_config.get("method", "contour_analysis")
    endpoint_min_conf = endpoint_config.get("min_confidence", 0.7)

    crossing_method = crossing_config.get("method", "skeleton_intersection")
    crossing_min_conf = crossing_config.get("min_confidence", 0.6)

    keypoints: List[Keypoint] = []

    skeleton: Optional[np.ndarray] = None
    if crossing_method == "skeleton_intersection":
        skeleton = skeletonize_rope(mask, config={})

    if endpoint_method == "contour_analysis":
        keypoints.extend(
            _detect_endpoints_from_contour(mask, min_confidence=endpoint_min_conf)
        )
        if not keypoints and skeleton is not None:
            keypoints.extend(
                _detect_endpoints_from_skeleton(
                    skeleton,
                    min_confidence=endpoint_min_conf,
                )
            )
    else:
        if skeleton is None:
            skeleton = skeletonize_rope(mask, config={})
        keypoints.extend(
            _detect_endpoints_from_skeleton(
                skeleton,
                min_confidence=endpoint_min_conf,
            )
        )

    if crossing_method == "skeleton_intersection":
        if skeleton is None:
            skeleton = skeletonize_rope(mask, config={})
        keypoints.extend(
            _detect_crossings_from_skeleton(
                skeleton,
                min_confidence=crossing_min_conf,
            )
        )

    return keypoints
