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
DEFAULT_MERGE_DISTANCE = 6.0
DEFAULT_CROSSING_MIN_AREA = 8
DEFAULT_CROSSING_MIN_NEIGHBORS = 3
DEFAULT_CROSSING_MIN_BRANCH_LENGTH = 8
DEFAULT_CROSSING_MIN_BRANCH_COUNT = 3

MAX_BRANCH_TRACE_STEPS = 200


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


def _find_contours(mask: np.ndarray) -> List[np.ndarray]:
    """Find contours in the binary mask."""
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    return contours if contours else []


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
    contours = _find_contours(mask)
    if not contours:
        return []

    points_list = [_sample_contour_points(contour) for contour in contours]
    points = np.vstack(points_list) if points_list else np.empty((0, 2))
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
    min_area: int,
    min_neighbor_count: int,
    min_branch_length: int,
    min_branch_count: int,
) -> List[Keypoint]:
    """Detect crossings from skeleton junction pixels."""
    if skeleton.size == 0 or np.count_nonzero(skeleton) == 0:
        return []

    neighbor_count = _count_neighbors(skeleton)
    junction_mask = (neighbor_count >= min_neighbor_count) & (
        skeleton > BINARY_THRESHOLD
    )

    if np.count_nonzero(junction_mask) == 0:
        return []

    junction_uint8 = (junction_mask.astype(np.uint8) * BINARY_MAX)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        junction_uint8,
        connectivity=8,
    )

    skeleton_binary = (skeleton > BINARY_THRESHOLD).astype(np.uint8)
    endpoint_mask = (neighbor_count == 1) & (skeleton_binary > 0)

    def _branch_reaches_length(
        start_rc: tuple[int, int],
        cluster_pixels: set[tuple[int, int]],
    ) -> bool:
        """Check if a branch reaches a minimum length away from a junction."""
        prev = None
        current = start_rc
        length = 0

        while length < min_branch_length and length < MAX_BRANCH_TRACE_STEPS:
            length += 1

            if endpoint_mask[current]:
                break

            neighbors = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = current[0] + dr, current[1] + dc
                    if nr < 0 or nr >= skeleton_binary.shape[0]:
                        continue
                    if nc < 0 or nc >= skeleton_binary.shape[1]:
                        continue
                    if skeleton_binary[nr, nc] == 0:
                        continue
                    if prev is not None and (nr, nc) == prev:
                        continue
                    if (nr, nc) in cluster_pixels:
                        continue
                    neighbors.append((nr, nc))

            if not neighbors:
                break
            if len(neighbors) != 1:
                # Hit a branch or ambiguous junction; count current length.
                break

            next_rc = neighbors[0]
            if junction_mask[next_rc]:
                # Reached another junction.
                break

            prev = current
            current = next_rc

        return length >= min_branch_length

    keypoints = []
    for label_idx in range(1, num_labels):
        area = float(stats[label_idx, cv2.CC_STAT_AREA])
        if area < float(min_area):
            continue

        # Collect cluster pixels for this junction.
        cluster_pixels = set(
            zip(
                *np.where(labels == label_idx)
            )
        )

        # Find branch starts: skeleton pixels adjacent to the cluster.
        branch_starts = set()
        for row, col in cluster_pixels:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = row + dr, col + dc
                    if nr < 0 or nr >= skeleton_binary.shape[0]:
                        continue
                    if nc < 0 or nc >= skeleton_binary.shape[1]:
                        continue
                    if skeleton_binary[nr, nc] == 0:
                        continue
                    if (nr, nc) in cluster_pixels:
                        continue
                    if junction_mask[nr, nc]:
                        continue
                    branch_starts.add((nr, nc))

        valid_branches = 0
        for start_rc in branch_starts:
            if _branch_reaches_length(start_rc, cluster_pixels):
                valid_branches += 1

        if valid_branches < min_branch_count:
            continue

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


def _merge_keypoints_by_distance(
    keypoints: List[Keypoint],
    merge_distance: float,
) -> List[Keypoint]:
    """Merge keypoints of the same type that are within merge_distance."""
    if not keypoints:
        return []

    merged: List[Keypoint] = []
    for kp in keypoints:
        found = False
        for idx, existing in enumerate(merged):
            if kp.keypoint_type != existing.keypoint_type:
                continue
            dx = kp.position[0] - existing.position[0]
            dy = kp.position[1] - existing.position[1]
            if (dx * dx + dy * dy) <= merge_distance * merge_distance:
                if kp.confidence > existing.confidence:
                    merged[idx] = kp
                found = True
                break
        if not found:
            merged.append(kp)

    return merged


def detect_keypoints(
    mask: np.ndarray,
    config: dict,
) -> List[Keypoint]:
    """Detect keypoints from a rope segmentation mask.

    Args:
        mask: Binary mask of the rope
        config: Configuration dictionary with detection parameters, including:
            - endpoint_detection: endpoint method and thresholds
            - crossing_detection: crossing method and thresholds
              (min_area, min_neighbor_count, min_branch_length, min_branch_count)
            - skeletonization: optional skeletonization config overrides

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
    endpoint_merge_distance = float(
        endpoint_config.get("merge_distance", DEFAULT_MERGE_DISTANCE)
    )

    crossing_method = crossing_config.get("method", "skeleton_intersection")
    crossing_min_conf = crossing_config.get("min_confidence", 0.6)
    crossing_min_area = int(
        crossing_config.get("min_area", DEFAULT_CROSSING_MIN_AREA)
    )
    crossing_min_neighbors = int(
        crossing_config.get("min_neighbor_count", DEFAULT_CROSSING_MIN_NEIGHBORS)
    )
    crossing_min_branch_length = int(
        crossing_config.get(
            "min_branch_length",
            DEFAULT_CROSSING_MIN_BRANCH_LENGTH,
        )
    )
    crossing_min_branch_count = int(
        crossing_config.get(
            "min_branch_count",
            DEFAULT_CROSSING_MIN_BRANCH_COUNT,
        )
    )

    keypoints: List[Keypoint] = []
    skeletonization_config = config.get("skeletonization", {})

    skeleton: Optional[np.ndarray] = None
    needs_skeleton = crossing_method == "skeleton_intersection" or endpoint_method in (
        "skeleton_endpoints",
        "combined",
    )
    if needs_skeleton:
        skeleton = skeletonize_rope(mask, config=skeletonization_config)

    endpoint_keypoints: List[Keypoint] = []
    if endpoint_method == "contour_analysis":
        endpoint_keypoints.extend(
            _detect_endpoints_from_contour(mask, min_confidence=endpoint_min_conf)
        )
        if not endpoint_keypoints and skeleton is not None:
            endpoint_keypoints.extend(
                _detect_endpoints_from_skeleton(
                    skeleton,
                    min_confidence=endpoint_min_conf,
                )
            )
    elif endpoint_method == "skeleton_endpoints":
        if skeleton is not None:
            endpoint_keypoints.extend(
                _detect_endpoints_from_skeleton(
                    skeleton,
                    min_confidence=endpoint_min_conf,
                )
            )
        if not endpoint_keypoints:
            endpoint_keypoints.extend(
                _detect_endpoints_from_contour(
                    mask,
                    min_confidence=endpoint_min_conf,
                )
            )
    elif endpoint_method == "combined":
        endpoint_keypoints.extend(
            _detect_endpoints_from_contour(mask, min_confidence=endpoint_min_conf)
        )
        if skeleton is not None:
            endpoint_keypoints.extend(
                _detect_endpoints_from_skeleton(
                    skeleton,
                    min_confidence=endpoint_min_conf,
                )
            )
        endpoint_keypoints = _merge_keypoints_by_distance(
            endpoint_keypoints,
            merge_distance=endpoint_merge_distance,
        )
    else:
        endpoint_keypoints.extend(
            _detect_endpoints_from_contour(mask, min_confidence=endpoint_min_conf)
        )

    keypoints.extend(endpoint_keypoints)

    if crossing_method == "skeleton_intersection":
        if skeleton is None:
            skeleton = skeletonize_rope(mask, config=skeletonization_config)
        keypoints.extend(
            _detect_crossings_from_skeleton(
                skeleton,
                min_confidence=crossing_min_conf,
                min_area=crossing_min_area,
                min_neighbor_count=crossing_min_neighbors,
                min_branch_length=crossing_min_branch_length,
                min_branch_count=crossing_min_branch_count,
            )
        )

    return keypoints
