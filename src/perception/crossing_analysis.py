"""Crossing over/under analysis using gradient rope color.

This module determines which strand is on top vs. bottom at each
crossing by sampling the original color image.  The visible color
at the crossing center belongs to the top (over) strand.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from typing import Optional

import cv2
import numpy as np

from src.perception.keypoint_detection import Keypoint

logger = logging.getLogger(__name__)

BINARY_THRESHOLD = 127
MAX_BRANCH_TRACE = 200


@dataclass
class CrossingInfo:
    """Over/under classification for a single crossing.

    Attributes:
        position: Crossing center (x, y) in image coordinates.
        over_branch_indices: Indices (into the local branch list) of
            the two branches that form the top strand.
        under_branch_indices: Indices of the two bottom-strand branches.
        over_color_hsv: Mean HSV color of the top strand.
        under_color_hsv: Mean HSV color of the bottom strand.
        center_color_hsv: Mean HSV color sampled at the crossing center.
        confidence: Classification confidence (0.0 – 1.0).
    """

    position: tuple[float, float]
    over_branch_indices: tuple[int, int]
    under_branch_indices: tuple[int, int]
    over_color_hsv: tuple[float, float, float]
    under_color_hsv: tuple[float, float, float]
    center_color_hsv: tuple[float, float, float]
    confidence: float


# ── helpers ──────────────────────────────────────────────────────────


def _circular_mean_hue(hues: np.ndarray) -> float:
    """Compute circular mean of OpenCV hue values (0-180).

    Args:
        hues: 1-D array of hue values in [0, 180).

    Returns:
        Circular mean hue in [0, 180).
    """
    if len(hues) == 0:
        return 0.0
    rads = hues.astype(np.float64) * (np.pi / 90.0)  # 0-180 → 0-2π
    sin_sum = np.sum(np.sin(rads))
    cos_sum = np.sum(np.cos(rads))
    mean_rad = np.arctan2(sin_sum, cos_sum)
    mean_hue = mean_rad * (90.0 / np.pi)  # back to 0-180
    return float(mean_hue % 180.0)


def _hsv_color_distance(
    c1: tuple[float, float, float],
    c2: tuple[float, float, float],
    hue_weight: float = 1.0,
    saturation_weight: float = 0.5,
    value_weight: float = 0.3,
) -> float:
    """Weighted HSV distance with circular hue handling.

    Args:
        c1: HSV tuple (H 0-180, S 0-255, V 0-255).
        c2: HSV tuple.
        hue_weight: Weight for hue channel.
        saturation_weight: Weight for saturation channel.
        value_weight: Weight for value channel.

    Returns:
        Scalar distance (lower is more similar).
    """
    h_diff = abs(c1[0] - c2[0])
    h_diff = min(h_diff, 180.0 - h_diff)  # circular
    s_diff = abs(c1[1] - c2[1])
    v_diff = abs(c1[2] - c2[2])
    return (
        hue_weight * (h_diff / 180.0)
        + saturation_weight * (s_diff / 255.0)
        + value_weight * (v_diff / 255.0)
    )


def _trace_branch_pixels(
    skeleton: np.ndarray,
    start: tuple[int, int],
    cluster: set[tuple[int, int]],
    max_length: int,
) -> list[tuple[int, int]]:
    """Trace skeleton pixels along a branch away from a junction cluster.

    Args:
        skeleton: Binary skeleton (uint8, 0/255).
        start: Starting pixel (row, col), adjacent to the cluster.
        cluster: Set of (row, col) pixels belonging to the junction.
        max_length: Maximum number of pixels to trace.

    Returns:
        List of (row, col) pixel coordinates along the branch.
    """
    skeleton_bin = (skeleton > BINARY_THRESHOLD).astype(np.uint8)
    h, w = skeleton_bin.shape
    path = [start]
    visited = set(cluster)
    visited.add(start)
    current = start

    for _ in range(min(max_length, MAX_BRANCH_TRACE) - 1):
        neighbors = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = current[0] + dr, current[1] + dc
                if 0 <= nr < h and 0 <= nc < w:
                    if skeleton_bin[nr, nc] and (nr, nc) not in visited:
                        neighbors.append((nr, nc))
        if not neighbors:
            break
        # Prefer the single continuation pixel; stop at branches.
        nxt = neighbors[0]
        visited.add(nxt)
        path.append(nxt)
        current = nxt
        if len(neighbors) > 1:
            break  # hit another junction or branch

    return path


def _sample_branch_color(
    image_hsv: np.ndarray,
    rope_mask: np.ndarray,
    pixels: list[tuple[int, int]],
    radius: int,
) -> tuple[float, float, float]:
    """Sample mean HSV color around a set of skeleton pixels.

    Args:
        image_hsv: HSV image (uint8).
        rope_mask: Binary rope mask (uint8, 0/255).
        pixels: List of (row, col) skeleton coordinates.
        radius: Neighborhood radius for sampling.

    Returns:
        (mean_H, mean_S, mean_V) tuple.
    """
    h, w = image_hsv.shape[:2]
    hues: list[float] = []
    sats: list[float] = []
    vals: list[float] = []

    mask_bin = rope_mask > BINARY_THRESHOLD

    for row, col in pixels:
        r0 = max(0, row - radius)
        r1 = min(h, row + radius + 1)
        c0 = max(0, col - radius)
        c1 = min(w, col + radius + 1)
        patch = image_hsv[r0:r1, c0:c1]
        mask_patch = mask_bin[r0:r1, c0:c1]
        if np.any(mask_patch):
            hues.extend(patch[mask_patch, 0].astype(float).tolist())
            sats.extend(patch[mask_patch, 1].astype(float).tolist())
            vals.extend(patch[mask_patch, 2].astype(float).tolist())

    if not hues:
        return (0.0, 0.0, 0.0)

    mean_h = _circular_mean_hue(np.array(hues))
    mean_s = float(np.mean(sats))
    mean_v = float(np.mean(vals))
    return (mean_h, mean_s, mean_v)


def _pair_branches_by_color(
    colors: list[tuple[float, float, float]],
    hue_weight: float = 1.0,
    saturation_weight: float = 0.5,
    value_weight: float = 0.3,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Pair 4 branches into 2 pairs minimising intra-pair color distance.

    Args:
        colors: List of 4 HSV color tuples.
        hue_weight: Weight for hue in distance.
        saturation_weight: Weight for saturation in distance.
        value_weight: Weight for value in distance.

    Returns:
        Two tuples of branch indices: (pair_a, pair_b).
    """
    indices = list(range(4))
    # 3 possible pairings of 4 items into 2 pairs
    pairings: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for pair_a in combinations(indices, 2):
        pair_b = tuple(i for i in indices if i not in pair_a)
        pairings.append((pair_a, tuple(pair_b)))  # type: ignore[arg-type]

    best_cost = float("inf")
    best_pairing = pairings[0]
    for pair_a, pair_b in pairings:
        cost_a = _hsv_color_distance(
            colors[pair_a[0]], colors[pair_a[1]],
            hue_weight, saturation_weight, value_weight,
        )
        cost_b = _hsv_color_distance(
            colors[pair_b[0]], colors[pair_b[1]],
            hue_weight, saturation_weight, value_weight,
        )
        total = cost_a + cost_b
        if total < best_cost:
            best_cost = total
            best_pairing = (pair_a, pair_b)

    return best_pairing


# ── main entry point ─────────────────────────────────────────────────


def analyze_crossing_over_under(
    image: np.ndarray,
    skeleton: np.ndarray,
    rope_mask: np.ndarray,
    crossings: list[Keypoint],
    config: Optional[dict] = None,
) -> list[CrossingInfo]:
    """Classify over/under for each crossing using gradient rope color.

    Args:
        image: Original BGR frame.
        skeleton: Binary skeleton mask (uint8, 0/255).
        rope_mask: Binary rope mask (uint8, 0/255).
        crossings: List of crossing ``Keypoint`` objects.
        config: Optional ``crossing_analysis`` config dict.

    Returns:
        List of ``CrossingInfo`` objects, one per analysed crossing.
    """
    if config is None:
        config = {}

    color_sample_length = int(config.get("color_sample_length", 15))
    color_sample_radius = int(config.get("color_sample_radius", 2))
    center_sample_radius = int(config.get("center_sample_radius", 4))
    junction_search_radius = int(
        config.get("junction_search_radius", 10)
    )
    min_branches = int(config.get("min_branches", 4))
    min_confidence = float(config.get("min_confidence", 0.2))
    hue_w = float(config.get("hue_weight", 1.0))
    sat_w = float(config.get("saturation_weight", 0.5))
    val_w = float(config.get("value_weight", 0.3))

    if image is None or skeleton is None or rope_mask is None:
        return []

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    skeleton_bin = (skeleton > BINARY_THRESHOLD).astype(np.uint8)
    h, w = skeleton_bin.shape

    # Pre-compute junction mask (pixels with >=3 neighbors)
    kernel = np.array(
        [[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8
    )
    neighbor_count = cv2.filter2D(
        skeleton_bin,
        -1,
        kernel,
        borderType=cv2.BORDER_CONSTANT,
    )
    junction_mask = (neighbor_count >= 3) & (skeleton_bin > 0)

    results: list[CrossingInfo] = []

    for crossing in crossings:
        cx, cy = crossing.position
        cr, cc = int(round(cy)), int(round(cx))

        # Find the junction cluster near the crossing position
        r0 = max(0, cr - junction_search_radius)
        r1 = min(h, cr + junction_search_radius + 1)
        c0 = max(0, cc - junction_search_radius)
        c1 = min(w, cc + junction_search_radius + 1)

        cluster: set[tuple[int, int]] = set()
        for r in range(r0, r1):
            for c in range(c0, c1):
                if junction_mask[r, c]:
                    cluster.add((r, c))

        if not cluster:
            continue

        # Find branch-start pixels adjacent to the cluster
        branch_starts: list[tuple[int, int]] = []
        for row, col in cluster:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        if (
                            skeleton_bin[nr, nc]
                            and (nr, nc) not in cluster
                            and not junction_mask[nr, nc]
                        ):
                            if (nr, nc) not in branch_starts:
                                branch_starts.append((nr, nc))

        if len(branch_starts) < min_branches:
            continue

        # Trace each branch and sample color
        branch_colors: list[tuple[float, float, float]] = []
        branch_pixels_list: list[list[tuple[int, int]]] = []
        for bs in branch_starts:
            pixels = _trace_branch_pixels(
                skeleton, bs, cluster, color_sample_length
            )
            color = _sample_branch_color(
                image_hsv, rope_mask, pixels, color_sample_radius
            )
            branch_colors.append(color)
            branch_pixels_list.append(pixels)

        # If more than 4 branches, keep the 4 with the most traced
        # pixels (most reliable color samples).
        if len(branch_colors) > 4:
            ranked = sorted(
                range(len(branch_colors)),
                key=lambda i: len(branch_pixels_list[i]),
                reverse=True,
            )[:4]
            branch_colors = [branch_colors[i] for i in ranked]
            branch_starts = [branch_starts[i] for i in ranked]

        if len(branch_colors) < 4:
            continue

        # Sample crossing center color
        cr0 = max(0, cr - center_sample_radius)
        cr1 = min(h, cr + center_sample_radius + 1)
        cc0 = max(0, cc - center_sample_radius)
        cc1 = min(w, cc + center_sample_radius + 1)
        center_patch = image_hsv[cr0:cr1, cc0:cc1]
        mask_patch = rope_mask[cr0:cr1, cc0:cc1] > BINARY_THRESHOLD
        if np.any(mask_patch):
            center_hues = center_patch[mask_patch, 0].astype(float)
            center_h = _circular_mean_hue(center_hues)
            center_s = float(np.mean(center_patch[mask_patch, 1]))
            center_v = float(np.mean(center_patch[mask_patch, 2]))
        else:
            center_h = float(image_hsv[cr, cc, 0])
            center_s = float(image_hsv[cr, cc, 1])
            center_v = float(image_hsv[cr, cc, 2])
        center_color = (center_h, center_s, center_v)

        # Pair branches by color similarity
        pair_a, pair_b = _pair_branches_by_color(
            branch_colors, hue_w, sat_w, val_w
        )

        # Mean color per pair
        def _pair_mean(
            pair: tuple[int, int],
        ) -> tuple[float, float, float]:
            hues = np.array(
                [branch_colors[pair[0]][0], branch_colors[pair[1]][0]]
            )
            mean_h = _circular_mean_hue(hues)
            mean_s = (
                branch_colors[pair[0]][1] + branch_colors[pair[1]][1]
            ) / 2.0
            mean_v = (
                branch_colors[pair[0]][2] + branch_colors[pair[1]][2]
            ) / 2.0
            return (mean_h, mean_s, mean_v)

        mean_a = _pair_mean(pair_a)
        mean_b = _pair_mean(pair_b)

        # The pair closer to center color is the over (top) strand
        dist_a = _hsv_color_distance(
            mean_a, center_color, hue_w, sat_w, val_w
        )
        dist_b = _hsv_color_distance(
            mean_b, center_color, hue_w, sat_w, val_w
        )

        if dist_a <= dist_b:
            over_pair, under_pair = pair_a, pair_b
            over_color, under_color = mean_a, mean_b
        else:
            over_pair, under_pair = pair_b, pair_a
            over_color, under_color = mean_b, mean_a

        # Confidence: how distinguishable the two pairs are
        total_dist = dist_a + dist_b
        if total_dist > 0:
            separation = abs(dist_a - dist_b) / total_dist
        else:
            separation = 0.0
        confidence = min(1.0, separation)

        if confidence < min_confidence:
            continue

        results.append(
            CrossingInfo(
                position=(cx, cy),
                over_branch_indices=over_pair,
                under_branch_indices=under_pair,
                over_color_hsv=over_color,
                under_color_hsv=under_color,
                center_color_hsv=center_color,
                confidence=confidence,
            )
        )

    return results
