"""Crossing over/under analysis using gradient rope color.

This module determines which strand is on top vs. bottom at each
crossing by sampling the original color image.  The visible color
at the crossing center belongs to the top (over) strand.

Detection strategy (Option B — color-guided):
1. Detect crossing *regions* from mask geometry (distance transform)
   rather than relying on skeleton junctions.
2. Find entry/exit points where the skeleton meets each region boundary.
3. Sample color along the skeleton outside the region (where it is clean).
4. Pair entries by color similarity, determine over/under from center color.
5. Patch the skeleton inside crossing regions with clean traced paths.
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


# ── dataclasses ─────────────────────────────────────────────────────


@dataclass
class CrossingRegion:
    """A detected crossing region from mask geometry.

    Attributes:
        center: (x, y) centroid of the region.
        bbox: (r0, c0, r1, c1) bounding box in row/col.
        region_mask: Boolean mask (full image size) of the region.
        normal_half_width: Estimated single-strand half-width.
    """

    center: tuple[float, float]
    bbox: tuple[int, int, int, int]
    region_mask: np.ndarray
    normal_half_width: float


@dataclass
class CrossingInfo:
    """Over/under classification for a single crossing.

    Attributes:
        position: Crossing center (x, y) in image coordinates.
        over_strand_path: (N, 2) traced path through the crossing
            for the top strand, in (row, col) coordinates.
        under_strand_path: (N, 2) traced path through the crossing
            for the bottom strand, in (row, col) coordinates.
        over_color_hsv: Mean HSV color of the top strand.
        under_color_hsv: Mean HSV color of the bottom strand.
        center_color_hsv: Mean HSV color sampled at the crossing center.
        confidence: Classification confidence (0.0 -- 1.0).
        region: The detected crossing region.
    """

    position: tuple[float, float]
    over_strand_path: np.ndarray
    under_strand_path: np.ndarray
    over_color_hsv: tuple[float, float, float]
    under_color_hsv: tuple[float, float, float]
    center_color_hsv: tuple[float, float, float]
    confidence: float
    region: CrossingRegion


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
    rads = hues.astype(np.float64) * (np.pi / 90.0)  # 0-180 -> 0-2pi
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
    excluded: set[tuple[int, int]],
    max_length: int,
) -> list[tuple[int, int]]:
    """Trace skeleton pixels along a branch away from excluded pixels.

    Args:
        skeleton: Binary skeleton (uint8, 0/255).
        start: Starting pixel (row, col), adjacent to the excluded set.
        excluded: Set of (row, col) pixels to avoid (e.g. crossing region).
        max_length: Maximum number of pixels to trace.

    Returns:
        List of (row, col) pixel coordinates along the branch.
    """
    skeleton_bin = (skeleton > BINARY_THRESHOLD).astype(np.uint8)
    h, w = skeleton_bin.shape
    path = [start]
    visited = set(excluded)
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
                    if (
                        skeleton_bin[nr, nc]
                        and (nr, nc) not in visited
                    ):
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
            hues.extend(
                patch[mask_patch, 0].astype(float).tolist()
            )
            sats.extend(
                patch[mask_patch, 1].astype(float).tolist()
            )
            vals.extend(
                patch[mask_patch, 2].astype(float).tolist()
            )

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
        pairings.append(
            (pair_a, tuple(pair_b))  # type: ignore[arg-type]
        )

    best_cost = float("inf")
    best_pairing = pairings[0]
    for pair_a, pair_b in pairings:
        cost_a = _hsv_color_distance(
            colors[pair_a[0]],
            colors[pair_a[1]],
            hue_weight,
            saturation_weight,
            value_weight,
        )
        cost_b = _hsv_color_distance(
            colors[pair_b[0]],
            colors[pair_b[1]],
            hue_weight,
            saturation_weight,
            value_weight,
        )
        total = cost_a + cost_b
        if total < best_cost:
            best_cost = total
            best_pairing = (pair_a, pair_b)

    return best_pairing


# ── Step 1: Detect crossing regions from mask geometry ───────────


def detect_crossing_regions(
    mask: np.ndarray,
    config: Optional[dict] = None,
) -> list[CrossingRegion]:
    """Detect crossing regions using the distance transform.

    At a crossing, two rope strands overlap and the mask is roughly
    twice as wide as a single strand.  The distance transform peaks
    at these locations.

    Args:
        mask: Binary rope mask (uint8, 0/255).
        config: Optional ``crossing_analysis`` config dict with keys
            ``width_ratio`` and ``min_crossing_area``.

    Returns:
        List of ``CrossingRegion`` objects.
    """
    if config is None:
        config = {}

    width_ratio = float(config.get("width_ratio", 1.4))
    min_area = int(config.get("min_crossing_area", 30))

    if mask is None or mask.size == 0:
        return []

    mask_bin = (mask > BINARY_THRESHOLD).astype(np.uint8)
    if np.count_nonzero(mask_bin) == 0:
        return []

    # Distance transform: distance from each mask pixel to background
    dist = cv2.distanceTransform(
        mask_bin * 255, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )

    # Estimate normal rope half-width = 75th percentile of distance
    # values.  The median is too low because many edge pixels have
    # distance ~1; the 75th percentile better captures the typical
    # single-strand center distance.
    rope_dists = dist[mask_bin > 0]
    if len(rope_dists) == 0:
        return []
    normal_half_width = float(np.percentile(rope_dists, 75))
    if normal_half_width < 1.0:
        return []

    # Crossing pixels: where distance > width_ratio * normal_half_width
    threshold = width_ratio * normal_half_width
    crossing_pixels = (dist > threshold).astype(np.uint8) * 255

    if np.count_nonzero(crossing_pixels) == 0:
        return []

    # Connected components of crossing pixels
    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(crossing_pixels, connectivity=8)
    )

    h, w = mask_bin.shape
    regions: list[CrossingRegion] = []
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        # Build region mask
        region_mask = labels == label_idx

        # Bounding box (row, col)
        x_stat = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y_stat = int(stats[label_idx, cv2.CC_STAT_TOP])
        w_stat = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        h_stat = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        bbox = (y_stat, x_stat, y_stat + h_stat, x_stat + w_stat)

        # Compactness check: a crossing region should be roughly
        # circular, not an elongated strip (which would indicate a
        # straight rope segment, not a crossing).
        cx_f, cy_f = centroids[label_idx]
        aspect = max(w_stat, h_stat) / max(1, min(w_stat, h_stat))
        if aspect > 4.0:
            continue

        regions.append(
            CrossingRegion(
                center=(cx_f, cy_f),
                bbox=bbox,
                region_mask=region_mask,
                normal_half_width=normal_half_width,
            )
        )

    return regions


# ── Step 2: Find entry/exit points at region boundaries ──────────


def _find_entry_points(
    skeleton: np.ndarray,
    crossing_region: CrossingRegion,
) -> list[tuple[int, int]]:
    """Find where the skeleton enters a crossing region.

    Skeleton pixels outside the region that are 8-connected neighbors
    of region boundary pixels are entry/exit points.

    Args:
        skeleton: Binary skeleton (uint8, 0/255).
        crossing_region: The crossing region to find entries for.

    Returns:
        List of (row, col) entry point coordinates.
    """
    skeleton_bin = (skeleton > BINARY_THRESHOLD).astype(np.uint8)
    h, w = skeleton_bin.shape
    region = crossing_region.region_mask

    # Dilate region by 1 pixel to find its boundary neighbors
    region_uint8 = region.astype(np.uint8) * 255
    dilated = cv2.dilate(
        region_uint8,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )
    # Boundary band: dilated minus original
    boundary_band = (dilated > 0) & (~region)

    # Entry points: skeleton pixels in the boundary band
    entry_mask = (skeleton_bin > 0) & boundary_band
    coords = np.column_stack(np.where(entry_mask))

    if len(coords) == 0:
        return []

    # Cluster nearby entry points and keep one per cluster.
    # The skeleton at a crossing is messy, so multiple pixels from
    # the same branch may touch the region boundary.  Use a merge
    # distance proportional to the rope width.
    entries: list[tuple[int, int]] = []
    used = set()
    merge_dist = max(5, int(crossing_region.normal_half_width * 2))
    for row, col in coords:
        if (row, col) in used:
            continue
        # Mark nearby points as used
        for row2, col2 in coords:
            dr = abs(row - row2)
            dc = abs(col - col2)
            if dr <= merge_dist and dc <= merge_dist:
                used.add((row2, col2))
        entries.append((int(row), int(col)))

    return entries


# ── Step 3 & 4: Analyze crossing over/under ──────────────────────


def _interpolate_path(
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    center: tuple[int, int],
) -> np.ndarray:
    """Interpolate a smooth path from pt1 through center to pt2.

    Returns an (N, 2) array of (row, col) coordinates.

    Args:
        pt1: Start point (row, col).
        pt2: End point (row, col).
        center: Center point (row, col) to pass through.

    Returns:
        Array of interpolated (row, col) coordinates.
    """
    # Use 3 control points and simple linear interpolation
    points = np.array([pt1, center, pt2], dtype=np.float64)

    # Generate path by interpolating between segments
    path_pixels: list[tuple[int, int]] = []
    for seg_idx in range(len(points) - 1):
        r0, c0 = points[seg_idx]
        r1, c1 = points[seg_idx + 1]
        n_steps = max(
            2, int(np.hypot(r1 - r0, c1 - c0))
        )
        for t_idx in range(n_steps):
            t = t_idx / max(1, n_steps - 1)
            r = r0 + t * (r1 - r0)
            c = c0 + t * (c1 - c0)
            rc = (int(round(r)), int(round(c)))
            if not path_pixels or path_pixels[-1] != rc:
                path_pixels.append(rc)

    # Add final point
    final = (int(round(points[-1, 0])), int(round(points[-1, 1])))
    if not path_pixels or path_pixels[-1] != final:
        path_pixels.append(final)

    return np.array(path_pixels, dtype=np.int32)


def analyze_crossing_over_under(
    image: np.ndarray,
    skeleton: np.ndarray,
    rope_mask: np.ndarray,
    config: Optional[dict] = None,
) -> list[CrossingInfo]:
    """Classify over/under for each crossing using gradient rope color.

    This function detects crossing regions internally via
    ``detect_crossing_regions`` (mask geometry), then uses color
    continuity to pair entry/exit points and determine over/under.

    Args:
        image: Original BGR frame.
        skeleton: Binary skeleton mask (uint8, 0/255).
        rope_mask: Binary rope mask (uint8, 0/255).
        config: Optional ``crossing_analysis`` config dict.

    Returns:
        List of ``CrossingInfo`` objects, one per analysed crossing.
    """
    if config is None:
        config = {}

    color_sample_length = int(config.get("color_sample_length", 15))
    color_sample_radius = int(config.get("color_sample_radius", 2))
    center_sample_radius = int(
        config.get("center_sample_radius", 4)
    )
    min_confidence = float(config.get("min_confidence", 0.2))
    hue_w = float(config.get("hue_weight", 1.0))
    sat_w = float(config.get("saturation_weight", 0.5))
    val_w = float(config.get("value_weight", 0.3))

    if image is None or skeleton is None or rope_mask is None:
        return []

    # Step 1: detect crossing regions from mask geometry
    crossing_regions = detect_crossing_regions(rope_mask, config)
    if not crossing_regions:
        return []

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, w = skeleton.shape[:2]

    results: list[CrossingInfo] = []

    for region in crossing_regions:
        # Step 2: find entry/exit points (skeleton-only).
        # A real crossing always has 4 skeleton branches entering the
        # region.  Tight curves only have 2.  No fallback — if the
        # skeleton doesn't give 4 entries, this is not a crossing.
        entries = _find_entry_points(skeleton, region)

        if len(entries) < 4:
            logger.debug(
                "Crossing region at (%.0f, %.0f): "
                "only %d skeleton entry points, skipping "
                "(likely a tight curve)",
                region.center[0],
                region.center[1],
                len(entries),
            )
            continue

        # Step 3: sample color at each entry point
        region_pixels = set(
            zip(*np.where(region.region_mask))
        )
        branch_colors: list[tuple[float, float, float]] = []
        branch_pixels_list: list[list[tuple[int, int]]] = []
        for entry in entries:
            pixels = _trace_branch_pixels(
                skeleton, entry, region_pixels,
                color_sample_length,
            )
            color = _sample_branch_color(
                image_hsv, rope_mask, pixels, color_sample_radius
            )
            branch_colors.append(color)
            branch_pixels_list.append(pixels)

        # Keep the 4 entries with the most traced pixels
        if len(branch_colors) > 4:
            ranked = sorted(
                range(len(branch_colors)),
                key=lambda i: len(branch_pixels_list[i]),
                reverse=True,
            )[:4]
            branch_colors = [branch_colors[i] for i in ranked]
            entries = [entries[i] for i in ranked]

        if len(branch_colors) < 4:
            continue

        # Step 4: pair branches by angular opposition.
        # At a simple crossing the two branches of the same strand
        # are roughly 180 degrees apart when viewed from the center.
        # Sort entry points by angle, then take the alternating
        # pairing: (0, 2) and (1, 3).
        cr_center = int(round(region.center[1]))
        cc_center = int(round(region.center[0]))
        angles = [
            np.arctan2(e[0] - cr_center, e[1] - cc_center)
            for e in entries
        ]
        sorted_indices = sorted(range(4), key=lambda i: angles[i])
        pair_a = (sorted_indices[0], sorted_indices[2])
        pair_b = (sorted_indices[1], sorted_indices[3])

        # Sample crossing center color
        cr = int(round(region.center[1]))
        cc = int(round(region.center[0]))
        cr0 = max(0, cr - center_sample_radius)
        cr1 = min(h, cr + center_sample_radius + 1)
        cc0 = max(0, cc - center_sample_radius)
        cc1 = min(w, cc + center_sample_radius + 1)
        center_patch = image_hsv[cr0:cr1, cc0:cc1]
        mask_patch = rope_mask[cr0:cr1, cc0:cc1] > BINARY_THRESHOLD
        if np.any(mask_patch):
            center_hues = center_patch[mask_patch, 0].astype(float)
            center_h = _circular_mean_hue(center_hues)
            center_s = float(
                np.mean(center_patch[mask_patch, 1])
            )
            center_v = float(
                np.mean(center_patch[mask_patch, 2])
            )
        else:
            cr_c = min(max(cr, 0), h - 1)
            cc_c = min(max(cc, 0), w - 1)
            center_h = float(image_hsv[cr_c, cc_c, 0])
            center_s = float(image_hsv[cr_c, cc_c, 1])
            center_v = float(image_hsv[cr_c, cc_c, 2])
        center_color = (center_h, center_s, center_v)

        # Mean color per pair
        def _pair_mean(
            pair: tuple[int, int],
        ) -> tuple[float, float, float]:
            hues = np.array(
                [
                    branch_colors[pair[0]][0],
                    branch_colors[pair[1]][0],
                ]
            )
            mean_h = _circular_mean_hue(hues)
            mean_s = (
                branch_colors[pair[0]][1]
                + branch_colors[pair[1]][1]
            ) / 2.0
            mean_v = (
                branch_colors[pair[0]][2]
                + branch_colors[pair[1]][2]
            ) / 2.0
            return (mean_h, mean_s, mean_v)

        mean_a = _pair_mean(pair_a)
        mean_b = _pair_mean(pair_b)

        # Pair closer to center color = over (top) strand
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

        # Generate traced paths through the crossing center
        center_rc = (cr, cc)
        over_path = _interpolate_path(
            entries[over_pair[0]],
            entries[over_pair[1]],
            center_rc,
        )
        under_path = _interpolate_path(
            entries[under_pair[0]],
            entries[under_pair[1]],
            center_rc,
        )

        results.append(
            CrossingInfo(
                position=(region.center[0], region.center[1]),
                over_strand_path=over_path,
                under_strand_path=under_path,
                over_color_hsv=over_color,
                under_color_hsv=under_color,
                center_color_hsv=center_color,
                confidence=confidence,
                region=region,
            )
        )

    return results


# ── Step 5: Patch skeleton at crossings ──────────────────────────


def patch_skeleton_at_crossings(
    skeleton: np.ndarray,
    crossing_results: list[CrossingInfo],
) -> np.ndarray:
    """Replace messy skeleton inside crossing regions with clean paths.

    Args:
        skeleton: Binary skeleton mask (uint8, 0/255).
        crossing_results: Results from ``analyze_crossing_over_under``.

    Returns:
        Patched skeleton with clean junction paths.
    """
    patched = skeleton.copy()
    h, w = patched.shape

    for ci in crossing_results:
        # Zero out skeleton inside the crossing region
        patched[ci.region.region_mask] = 0

        # Draw the over and under strand paths as 1-pixel lines
        for path in (ci.over_strand_path, ci.under_strand_path):
            if path is None or len(path) < 2:
                continue
            for i in range(len(path) - 1):
                r0, c0 = int(path[i, 0]), int(path[i, 1])
                r1, c1 = int(path[i + 1, 0]), int(path[i + 1, 1])
                cv2.line(
                    patched,
                    (c0, r0),
                    (c1, r1),
                    255,
                    1,
                )

    return patched
