"""Visualization utilities for perception results.

This module provides functions to visualize rope perception results.
Note: Display functions are only called when explicitly requested.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from src.perception.keypoint_mask import (
    BACKGROUND_CLASS,
    CROSSING_CLASS,
    ENDPOINT_CLASS,
    ROPE_CLASS,
)
from src.perception.rope_segmentation import RopeMask
from src.perception.video_processor import ProcessingResult

if TYPE_CHECKING:
    from src.perception.crossing_analysis import CrossingInfo


def draw_rope_mask(
    image: np.ndarray,
    mask: RopeMask,
    alpha: float = 0.2,
) -> np.ndarray:
    """Draw rope mask overlay on image.

    Args:
        image: Original image (BGR format)
        mask: RopeMask object
        alpha: Transparency of mask overlay (0.0 to 1.0)

    Returns:
        Image with mask overlay
    """
    mask_colored = cv2.applyColorMap(mask.mask, cv2.COLORMAP_JET)
    mask_colored = cv2.bitwise_and(mask_colored, mask_colored, mask=mask.mask)

    # Blend only where the rope mask exists to avoid darkening the background.
    blended = cv2.addWeighted(image, 1.0 - alpha, mask_colored, alpha, 0)
    result = image.copy()
    rope_region = mask.mask > 0
    result[rope_region] = blended[rope_region]
    return result


def draw_keypoint_mask_overlay(
    image: np.ndarray,
    class_mask: np.ndarray,
    alpha: float = 0.5,
    rope_color: tuple = (255, 255, 255),
    endpoint_color: tuple = (0, 255, 0),
    crossing_color: tuple = (0, 0, 255),
) -> np.ndarray:
    """Overlay a color-coded keypoint mask on the image.

    Args:
        image: Image to draw on (BGR format)
        class_mask: Class-labeled mask (0=bg, 1=rope, 2=endpoint, 3=crossing)
        alpha: Transparency of overlay
        rope_color: BGR color for rope pixels
        endpoint_color: BGR color for endpoint pixels
        crossing_color: BGR color for crossing pixels

    Returns:
        Image with keypoint mask overlay
    """
    if class_mask is None or class_mask.size == 0:
        return image

    color_mask = np.zeros_like(image, dtype=np.uint8)
    color_mask[class_mask == ROPE_CLASS] = rope_color
    color_mask[class_mask == ENDPOINT_CLASS] = endpoint_color
    color_mask[class_mask == CROSSING_CLASS] = crossing_color

    # Only blend where we have rope pixels
    mask_region = (class_mask != BACKGROUND_CLASS)
    overlay = np.zeros_like(image, dtype=np.uint8)
    overlay[mask_region] = color_mask[mask_region]

    blended = cv2.addWeighted(image, 1.0 - alpha, overlay, alpha, 0)
    result = image.copy()
    result[mask_region] = blended[mask_region]
    return result


def draw_rope_path(
    image: np.ndarray,
    path: np.ndarray,
    color: tuple = (255, 0, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw rope path on image.

    Args:
        image: Image to draw on (BGR format)
        path: Array of (x, y) coordinates
        color: Path color (BGR tuple)
        thickness: Line thickness

    Returns:
        Image with path drawn
    """
    result = image.copy()

    if len(path) < 2:
        return result

    # Convert to integer coordinates
    points = path.astype(np.int32)

    # Draw path as connected line segments
    for i in range(len(points) - 1):
        pt1 = tuple(points[i])
        pt2 = tuple(points[i + 1])
        cv2.line(result, pt1, pt2, color, thickness)

    return result


def draw_rope_edges(
    image: np.ndarray,
    edges: list[np.ndarray],
    color: tuple = (255, 0, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw rope edges (graph paths) on image.

    Args:
        image: Image to draw on (BGR format)
        edges: List of (N, 2) arrays, each an edge path
        color: Path color (BGR tuple)
        thickness: Line thickness

    Returns:
        Image with edge paths drawn
    """
    result = image.copy()

    if not edges:
        return result

    for edge in edges:
        if edge is None or len(edge) < 2:
            continue
        points = edge.astype(np.int32)
        for i in range(len(points) - 1):
            pt1 = tuple(points[i])
            pt2 = tuple(points[i + 1])
            cv2.line(result, pt1, pt2, color, thickness)

    return result


def draw_crossing_over_under(
    image: np.ndarray,
    crossing_details: list[CrossingInfo],
    over_color: tuple[int, int, int] = (0, 255, 0),
    under_color: tuple[int, int, int] = (0, 0, 255),
    gap_radius: int = 8,
    thickness: int = 3,
) -> np.ndarray:
    """Draw knot-diagram-style over/under at each crossing.

    Uses the standard knot-diagram convention: the over (top) strand
    is drawn as a continuous line through the crossing, while the
    under (bottom) strand has a gap at the crossing center, making
    the topology unambiguous in a single image — suitable as an RL
    observation for a robot arm learning to untie ropes.

    Args:
        image: Image to draw on (BGR format).
        crossing_details: List of ``CrossingInfo`` objects.
        over_color: BGR color for the over (top) strand.
        under_color: BGR color for the under (bottom) strand.
        gap_radius: Pixel radius of the gap cut from the under
            strand at the crossing center.
        thickness: Line thickness for both strands.

    Returns:
        Image with crossing indicators drawn.
    """
    result = image.copy()
    for ci in crossing_details:
        cx = int(round(ci.position[0]))
        cy = int(round(ci.position[1]))

        over_path = getattr(ci, "over_strand_path", None)
        under_path = getattr(ci, "under_strand_path", None)

        # --- Under strand first (with gap at center) ---
        if under_path is not None and len(under_path) >= 2:
            pts = np.array(
                [[p[1], p[0]] for p in under_path],
                dtype=np.int32,
            )
            # Split into segments outside the gap radius
            before = []
            after = []
            past_gap = False
            for pt in pts:
                dist = np.hypot(pt[0] - cx, pt[1] - cy)
                if dist > gap_radius:
                    if not past_gap:
                        before.append(pt)
                    else:
                        after.append(pt)
                else:
                    past_gap = True
            if len(before) >= 2:
                cv2.polylines(
                    result,
                    [np.array(before, dtype=np.int32)],
                    False,
                    under_color,
                    thickness,
                )
            if len(after) >= 2:
                cv2.polylines(
                    result,
                    [np.array(after, dtype=np.int32)],
                    False,
                    under_color,
                    thickness,
                )

        # --- Over strand on top (continuous, no gap) ---
        if over_path is not None and len(over_path) >= 2:
            pts = np.array(
                [[p[1], p[0]] for p in over_path],
                dtype=np.int32,
            )
            cv2.polylines(
                result, [pts], False, over_color, thickness
            )

        # Small filled dot at crossing center
        cv2.circle(result, (cx, cy), 3, over_color, -1)

    return result


def visualize_result(
    result: ProcessingResult,
    show_mask: bool = True,
    show_keypoints: bool = True,
    show_path: bool = True,
) -> np.ndarray:
    """Create visualization of processing result.

    Args:
        result: ProcessingResult object
        show_mask: Whether to overlay segmentation mask
        show_keypoints: Whether to overlay keypoint mask
        show_path: Whether to draw rope path

    Returns:
        Visualized image
    """
    vis_image = result.frame.copy()

    if show_mask:
        vis_image = draw_rope_mask(vis_image, result.rope_mask)

    if show_path:
        path_graph = getattr(result.rope_state, "path_graph", None)
        if path_graph is not None and len(path_graph.get("edges", [])) > 0:
            vis_image = draw_rope_edges(vis_image, path_graph["edges"])
        elif len(result.rope_state.path) > 0:
            vis_image = draw_rope_path(vis_image, result.rope_state.path)

    if show_keypoints:
        vis_image = draw_keypoint_mask_overlay(
            vis_image,
            result.keypoint_mask,
        )

    # Draw crossing over/under indicators
    crossing_details = getattr(result.rope_state, "crossing_details", [])
    if crossing_details:
        vis_image = draw_crossing_over_under(vis_image, crossing_details)

    # Add processing info text
    info_text = (
        f"Frame: {result.frame_number} | "
        f"Time: {result.processing_time:.3f}s | "
        f"Endpoints: {len(result.rope_state.endpoints)} | "
        f"Crossings: {len(result.rope_state.crossings)}"
    )
    cv2.putText(
        vis_image,
        info_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
    )

    return vis_image


def display_result(
    result: ProcessingResult,
    window_name: str = "Rope Perception",
    show_mask: bool = True,
    show_keypoints: bool = True,
    show_path: bool = True,
) -> None:
    """Display processing result in a window.

    This function explicitly displays images, which is only called
    when visualization is requested.

    Args:
        result: ProcessingResult object
        window_name: Name of display window
        show_mask: Whether to overlay segmentation mask
        show_keypoints: Whether to overlay keypoint mask
        show_path: Whether to draw rope path
    """
    vis_image = visualize_result(
        result,
        show_mask=show_mask,
        show_keypoints=show_keypoints,
        show_path=show_path,
    )

    cv2.imshow(window_name, vis_image)
