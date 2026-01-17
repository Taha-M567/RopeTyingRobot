"""Visualization utilities for perception results.

This module provides functions to visualize rope perception results.
Note: Display functions are only called when explicitly requested.
"""

from typing import List, Optional

import cv2
import numpy as np

from src.perception.keypoint_detection import Keypoint
from src.perception.rope_segmentation import RopeMask
from src.perception.state_estimation import RopeState
from src.perception.video_processor import ProcessingResult


def draw_rope_mask(
    image: np.ndarray,
    mask: RopeMask,
    alpha: float = 0.5,
) -> np.ndarray:
    """Draw rope mask overlay on image.

    Args:
        image: Original image (BGR format)
        mask: RopeMask object
        alpha: Transparency of mask overlay (0.0 to 1.0)

    Returns:
        Image with mask overlay
    """
    overlay = image.copy()
    mask_colored = cv2.applyColorMap(mask.mask, cv2.COLORMAP_JET)
    mask_colored = cv2.bitwise_and(mask_colored, mask_colored, mask=mask.mask)

    # Blend mask with original image
    result = cv2.addWeighted(overlay, 1.0 - alpha, mask_colored, alpha, 0)
    return result


def draw_keypoints(
    image: np.ndarray,
    keypoints: List[Keypoint],
    radius: int = 5,
) -> np.ndarray:
    """Draw keypoints on image.

    Args:
        image: Image to draw on (BGR format)
        keypoints: List of Keypoint objects
        radius: Radius of keypoint circles

    Returns:
        Image with keypoints drawn
    """
    result = image.copy()

    # Color mapping for keypoint types
    colors = {
        "endpoint": (0, 255, 0),  # Green
        "crossing": (0, 0, 255),  # Red
        "knot": (255, 0, 255),  # Magenta
    }

    for kp in keypoints:
        x, y = int(kp.position[0]), int(kp.position[1])
        color = colors.get(kp.keypoint_type, (255, 255, 255))

        # Draw circle
        cv2.circle(result, (x, y), radius, color, -1)

        # Draw label
        label = f"{kp.keypoint_type}: {kp.confidence:.2f}"
        cv2.putText(
            result,
            label,
            (x + radius + 2, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
        )

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
        show_keypoints: Whether to draw keypoints
        show_path: Whether to draw rope path

    Returns:
        Visualized image
    """
    vis_image = result.frame.copy()

    if show_mask:
        vis_image = draw_rope_mask(vis_image, result.rope_mask)

    if show_path and len(result.rope_state.path) > 0:
        vis_image = draw_rope_path(vis_image, result.rope_state.path)

    if show_keypoints:
        vis_image = draw_keypoints(vis_image, result.keypoints)

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
        0.7,
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
        show_keypoints: Whether to draw keypoints
        show_path: Whether to draw rope path
    """
    vis_image = visualize_result(
        result,
        show_mask=show_mask,
        show_keypoints=show_keypoints,
        show_path=show_path,
    )

    cv2.imshow(window_name, vis_image)
