"""Visualization utilities for perception results.

This module provides functions to visualize rope perception results.
Note: Display functions are only called when explicitly requested.
"""

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
    mask_region = (class_mask != BACKGROUND_CLASS).astype(np.uint8) * 255
    mask_region = cv2.cvtColor(mask_region, cv2.COLOR_GRAY2BGR)
    overlay = cv2.bitwise_and(color_mask, mask_region)

    return cv2.addWeighted(image, 1.0 - alpha, overlay, alpha, 0)


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
        show_keypoints: Whether to overlay keypoint mask
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
        vis_image = draw_keypoint_mask_overlay(
            vis_image,
            result.keypoint_mask,
        )

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
