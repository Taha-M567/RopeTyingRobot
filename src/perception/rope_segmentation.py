"""Rope segmentation using OpenCV.

This module provides functions for segmenting ropes from images.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class RopeMask:
    """Data structure for rope segmentation results.

    Attributes:
        mask: Binary mask of the rope (numpy array, dtype=uint8)
        confidence: Confidence score for the segmentation (0.0 to 1.0)
        image_shape: Original image shape (height, width)
    """

    mask: np.ndarray
    confidence: float
    image_shape: Tuple[int, int]


def _preprocess_image(
    image: np.ndarray,
    blur_kernel_size: int = 5,
) -> np.ndarray:
    """Preprocess image with blur and noise reduction.

    Args:
        image: Input image (BGR or grayscale)
        blur_kernel_size: Size of Gaussian blur kernel (must be odd)

    Returns:
        Preprocessed image
    """
    # Ensure kernel size is odd
    if blur_kernel_size % 2 == 0:
        blur_kernel_size += 1

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(
        image,
        (blur_kernel_size, blur_kernel_size),
        0,
    )

    return blurred


def _apply_morphology(
    mask: np.ndarray,
    opening_kernel_size: int = 3,
    closing_kernel_size: int = 5,
) -> np.ndarray:
    """Apply morphological operations to clean mask.

    Args:
        mask: Binary mask
        opening_kernel_size: Size of opening kernel (removes noise)
        closing_kernel_size: Size of closing kernel (fills gaps)

    Returns:
        Cleaned binary mask
    """
    # Create kernels
    opening_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (opening_kernel_size, opening_kernel_size),
    )
    closing_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (closing_kernel_size, closing_kernel_size),
    )

    # Apply opening (erosion then dilation) to remove noise
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, opening_kernel)

    # Apply closing (dilation then erosion) to fill gaps
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, closing_kernel)

    return closed


def _filter_contours(
    mask: np.ndarray,
    min_area: int = 100,
    max_area: Optional[int] = None,
    min_aspect_ratio: float = 2.0,
    hole_fill_max_area: Optional[int] = None,
    hollow_fill_ratio_threshold: Optional[float] = None,
) -> np.ndarray:
    """Filter contours to keep only rope-like shapes.

    Args:
        mask: Binary mask
        min_area: Minimum contour area in pixels
        max_area: Maximum contour area in pixels (None for no limit)
        min_aspect_ratio: Minimum aspect ratio (length/width)
        hole_fill_max_area: Fill holes up to this area (pixels); larger holes
            (e.g., rope loops) are preserved. None disables hole filling.
        hollow_fill_ratio_threshold: If set, treat contours with fill ratios
            below this value as hollow (loop-like) and skip aspect ratio checks.

    Returns:
        Filtered binary mask
    """
    # Find contours
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return mask

    # Start from the original mask so holes (e.g., rope loops) are preserved.
    filtered_mask = mask.copy()

    for contour in contours:
        area = cv2.contourArea(contour)

        # Check area
        if area < min_area:
            cv2.drawContours(filtered_mask, [contour], -1, 0, -1)
            continue
        if max_area is not None and area > max_area:
            cv2.drawContours(filtered_mask, [contour], -1, 0, -1)
            continue

        is_hollow = False
        if hollow_fill_ratio_threshold is not None and hollow_fill_ratio_threshold > 0:
            contour_mask = np.zeros_like(mask, dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 255, -1)
            contour_area_pixels = int(np.count_nonzero(contour_mask))
            if contour_area_pixels > 0:
                rope_area_pixels = int(
                    np.count_nonzero(cv2.bitwise_and(mask, contour_mask))
                )
                fill_ratio = rope_area_pixels / contour_area_pixels
                is_hollow = fill_ratio < hollow_fill_ratio_threshold

        # Check aspect ratio (skip for hollow/loop-like contours)
        if not is_hollow and len(contour) >= 5:  # Need at least 5 points for fitEllipse
            try:
                (_, _), (w, h), _ = cv2.fitEllipse(contour)
                aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
                if aspect_ratio < min_aspect_ratio:
                    cv2.drawContours(filtered_mask, [contour], -1, 0, -1)
            except Exception:
                # If fitEllipse fails, use bounding box
                _, _, w, h = cv2.boundingRect(contour)
                aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
                if aspect_ratio < min_aspect_ratio:
                    cv2.drawContours(filtered_mask, [contour], -1, 0, -1)

    if hole_fill_max_area is not None and hole_fill_max_area > 0:
        # Fill only small holes to reduce skeleton noise while preserving
        # large loop interiors.
        holes = cv2.bitwise_not(filtered_mask)
        hole_contours, _ = cv2.findContours(
            holes,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        for hole_contour in hole_contours:
            area = cv2.contourArea(hole_contour)
            if area <= hole_fill_max_area:
                cv2.drawContours(filtered_mask, [hole_contour], -1, 255, -1)

    return filtered_mask


def _calculate_confidence(
    mask: np.ndarray,
    image_shape: Tuple[int, int],
    method_agreement: Optional[float] = None,
) -> float:
    """Calculate confidence score for segmentation.

    Args:
        mask: Binary mask
        image_shape: Original image shape (height, width)
        method_agreement: Agreement between methods (0.0 to 1.0) if combined

    Returns:
        Confidence score (0.0 to 1.0)
    """
    height, width = image_shape
    total_pixels = height * width

    # Calculate mask coverage ratio
    mask_pixels = np.count_nonzero(mask)
    coverage_ratio = mask_pixels / total_pixels if total_pixels > 0 else 0.0

    # Coverage should be reasonable (not too small, not too large)
    # Typical rope might be 1-10% of image
    if coverage_ratio < 0.001 or coverage_ratio > 0.3:
        coverage_score = 0.0
    elif coverage_ratio < 0.01:
        coverage_score = coverage_ratio * 50  # Scale up small coverage
    else:
        coverage_score = min(1.0, 1.0 - (coverage_ratio - 0.01) * 5)

    # Contour quality score
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        contour_score = 0.0
    else:
        # Prefer fewer, larger contours (single rope)
        largest_contour = max(contours, key=cv2.contourArea)
        largest_area = cv2.contourArea(largest_contour)
        total_contour_area = sum(cv2.contourArea(c) for c in contours)

        if total_contour_area > 0:
            # Score based on how much of mask is in largest contour
            contour_score = largest_area / total_contour_area
        else:
            contour_score = 0.0

    # Combine scores
    base_confidence = (coverage_score * 0.6 + contour_score * 0.4)

    # Boost confidence if methods agree (for combined method)
    if method_agreement is not None:
        base_confidence = base_confidence * 0.7 + method_agreement * 0.3

    return min(1.0, max(0.0, base_confidence))


def _segment_by_color(
    image: np.ndarray,
    config: dict,
) -> np.ndarray:
    """Segment rope using color thresholding.

    Args:
        image: Input image in BGR format
        config: Configuration dictionary with segmentation parameters

    Returns:
        Binary mask of segmented rope
    """
    # Get config parameters with defaults
    blur_kernel_size = config.get("blur_kernel_size", 5)
    morph_ops = config.get("morph_operations", {})
    opening_size = morph_ops.get("opening_kernel_size", 3)
    closing_size = morph_ops.get("closing_kernel_size", 5)
    contour_filter = config.get("contour_filter", {})
    min_area = contour_filter.get("min_area", 100)
    min_aspect_ratio = contour_filter.get("min_aspect_ratio", 2.0)
    hole_fill_max_area = contour_filter.get("hole_fill_max_area", None)
    hollow_fill_ratio_threshold = contour_filter.get(
        "hollow_fill_ratio_threshold",
        None,
    )

    # Convert to HSV for better color separation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Get color range (default to black rope on wood)
    color_range = config.get("color_range", {})
    hsv_lower = np.array(color_range.get("hsv_lower", [0, 0, 0]))
    hsv_upper = np.array(color_range.get("hsv_upper", [180, 50, 80]))

    # Preprocess
    blurred = _preprocess_image(hsv, blur_kernel_size)

    # Create mask using color threshold
    mask = cv2.inRange(blurred, hsv_lower, hsv_upper)

    # Apply morphological operations
    mask = _apply_morphology(mask, opening_size, closing_size)

    # Filter contours
    mask = _filter_contours(
        mask,
        min_area,
        None,
        min_aspect_ratio,
        hole_fill_max_area,
        hollow_fill_ratio_threshold,
    )

    return mask


def _segment_by_edges(
    image: np.ndarray,
    config: dict,
) -> np.ndarray:
    """Segment rope using edge detection.

    Args:
        image: Input image in BGR format
        config: Configuration dictionary with segmentation parameters

    Returns:
        Binary mask of segmented rope
    """
    # Get config parameters with defaults
    blur_kernel_size = config.get("blur_kernel_size", 5)
    edge_detection = config.get("edge_detection", {})
    canny_low = edge_detection.get("canny_low_threshold", 50)
    canny_high = edge_detection.get("canny_high_threshold", 150)
    morph_ops = config.get("morph_operations", {})
    closing_size = morph_ops.get("closing_kernel_size", 5)
    contour_filter = config.get("contour_filter", {})
    min_area = contour_filter.get("min_area", 100)
    min_aspect_ratio = contour_filter.get("min_aspect_ratio", 2.0)
    hole_fill_max_area = contour_filter.get("hole_fill_max_area", None)
    hollow_fill_ratio_threshold = contour_filter.get(
        "hollow_fill_ratio_threshold",
        None,
    )

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Preprocess
    blurred = _preprocess_image(gray, blur_kernel_size)

    # Canny edge detection
    edges = cv2.Canny(blurred, canny_low, canny_high)

    # Connect edges using morphological closing
    closing_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (closing_size, closing_size),
    )
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, closing_kernel)

    # Fill enclosed regions
    contours, _ = cv2.findContours(
        closed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    mask = np.zeros_like(gray)
    if contours:
        # Fill all contours
        cv2.drawContours(mask, contours, -1, 255, -1)

    # Filter contours
    mask = _filter_contours(
        mask,
        min_area,
        None,
        min_aspect_ratio,
        hole_fill_max_area,
        hollow_fill_ratio_threshold,
    )

    return mask


def _segment_combined(
    image: np.ndarray,
    config: dict,
) -> tuple[np.ndarray, float]:
    """Segment rope using combined color and edge methods.

    Args:
        image: Input image in BGR format
        config: Configuration dictionary with segmentation parameters

    Returns:
        Tuple of (binary mask, method agreement score)
    """
    # Run both methods
    color_mask = _segment_by_color(image, config)
    edge_mask = _segment_by_edges(image, config)

    # Combine masks using logical OR
    combined_mask = cv2.bitwise_or(color_mask, edge_mask)

    # Calculate agreement (intersection over union of non-zero regions)
    intersection = cv2.bitwise_and(color_mask, edge_mask)
    union = cv2.bitwise_or(color_mask, edge_mask)

    intersection_pixels = np.count_nonzero(intersection)
    union_pixels = np.count_nonzero(union)

    if union_pixels > 0:
        agreement = intersection_pixels / union_pixels
    else:
        agreement = 0.0

    # Apply additional morphological operations
    morph_ops = config.get("morph_operations", {})
    closing_size = morph_ops.get("closing_kernel_size", 5)
    combined_mask = _apply_morphology(
        combined_mask,
        opening_kernel_size=3,
        closing_kernel_size=closing_size,
    )

    return combined_mask, agreement


def segment_rope(
    image: np.ndarray,
    config: Optional[dict] = None,
) -> RopeMask:
    """Segment rope from an input image.

    Args:
        image: Input image as numpy array (BGR format)
        config: Optional configuration dictionary with segmentation parameters

    Returns:
        RopeMask object containing the segmentation mask and metadata

    Raises:
        ValueError: If image is empty or invalid
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or invalid")

    # Default config
    if config is None:
        config = {}

    # Get segmentation method (default to color_threshold)
    method = config.get("method", "color_threshold")

    # Get image shape
    height, width = image.shape[:2]
    image_shape = (height, width)

    # Dispatch to appropriate method
    method_agreement = None

    if method == "color_threshold":
        mask = _segment_by_color(image, config)
    elif method == "edge_detection":
        mask = _segment_by_edges(image, config)
    elif method == "combined":
        mask, method_agreement = _segment_combined(image, config)
    elif method == "adaptive":
        # Placeholder for future adaptive method
        # For now, fall back to color thresholding
        mask = _segment_by_color(image, config)
    else:
        # Unknown method, default to color thresholding
        mask = _segment_by_color(image, config)

    # Calculate confidence
    confidence = _calculate_confidence(mask, image_shape, method_agreement)

    return RopeMask(
        mask=mask,
        confidence=confidence,
        image_shape=image_shape,
    )
