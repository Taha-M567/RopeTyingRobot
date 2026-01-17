"""Perception module for rope detection, segmentation, and tracking.

This module handles all computer vision tasks using OpenCV:
- Rope segmentation
- Keypoint extraction
- Skeletonization
- State estimation (crossings, endpoints)
- Live video processing
"""

# Import core classes and functions
from src.perception.keypoint_detection import Keypoint, detect_keypoints
from src.perception.rope_segmentation import RopeMask, segment_rope
from src.perception.skeletonization import extract_path, skeletonize_rope
from src.perception.state_estimation import RopeState, estimate_rope_state

# Lazy imports for video processing to avoid circular dependencies
# These can be imported directly when needed:
# from src.perception.video_processor import LiveVideoProcessor
# from src.perception.visualization import display_result

__all__ = [
    "Keypoint",
    "RopeMask",
    "RopeState",
    "segment_rope",
    "detect_keypoints",
    "skeletonize_rope",
    "extract_path",
    "estimate_rope_state",
]
