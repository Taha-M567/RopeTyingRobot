"""Reusable perception-pipeline runner for Isaac Lab camera frames.

Wraps the OpenCV stages from ``src/perception/`` so both
``so100_sandbox.py`` and ``play_with_perception.py`` can share a single
implementation.  Pure Python / numpy — no Isaac Lab import side effects,
so this module is safe to import before ``AppLauncher`` is called.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from src.perception.keypoint_detection import detect_keypoints
from src.perception.keypoint_mask import create_keypoint_class_mask
from src.perception.rope_segmentation import segment_rope
from src.perception.skeletonization import extract_path, skeletonize_rope
from src.perception.state_estimation import RopeState, estimate_rope_state

logger = logging.getLogger(__name__)


@dataclass
class PerceptionOutput:
    """Result of running the perception pipeline on one frame."""

    rope_state: RopeState
    overlay: np.ndarray
    metrics: dict[str, Any]


def tensor_rgb_to_bgr(rgb_tensor: "Any") -> np.ndarray:
    """Convert an Isaac Lab camera RGB(A) tensor to an OpenCV BGR uint8 image.

    Accepts torch tensors with shape ``(H, W, 3)`` or ``(H, W, 4)`` and
    either ``uint8`` or floating-point dtypes (values in ``[0, 1]`` or
    ``[0, 255]``).
    """
    rgb_np = rgb_tensor.detach().cpu().numpy()
    if rgb_np.shape[-1] == 4:
        rgb_np = rgb_np[..., :3]
    if rgb_np.dtype != np.uint8:
        max_value = float(np.max(rgb_np)) if rgb_np.size > 0 else 0.0
        if max_value <= 1.0:
            rgb_np = np.clip(rgb_np * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            rgb_np = np.clip(rgb_np, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)


def _draw_overlay(
    frame_bgr: np.ndarray,
    rope_mask: np.ndarray,
    keypoints: list[Any],
    path_graph: dict[str, Any] | None,
    fallback_path: np.ndarray,
) -> np.ndarray:
    """Draw segmentation, keypoints, and path information on a copy of the frame."""
    vis = frame_bgr.copy()

    mask_region = rope_mask > 0
    if np.any(mask_region):
        overlay = vis.copy()
        overlay[mask_region] = (0, 120, 255)
        vis = cv2.addWeighted(vis, 0.75, overlay, 0.25, 0.0)

    for kp in keypoints:
        x = int(round(kp.position[0]))
        y = int(round(kp.position[1]))
        color = (0, 255, 0) if kp.keypoint_type == "endpoint" else (0, 0, 255)
        radius = 4 if kp.keypoint_type == "endpoint" else 6
        cv2.circle(vis, (x, y), radius, color, -1)

    edges: list[np.ndarray] = []
    if path_graph is not None:
        edges = path_graph.get("edges", [])
    if edges:
        for edge in edges:
            if edge is None or len(edge) < 2:
                continue
            points = edge.astype(np.int32)
            for idx in range(len(points) - 1):
                cv2.line(
                    vis,
                    tuple(points[idx]),
                    tuple(points[idx + 1]),
                    (255, 50, 50),
                    2,
                )
    elif fallback_path is not None and len(fallback_path) >= 2:
        points = fallback_path.astype(np.int32)
        for idx in range(len(points) - 1):
            cv2.line(
                vis,
                tuple(points[idx]),
                tuple(points[idx + 1]),
                (255, 50, 50),
                2,
            )

    return vis


def run_perception(
    frame_bgr: np.ndarray,
    perception_cfg: dict[str, Any],
) -> PerceptionOutput:
    """Run segmentation → keypoints → skeleton → state-estimation on one frame.

    Args:
        frame_bgr: BGR image (H, W, 3) uint8.
        perception_cfg: ``perception`` block from ``perception_config.yaml``.
    """
    pipeline_cfg = perception_cfg.get("pipeline", {})
    segmentation_cfg = perception_cfg.get("segmentation", {})
    keypoint_cfg = perception_cfg.get("keypoint_detection", {})
    keypoint_mask_cfg = perception_cfg.get("keypoint_mask", {})
    skeleton_cfg = perception_cfg.get("skeletonization", {})

    disable_keypoints = bool(pipeline_cfg.get("disable_keypoint_extraction", False))
    disable_skeleton = bool(pipeline_cfg.get("disable_skeletonization", False))

    rope_mask_obj = segment_rope(frame_bgr, config=segmentation_cfg)
    rope_mask = rope_mask_obj.mask

    if disable_keypoints:
        keypoints: list[Any] = []
    else:
        keypoints = detect_keypoints(rope_mask, config=keypoint_cfg)

    _ = create_keypoint_class_mask(rope_mask, keypoints, config=keypoint_mask_cfg)

    if disable_skeleton:
        path = np.array([], dtype=np.float32).reshape(0, 2)
    else:
        skeleton = skeletonize_rope(rope_mask, config=skeleton_cfg)
        path = extract_path(skeleton)

    rope_state = estimate_rope_state(keypoints, path)

    overlay = _draw_overlay(
        frame_bgr=frame_bgr,
        rope_mask=rope_mask,
        keypoints=keypoints,
        path_graph=rope_state.path_graph,
        fallback_path=rope_state.path,
    )

    metrics = {
        "confidence": float(rope_mask_obj.confidence),
        "endpoints": len(rope_state.endpoints),
        "crossings": len(rope_state.crossings),
        "path_points": int(len(rope_state.path)),
    }

    return PerceptionOutput(
        rope_state=rope_state,
        overlay=overlay,
        metrics=metrics,
    )


def select_two_endpoints(
    keypoints: list[Any],
    endpoints_xy: list[tuple[float, float]],
    fallback: np.ndarray | None,
) -> np.ndarray:
    """Pick the two highest-confidence endpoints; fall back when missing.

    Args:
        keypoints: full list returned by ``detect_keypoints`` (used to
            look up per-endpoint confidence).
        endpoints_xy: the pixel-coordinate list inside ``RopeState``.
        fallback: ``(2, 2)`` array of last-known endpoint pixels, or
            ``None`` to fail when fewer than two endpoints are detected.

    Returns:
        ``(2, 2)`` array of pixel coordinates ordered by descending
        confidence.  Raises ``ValueError`` when there are fewer than
        two endpoints AND no fallback is supplied.
    """
    endpoint_kps = [kp for kp in keypoints if kp.keypoint_type == "endpoint"]
    endpoint_kps.sort(key=lambda kp: kp.confidence, reverse=True)

    if len(endpoint_kps) >= 2:
        return np.array(
            [endpoint_kps[0].position, endpoint_kps[1].position],
            dtype=np.float64,
        )

    if fallback is not None:
        if len(endpoint_kps) == 1:
            # Use the fresh detection plus the older second endpoint.
            return np.array(
                [endpoint_kps[0].position, fallback[1]],
                dtype=np.float64,
            )
        return fallback.astype(np.float64)

    # Backstop: if endpoints_xy has anything, use what we can.
    if len(endpoints_xy) >= 2:
        return np.array(endpoints_xy[:2], dtype=np.float64)

    raise ValueError(
        f"Need 2 rope endpoints, got {len(endpoint_kps)} with no fallback."
    )
