"""Chessboard-based intrinsic calibration for the deployment camera.

Captures *N* images of a printed chessboard from the production camera,
runs OpenCV's standard calibration, and writes the resulting intrinsic
matrix + distortion coefficients back into ``camera_config.yaml``.

This script does *not* touch ``extrinsics`` — measure those by hand or
with a known fiducial and edit the YAML separately.

Usage::

    python scripts/calibrate_camera.py \\
        --camera-id 0 --rows 6 --cols 9 --square-size 0.025 \\
        --num-frames 25 \\
        --config src/configs/camera_config.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)


def _build_object_points(rows: int, cols: int, square_size: float) -> np.ndarray:
    """Return ``(rows*cols, 3)`` of (X, Y, 0) chessboard corner positions."""
    pts = np.zeros((rows * cols, 3), dtype=np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    pts[:, :2] = grid * square_size
    return pts


def _capture_calibration_frames(
    camera_id: int,
    rows: int,
    cols: int,
    num_frames: int,
) -> tuple[list[np.ndarray], list[np.ndarray], tuple[int, int]]:
    """Interactively capture chessboard corners; SPACE to keep, q to quit."""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}")

    obj_pts: list[np.ndarray] = []
    img_pts: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    pattern_size = (cols, rows)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        1e-3,
    )

    obj_template = _build_object_points(rows, cols, square_size=1.0)

    try:
        while len(obj_pts) < num_frames:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Frame grab failed; retrying...")
                time.sleep(0.05)
                continue
            if image_size is None:
                image_size = (frame.shape[1], frame.shape[0])
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, pattern_size)

            display = frame.copy()
            if found:
                refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), criteria,
                )
                cv2.drawChessboardCorners(display, pattern_size, refined, found)
                cv2.putText(
                    display,
                    f"SPACE to capture ({len(obj_pts)}/{num_frames})",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
            else:
                cv2.putText(
                    display,
                    "Chessboard not detected",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("calibrate_camera", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" ") and found:
                img_pts.append(refined)
                obj_pts.append(obj_template.copy())
                logger.info("Captured %d/%d", len(obj_pts), num_frames)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if image_size is None:
        raise RuntimeError("No frames captured.")
    return obj_pts, img_pts, image_size


def _scale_object_points(
    obj_pts: list[np.ndarray], square_size: float,
) -> list[np.ndarray]:
    return [pts * square_size for pts in obj_pts]


def _write_intrinsics_yaml(
    config_path: Path,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    image_size: tuple[int, int],
) -> None:
    """Merge intrinsics into the existing YAML, leaving other keys untouched."""
    if config_path.exists():
        with open(config_path) as fh:
            data = yaml.safe_load(fh) or {}
    else:
        data = {}

    camera = data.setdefault("camera", {})
    camera["image_size"] = {"width": int(image_size[0]), "height": int(image_size[1])}
    calibration = camera.setdefault("calibration", {})
    calibration["camera_matrix"] = camera_matrix.tolist()
    calibration["dist_coeffs"] = dist_coeffs.flatten().tolist()

    with open(config_path, "w") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--rows", type=int, default=6, help="Inner chessboard rows.")
    parser.add_argument("--cols", type=int, default=9, help="Inner chessboard cols.")
    parser.add_argument(
        "--square-size", type=float, default=0.025,
        help="Side length of one chessboard square in metres.",
    )
    parser.add_argument("--num-frames", type=int, default=25)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/configs/camera_config.yaml"),
    )
    args = parser.parse_args()

    obj_pts, img_pts, image_size = _capture_calibration_frames(
        camera_id=args.camera_id,
        rows=args.rows,
        cols=args.cols,
        num_frames=args.num_frames,
    )
    if len(obj_pts) < 5:
        logger.error("Need at least 5 frames; got %d.", len(obj_pts))
        return 1

    obj_pts_scaled = _scale_object_points(obj_pts, args.square_size)
    rms, K, dist, _, _ = cv2.calibrateCamera(
        obj_pts_scaled, img_pts, image_size, None, None,
    )
    logger.info("Calibration RMS reprojection error: %.4f px", rms)
    logger.info("Camera matrix:\n%s", K)
    logger.info("Distortion:\n%s", dist.flatten())

    _write_intrinsics_yaml(args.config, K, dist, image_size)
    logger.info("Wrote intrinsics to %s", args.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
