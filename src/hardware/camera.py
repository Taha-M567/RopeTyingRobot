"""Camera interface for real hardware.

This module provides camera capture functionality with calibration.
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class CameraCalibration:
    """Camera calibration parameters.

    Attributes:
        camera_matrix: 3x3 camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        image_size: (width, height) of images
    """

    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: tuple


class Camera:
    """Interface for camera hardware."""

    def __init__(
        self,
        camera_id: int,
        calibration: Optional[CameraCalibration] = None,
    ):
        """Initialize camera.

        Args:
            camera_id: Camera device ID
            calibration: Optional camera calibration parameters
        """
        self.camera_id = camera_id
        self.calibration = calibration
        self.cap = None

    def connect(self) -> None:
        """Connect to camera hardware."""
        # TODO: Implement camera connection
        self.cap = cv2.VideoCapture(self.camera_id)

    def disconnect(self) -> None:
        """Disconnect from camera."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def capture(self) -> np.ndarray:
        """Capture a single frame.

        Returns:
            Image as numpy array (BGR format)

        Raises:
            RuntimeError: If camera is not connected
        """
        if self.cap is None:
            raise RuntimeError("Camera not connected")

        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame")

        # Apply undistortion if calibration available
        if self.calibration is not None:
            frame = cv2.undistort(
                frame,
                self.calibration.camera_matrix,
                self.calibration.dist_coeffs,
            )

        return frame

    def load_calibration(self, config_path: str) -> None:
        """Load camera calibration from config file.

        Args:
            config_path: Path to calibration config file
        """
        # TODO: Load calibration from YAML/JSON
        pass
