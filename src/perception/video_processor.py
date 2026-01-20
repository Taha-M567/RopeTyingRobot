"""Live video processing for rope perception.

This module processes live video streams through the perception pipeline.
"""

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Callable, Optional

import cv2
import numpy as np

from src.hardware.camera import Camera
from src.perception.keypoint_detection import Keypoint, detect_keypoints
from src.perception.rope_segmentation import RopeMask, segment_rope
from src.perception.skeletonization import extract_path, skeletonize_rope
from src.perception.state_estimation import RopeState, estimate_rope_state
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result of processing a single video frame.

    Attributes:
        frame: Original frame (BGR format)
        rope_mask: Segmentation result
        keypoints: Detected keypoints
        rope_state: Estimated rope state
        processing_time: Time taken to process frame (seconds)
        frame_number: Sequential frame number
    """

    frame: np.ndarray
    rope_mask: RopeMask
    keypoints: list[Keypoint]
    rope_state: RopeState
    processing_time: float
    frame_number: int


class LiveVideoProcessor:
    """Process live video streams through the perception pipeline.

    This class captures frames from a camera and processes them through:
    1. Rope segmentation
    2. Keypoint detection
    3. Skeletonization
    4. State estimation

    Uses a separate thread for frame capture to maintain real-time performance.
    """

    def __init__(
        self,
        camera: Camera,
        perception_config: dict,
        callback: Optional[Callable[[ProcessingResult], None]] = None,
    ):
        """Initialize live video processor.

        Args:
            camera: Camera object for frame capture
            perception_config: Configuration dictionary for perception
            callback: Optional callback function called with each result
        """
        self.camera = camera
        self.perception_config = perception_config
        self.callback = callback

        self.is_running = False
        self.frame_queue: Queue[tuple[np.ndarray, int]] = Queue(maxsize=2)
        self.capture_thread: Optional[threading.Thread] = None
        self.frame_number = 0

    def start(self) -> None:
        """Start video processing.

        Raises:
            RuntimeError: If camera is not connected
        """
        if self.camera.cap is None:
            raise RuntimeError("Camera not connected. Call camera.connect() first.")

        if self.is_running:
            logger.warning("Video processor already running")
            return

        self.is_running = True
        self.frame_number = 0

        # Start capture thread
        self.capture_thread = threading.Thread(
            target=self._capture_frames,
            daemon=True,
        )
        self.capture_thread.start()

        logger.info("Live video processing started")

    def stop(self) -> None:
        """Stop video processing."""
        if not self.is_running:
            return

        self.is_running = False

        # Wait for capture thread to finish
        if self.capture_thread is not None:
            self.capture_thread.join(timeout=2.0)

        # Clear frame queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Exception:
                pass

        logger.info("Live video processing stopped")

    def _capture_frames(self) -> None:
        """Capture frames in a separate thread.

        This method runs in a background thread to maintain real-time
        frame capture while processing happens on the main thread.
        """
        while self.is_running:
            try:
                frame = self.camera.capture()
                self.frame_number += 1

                # Add frame to queue (non-blocking, drop if queue full)
                try:
                    self.frame_queue.put_nowait((frame, self.frame_number))
                except Exception:
                    # Queue full, drop frame to maintain real-time performance
                    logger.debug("Frame queue full, dropping frame")

            except RuntimeError as e:
                logger.error(f"Failed to capture frame: {e}")
                time.sleep(0.1)  # Brief pause before retry
            except Exception as e:
                logger.error(f"Unexpected error in capture thread: {e}")
                break

    def process_next_frame(self) -> Optional[ProcessingResult]:
        """Process the next available frame.

        Returns:
            ProcessingResult if frame available, None otherwise
        """
        if not self.is_running:
            return None

        # Get frame from queue (non-blocking)
        try:
            frame, frame_num = self.frame_queue.get_nowait()
        except Exception:
            return None

        # Process frame through perception pipeline
        start_time = time.time()

        try:
            # 1. Segment rope
            # Apply box filter to reduce noise
            frame = cv2.boxFilter(frame, -1, (4,4) , normalize=True)
            rope_mask = segment_rope(
                frame,
                config=self.perception_config.get("segmentation"),
            )

            # 2. Detect keypoints
            keypoints = detect_keypoints(
                rope_mask.mask,
                config=self.perception_config.get("keypoint_detection", {}),
            )

            # 3. Skeletonize
            skeleton = skeletonize_rope(
                rope_mask.mask,
                config=self.perception_config.get("skeletonization", {}),
            )

            # 4. Extract path
            path = extract_path(skeleton)

            # 5. Estimate state
            rope_state = estimate_rope_state(keypoints, path)

            processing_time = time.time() - start_time

            result = ProcessingResult(
                frame=frame,
                rope_mask=rope_mask,
                keypoints=keypoints,
                rope_state=rope_state,
                processing_time=processing_time,
                frame_number=frame_num,
            )

            # Call callback if provided
            if self.callback is not None:
                self.callback(result)

            return result

        except Exception as e:
            logger.error(f"Error processing frame {frame_num}: {e}")
            return None

    def run_continuous(
        self,
        max_frames: Optional[int] = None,
        target_fps: Optional[float] = None,
        check_quit_key: bool = False,
    ) -> None:
        """Run continuous processing loop.

        Args:
            max_frames: Maximum number of frames to process (None for unlimited)
            target_fps: Target processing rate in frames per second
            check_quit_key: If True, check for 'q' key press to quit
        """
        if not self.is_running:
            self.start()

        frame_count = 0
        frame_interval = 1.0 / target_fps if target_fps else 0.0
        last_frame_time = time.time()

        logger.info("Starting continuous processing loop")

        try:
            while self.is_running:
                # Check for quit key (if visualization is being used)
                if check_quit_key:
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("Quit key pressed")
                        break

                # Rate limiting
                if target_fps:
                    elapsed = time.time() - last_frame_time
                    if elapsed < frame_interval:
                        time.sleep(frame_interval - elapsed)
                    last_frame_time = time.time()

                # Process frame
                result = self.process_next_frame()

                if result is not None:
                    frame_count += 1
                    logger.debug(
                        f"Processed frame {result.frame_number} "
                        f"in {result.processing_time:.3f}s"
                    )

                    # Check max frames limit
                    if max_frames and frame_count >= max_frames:
                        logger.info(f"Reached max frames limit: {max_frames}")
                        break

                # Small sleep to prevent busy waiting
                if result is None:
                    time.sleep(0.01)

        except KeyboardInterrupt:
            logger.info("Processing interrupted by user")
        finally:
            self.stop()


def create_processor_from_config(
    camera_id: int,
    config_path: Path,
    callback: Optional[Callable[[ProcessingResult], None]] = None,
) -> LiveVideoProcessor:
    """Create a LiveVideoProcessor from configuration files.

    Args:
        camera_id: Camera device ID
        config_path: Path to perception config YAML file
        callback: Optional callback function for results

    Returns:
        Configured LiveVideoProcessor instance
    """
    # Load configuration
    config = load_config(config_path)
    perception_config = config.get("perception", {})

    # Create camera
    camera = Camera(camera_id)
    camera.connect()

    # Create processor
    processor = LiveVideoProcessor(
        camera=camera,
        perception_config=perception_config,
        callback=callback,
    )

    return processor
