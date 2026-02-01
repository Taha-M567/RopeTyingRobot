"""Example script for live video perception.

This script demonstrates how to use the LiveVideoProcessor for real-time
rope perception from a camera feed.
"""

import logging
import sys
from pathlib import Path

import cv2

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


from src.perception.video_processor import LiveVideoProcessor, ProcessingResult, create_processor_from_config

from src.perception.visualization import display_result
from src.hardware.camera import Camera
from src.utils.logging_config import setup_logging


# Setup logging
setup_logging(log_level=logging.INFO)
logger = logging.getLogger(__name__)


def result_callback(result: ProcessingResult) -> None:
    """Callback function called for each processed frame.

    Args:
        result: Processing result for the frame
    """
    # Log key information
    logger.info(
        f"Frame {result.frame_number}: "
        f"{len(result.rope_state.endpoints)} endpoints, "
        f"{len(result.rope_state.crossings)} crossings, "
        f"processed in {result.processing_time:.3f}s"
    )

    # Display result (optional - comment out if no display available)
    display_result(result, window_name="Live Rope Perception")


def main() -> None:
    """Main function for live perception demo."""
    # Configuration
    camera_id = 0  # Change to your camera ID if needed
    config_path = Path("src/configs/perception_config.yaml")

    logger.info("Starting live perception demo")

    # Check if config file exists
    if not config_path.exists():
        logger.warning(
            f"Config file not found: {config_path}. "
            "Using default empty config."
        )
        perception_config = {}
    else:
        try:
            from src.utils.config_loader import load_config
            config = load_config(config_path)
            perception_config = config.get("perception", {})
        except ImportError:
            logger.warning(
                "PyYAML not installed. Using default empty config. "
                "Install with: pip install PyYAML"
            )
            perception_config = {}

    try:
        # Create camera
        logger.info(f"Connecting to camera {camera_id}...")
        camera = Camera(camera_id)
        camera.connect()

        # Verify camera works
        try:
            test_frame = camera.capture()
            logger.info(
                f"Camera connected successfully. "
                f"Frame size: {test_frame.shape}"
            )
        except Exception as e:
            logger.error(f"Failed to capture test frame: {e}")
            logger.error(
                "Make sure your camera is connected and not being used "
                "by another application."
            )
            camera.disconnect()
            return

        # Create processor
        processor = LiveVideoProcessor(
            camera=camera,
            perception_config=perception_config,
            callback=result_callback,
        )

        # Run continuous processing
        # Press 'q' to quit
        logger.info("Press 'q' to quit")
        processor.run_continuous(
            target_fps=1,  # Process at 10 FPS
            check_quit_key=True,  # Check for 'q' key to quit
        )


    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Error in demo: {e}", exc_info=True)
    finally:
        # Cleanup
        cv2.destroyAllWindows()
        if "processor" in locals():
            processor.stop()
        if "camera" in locals():
            camera.disconnect()
        logger.info("Demo finished")


if __name__ == "__main__":
    main()
