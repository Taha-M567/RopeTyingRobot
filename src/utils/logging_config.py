"""Logging configuration for the project.

This module sets up consistent logging across all modules.
"""

import logging
from pathlib import Path
from typing import Optional


def setup_logging(
    log_dir: Optional[Path] = None,
    log_level: int = logging.INFO,
) -> None:
    """Configure logging for the project.

    Args:
        log_dir: Optional directory to save log files
        log_level: Logging level (default: INFO)
    """
    # Create log directory if specified
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "rope_tying_robot.log"
    else:
        log_file = None

    # Configure logging format
    log_format = (
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    date_format = "%Y-%m-%d %H:%M:%S"

    # Configure handlers
    handlers = [logging.StreamHandler()]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
    )
